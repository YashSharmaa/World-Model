# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The statistical-safety chance-constraint gate (doc M, ``risk_objective``).

Hard feasibility is owned by the exact kernel. *Statistical* safety — the
genuinely uncertain hazards the learned residual model forecasts (hotspot,
headroom violation, pods-ready timeout, requeue, XID/device fault) — is enforced
**here**, as a gate, not as a weighted reward term (doc M, T1; the project spine:
"safety is a gate, not a weighted metric").

A candidate survives only if, for *every* modeled hazard family ``ℓ``:

  * its **one-sided conformal upper bound** ``conformal_upper(ℓ)`` does not exceed
    the operator's configured budget ``δ_ℓ`` (doc M, E2 / doc L). We gate on the
    conformal upper bound, never the bare point estimate, because the point
    estimate can look safe on average while the tail is unsafe; and
  * (optionally) a **CVaR tail term** ``CVaR_α`` over the hazard-margin spread is
    within ``cvar_budget`` (doc M, E5; Rockafellar-Uryasev form). ``cvar_alpha=0``
    disables the tail gate (the v1 default).

Because VaR ≤ CVaR for non-negative violation margins, a CVaR gate implies a
lower bound on the probability of no violation (doc M, E1) — so the two gates are
coherent, not contradictory.

Dropping a candidate here is *not* a hard-infeasibility blocker; it is a
statistical-risk rejection, and the reason is recorded so :mod:`why_not` can
compose it into the user-facing card.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schema.kpis import HazardChannel, HazardForecast, KPIForecastBundle
from ..schema.policy import RiskThresholds

__all__ = [
    "HazardBudget",
    "RiskAssessment",
    "assess_risk",
    "conformal_upper_for",
    "cvar_from_margin",
    "hazard_budget_map",
    "passes_risk_gate",
]


# ──────────────────────────────────────────────────────────────────────────────
# Hazard → budget wiring
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class HazardBudget:
    """The configured one-sided budget ``δ_ℓ`` for one hazard family."""

    channel: HazardChannel
    budget: float


def hazard_budget_map(risk: RiskThresholds) -> dict[HazardChannel, float]:
    """Map each modeled hazard channel to its configured chance-constraint budget.

    Channels without an explicit operator budget (e.g. ``NODE_FAILURE``) inherit
    the most conservative configured XID-fault budget, since a node failure is at
    least as severe as a single device XID fault (doc U severity ordering).
    """
    return {
        HazardChannel.HOTSPOT: risk.max_hotspot_prob,
        HazardChannel.HEADROOM_VIOLATION: risk.max_headroom_violation_prob,
        HazardChannel.PODS_READY_TIMEOUT: risk.max_pods_ready_timeout_prob,
        HazardChannel.REQUEUE: risk.max_requeue_prob,
        HazardChannel.XID_FAULT: risk.max_xid_fault_prob,
        HazardChannel.NODE_FAILURE: risk.max_xid_fault_prob,
    }


def conformal_upper_for(
    bundle: KPIForecastBundle, channel: HazardChannel, horizon_s: int
) -> float | None:
    """Return the one-sided conformal upper bound for a hazard at a horizon.

    Falls back to the nearest forecast horizon for the channel; returns ``None``
    if the model published no forecast for the channel at all (an *absence* of a
    forecast is handled by the caller as conservatively unknown, not as zero).
    """
    best: HazardForecast | None = None
    for h in bundle.hazards:
        if h.channel is channel and h.horizon_s == horizon_s:
            return float(h.conformal_upper)
        if h.channel is channel and (
            best is None
            or abs(h.horizon_s - horizon_s) < abs(best.horizon_s - horizon_s)
        ):
            best = h
    return None if best is None else float(best.conformal_upper)


# ──────────────────────────────────────────────────────────────────────────────
# CVaR tail term (Rockafellar–Uryasev form, doc M E5)
# ──────────────────────────────────────────────────────────────────────────────
def cvar_from_margin(
    bundle: KPIForecastBundle, channel: HazardChannel, horizon_s: int, alpha: float
) -> float:
    """Estimate ``CVaR_alpha`` of a hazard's non-negative violation margin.

    We do not have full posterior samples in the bundle contract, so we form a
    conservative two-point margin from what the bundle exposes: the point hazard
    ``probability`` (a central estimate of the violation margin) and the conformal
    upper bound (a tail value). The Rockafellar-Uryasev CVaR of this non-negative
    margin reduces, for the worst tail mass ``(1 - alpha)``, to a convex blend
    that is monotone in the conformal gap — i.e. candidates with a wider
    point-to-conformal spread (more epistemic tail risk) score worse. With
    ``alpha → 1`` this collapses to the conformal upper bound (the worst case),
    matching ``VaR_α ≤ CVaR_α`` (doc M, E5).
    """
    if alpha <= 0.0:
        return 0.0
    point = 0.0
    upper = 0.0
    for h in bundle.hazards:
        if h.channel is channel and h.horizon_s == horizon_s:
            point = max(0.0, float(h.probability))
            upper = max(0.0, float(h.conformal_upper))
            break
    else:
        # Nearest horizon fallback.
        cands = [h for h in bundle.hazards if h.channel is channel]
        if cands:
            h = min(cands, key=lambda x: abs(x.horizon_s - horizon_s))
            point = max(0.0, float(h.probability))
            upper = max(0.0, float(h.conformal_upper))
    # Tail-weighted blend: as alpha grows the conformal (tail) value dominates.
    alpha = min(max(alpha, 0.0), 0.999999)
    return (1.0 - alpha) * point + alpha * upper


# ──────────────────────────────────────────────────────────────────────────────
# Per-candidate risk assessment
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class RiskAssessment:
    """The full statistical-safety verdict for one candidate at one horizon."""

    passes: bool
    horizon_s: int
    # per-channel: (conformal_upper, budget, exceeds?)
    chance_checks: dict[str, tuple[float, float, bool]] = field(default_factory=dict)
    cvar_checks: dict[str, tuple[float, float, bool]] = field(default_factory=dict)
    violated_channels: list[str] = field(default_factory=list)

    def reason(self) -> str:
        """A short human-readable rejection reason (empty if it passed)."""
        if self.passes:
            return ""
        parts = []
        for ch in self.violated_channels:
            if ch in self.chance_checks:
                up, bud, _ = self.chance_checks[ch]
                parts.append(f"{ch} conformal P̂≤{up:.3f} > budget {bud:.3f}")
            elif ch in self.cvar_checks:
                cv, bud, _ = self.cvar_checks[ch]
                parts.append(f"{ch} CVaR {cv:.3f} > budget {bud:.3f}")
        return "; ".join(parts)


def assess_risk(
    bundle: KPIForecastBundle, risk: RiskThresholds, horizon_s: int
) -> RiskAssessment:
    """Evaluate every chance constraint (+ optional CVaR tail) for a candidate.

    A *missing* forecast for a modeled hazard is treated conservatively: if the
    operator set a finite budget for that hazard and the model is silent, we do
    **not** silently pass it; we record an unknown-risk violation so the candidate
    is held out (better to abstain/fall back than to assume zero hazard). This is
    the gate's contribution to keeping the false-safe rate below 1% (doc F).
    """
    budgets = hazard_budget_map(risk)
    chance: dict[str, tuple[float, float, bool]] = {}
    cvar: dict[str, tuple[float, float, bool]] = {}
    violated: list[str] = []

    for channel, budget in budgets.items():
        # A budget of >= 1.0 means "no constraint" for this hazard.
        constrained = budget < 1.0
        upper = conformal_upper_for(bundle, channel, horizon_s)
        if upper is None:
            if constrained:
                # Unknown hazard under an active budget → conservative reject.
                chance[channel.value] = (1.0, budget, True)
                violated.append(channel.value)
            continue
        exceeds = constrained and (upper > budget + 1e-12)
        chance[channel.value] = (upper, budget, exceeds)
        if exceeds:
            violated.append(channel.value)

    # Optional CVaR tail gate (doc M, E5). alpha=0 disables it.
    if risk.cvar_alpha > 0.0:
        for channel in budgets:
            cv = cvar_from_margin(bundle, channel, horizon_s, risk.cvar_alpha)
            exceeds = cv > risk.cvar_budget + 1e-12
            cvar[channel.value] = (cv, risk.cvar_budget, exceeds)
            if exceeds and channel.value not in violated:
                violated.append(channel.value)

    return RiskAssessment(
        passes=not violated,
        horizon_s=horizon_s,
        chance_checks=chance,
        cvar_checks=cvar,
        violated_channels=violated,
    )


def passes_risk_gate(
    bundle: KPIForecastBundle, risk: RiskThresholds, horizon_s: int
) -> bool:
    """Convenience boolean: does this candidate clear every statistical gate?"""
    return assess_risk(bundle, risk, horizon_s).passes
