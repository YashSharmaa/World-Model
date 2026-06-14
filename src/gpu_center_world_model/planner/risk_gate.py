# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The statistical-safety risk gate (doc M, ``risk_objective`` / ``cmdp_formulation``).

Hard feasibility is owned by the exact kernel. *Statistical* safety — the
genuinely uncertain hazards the learned residual model forecasts (hotspot,
headroom violation, pods-ready timeout, requeue, XID/device fault, node failure)
— is enforced **here**, as a gate, not as a weighted reward term (doc M, T1; the
project spine: "safety is a gate, not a weighted metric").

Doc M ``risk_objective`` E1 enumerates **three distinct** statistical-safety
tests, and the operator may demand any combination of them per hazard family
``ℓ``. This module keeps them genuinely separate (an earlier version conflated
them, and computed CVaR as a single two-point blend):

  1. **Chance constraint** (doc M E4) — ``P(Z_ℓ > 0) ≤ δ_ℓ``. We read the
     calibrated upper bound on the *probability that the hazard fires at all*
     (``conformal_upper`` on the :class:`HazardForecast`) and compare it to the
     operator budget ``δ_ℓ``. This gates a probability, never a margin.
  2. **Conformal margin** (doc M E2 / doc L D6) — ``U_ℓ = score + q̂_{1-δ} ≤ 0``.
     A *one-sided calibrated margin* test on the signed violation margin. The
     forecast's ``conformal_upper`` is exactly ``score + q̂_{1-δ}`` (doc L D6
     E13), so the margin a budget ``δ_ℓ`` leaves is ``U_ℓ = conformal_upper - δ_ℓ``
     and the gate is ``U_ℓ ≤ tol``. This is a *different* object from the chance
     constraint: a hazard can sit under its probability budget yet have its
     calibrated margin poke above zero (or vice-versa).
  3. **CVaR tail** (doc M E5) — ``CVaR_α(Z_ℓ) ≤ β_ℓ`` in Rockafellar–Uryasev
     form ``CVaR_α(Z) = inf_η { η + (1/(1-α)) E[(Z-η)_+] }``. We take the
     infimum over a *distribution* of the violation margin — a quantile grid /
     samples published by the forecast bundle when available, else a documented
     two-point (point, conformal-tail) fallback.

Because ``VaR_α ≤ CVaR_α`` for non-negative violation margins, a CVaR gate
implies a lower bound on the probability of no violation (doc M, E1) — so the
three gates are coherent, not contradictory.

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
    "MarginDistribution",
    "RiskAssessment",
    "assess_risk",
    "conformal_margin_for",
    "conformal_upper_for",
    "cvar_from_margin",
    "hazard_budget_map",
    "margin_distribution_for",
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

    Every modeled hazard has its *own* operator budget ``δ_ℓ``. In particular
    :attr:`HazardChannel.NODE_FAILURE` maps to the dedicated
    :attr:`RiskThresholds.max_node_failure_prob` rather than borrowing the
    XID-fault budget (the old, conflated behaviour): a whole-node failure is
    strictly more severe than a single device XID fault (doc U severity ordering),
    so it deserves its own independently-sizable budget. Its default is permissive
    (opt-in) so pre-existing policies stay byte-stable; operators enforcing it
    should size it at or below the XID-fault budget.
    """
    return {
        HazardChannel.HOTSPOT: risk.max_hotspot_prob,
        HazardChannel.HEADROOM_VIOLATION: risk.max_headroom_violation_prob,
        HazardChannel.PODS_READY_TIMEOUT: risk.max_pods_ready_timeout_prob,
        HazardChannel.REQUEUE: risk.max_requeue_prob,
        HazardChannel.XID_FAULT: risk.max_xid_fault_prob,
        HazardChannel.NODE_FAILURE: risk.max_node_failure_prob,
    }


def _hazard_at(
    bundle: KPIForecastBundle, channel: HazardChannel, horizon_s: int
) -> HazardForecast | None:
    """Return the hazard forecast for ``channel`` at (or nearest) ``horizon_s``.

    Returns ``None`` only when the model published no forecast for the channel at
    all (an *absence* is handled by the caller as conservatively unknown, not as
    zero risk).
    """
    best: HazardForecast | None = None
    for h in bundle.hazards:
        if h.channel is channel and h.horizon_s == horizon_s:
            return h
        if h.channel is channel and (
            best is None
            or abs(h.horizon_s - horizon_s) < abs(best.horizon_s - horizon_s)
        ):
            best = h
    return best


def conformal_upper_for(
    bundle: KPIForecastBundle, channel: HazardChannel, horizon_s: int
) -> float | None:
    """Return the one-sided conformal upper bound for a hazard at a horizon.

    Falls back to the nearest forecast horizon for the channel; returns ``None``
    if the model published no forecast for the channel at all (an *absence* of a
    forecast is handled by the caller as conservatively unknown, not as zero).
    """
    h = _hazard_at(bundle, channel, horizon_s)
    return None if h is None else float(h.conformal_upper)


def conformal_margin_for(
    bundle: KPIForecastBundle,
    channel: HazardChannel,
    horizon_s: int,
    budget: float,
) -> float | None:
    """Return the conformal *margin* ``U_ℓ = score + q̂_{1-δ} - δ_ℓ`` (doc M E2).

    The forecast's ``conformal_upper`` *is* the one-sided calibrated upper bound
    ``score + q̂_{1-δ}`` (doc L D6 E13). Centring it on the operator budget
    ``δ_ℓ`` gives the signed margin the doc-M gate tests: ``U_ℓ ≤ 0`` means the
    calibrated upper bound is within budget with room to spare; ``U_ℓ > 0`` means
    the margin pokes over budget. Returns ``None`` when the channel was never
    forecast (caller treats absence conservatively).

    This is deliberately a *different* quantity from the chance constraint: the
    chance constraint asks whether the *probability* ``P(Z_ℓ > 0)`` is within
    budget; the margin asks whether the calibrated *score margin* is non-positive.
    They coincide numerically here because ``conformal_upper`` plays both the role
    of a probability bound and a margin score, but the operator can size, relax,
    or disable each gate independently (``conformal_margin_tol`` vs ``δ_ℓ``).
    """
    upper = conformal_upper_for(bundle, channel, horizon_s)
    if upper is None:
        return None
    return upper - float(budget)


# ──────────────────────────────────────────────────────────────────────────────
# Violation-margin distribution (doc M D2 / E5) — quantile grid or samples
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class MarginDistribution:
    """A discrete distribution of a hazard's non-negative violation margin ``Z_ℓ``.

    Doc M E5 computes ``CVaR_α(Z_ℓ)`` over the *distribution* of the violation
    margin, not over a single point. The forecast contract exposes the margin via
    a quantile grid / samples when the model publishes them (the rich path), and
    always via the calibrated two-point ``(point, conformal-tail)`` pair (the
    documented fallback). This object carries either form as an explicit, weighted
    set of ``(value, weight)`` atoms so the Rockafellar–Uryasev infimum below is a
    plain finite optimisation.

    Attributes
    ----------
    values:
        The margin atoms ``z_i ≥ 0`` (sorted ascending), non-negative because the
        violation margin is a positive part (doc M D2 ``V_ℓ ≥ 0``).
    weights:
        The probability mass ``w_i ≥ 0`` on each atom, summing to 1.
    source:
        ``"grid"`` / ``"samples"`` when the bundle published a real margin
        distribution; ``"two_point"`` for the documented fallback.
    """

    values: tuple[float, ...]
    weights: tuple[float, ...]
    source: str

    def cvar(self, alpha: float) -> float:
        """``CVaR_α`` via the Rockafellar–Uryasev infimum (doc M E5).

        ``CVaR_α(Z) = inf_η { η + (1/(1-α)) E[(Z - η)_+] }``. The objective is
        convex and piecewise-linear in ``η`` with breakpoints exactly at the atom
        values, so its infimum is attained at one of them; we evaluate the
        objective at each atom and take the minimum. For ``α → 1`` the
        ``1/(1-α)`` weight dominates and the minimiser collapses onto the largest
        atom, recovering ``CVaR_1 = max Z = VaR_1`` and the inequality
        ``VaR_α ≤ CVaR_α`` (doc M E5). For ``α ≤ 0`` it reduces to the mean.
        """
        if not self.values:
            return 0.0
        a = min(max(float(alpha), 0.0), 1.0 - 1e-12)
        inv = 1.0 / (1.0 - a)
        best = float("inf")
        # The infimum over η of a convex piecewise-linear function is at a
        # breakpoint (an atom). Evaluating at each atom is exact and O(n^2) over
        # the tiny per-channel grid (a handful of atoms).
        for eta in self.values:
            expect_pos = 0.0
            for z, w in zip(self.values, self.weights, strict=False):
                expect_pos += w * max(0.0, z - eta)
            obj = eta + inv * expect_pos
            if obj < best:
                best = obj
        return max(0.0, best)

    def var(self, alpha: float) -> float:
        """``VaR_α`` — the smallest atom whose cumulative mass reaches ``α``.

        The lower ``α``-quantile of the margin distribution. Used to expose the
        ``VaR_α ≤ CVaR_α`` relation (doc M E5) for diagnostics; the gate itself
        uses :meth:`cvar`.
        """
        if not self.values:
            return 0.0
        a = min(max(float(alpha), 0.0), 1.0)
        cum = 0.0
        for z, w in zip(self.values, self.weights, strict=False):
            cum += w
            if cum >= a - 1e-12:
                return float(z)
        return float(self.values[-1])


def _read_published_margin(
    h: HazardForecast,
) -> tuple[tuple[float, float], ...] | None:
    """Read a published violation-margin quantile grid / samples off a forecast.

    The forecast contract may publish a richer view of the violation margin than
    the two-point ``(probability, conformal_upper)`` pair — either an explicit
    quantile grid ``{tau: value}`` or a list of Monte-Carlo samples — under one of
    a small set of additive, optional attribute names. We read whichever is
    present (without requiring it, so the schema stays add-only and bundles that
    only carry the two scalars still work):

    * ``margin_quantiles`` — a ``{tau: margin_value}`` grid (the doc-L quantile
      grid analogue for the violation margin); each consecutive quantile band
      contributes its probability mass to the band's value.
    * ``margin_samples`` — an iterable of margin samples; each gets equal weight.

    Returns a tuple of ``(value, weight)`` atoms, or ``None`` when neither is
    published (the caller then uses the two-point fallback).
    """
    grid = getattr(h, "margin_quantiles", None)
    if isinstance(grid, dict) and grid:
        taus = sorted(float(t) for t in grid)
        atoms: list[tuple[float, float]] = []
        prev = 0.0
        for t in taus:
            w = max(0.0, t - prev)
            prev = t
            atoms.append((max(0.0, float(grid[t])), w))
        # Any residual mass above the top quantile sits on the top value (tail).
        tail = max(0.0, 1.0 - prev)
        if tail > 0.0:
            top_val = max(0.0, float(grid[taus[-1]]))
            atoms.append((top_val, tail))
        return tuple(atoms)

    samples = getattr(h, "margin_samples", None)
    if samples is not None:
        try:
            vals = [max(0.0, float(s)) for s in samples]
        except (TypeError, ValueError):
            vals = []
        if vals:
            w = 1.0 / len(vals)
            return tuple((v, w) for v in vals)
    return None


def margin_distribution_for(
    bundle: KPIForecastBundle, channel: HazardChannel, horizon_s: int
) -> MarginDistribution:
    """Build the violation-margin distribution ``Z_ℓ`` for a hazard (doc M D2/E5).

    Prefers a *published* margin quantile grid / samples on the matching
    :class:`HazardForecast` (the rich path, doc M E5 — CVaR over the real
    distribution). When the bundle exposes only the two scalars the contract
    guarantees, falls back to the **documented two-point distribution**: an atom
    at the calibrated point hazard ``probability`` (the central violation-margin
    estimate) carrying mass ``α``-complement, and an atom at the ``conformal_upper``
    tail carrying the epistemic tail mass. The two-point split here is uniform
    ``0.5/0.5`` so the distribution is alpha-agnostic; the Rockafellar–Uryasev
    infimum in :meth:`MarginDistribution.cvar` then does the alpha-weighting,
    rather than baking alpha into a blend (the old, conflated behaviour).

    An absent channel yields an all-zero margin (a benign degenerate distribution
    whose CVaR is 0); the caller's chance / margin gates handle the *missing-under-
    budget* case as a conservative reject separately.
    """
    h = _hazard_at(bundle, channel, horizon_s)
    if h is None:
        return MarginDistribution(values=(0.0,), weights=(1.0,), source="two_point")

    published = _read_published_margin(h)
    if published is not None:
        vals = [v for v, _ in published]
        ws = [w for _, w in published]
        total = sum(ws)
        if total > 0.0:
            ws = [w / total for w in ws]
            order = sorted(range(len(vals)), key=lambda i: vals[i])
            return MarginDistribution(
                values=tuple(vals[i] for i in order),
                weights=tuple(ws[i] for i in order),
                source="grid" if getattr(h, "margin_quantiles", None) else "samples",
            )

    # Documented two-point fallback: (central point, conformal tail), 0.5/0.5.
    point = max(0.0, float(h.probability))
    upper = max(0.0, float(h.conformal_upper))
    lo, hi = (point, upper) if point <= upper else (upper, point)
    if abs(hi - lo) < 1e-15:
        return MarginDistribution(values=(lo,), weights=(1.0,), source="two_point")
    return MarginDistribution(
        values=(lo, hi), weights=(0.5, 0.5), source="two_point"
    )


# ──────────────────────────────────────────────────────────────────────────────
# CVaR tail term (Rockafellar–Uryasev form, doc M E5)
# ──────────────────────────────────────────────────────────────────────────────
def cvar_from_margin(
    bundle: KPIForecastBundle, channel: HazardChannel, horizon_s: int, alpha: float
) -> float:
    """Estimate ``CVaR_alpha`` of a hazard's non-negative violation margin (doc M E5).

    Builds the violation-margin distribution (:func:`margin_distribution_for`) —
    a published quantile grid / samples when available, else the documented
    two-point ``(point, conformal-tail)`` fallback — and returns the
    Rockafellar–Uryasev infimum ``inf_η { η + (1/(1-α)) E[(Z-η)_+] }`` over it
    (:meth:`MarginDistribution.cvar`). With ``alpha → 1`` this collapses to the
    distribution's max (the worst tail), matching ``VaR_α ≤ CVaR_α``; with
    ``alpha ≤ 0`` it disables the tail term and returns 0.
    """
    if alpha <= 0.0:
        return 0.0
    dist = margin_distribution_for(bundle, channel, horizon_s)
    return dist.cvar(alpha)


# ──────────────────────────────────────────────────────────────────────────────
# Per-candidate risk assessment
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class RiskAssessment:
    """The full statistical-safety verdict for one candidate at one horizon.

    Carries the three doc-M E1 gate families separately:

    * :attr:`chance_checks` — ``P(Z_ℓ > 0) ≤ δ_ℓ`` (doc M E4); each entry is
      ``(conformal_upper, budget, exceeds)``.
    * :attr:`margin_checks` — ``U_ℓ = score + q̂ ≤ tol`` (doc M E2); each entry is
      ``(U_ℓ, tol, exceeds)``. Empty when the operator disabled the margin gate
      (``conformal_margin_tol = +inf``).
    * :attr:`cvar_checks` — ``CVaR_α(Z_ℓ) ≤ β_ℓ`` (doc M E5); each entry is
      ``(cvar, budget, exceeds)``. Empty when ``cvar_alpha = 0``.
    """

    passes: bool
    horizon_s: int
    # per-channel: (conformal_upper, budget, exceeds?) — chance constraint (E4)
    chance_checks: dict[str, tuple[float, float, bool]] = field(default_factory=dict)
    # per-channel: (cvar, budget, exceeds?) — CVaR tail (E5)
    cvar_checks: dict[str, tuple[float, float, bool]] = field(default_factory=dict)
    # per-channel: (U_ell, tol, exceeds?) — conformal margin (E2)
    margin_checks: dict[str, tuple[float, float, bool]] = field(default_factory=dict)
    violated_channels: list[str] = field(default_factory=list)

    def reason(self) -> str:
        """A short human-readable rejection reason (empty if it passed)."""
        if self.passes:
            return ""
        parts = []
        for ch in self.violated_channels:
            if ch in self.chance_checks and self.chance_checks[ch][2]:
                up, bud, _ = self.chance_checks[ch]
                parts.append(f"{ch} conformal P̂≤{up:.3f} > budget {bud:.3f}")
            elif ch in self.margin_checks and self.margin_checks[ch][2]:
                u, tol, _ = self.margin_checks[ch]
                parts.append(f"{ch} conformal margin U={u:.3f} > tol {tol:.3f}")
            elif ch in self.cvar_checks and self.cvar_checks[ch][2]:
                cv, bud, _ = self.cvar_checks[ch]
                parts.append(f"{ch} CVaR {cv:.3f} > budget {bud:.3f}")
        return "; ".join(parts)


def assess_risk(
    bundle: KPIForecastBundle, risk: RiskThresholds, horizon_s: int
) -> RiskAssessment:
    """Evaluate every statistical-safety gate for a candidate (doc M E1/E2/E4/E5).

    Runs the three *separate* gates in doc-M order:

    1. **Chance constraint (E4)** — ``P(Z_ℓ > 0) ≤ δ_ℓ`` for every modeled hazard.
       A *missing* forecast under an active budget is treated conservatively: the
       model is silent on a hazard the operator constrains, so we record an
       unknown-risk violation rather than silently pass it (better to abstain /
       fall back than to assume zero hazard — the gate's contribution to keeping
       the false-safe rate below 1%, doc F).
    2. **Conformal margin (E2)** — ``U_ℓ = score + q̂ - δ_ℓ ≤ tol``, when the
       operator enabled it (``conformal_margin_tol`` finite). Distinct from (1):
       it tests the calibrated signed margin, not the probability.
    3. **CVaR tail (E5)** — ``CVaR_α(Z_ℓ) ≤ β_ℓ`` over the violation-margin
       distribution, when ``cvar_alpha > 0``. The budget is resolved through
       :meth:`RiskThresholds.usable_cvar_budget` so enabling tail aversion without
       sizing a budget can never reject every candidate.
    """
    budgets = hazard_budget_map(risk)
    chance: dict[str, tuple[float, float, bool]] = {}
    margin: dict[str, tuple[float, float, bool]] = {}
    cvar: dict[str, tuple[float, float, bool]] = {}
    violated: list[str] = []

    margin_tol = float(risk.conformal_margin_tol)
    margin_gate_on = margin_tol != float("inf")
    cvar_alpha = float(risk.cvar_alpha)
    cvar_budget = risk.usable_cvar_budget()

    for channel, budget in budgets.items():
        # A budget of >= 1.0 means "no constraint" for this hazard.
        constrained = budget < 1.0
        upper = conformal_upper_for(bundle, channel, horizon_s)

        if upper is None:
            if constrained:
                # Unknown hazard under an active budget → conservative reject on
                # the chance constraint (the probability could be anything).
                chance[channel.value] = (1.0, budget, True)
                if channel.value not in violated:
                    violated.append(channel.value)
            # No forecast ⇒ no margin / CVaR check is possible; the conservative
            # chance reject above already holds the candidate out.
            continue

        # 1. Chance constraint (doc M E4): P(Z_ℓ > 0) ≤ δ_ℓ.
        chance_exceeds = constrained and (upper > budget + 1e-12)
        chance[channel.value] = (upper, budget, chance_exceeds)
        if chance_exceeds and channel.value not in violated:
            violated.append(channel.value)

        # 2. Conformal margin (doc M E2): U_ℓ = score + q̂ - δ_ℓ ≤ tol.
        if margin_gate_on and constrained:
            u = upper - budget
            margin_exceeds = u > margin_tol + 1e-12
            margin[channel.value] = (u, margin_tol, margin_exceeds)
            if margin_exceeds and channel.value not in violated:
                violated.append(channel.value)

    # 3. CVaR tail gate (doc M E5). alpha=0 disables it.
    if cvar_alpha > 0.0:
        for channel in budgets:
            cv = cvar_from_margin(bundle, channel, horizon_s, cvar_alpha)
            cvar_exceeds = cv > cvar_budget + 1e-12
            cvar[channel.value] = (cv, cvar_budget, cvar_exceeds)
            if cvar_exceeds and channel.value not in violated:
                violated.append(channel.value)

    return RiskAssessment(
        passes=not violated,
        horizon_s=horizon_s,
        chance_checks=chance,
        cvar_checks=cvar,
        margin_checks=margin,
        violated_channels=violated,
    )


def passes_risk_gate(
    bundle: KPIForecastBundle, risk: RiskThresholds, horizon_s: int
) -> bool:
    """Convenience boolean: does this candidate clear every statistical gate?"""
    return assess_risk(bundle, risk, horizon_s).passes
