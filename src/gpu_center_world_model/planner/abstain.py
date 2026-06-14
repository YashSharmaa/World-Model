# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The abstention gate — *knowing when not to recommend* (doc F, doc L, doc T/U).

A risk-sensitive planner must decline to recommend when it is operating outside
the regime where its learned forecasts are trustworthy. Three signals drive
abstention, each compared to an operator-configured threshold in
:class:`~..schema.policy.AbstentionConfig`:

1. **Epistemic disagreement** — ensemble / posterior spread on the candidate's
   forecast (``KPIForecastBundle.epistemic_score``). High disagreement means the
   model itself is unsure (doc L).
2. **OOD score** — distance from the training support / topology-and-workload
   regime (``KPIForecastBundle.ood_score``). Models that look calibrated
   in-distribution become overconfident under shift (doc F shift slices).
3. **Observation integrity** — degraded telemetry (stale / dropped collectors)
   read from the world state's source-health block ``M_t`` (doc T/U). The
   missingness *is* the signal; never read through it.

When the gate fires, the planner falls back per ``AbstentionConfig.fallback``:
``conservative_rule`` (rank on the conformal-safe-but-cautious key) or ``hold``
(emit the always-safe hold). Either way the emitted action remains drawn from the
kernel-safe set, so abstention never breaks the hard-safety certificate (doc M,
G1). Crucially, the model bundle may *already* carry ``abstain=True`` from the
learned head — we honour that, but the planner's gate is authoritative and can
abstain even when the model did not.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schema.kpis import KPIForecastBundle
from ..schema.observations import SourceHealth
from ..schema.policy import AbstentionConfig
from ..schema.state import SourceState, WorldState

__all__ = [
    "AbstainDecision",
    "aggregate_abstention",
    "observation_integrity_bad_prob",
    "should_abstain",
]


# ──────────────────────────────────────────────────────────────────────────────
# Observation-integrity signal (M_t)
# ──────────────────────────────────────────────────────────────────────────────
#: Source-health regimes that count as "bad" telemetry for the integrity gate.
_BAD_HEALTH: frozenset[str] = frozenset(
    {SourceHealth.STALE.value, SourceHealth.DELAYED.value, SourceHealth.DROPPED.value}
)
#: Free-form collector status strings that also count as degraded (doc T).
_BAD_STATUS_TOKENS: frozenset[str] = frozenset(
    {"bad", "down", "degraded", "stale", "dropped", "delayed", "unhealthy"}
)


def observation_integrity_bad_prob(source: SourceState) -> float:
    """Estimate the probability that decision-relevant telemetry is degraded.

    Combines three observable facets of the source-health block ``M_t``:

    * fraction of collectors reporting a degraded status,
    * mean per-channel missing probability, and
    * fraction of channels whose age-of-information is large relative to its
      transport delay (a staleness proxy).

    Returns a value in ``[0, 1]``; the abstention gate compares it to
    ``AbstentionConfig.max_source_health_bad_prob``. With an empty/clean source
    block this is 0.0 (telemetry is trusted).
    """
    signals: list[float] = []

    health = source.collector_health
    if health:
        bad = sum(1 for s in health.values() if str(s).lower() in _BAD_STATUS_TOKENS)
        signals.append(bad / len(health))

    miss = source.channel_missing_prob
    if miss:
        signals.append(min(1.0, max(0.0, sum(miss.values()) / len(miss))))

    ages = source.channel_age_s
    if ages:
        stale = 0
        for name, age in ages.items():
            # A channel is "stale" when its age exceeds twice its transport delay
            # (or 60 s when no transport delay is recorded).
            delay = source.transport_delay_s.get(name, 0.0)
            threshold = max(2.0 * delay, 60.0)
            if age > threshold:
                stale += 1
        signals.append(stale / len(ages))

    if not signals:
        return 0.0
    # The integrity verdict is driven by the *worst* facet — one dead collector
    # is enough to make a recommendation untrustworthy.
    return max(signals)


# ──────────────────────────────────────────────────────────────────────────────
# Per-candidate abstention decision
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class AbstainDecision:
    """Whether to abstain for one candidate, with the triggering evidence."""

    abstain: bool
    reason: str
    epistemic: float
    ood: float
    source_bad_prob: float
    model_flagged: bool

    @property
    def fallback_needed(self) -> bool:
        return self.abstain


def should_abstain(
    bundle: KPIForecastBundle,
    cfg: AbstentionConfig,
    source_bad_prob: float,
) -> AbstainDecision:
    """Evaluate the abstention gate for a single candidate forecast.

    Order of checks (first hit wins for the reason string): model-flagged →
    epistemic disagreement → OOD → degraded observation integrity. The
    ``source_bad_prob`` is computed once per decision window (it is a property of
    the world state, not of any single candidate) and passed in here.
    """
    epi = float(bundle.epistemic_score)
    ood = float(bundle.ood_score)
    model_flagged = bool(bundle.abstain)

    reason = ""
    abstain = False
    if model_flagged:
        abstain = True
        reason = bundle.abstain_reason or "model_self_abstain"
    elif epi > cfg.max_epistemic_disagreement:
        abstain = True
        reason = (
            f"epistemic_disagreement {epi:.3f} > {cfg.max_epistemic_disagreement:.3f}"
        )
    elif ood > cfg.max_ood_score:
        abstain = True
        reason = f"ood_score {ood:.3f} > {cfg.max_ood_score:.3f}"
    elif source_bad_prob > cfg.max_source_health_bad_prob:
        abstain = True
        reason = (
            f"observation_integrity {source_bad_prob:.3f} "
            f"> {cfg.max_source_health_bad_prob:.3f}"
        )

    return AbstainDecision(
        abstain=abstain,
        reason=reason,
        epistemic=epi,
        ood=ood,
        source_bad_prob=source_bad_prob,
        model_flagged=model_flagged,
    )


def aggregate_abstention(
    decisions: dict[str, AbstainDecision],
    state: WorldState,
    cfg: AbstentionConfig,
) -> tuple[bool, str]:
    """Decide whether the *whole decision window* should abstain.

    Window-level abstention fires when either:

    * the observation-integrity signal exceeds threshold (degraded telemetry
      taints every candidate equally — abstain the whole window), or
    * the *top-ranked* candidates we would otherwise recommend are themselves
      flagged. We adopt the conservative policy that the window abstains if **all**
      surviving candidates are individually flagged for abstention, since there is
      then no trustworthy action to recommend.

    Returns ``(abstain, reason)``. The actual fallback action selection (hold vs
    conservative rule) is performed by the planner using
    ``AbstentionConfig.fallback``.
    """
    source_bad = observation_integrity_bad_prob(state.source)
    if source_bad > cfg.max_source_health_bad_prob:
        threshold = cfg.max_source_health_bad_prob
        return True, f"degraded_telemetry {source_bad:.3f} > {threshold:.3f}"

    if not decisions:
        return False, ""

    if all(d.abstain for d in decisions.values()):
        # Surface the most common / first reason for the window.
        first = next(iter(decisions.values()))
        return True, f"all_candidates_flagged ({first.reason})"

    return False, ""
