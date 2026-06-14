# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Lexicographic energy scoring of safe candidates (doc M, E1).

For every safe candidate the injected world model returns a
:class:`~..schema.kpis.KPIForecastBundle`. From that bundle we build the
**lexicographic energy key** the planner ranks on (smaller is better):

    E_lex = ( hard_safety ,            # always 0 here — the set is kernel-safe
              statistical_safety ,     # residual conformal hazard pressure
              p95_queue_wait ,         # operator's primary utility
              -admitted_gpu_hours ,    # admit more useful work
              cross_rack / locality ,  # topology quality
              fragmentation )          # leftover-gap cost

This is the *barrier-plus-lexicographic* objective from doc M (E1): hard
infeasibility first (kept in the tuple for provenance even though it is 0 on the
safe set), then statistical safety, then queue/admission utility, then
efficiency/locality. Comparing tuples lexicographically is exactly
``lex-argmin`` over the safe candidate set (doc M, E2).

This module also exposes an **order-preserving scalar embedding** (doc M / F,
``safe_regret_definition_and_bounds`` D2): bounded KPI ranges and dominating
weights ``M_1 >> M_2 >> ...`` so that ``S(a) = Σ_i M_i ψ_i(a)`` reproduces the
lexicographic order on the bounded candidate set. Safe-regret is numeric, so the
scalar embedding is what the evaluation harness differences against the oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schema.kpis import (
    HazardChannel,
    HazardForecast,
    KPIChannel,
    KPIForecastBundle,
    QuantileForecast,
)
from ..schema.policy import RiskThresholds

__all__ = [
    "LexWeights",
    "admitted_gpu_hours",
    "build_lex_key",
    "fragmentation_cost",
    "lex_key_components",
    "locality_cost",
    "p95_queue_wait",
    "scalar_embedding",
    "statistical_safety_pressure",
]


# ──────────────────────────────────────────────────────────────────────────────
# KPI extraction helpers (one bundle -> one scalar per lexicographic field)
# ──────────────────────────────────────────────────────────────────────────────
def _quantile(
    bundle: KPIForecastBundle,
    channel: KPIChannel,
    horizon_s: int,
    tau: float,
) -> float | None:
    """Return quantile ``tau`` of ``channel`` at ``horizon_s`` if forecast."""
    best: QuantileForecast | None = None
    for q in bundle.quantiles:
        if q.channel is channel and q.horizon_s == horizon_s:
            best = q
            break
    if best is None:
        # Fall back to the nearest available horizon for this channel.
        cands = [q for q in bundle.quantiles if q.channel is channel]
        if not cands:
            return None
        best = min(cands, key=lambda q: abs(q.horizon_s - horizon_s))
    if tau in best.quantiles:
        return best.quantiles[tau]
    # Nearest available quantile (defensive — heads may publish a sparse grid).
    if not best.quantiles:
        return None
    nearest_tau = min(best.quantiles, key=lambda t: abs(t - tau))
    return best.quantiles[nearest_tau]


def _hazard(
    bundle: KPIForecastBundle, channel: HazardChannel, horizon_s: int
) -> HazardForecast | None:
    """Return the hazard forecast for ``channel`` at ``horizon_s`` if present."""
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


def p95_queue_wait(bundle: KPIForecastBundle, horizon_s: int) -> float:
    """p95 queue-wait delta (seconds). The operator's primary utility term.

    We use the upper quantile (p95) rather than the median because the planner is
    risk-sensitive about the *tail* of waiting time, consistent with the p95 SLO
    operators actually track (doc F D9 / doc K).
    """
    v = _quantile(bundle, KPIChannel.QUEUE_WAIT_DELTA_S, horizon_s, 0.95)
    if v is None:
        v = _quantile(bundle, KPIChannel.QUEUE_WAIT_DELTA_S, horizon_s, 0.5)
    return float(v) if v is not None else 0.0


def admitted_gpu_hours(bundle: KPIForecastBundle, horizon_s: int) -> float:
    """Median admitted-GPU-hours delta (more is better → negated in the key)."""
    v = _quantile(bundle, KPIChannel.ADMITTED_GPU_HOURS_DELTA, horizon_s, 0.5)
    return float(v) if v is not None else 0.0


def locality_cost(bundle: KPIForecastBundle, horizon_s: int) -> float:
    """Cross-rack / locality cost (lower is better → tighter gang placement)."""
    v = _quantile(bundle, KPIChannel.CROSS_RACK_RATE, horizon_s, 0.5)
    return float(v) if v is not None else 0.0


def fragmentation_cost(bundle: KPIForecastBundle, horizon_s: int) -> float:
    """Topology-fragmentation delta (lower is better → fewer stranded gaps)."""
    v = _quantile(bundle, KPIChannel.TOPOLOGY_FRAGMENTATION_DELTA, horizon_s, 0.5)
    return float(v) if v is not None else 0.0


#: Map each modeled hazard channel to its operator chance-constraint budget.
#: ``NODE_FAILURE`` maps to its *own* dedicated budget (doc U severity ordering),
#: not the aliased XID-fault budget — kept in lock-step with the risk gate's
#: :func:`~.risk_gate.hazard_budget_map` so the lex statistical-safety pressure
#: term and the gate read the *same* per-channel budget.
_HAZARD_BUDGET_ATTR: dict[HazardChannel, str] = {
    HazardChannel.HOTSPOT: "max_hotspot_prob",
    HazardChannel.HEADROOM_VIOLATION: "max_headroom_violation_prob",
    HazardChannel.PODS_READY_TIMEOUT: "max_pods_ready_timeout_prob",
    HazardChannel.REQUEUE: "max_requeue_prob",
    HazardChannel.XID_FAULT: "max_xid_fault_prob",
    HazardChannel.NODE_FAILURE: "max_node_failure_prob",
}


def statistical_safety_pressure(
    bundle: KPIForecastBundle,
    horizon_s: int,
    risk: RiskThresholds | None = None,
) -> float:
    """Residual statistical-safety pressure on the *safe* set (doc M, E1 R_stat).

    Doc M defines the statistical-safety lexicographic term as the **positive
    excess over budget**: ``R_stat = Σ_ℓ (conformal_upper_ℓ - δ_ℓ)_+``. Every
    survivor of the risk gate is within budget, so this term is ~0 for all of
    them — it acts as a *tie-breaker* that prefers the candidate with the most
    headroom under its budgets, **without** dominating the queue-wait / admission
    utility terms below it (which would otherwise make the planner trivially
    prefer ``hold`` every time, since ``hold`` has near-zero hazards by nature).

    When no ``risk`` budgets are supplied (e.g. a raw scoring call outside the
    planner), we fall back to the conformal sum so the term is still a meaningful
    least-risk ordering — but the planner always passes the operator's budgets.
    """
    total = 0.0
    for channel in HazardChannel:
        h = _hazard(bundle, channel, horizon_s)
        if h is None:
            continue
        upper = max(0.0, float(h.conformal_upper))
        if risk is None:
            total += upper
            continue
        budget = float(getattr(risk, _HAZARD_BUDGET_ATTR[channel], 1.0))
        # Positive part of the excess over the chance-constraint budget (doc M).
        total += max(0.0, upper - budget)
    return total


# ──────────────────────────────────────────────────────────────────────────────
# The lexicographic energy key
# ──────────────────────────────────────────────────────────────────────────────
def lex_key_components(
    bundle: KPIForecastBundle,
    horizon_s: int,
    *,
    hard_safety: float = 0.0,
    risk: RiskThresholds | None = None,
) -> dict[str, float]:
    """Return the named lexicographic key components (for ledgers / debugging)."""
    return {
        "hard_safety": float(hard_safety),
        "statistical_safety": statistical_safety_pressure(bundle, horizon_s, risk),
        "p95_queue_wait": p95_queue_wait(bundle, horizon_s),
        "neg_admitted_gpu_hours": -admitted_gpu_hours(bundle, horizon_s),
        "locality": locality_cost(bundle, horizon_s),
        "fragmentation": fragmentation_cost(bundle, horizon_s),
    }


#: Canonical lexicographic field order (highest priority first), smaller better.
LEX_FIELDS: tuple[str, ...] = (
    "hard_safety",
    "statistical_safety",
    "p95_queue_wait",
    "neg_admitted_gpu_hours",
    "locality",
    "fragmentation",
)


def build_lex_key(
    bundle: KPIForecastBundle,
    horizon_s: int,
    *,
    hard_safety: float = 0.0,
    risk: RiskThresholds | None = None,
) -> tuple[float, ...]:
    """Build the lexicographic energy tuple (lower is better) for one candidate.

    The tuple ordering *is* the priority ordering: Python compares tuples
    lexicographically, so ``min(candidates, key=lambda c: c.lex_key)`` is exactly
    ``lex-argmin E_lex`` over the safe set (doc M, E1/E2). ``hard_safety`` is kept
    as the leading element for provenance (and to make the key total-ordering
    correct if a non-safe action is ever scored), but it is 0 for every member of
    the kernel-certified safe set. ``risk`` supplies the chance-constraint budgets
    so the statistical-safety term is the doc-M positive-excess-over-budget form.
    """
    comp = lex_key_components(bundle, horizon_s, hard_safety=hard_safety, risk=risk)
    return tuple(comp[f] for f in LEX_FIELDS)


# ──────────────────────────────────────────────────────────────────────────────
# Order-preserving scalar embedding (for numeric safe-regret reporting)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class LexWeights:
    """Bounded KPI ranges + dominating weights for the scalar embedding.

    The scalar embedding ``S(a) = Σ_i M_i ψ_i(a)`` preserves the lexicographic
    order on the bounded candidate set iff each weight dominates the maximum
    possible contribution of all lower-priority terms (doc M / F D2):
    ``M_i > Σ_{j>i} M_j · R_j`` where ``R_j`` is the range of field ``j``.

    The defaults below bound each field generously for a 2-4 rack demo pod. The
    embedding is only used for *reporting* safe-regret as a single number; the
    planner itself ranks on the raw lexicographic tuple, so a loose range only
    affects the printed regret scale, never the chosen action.
    """

    # Per-field bounded magnitude ranges R_j (worst-case |value|), highest first.
    ranges: dict[str, float] = field(
        default_factory=lambda: {
            "hard_safety": 1.0,  # 0/1 indicator
            "statistical_safety": float(len(HazardChannel)),  # sum of [0,1] bounds
            "p95_queue_wait": 3600.0,  # one hour of wait delta
            "neg_admitted_gpu_hours": 1000.0,  # demo-pod GPU-hours bound
            "locality": 1.0,  # cross-rack rate in [0,1]
            "fragmentation": 1.0,  # fragmentation delta in [0,1]
        }
    )

    def weights(self) -> dict[str, float]:
        """Return dominating weights ``M_i`` from the field ranges.

        Computed bottom-up: the lowest-priority field gets weight 1; each higher
        field gets a weight strictly greater than the total possible magnitude of
        everything below it. This guarantees order preservation (doc M / F D2).
        """
        order = list(LEX_FIELDS)
        w: dict[str, float] = {}
        running = 0.0  # Σ_{j>i} M_j · R_j accumulated from the bottom up
        for fname in reversed(order):
            weight = running + 1.0
            w[fname] = weight
            running += weight * float(self.ranges.get(fname, 1.0))
        return w


def scalar_embedding(
    lex_key: tuple[float, ...], weights: LexWeights | None = None
) -> float:
    """Map a lexicographic key to an order-preserving scalar score.

    Used for numeric safe-regret (doc M, ``safe_regret``): ``r = S(â) - S(a*)``.
    Lower is better, exactly like the lexicographic tuple it embeds.
    """
    w = (weights or LexWeights()).weights()
    score = 0.0
    for fname, value in zip(LEX_FIELDS, lex_key, strict=False):
        score += w[fname] * float(value)
    return score
