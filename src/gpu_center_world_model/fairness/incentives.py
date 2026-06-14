# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Incentives, restricted strategy-proofness, and gaming diagnostics (doc V).

Manipulation surfaces in this project are broader than classical demand
misreporting (doc V, D24): M1 priority inflation, M2 queue hopping, M3 job
splitting / merging, M4 under-requesting resources / runtime, M5 topology-
preference gaming, M6 burst-submission timing.

Two things are provided:

1. **Restricted strategy-proofness conditions** (doc V, D25 / C1-C6). Under these
   operator-controlled restrictions the weighted-DRF benchmark retains its
   classical truthfulness interpretation for demand-ray revelation (Th6). The
   theorem does **not** extend to the full advisor once priorities, queue
   identity, topology claims, or unverified runtimes are manipulable — so this
   module *checks the preconditions* rather than asserting blanket guarantees.

2. **Empirical gaming diagnostics + replay price-of-anarchy** (doc V,
   price_of_anarchy). No universal closed-form PoA bound is claimed for this
   Kueue-style game (gangs, topology, borrowing, mutable labels make the strategy
   space too domain-specific). The right v2 object is the empirical replay
   estimator (Eq16):

       hat_PoA_H = W_H(centralized fairness-constrained advisor)
                   / W_H(realized decentralized tenant-driven baseline),

   where ``W_H`` is fairness-constrained welfare: throughput if the drift and
   debt budgets hold, ``-inf`` otherwise (D27).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..schema.policy import PolicyConfig

#: ``base_payoff(tenant_id) -> payoff`` at the realized strategy profile (D28).
PayoffFn = Callable[[str], float]
#: ``deviations(tenant_id) -> iterable of (label, payoff)`` unilateral deviations.
DeviationEnumerator = Callable[[str], Iterable[tuple[str, float]]]

#: The restricted strategy-proofness conditions C1-C6 (doc V, D25), in order.
SP_CONDITIONS: tuple[str, ...] = (
    "C1_operator_controlled_priority_and_weights",
    "C2_tenant_scoped_queue_membership",
    "C3_verified_resource_requests",
    "C4_operator_defined_disagreement",
    "C5_verifiable_demand_rays_only",
    "C6_fairness_ranks_safe_actions_only",
)


# ──────────────────────────────────────────────────────────────────────────────
# Restricted strategy-proofness preconditions (doc V, D25 / Th6)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class StrategyProofnessReport:
    """Which of the restricted SP conditions C1-C6 currently hold (doc V, D25).

    ``benchmark_sp_holds`` is True only when *all* conditions hold, in which case
    weighted DRF retains its classical demand-ray truthfulness interpretation
    (Th6). It is deliberately conservative: any unmet condition flips it False,
    because the doc explicitly warns the theorem does not survive priority /
    queue / topology / runtime manipulability.
    """

    conditions: dict[str, bool]
    benchmark_sp_holds: bool
    notes: list[str] = field(default_factory=list)


def check_strategy_proofness(
    policy: PolicyConfig,
    *,
    requests_verified: bool = False,
    demand_rays_verifiable: bool = True,
    disagreement_operator_defined: bool = True,
    fairness_over_safe_set: bool = True,
) -> StrategyProofnessReport:
    """Evaluate restricted strategy-proofness conditions C1-C6 (doc V, D25/Th6).

    The conditions are read primarily off the frozen :class:`PolicyConfig`:

    * **C1** ``priority_trust_mode == "operator_only"`` (priorities & weights are
      operator-controlled, not tenant-writable).
    * **C2** ``tenant_scope != "local_queue"`` (accounting is tenant- or
      ClusterQueue-scoped, so a tenant cannot reset history by spawning fresh
      LocalQueues — the queue-hopping exploit).
    * **C3** ``requests_verified`` (resource requests hard-limited / checked
      against realized telemetry — caller-supplied, from the data engine).
    * **C4** ``disagreement_operator_defined`` (disagreement points come from
      quota / SLA policy, not self-report).
    * **C5** ``demand_rays_verifiable`` (only verifiable demand rays feed the
      mechanism — no tenant-declared soft features like hoped-for runtime).
    * **C6** ``fairness_over_safe_set`` (fairness ranks only kernel-safe actions).
    """
    cond = {
        "C1_operator_controlled_priority_and_weights": (
            policy.priority_trust_mode == "operator_only"
        ),
        "C2_tenant_scoped_queue_membership": policy.tenant_scope != "local_queue",
        "C3_verified_resource_requests": bool(requests_verified),
        "C4_operator_defined_disagreement": bool(disagreement_operator_defined),
        "C5_verifiable_demand_rays_only": bool(demand_rays_verifiable),
        "C6_fairness_ranks_safe_actions_only": bool(fairness_over_safe_set),
    }
    notes: list[str] = []
    if not cond["C1_operator_controlled_priority_and_weights"]:
        notes.append(
            "priority_trust_mode allows tenant claims → priority inflation is a "
            "live exploit; benchmark SP guarantee void (doc V, M1/D25)."
        )
    if not cond["C2_tenant_scoped_queue_membership"]:
        notes.append(
            "LocalQueue-scoped accounting → queue hopping resets fair-sharing "
            "history; aggregate at tenant_id (doc V, H1/M2)."
        )
    if not cond["C3_verified_resource_requests"]:
        notes.append(
            "requests not verified against telemetry → under-requesting is "
            "profitable (doc V, M4)."
        )
    holds = all(cond.values())
    return StrategyProofnessReport(
        conditions=cond, benchmark_sp_holds=holds, notes=notes
    )


# ──────────────────────────────────────────────────────────────────────────────
# Gaming diagnostics (doc V, D24 / M1-M6)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class TenantBehavior:
    """Observed behavioral signals for one tenant over a replay window.

    All fields are read from retained provenance / telemetry hooks (doc V, H5/H6):
    priority provenance & mutability, requested-vs-realized resource and runtime
    envelopes, queue membership counts, and submission timestamps.
    """

    tenant_id: str
    # M1 priority inflation
    priority_changes_while_pending: int = 0
    tenant_set_priority: bool = False  # tenant (not operator) wrote priority class
    # M2 queue hopping
    distinct_local_queues: int = 1
    new_local_queues_in_window: int = 0
    # M4 under-requesting (requested vs realized peak resource use)
    requested_gpu: float = 0.0
    realized_peak_gpu: float = 0.0
    requested_runtime_s: float = 0.0
    realized_runtime_s: float = 0.0
    # M6 burst submission
    submit_timestamps_s: Sequence[float] = field(default_factory=tuple)


@dataclass(slots=True)
class GamingDiagnostics:
    """Per-tenant gaming scores in [0, 1] for the v2 evaluation hooks (doc V, H9)."""

    tenant_id: str
    priority_inflation: float
    queue_hopping: float
    under_requesting: float
    burst_submission: float

    def flagged(self, threshold: float = 0.5) -> list[str]:
        """Surfaces (M*) whose score exceeds ``threshold``."""
        out = []
        if self.priority_inflation > threshold:
            out.append("M1_priority_inflation")
        if self.queue_hopping > threshold:
            out.append("M2_queue_hopping")
        if self.under_requesting > threshold:
            out.append("M4_under_requesting")
        if self.burst_submission > threshold:
            out.append("M6_burst_submission")
        return out


def diagnose_gaming(
    behavior: TenantBehavior,
    *,
    burst_window_s: float = 60.0,
    burst_count_for_full: int = 10,
) -> GamingDiagnostics:
    """Score the four observable gaming surfaces for one tenant (doc V, D24).

    Heuristic, bounded-in-[0,1] detectors grounded in the doc's exploit analysis:

    * **Priority inflation (M1).** Tenant-written priority class and/or priority
      mutated while the workload was pending — Kueue uses priority directly for
      sorting / preemption / borrowing order, so this is profitable whenever a
      tenant can raise it (doc V, D24/M1).
    * **Queue hopping (M2).** Fresh LocalQueues spun up within the window;
      LocalQueue-scoped fair-sharing history makes this profitable unless
      accounting is tenant-scoped (doc V, M2/H1).
    * **Under-requesting (M4).** Realized peak use exceeds the request; admission
      and weighted-share accounting key off requests, so under-reporting is
      profitable when enforcement is weak (doc V, D24/M4).
    * **Burst submission (M6).** Many submissions inside a short window before
      usage statistics update — the exploit Kueue's entry penalty defends against
      (doc V, D24/M6).
    """
    # M1 priority inflation
    pi = 0.0
    if behavior.tenant_set_priority:
        pi += 0.6
    pi += min(behavior.priority_changes_while_pending, 4) / 4.0 * 0.4
    pi = float(min(pi, 1.0))

    # M2 queue hopping — newly created queues this window dominate the score,
    # with a milder contribution from simply spreading across many queues.
    qh_new = min(behavior.new_local_queues_in_window, 5) / 5.0
    qh_spread = max(behavior.distinct_local_queues - 1, 0)
    qh_spread = min(qh_spread, 5) / 5.0
    queue_hopping = float(min(0.7 * qh_new + 0.3 * qh_spread, 1.0))

    # M4 under-requesting — ratio of realized peak over request, both gpu & time.
    def _under(realized: float, requested: float) -> float:
        if requested <= 0:
            # Requested ~nothing but realized something ⇒ strong under-request.
            return 1.0 if realized > 0 else 0.0
        ratio = realized / requested
        # ratio <= 1 ⇒ no under-request; ratio 2x ⇒ full score.
        return float(np.clip(ratio - 1.0, 0.0, 1.0))

    ur_gpu = _under(behavior.realized_peak_gpu, behavior.requested_gpu)
    ur_time = _under(behavior.realized_runtime_s, behavior.requested_runtime_s)
    under_requesting = float(max(ur_gpu, ur_time))

    # M6 burst submission — densest count of submissions in any burst_window_s.
    ts = np.sort(np.asarray(behavior.submit_timestamps_s, dtype=float))
    burst = 0
    if ts.size:
        for i in range(ts.size):
            j = np.searchsorted(ts, ts[i] + burst_window_s, side="right")
            burst = max(burst, int(j - i))
    burst_submission = float(min(burst / max(burst_count_for_full, 1), 1.0))

    return GamingDiagnostics(
        tenant_id=behavior.tenant_id,
        priority_inflation=pi,
        queue_hopping=queue_hopping,
        under_requesting=under_requesting,
        burst_submission=burst_submission,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Empirical price-of-anarchy replay estimator (doc V, Eq15-Eq16 / D27)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class PriceOfAnarchy:
    """Empirical replay PoA estimate ``hat_PoA_H`` (doc V, Eq16)."""

    poa: float
    centralized_welfare: float
    decentralized_welfare: float
    centralized_feasible: bool
    decentralized_feasible: bool
    note: str = ""


def fairness_constrained_welfare(
    throughput_gpu_hours: float,
    fairness_drift: float,
    max_positive_debt: float,
    *,
    drift_budget: float,
    debt_budget: float,
) -> float:
    """Fairness-constrained welfare ``W_H`` (doc V, D27).

    Returns the throughput if both the drift budget ``δ_F`` and the worst-tenant
    debt budget ``δ_D`` hold, and ``-inf`` otherwise. This is the lexicographic
    "safety → fairness budget → throughput" discipline: budgets are *gates*, not
    weighted-sum penalties (doc V, Eq11 / rejected scalarization).
    """
    if fairness_drift <= drift_budget and max_positive_debt <= debt_budget:
        return float(throughput_gpu_hours)
    return float("-inf")


def empirical_price_of_anarchy(
    centralized_throughput: float,
    centralized_drift: float,
    centralized_max_debt: float,
    decentralized_throughput: float,
    decentralized_drift: float,
    decentralized_max_debt: float,
    *,
    drift_budget: float = float("inf"),
    debt_budget: float = float("inf"),
) -> PriceOfAnarchy:
    """Replay PoA: centralized advisor welfare / decentralized baseline (Eq16).

    ``hat_PoA_H = W_H(centralized fairness-constrained advisor) /
    W_H(realized decentralized tenant-driven baseline)``. A value > 1 means the
    centralized fairness-constrained advisor would deliver strictly more
    constrained welfare than the realized strategic equilibrium — the efficiency
    left on the table by decentralized gaming. The estimator is empirical and
    branch-reference-relative; no universal closed-form bound is claimed (doc V).
    """
    w_c = fairness_constrained_welfare(
        centralized_throughput,
        centralized_drift,
        centralized_max_debt,
        drift_budget=drift_budget,
        debt_budget=debt_budget,
    )
    w_d = fairness_constrained_welfare(
        decentralized_throughput,
        decentralized_drift,
        decentralized_max_debt,
        drift_budget=drift_budget,
        debt_budget=debt_budget,
    )
    c_feas = np.isfinite(w_c)
    d_feas = np.isfinite(w_d)

    note = ""
    if not c_feas:
        poa = float("inf")
        note = "centralized advisor violated a fairness budget (W_H = -inf)."
    elif not d_feas:
        poa = float("inf")
        note = (
            "decentralized baseline violated a fairness budget; advisor strictly "
            "dominates (PoA → ∞)."
        )
    elif w_d <= 0:
        poa = float("inf") if w_c > 0 else 1.0
        note = "degenerate decentralized welfare (<= 0)."
    else:
        poa = float(w_c / w_d)
    return PriceOfAnarchy(
        poa=poa,
        centralized_welfare=float(w_c),
        decentralized_welfare=float(w_d),
        centralized_feasible=bool(c_feas),
        decentralized_feasible=bool(d_feas),
        note=note,
    )


def aggregate_gaming(
    diagnostics: Sequence[GamingDiagnostics],
) -> dict[str, float]:
    """Replay-level summary of gaming pressure across tenants (doc V, H9).

    Returns the mean of each surface plus the count of tenants flagged on any
    surface — a compact slice for the v2 evaluation hooks alongside FD / debt.
    """
    if not diagnostics:
        return {
            "mean_priority_inflation": 0.0,
            "mean_queue_hopping": 0.0,
            "mean_under_requesting": 0.0,
            "mean_burst_submission": 0.0,
            "n_flagged": 0.0,
        }
    arr = np.array(
        [
            [
                d.priority_inflation,
                d.queue_hopping,
                d.under_requesting,
                d.burst_submission,
            ]
            for d in diagnostics
        ]
    )
    means = arr.mean(axis=0)
    n_flagged = sum(1 for d in diagnostics if d.flagged())
    return {
        "mean_priority_inflation": float(means[0]),
        "mean_queue_hopping": float(means[1]),
        "mean_under_requesting": float(means[2]),
        "mean_burst_submission": float(means[3]),
        "n_flagged": float(n_flagged),
    }


__all__ = [
    "SP_CONDITIONS",
    "DeviationEnumerator",
    "DeviationResult",
    "GamingDiagnostics",
    "PayoffFn",
    "PriceOfAnarchy",
    "ShareStabilityReport",
    "StrategyProofnessReport",
    "TenantBehavior",
    "aggregate_gaming",
    "check_epsilon_share_stability",
    "check_strategy_proofness",
    "diagnose_gaming",
    "empirical_price_of_anarchy",
    "fairness_constrained_welfare",
]


# ──────────────────────────────────────────────────────────────────────────────
# Epsilon-share-stability: best-response / unilateral-deviation checker (D28)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class DeviationResult:
    """One tenant's most-profitable unilateral deviation (doc V D28)."""

    tenant_id: str
    base_payoff: float
    best_deviation: str | None
    best_payoff: float
    gain: float  # best_payoff - base_payoff
    profitable: bool  # gain > epsilon


@dataclass(slots=True)
class ShareStabilityReport:
    """Empirical ε-share-stability of a realized state (doc V D28).

    A state is **ε-share-stable** when no tenant can obtain a strictly better
    payoff (by more than ``epsilon``) via a unilateral deviation under the
    configured share-based admission / preemption rule, with all other tenants'
    strategies fixed (doc V D28). This complements the empirical replay PoA: PoA
    measures the *efficiency* left on the table by the decentralized equilibrium,
    while ε-share-stability checks whether the realized profile *is* an
    (approximate) equilibrium at all.

    The check is empirical and deviation-set-relative — no closed-form bound is
    claimed for this Kueue-style game (gangs / topology / borrowing make the
    strategy space domain-specific, doc V price_of_anarchy / D26).
    """

    epsilon: float
    is_stable: bool
    deviations: list[DeviationResult] = field(default_factory=list)
    max_gain: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "epsilon": self.epsilon,
            "is_stable": self.is_stable,
            "max_gain": self.max_gain,
            "n_profitable_deviations": sum(
                1 for d in self.deviations if d.profitable
            ),
        }


def check_epsilon_share_stability(
    tenants: Sequence[str],
    base_payoff: PayoffFn,
    deviations: DeviationEnumerator,
    *,
    epsilon: float = 0.0,
) -> ShareStabilityReport:
    """Best-response / unilateral-deviation ε-share-stability checker (doc V D28).

    For every tenant ``t`` we evaluate its base payoff at the realized profile and
    the payoff of each unilateral deviation in ``deviations(t)`` (others fixed),
    keeping the most profitable. The state is ε-share-stable iff no tenant's best
    deviation gain exceeds ``epsilon`` (doc V D28: "no tenant can obtain a
    strictly better payoff by unilateral deviation under fixed others").

    Parameters
    ----------
    tenants:
        Tenant identities in the realized profile.
    base_payoff:
        ``base_payoff(t)`` → the tenant's realized payoff (e.g. its long-run
        service ``U_t`` or weighted share) under the *current* strategy profile.
    deviations:
        ``deviations(t)`` → an iterable of ``(label, payoff)`` candidate unilateral
        deviations for tenant ``t`` (priority inflation, queue hopping, job
        splitting, burst timing — the manipulation surfaces D24/M1-M6), each with
        the payoff the tenant would obtain *if it alone deviated*. The caller
        supplies these from a replay or the share-based admission rule simulation
        so the check stays grounded in the platform's actual mechanics.
    epsilon:
        The stability slack ``ε`` (doc V D28). ``0`` ⇒ exact best-response check.
    """
    results: list[DeviationResult] = []
    max_gain = 0.0
    for t in tenants:
        base = float(base_payoff(t))
        best_label: str | None = None
        best_payoff = base
        for label, payoff in deviations(t):
            p = float(payoff)
            if p > best_payoff:
                best_payoff, best_label = p, label
        gain = best_payoff - base
        max_gain = max(max_gain, gain)
        results.append(
            DeviationResult(
                tenant_id=t,
                base_payoff=base,
                best_deviation=best_label,
                best_payoff=best_payoff,
                gain=gain,
                profitable=gain > epsilon,
            )
        )
    return ShareStabilityReport(
        epsilon=epsilon,
        is_stable=max_gain <= epsilon,
        deviations=results,
        max_gain=max_gain,
    )
