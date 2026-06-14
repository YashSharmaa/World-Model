# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The v1 → v2 fairness hooks, and the safe-set re-ranker (doc V, v1_hooks_for_v2).

Fairness is a **v2 ranking layer over the exact-kernel-safe set** — disabled by
default in v1 (``PolicyConfig.fairness_mode == "off"``). This module wires the
minimal hooks v1 must leave in place so v2 can be switched on without breaking
the exact-kernel / learned-residual split (doc V, H1-H10):

* **H1 tenant-level accounting** — :func:`tenant_demands_from_state` aggregates
  fair-division inputs at ``tenant_id`` (not LocalQueue), so queue hopping cannot
  reset history (doc V, H1/M2).
* **H2 retained Kueue observables** — :func:`fairness_observables` exports the
  native shares (ClusterQueue / cohort weighted share, LocalQueue history share,
  usage, reservation) already retained on the schema.
* **H3 entitlement primitives** — weights, nominal quota, borrowing / lending
  limits are read straight from :class:`ClusterQueue`.
* **H7 policy knobs** — everything keys off the frozen
  :class:`PolicyConfig.fairness_mode` and friends.
* **H8 explanation ledger** — :func:`fairness_rank` annotates each ranked action
  with projected drift / debt evidence.
* **H10 ranker over safe actions only** — :func:`fairness_rank` re-orders the
  planner's *already-safe* ``ScoredAction`` list; it never adds, removes, or
  un-gates an action. When ``fairness_mode == "off"`` it is a strict no-op.

The re-ranking discipline follows doc V Eq11: among safe actions, prefer those
that keep the fairness-drift delta within budget and reduce (or least worsen) the
worst-served tenant's debt, then fall back to the planner's own ordering. This is
a *budgeted lexicographic* re-rank, not a weighted-sum scalarization (rejected in
doc V).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..planner.contracts import LedgerEntry, ScoredAction
from ..schema.actions import Action, ActionType, action_workload_uid
from ..schema.policy import PolicyConfig
from ..schema.state import SymbolicState, WorldState
from .benchmarks import RESOURCES, TenantDemand, fair_division
from .debt import FairnessDebtTracker

#: Fairness modes that activate the v2 ranker (everything except ``off``).
ACTIVE_MODES: frozenset[str] = frozenset({"drf", "maxmin", "pf", "nbs"})


# ──────────────────────────────────────────────────────────────────────────────
# H1 — tenant-level fair-division inputs from the symbolic state
# ──────────────────────────────────────────────────────────────────────────────
def tenant_demand_ray(podset_reqs: Sequence[Mapping[str, float]]) -> np.ndarray:
    """Aggregate a tenant's pending podset requests into a demand ray ``a_t``.

    The ray is the *direction* of multi-resource demand over :data:`RESOURCES`;
    its scale is irrelevant to the Leontief utility, so we keep the raw summed
    resource-hours (the benchmarks normalize internally). Resources never
    requested stay zero (non-binding in the Leontief min).
    """
    ray = np.zeros(len(RESOURCES), dtype=float)
    for req in podset_reqs:
        for i, r in enumerate(RESOURCES):
            ray[i] += float(req.get(r, 0.0))
    return ray


def tenant_demands_from_state(
    state: SymbolicState | WorldState,
    *,
    weights: Mapping[str, float] | None = None,
    nominal_service: Mapping[str, float] | None = None,
    only_pending: bool = True,
) -> list[TenantDemand]:
    """Build per-tenant :class:`TenantDemand` from the symbolic world state (H1).

    Demand rays come from each tenant's pending workloads' podsets (GPU, CPU,
    memory; network is approximated from the comm class as a light proxy when no
    explicit request exists). Weights default to the tenant's ClusterQueue
    ``fairshare_weight`` (operator policy, doc V A5/H3); nominal service defaults
    to 0 unless supplied by the caller (the NBS disagreement baseline, doc V D11,
    which is operator-defined, not self-reported).
    """
    sym = state.symbolic if isinstance(state, WorldState) else state

    # tenant -> list of podset request dicts
    reqs: dict[str, list[dict[str, float]]] = {}
    # tenant -> representative cluster_queue (for weight lookup)
    tenant_cq: dict[str, str] = {}
    for wl in sym.workloads.values():
        if only_pending and wl.status.value not in ("pending", "reserving"):
            continue
        tenant_cq.setdefault(wl.tenant_id, wl.cluster_queue_id)
        for ps in wl.podsets:
            net_proxy = 0.0
            # crude network demand proxy from comm class (doc V A4: effective
            # bundle includes network complements for gang jobs).
            cc = wl.comm_class.value
            if cc == "allreduce_heavy":
                net_proxy = float(ps.gpu_req) * ps.count
            elif cc == "pipeline":
                net_proxy = 0.5 * float(ps.gpu_req) * ps.count
            elif cc == "light":
                net_proxy = 0.1 * float(ps.gpu_req) * ps.count
            reqs.setdefault(wl.tenant_id, []).append(
                {
                    "gpu": float(ps.gpu_req) * ps.count,
                    "cpu": float(ps.cpu_req) * ps.count,
                    "mem": float(ps.mem_req_gb) * ps.count,
                    "net": net_proxy,
                }
            )

    demands: list[TenantDemand] = []
    for tenant_id, podset_reqs in reqs.items():
        ray = tenant_demand_ray(podset_reqs)
        if not np.any(ray > 0):
            continue
        # weight from ClusterQueue fairshare_weight, then caller override.
        w = 1.0
        cq_name = tenant_cq.get(tenant_id)
        if cq_name is not None and cq_name in sym.cluster_queues:
            w = float(sym.cluster_queues[cq_name].fairshare_weight) or 1.0
        if weights is not None and tenant_id in weights:
            w = float(weights[tenant_id])
        d = (
            float(nominal_service[tenant_id])
            if nominal_service is not None and tenant_id in nominal_service
            else 0.0
        )
        demands.append(
            TenantDemand(
                tenant_id=tenant_id,
                demand_ray=ray,
                weight=max(w, 1e-9),
                nominal_service=d,
            )
        )
    return demands


def horizon_capacities(
    state: SymbolicState | WorldState,
    horizon_s: float,
    *,
    gpu_hour_basis: bool = True,
) -> dict[str, float]:
    """Horizon-aggregated free capacity ``C_r(H)`` from topology domains (doc V A3).

    Sums currently-free GPU / CPU / memory across topology domains (Kueue TAS
    convention: ready & schedulable only) and multiplies by the horizon in hours
    to get resource-*hours* (the bundle unit, Eq1). Network capacity has no
    first-class schema field at demo-pod scale, so it is approximated as
    proportional to GPU capacity (cross-rack fabric scales with accelerators).
    """
    sym = state.symbolic if isinstance(state, WorldState) else state
    free_gpu = sum(d.free_gpu for d in sym.domains.values())
    free_cpu = sum(d.free_cpu for d in sym.domains.values())
    free_mem = sum(d.free_mem_gb for d in sym.domains.values())
    hours = (horizon_s / 3600.0) if gpu_hour_basis else 1.0
    return {
        "gpu": float(free_gpu) * hours,
        "cpu": float(free_cpu) * hours,
        "mem": float(free_mem) * hours,
        "net": float(free_gpu) * hours,  # proxy: fabric ∝ accelerators
    }


# ──────────────────────────────────────────────────────────────────────────────
# H2 — retained Kueue fairness observables
# ──────────────────────────────────────────────────────────────────────────────
def fairness_observables(
    state: SymbolicState | WorldState,
) -> dict[str, dict[str, float]]:
    """Export the native Kueue fairness observables retained on the schema (H2).

    Returns the share / usage signals v2 keys off, grouped by surface:
    ``cluster_queue_weighted_share``, ``cohort_weighted_share`` (max over member
    CQ shares as a cohort proxy), ``local_queue_weighted_share``,
    ``local_queue_usage`` (historical usage), and the entitlement primitives
    (weights, reserved / borrowed / pending GPU). These map 1:1 to
    ``kueue_cluster_queue_weighted_share``, ``kueue_cohort_weighted_share``,
    ``status.fairSharing.weightedShare``, and ``kueue_local_queue_resource_usage``.
    """
    sym = state.symbolic if isinstance(state, WorldState) else state
    cq_ws: dict[str, float] = {}
    cq_weight: dict[str, float] = {}
    cohort_ws: dict[str, float] = {}
    for cq in sym.cluster_queues.values():
        cq_ws[cq.name] = float(cq.weighted_share)
        cq_weight[cq.name] = float(cq.fairshare_weight)
        if cq.cohort_id is not None:
            cohort_ws[cq.cohort_id] = max(
                cohort_ws.get(cq.cohort_id, 0.0), float(cq.weighted_share)
            )
    lq_ws: dict[str, float] = {}
    lq_usage: dict[str, float] = {}
    for lq in sym.local_queues.values():
        lq_ws[lq.name] = float(lq.fairshare_weighted_share)
        lq_usage[lq.name] = float(lq.historical_usage)
    return {
        "cluster_queue_weighted_share": cq_ws,
        "cluster_queue_weight": cq_weight,
        "cohort_weighted_share": cohort_ws,
        "local_queue_weighted_share": lq_ws,
        "local_queue_usage": lq_usage,
    }


# ──────────────────────────────────────────────────────────────────────────────
# H8 / H10 — the safe-set re-ranker
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class FairnessRankResult:
    """Output of :func:`fairness_rank`: re-ordered safe actions + a ledger (H8).

    Attributes
    ----------
    ranked:
        The fairness-re-ordered safe ``ScoredAction`` list (winner first).
    ledger:
        One ``ranked`` :class:`LedgerEntry` per candidate with the Eq11 fairness
        evidence (serving tenant, benchmark service, under-served gap ``B - U``,
        positive debt, the action-conditioned drift delta, and the two budget
        verdicts) — the H8 explanation surface.
    applied:
        ``True`` only when ``fairness_mode`` is an active mode; ``False`` (no-op)
        when fairness is off.
    fairness_mode:
        The mode that produced the ranking (echo of ``policy.fairness_mode``).
    benchmark_service:
        ``tenant_id -> B_t`` benchmark Leontief service from :func:`fair_division`.
    under_served_gap:
        ``tenant_id -> (B_t - U_t)_+`` true under-served gap given the supplied
        realized-service history (doc V D31). Empty when no history was passed.
    drift_delta:
        ``action_id -> δ_FD(a|s)`` the action-conditioned drift delta vs hold
        (doc V D23). Populated only when per-action drift was supplied.
    within_budget:
        ``action_id -> bool`` — whether the candidate satisfies *both* Eq11
        budgets (drift ≤ δ_F and ``max_t Debt_t^+`` ≤ δ_D). Candidates that
        violate a budget are sorted *after* all in-budget candidates (a soft
        lexicographic demotion; fairness is a ranker, never a gate — doc V H10).
    """

    ranked: list[ScoredAction]
    ledger: list[LedgerEntry] = field(default_factory=list)
    applied: bool = False  # False when fairness_mode == off (no-op)
    fairness_mode: str = "off"
    benchmark_service: dict[str, float] = field(default_factory=dict)
    under_served_gap: dict[str, float] = field(default_factory=dict)
    drift_delta: dict[str, float] = field(default_factory=dict)
    within_budget: dict[str, bool] = field(default_factory=dict)


def _action_tenant(
    action: Action,
    sym: SymbolicState,
) -> str | None:
    """Resolve the tenant an action serves via its workload (H1 accounting)."""
    uid = action_workload_uid(action)
    if uid is None:
        return None
    wl = sym.workloads.get(uid)
    return wl.tenant_id if wl is not None else None


def fairness_rank(
    candidates: Sequence[ScoredAction],
    state: SymbolicState | WorldState,
    policy: PolicyConfig,
    *,
    horizon_s: float = 1800.0,
    debt_tracker: FairnessDebtTracker | None = None,
    weights: Mapping[str, float] | None = None,
    nominal_service: Mapping[str, float] | None = None,
    realized_service: Mapping[str, float] | None = None,
    action_drift_delta: Mapping[str, float] | None = None,
) -> FairnessRankResult:
    """Re-rank a planner safe set by the Eq11 budgeted-lexicographic rule (H10).

    This is the single public entry point the planner calls. Contract:

    * **No-op when off.** If ``policy.fairness_mode == "off"`` the input order is
      returned unchanged with ``applied=False`` (v1 default; doc V).
    * **Safe-set only.** Operates strictly on the already-kernel-safe, already-
      risk-gated ``ScoredAction`` list. It never adds, drops, or un-gates an
      action — fairness is a ranker, not a gate (doc V, H10 / A1).
    * **Budgeted lexicographic re-rank** (doc V, Eq11). The rule is

        ``a* ∈ argmax_{a ∈ A_safe : FD_H(a) ≤ δ_F and max_t Debt_t^+(a) ≤ δ_D}
          T_H(a)``, then tie-break by lower p95 wait, higher locality, lower
        fragmentation.

      Realised as a budgeted lexicographic sort key per candidate (lower sorts
      earlier ⇒ better):

        0. ``budget_violation`` — ``0`` if the candidate satisfies *both* Eq11
           budgets (its action-conditioned drift ``δ_FD(a|s) ≤ δ_F`` and the
           post-action worst-served debt ``max_t Debt_t^+ ≤ δ_D``), else ``1``.
           Budget-respecting candidates are always preferred; this is the
           lexicographic gate of Eq11, kept *soft* so the safe set is never
           emptied (a ranker, not a hard gate — doc V H10);
        1. ``-served_gap`` — actions toward the tenant with the largest true
           under-served gap ``(B_t - U_t)_+`` sort first (worst-served-first);
        2. ``-positive_debt`` — among equal gaps, serve the more-indebted tenant;
        3. the planner's own lexicographic key (preserves safety → statistical
           safety → wait → GPU-hours → locality → fragmentation as the Eq11
           throughput-first tie-break).

      ``hold`` actions are anchored at their planner position (they neither serve
      nor starve a tenant) so fairness never forces churn for its own sake.

    Parameters new in the wiring item
    ----------------------------------
    realized_service:
        ``tenant_id -> U_t`` realised cumulative Leontief service so far. When
        supplied the under-served gap is the *true* ``(B_t - U_t)_+`` (doc V
        D31), not the raw benchmark service; absent, it degrades to ``B_t``.
    action_drift_delta:
        ``action_id -> δ_FD(a|s)`` the action-conditioned fairness-drift delta
        vs hold (doc V D23/Eq11). When supplied it gates each candidate against
        ``policy.fairness_drift_budget``; absent, the drift budget is treated as
        satisfied (no per-action drift evidence to gate on).
    """
    sym = state.symbolic if isinstance(state, WorldState) else state
    mode = policy.fairness_mode
    cand_list = list(candidates)

    if mode not in ACTIVE_MODES:
        # v1 default path: strict no-op, preserve order and ranks.
        return FairnessRankResult(
            ranked=cand_list,
            ledger=[],
            applied=False,
            fairness_mode=mode,
        )

    # ---- benchmark service per tenant (H1 + benchmarks) -------------------
    demands = tenant_demands_from_state(
        state,
        weights=weights,
        nominal_service=nominal_service,
        only_pending=True,
    )
    benchmark_service: dict[str, float] = {}
    if demands:
        caps = horizon_capacities(state, horizon_s)
        try:
            div = fair_division(mode, demands, caps)
            benchmark_service = div.service_map()
        except ValueError:
            benchmark_service = {}

    # ---- true under-served gap (B - U) per tenant (doc V D31) -------------
    # When realised-service history is supplied, the under-served gap is the
    # positive part of (benchmark - realised). Without history we fall back to
    # the benchmark service itself (the gap a tenant *would* face from zero), so
    # the ranker is still worst-served-first but the ledger flags it as a proxy.
    under_served_gap: dict[str, float] = {}
    have_history = realized_service is not None
    for tid, bench in benchmark_service.items():
        realised = float(realized_service.get(tid, 0.0)) if have_history else 0.0
        under_served_gap[tid] = max(float(bench) - realised, 0.0)

    drift_budget = float(policy.fairness_drift_budget)
    debt_budget = float(policy.debt_budget)

    # ---- per-tenant current positive debt (slow-core memory) --------------
    def _pos_debt(tenant_id: str | None) -> float:
        if tenant_id is None or debt_tracker is None:
            return 0.0
        return debt_tracker.positive_debt(tenant_id)

    # Worst-served standing debt across all tracked tenants — the LHS of the
    # Eq11/D27 debt budget ``max_t Debt_t^+ ≤ δ_D``. Serving an under-served
    # tenant relieves debt; serving an already-satisfied tenant (or holding while
    # someone starves) leaves it standing, so we charge each candidate the
    # post-action worst-served debt (debt minus what *this* action repays).
    base_max_debt = (
        debt_tracker.max_positive_debt() if debt_tracker is not None else 0.0
    )

    ledger: list[LedgerEntry] = []
    drift_delta_out: dict[str, float] = {}
    within_budget: dict[str, bool] = {}
    decorated: list[tuple[tuple[float, ...], int, ScoredAction]] = []
    for idx, sc in enumerate(cand_list):
        tenant = _action_tenant(sc.action, sym)
        is_hold = sc.action.type == ActionType.HOLD
        pos_debt = _pos_debt(tenant)
        bench = float(benchmark_service.get(tenant, 0.0)) if tenant else 0.0
        gap = float(under_served_gap.get(tenant, 0.0)) if tenant else 0.0

        # δ_FD(a|s): per-action drift delta vs hold (doc V D23). Hold is its own
        # reference, so its delta is exactly 0; other actions take the supplied
        # value (None ⇒ unknown ⇒ treated as drift-neutral for the budget).
        if action_drift_delta is not None and sc.action_id in action_drift_delta:
            drift = float(action_drift_delta[sc.action_id])
        else:
            drift = 0.0
        drift_delta_out[sc.action_id] = drift

        # Post-action worst-served debt: an action serving an under-served tenant
        # repays up to its benchmark service this window, so the worst-served
        # debt it leaves standing is ``max(base_max_debt - repaid, debt_of_a)``
        # bounded below by 0. Hold repays nothing.
        repaid = 0.0 if is_hold or tenant is None else min(pos_debt, bench)
        post_max_debt = max(base_max_debt - repaid, 0.0)

        drift_ok = drift <= drift_budget
        debt_ok = post_max_debt <= debt_budget
        budget_ok = drift_ok and debt_ok
        within_budget[sc.action_id] = budget_ok
        violation = 0 if budget_ok else 1

        if is_hold or tenant is None:
            # neutral: keep at planner position, do not let fairness reorder it.
            served_gap = 0.0
            helps = 0.0
        else:
            # lower key sorts earlier ⇒ negate "good for fairness" signals.
            served_gap = -gap
            helps = -pos_debt

        planner_key = sc.lex_key if sc.lex_key else (float(sc.rank or idx),)
        # Eq11 budgeted-lexicographic key: budget verdict dominates, then
        # worst-served gap, then standing debt, then the planner ordering.
        fairness_key = (float(violation), served_gap, helps, *planner_key)
        decorated.append((fairness_key, idx, sc))

        ledger.append(
            LedgerEntry(
                kind="ranked",
                action_id=sc.action_id,
                detail=(
                    f"fairness_mode={mode}; tenant={tenant}; "
                    f"benchmark_service={bench:.4g}; under_served_gap={gap:.4g}; "
                    f"positive_debt={pos_debt:.4g}; drift_delta={drift:.4g}; "
                    f"within_budget={budget_ok}"
                ),
                evidence={
                    "tenant_id": tenant,
                    "fairness_mode": mode,
                    "benchmark_service": bench,
                    "under_served_gap": gap,
                    "positive_debt": pos_debt,
                    "drift_delta": drift,
                    "drift_budget": drift_budget,
                    "drift_within_budget": drift_ok,
                    "post_action_max_debt": post_max_debt,
                    "debt_budget": debt_budget,
                    "debt_within_budget": debt_ok,
                    "within_budget": budget_ok,
                    "is_hold": is_hold,
                    "has_realized_history": have_history,
                },
            )
        )

    # stable sort: fairness key first, original planner index as final tiebreak.
    decorated.sort(key=lambda item: (item[0], item[1]))
    ranked = [sc for _, _, sc in decorated]
    for new_rank, sc in enumerate(ranked):
        sc.rank = new_rank

    return FairnessRankResult(
        ranked=ranked,
        ledger=ledger,
        applied=True,
        fairness_mode=mode,
        benchmark_service=benchmark_service,
        under_served_gap=under_served_gap,
        drift_delta=drift_delta_out,
        within_budget=within_budget,
    )


__all__ = [
    "ACTIVE_MODES",
    "FairnessRankResult",
    "fairness_observables",
    "fairness_rank",
    "horizon_capacities",
    "tenant_demand_ray",
    "tenant_demands_from_state",
]
