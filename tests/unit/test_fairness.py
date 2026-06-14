# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Unit tests for the fairness wiring into the planner + eval (doc V Eq11/H9/H10).

These pin the PLAN-FAIRNESS-WIRING behaviour:

* **Off is a strict no-op.** With the default ``fairness_mode == "off"`` the
  planner's lexicographic order is unchanged — the golden stays byte-stable.
* **Eq11 budgeted-lexicographic re-rank.** When a mode is on, ``fairness_rank``
  prefers actions that serve the worst-served / most-indebted tenant first,
  gated by the drift budget δ_F and the worst-served debt budget δ_D, and never
  drops or un-gates a safe action (H10).
* **True under-served gap (B − U).** Supplied realised service shrinks the gap.
* **Ledger evidence.** Every re-ranked action carries the Eq11 fairness fields.
* **Runner slice-wise fairness.** FD_H / fair-regret / debt / gaming / PoA are
  aggregated and reported slice-wise (H9).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gpu_center_world_model.evaluation.metrics.planning import (
    FairnessEvalWindow,
    evaluate_fairness,
    fairness_windows_by_slice,
)
from gpu_center_world_model.evaluation.runner import (
    TrackInputs,
    compute_fairness_slices,
    run_benchmark,
    run_track_fairness,
    track_inputs_from_records,
)
from gpu_center_world_model.fairness.benchmarks import (
    TenantDemand,
    effective_bundle_utility,
    optimality_certificate,
    verify_axioms,
    weighted_drf,
    weighted_lexi_max_min,
    weighted_nash_bargaining,
    weighted_proportional_fair,
)
from gpu_center_world_model.fairness.debt import (
    FairnessDebtTracker,
    decay_factor,
    fair_regret,
    welfare_of_service,
)
from gpu_center_world_model.fairness.debt import (
    FairnessDebtTracker as _DebtTracker,
)
from gpu_center_world_model.fairness.drift import (
    DriftSample,
    FairnessDrift,
    action_conditioned_drift_delta,
    cohort_drift,
    fairness_drift_kpi,
    local_queue_drift,
)
from gpu_center_world_model.fairness.hooks import fairness_rank
from gpu_center_world_model.fairness.incentives import (
    TenantBehavior,
    aggregate_gaming,
    check_epsilon_share_stability,
    check_strategy_proofness,
    diagnose_gaming,
    empirical_price_of_anarchy,
    fairness_constrained_welfare,
)
from gpu_center_world_model.planner.contracts import ScoredAction
from gpu_center_world_model.planner.risk_gate import (
    MarginDistribution,
    assess_risk,
    cvar_from_margin,
    passes_risk_gate,
)
from gpu_center_world_model.schema.actions import AdmitAndPlace, Hold
from gpu_center_world_model.schema.entities import (
    LocalQueue,
    PodSet,
    TopologyDomain,
    TopologyLevel,
    Workload,
    WorkloadStatus,
)
from gpu_center_world_model.schema.kpis import (
    HazardChannel,
    HazardForecast,
    KPIForecastBundle,
)
from gpu_center_world_model.schema.policy import (
    DEFAULT_CVAR_BUDGET,
    PolicyConfig,
    RiskThresholds,
)
from gpu_center_world_model.schema.state import SymbolicState


# ──────────────────────────────────────────────────────────────────────────────
# Tiny two-tenant fixture built directly from the schema (no data engine).
# ──────────────────────────────────────────────────────────────────────────────
def _wl(uid: str, tenant: str, gpu: int) -> Workload:
    return Workload(
        workload_uid=uid,
        tenant_id=tenant,
        local_queue_id=f"lq_{tenant}",
        cluster_queue_id="cq1",
        status=WorkloadStatus.PENDING,
        podsets=[
            PodSet(
                workload_uid=uid, name="p", count=1,
                gpu_req=gpu, cpu_req=1, mem_req_gb=8,
            )
        ],
    )


def _two_tenant_state(
    *, usage_a: float = 0.0, usage_b: float = 0.0
) -> SymbolicState:
    dom = TopologyDomain(
        domain_id="rack0",
        level=TopologyLevel.RACK,
        free_gpu=16,
        free_cpu=64.0,
        free_mem_gb=256.0,
    )
    return SymbolicState(
        workloads={"wA": _wl("wA", "tenantA", 4), "wB": _wl("wB", "tenantB", 4)},
        domains={"rack0": dom},
        local_queues={
            "lq_tenantA": LocalQueue(
                name="lq_tenantA",
                namespace="ns",
                cluster_queue_ref="cq1",
                tenant_id="tenantA",
                historical_usage=usage_a,
            ),
            "lq_tenantB": LocalQueue(
                name="lq_tenantB",
                namespace="ns",
                cluster_queue_ref="cq1",
                tenant_id="tenantB",
                historical_usage=usage_b,
            ),
        },
    )


def _place(uid: str) -> AdmitAndPlace:
    return AdmitAndPlace(
        workload_uid=uid,
        cluster_queue_id="cq1",
        resource_flavor_id="f",
        domain_policy="preferred",
        domain_id_or_set=("rack0",),
    )


def _sa(aid: str, action, lex=(0.0, 0.0, 5.0)) -> ScoredAction:
    return ScoredAction(
        action_id=aid,
        action=action,
        forecast=KPIForecastBundle(action_id=aid),
        lex_key=lex,
        delta_kpis={"locality": 0.0, "fragmentation": 0.0},
    )


def _candidates() -> list[ScoredAction]:
    # Equal planner lex keys so fairness is the deciding factor.
    return [
        _sa("hold", Hold(workload_uid="wA")),
        _sa("placeA", _place("wA")),
        _sa("placeB", _place("wB")),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# fairness_rank: off is a strict no-op
# ──────────────────────────────────────────────────────────────────────────────
def test_fairness_off_is_strict_noop():
    cands = _candidates()
    res = fairness_rank(cands, _two_tenant_state(), PolicyConfig())  # default off
    assert not res.applied
    assert res.fairness_mode == "off"
    assert [c.action_id for c in res.ranked] == ["hold", "placeA", "placeB"]
    assert res.ledger == []


# ──────────────────────────────────────────────────────────────────────────────
# Eq11 worst-served-first re-rank
# ──────────────────────────────────────────────────────────────────────────────
def test_fairness_prefers_more_indebted_tenant():
    # tenantB carries standing positive debt; tenantA does not. The placement
    # serving tenantB must rank ahead of the one serving tenantA.
    state = _two_tenant_state()
    tracker = FairnessDebtTracker(half_life_s=3600.0)
    tracker.update(
        {"tenantA": 10.0, "tenantB": 0.0},
        {"tenantA": 5.0, "tenantB": 8.0},
        ts_s=0.0,
    )
    pol = PolicyConfig(fairness_mode="pf")
    res = fairness_rank(_candidates(), state, pol, debt_tracker=tracker)
    assert res.applied
    order = [c.action_id for c in res.ranked]
    assert order.index("placeB") < order.index("placeA")
    # Ranks are dense from 0 after re-ranking.
    assert [c.rank for c in res.ranked] == list(range(len(res.ranked)))


def test_true_under_served_gap_uses_realized_history():
    # With realised history, the gap is (B - U)_+: a tenant already served above
    # benchmark has zero gap, so the other tenant's placement is preferred.
    state = _two_tenant_state()
    pol = PolicyConfig(fairness_mode="pf")
    res = fairness_rank(
        _candidates(),
        state,
        pol,
        realized_service={"tenantA": 1000.0, "tenantB": 0.0},
    )
    assert res.applied
    # tenantA is fully served (gap 0), tenantB under-served → placeB first.
    order = [c.action_id for c in res.ranked]
    assert order.index("placeB") < order.index("placeA")
    assert res.under_served_gap["tenantA"] == 0.0
    assert res.under_served_gap["tenantB"] >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Budget gates (drift δ_F and debt δ_D)
# ──────────────────────────────────────────────────────────────────────────────
def test_drift_budget_demotes_over_budget_action():
    state = _two_tenant_state()
    # placeA worsens drift beyond budget; placeB is within budget.
    pol = PolicyConfig(fairness_mode="pf", fairness_drift_budget=0.1)
    res = fairness_rank(
        _candidates(),
        state,
        pol,
        action_drift_delta={"placeA": 0.5, "placeB": 0.0, "hold": 0.0},
    )
    assert res.within_budget["placeA"] is False
    assert res.within_budget["placeB"] is True
    order = [c.action_id for c in res.ranked]
    # Over-budget placeA is demoted behind in-budget placeB.
    assert order.index("placeB") < order.index("placeA")


def test_debt_budget_flags_over_budget_when_debt_exceeds_delta_d():
    state = _two_tenant_state()
    tracker = FairnessDebtTracker(half_life_s=0.0)  # no decay
    # Push worst-served standing debt to 100, far above a tight δ_D.
    tracker.update(
        {"tenantA": 0.0, "tenantB": 0.0},
        {"tenantA": 100.0, "tenantB": 0.0},
        ts_s=0.0,
    )
    pol = PolicyConfig(fairness_mode="pf", debt_budget=1.0)
    res = fairness_rank(_candidates(), state, pol, debt_tracker=tracker)
    # Hold repays nothing, so it leaves the worst-served debt standing > δ_D.
    assert res.within_budget["hold"] is False
    # Every candidate still survives (ranker, not a gate): all three present.
    assert {c.action_id for c in res.ranked} == {"hold", "placeA", "placeB"}


def test_fairness_never_drops_or_adds_actions():
    state = _two_tenant_state()
    pol = PolicyConfig(fairness_mode="drf")
    res = fairness_rank(_candidates(), state, pol)
    assert {c.action_id for c in res.ranked} == {"hold", "placeA", "placeB"}


# ──────────────────────────────────────────────────────────────────────────────
# Ledger evidence (H8)
# ──────────────────────────────────────────────────────────────────────────────
def test_ledger_carries_eq11_fairness_evidence():
    state = _two_tenant_state()
    pol = PolicyConfig(fairness_mode="pf")
    res = fairness_rank(_candidates(), state, pol)
    assert res.applied
    assert len(res.ledger) == 3
    ev = res.ledger[0].evidence
    for key in (
        "tenant_id",
        "fairness_mode",
        "benchmark_service",
        "under_served_gap",
        "positive_debt",
        "drift_delta",
        "drift_within_budget",
        "post_action_max_debt",
        "debt_within_budget",
        "within_budget",
    ):
        assert key in ev


# ──────────────────────────────────────────────────────────────────────────────
# Planner plan() integration: off byte-stable, on re-ranks + ledger
# ──────────────────────────────────────────────────────────────────────────────
def test_plan_fairness_off_matches_default_path(
    kernel, demo_state, queued_workload_uid, permissive_policy
):
    # Use a benign in-process model so the planner produces a clean recommendation.
    from test_planner import _BenignModel

    from gpu_center_world_model.planner import ShieldedPlanner

    planner = ShieldedPlanner(kernel, _BenignModel(), model_version="x")
    res_default = planner.plan(
        demo_state, queued_workload_uid, 900, permissive_policy
    )
    # Explicit off policy must give an identical recommendation + ordering.
    off_policy = PolicyConfig(
        risk=permissive_policy.risk,
        abstention=permissive_policy.abstention,
        fairness_mode="off",
    )
    res_off = planner.plan(demo_state, queued_workload_uid, 900, off_policy)
    assert (
        res_off.recommended_action_id == res_default.recommended_action_id
    )
    assert [c.action_id for c in res_off.candidates] == [
        c.action_id for c in res_default.candidates
    ]
    # No fairness ledger entries on the off path.
    assert not any(
        e.kind == "ranked" and "within_budget" in e.evidence
        for e in res_off.ledger
    )


def test_plan_fairness_on_records_ledger_and_stays_in_safe_set(
    kernel, demo_state, queued_workload_uid, permissive_policy
):
    from test_planner import _BenignModel

    from gpu_center_world_model.planner import (
        ShieldedPlanner,
        enumerate_safe_candidates,
    )

    pol = PolicyConfig(
        risk=permissive_policy.risk,
        abstention=permissive_policy.abstention,
        fairness_mode="pf",
    )
    tracker = FairnessDebtTracker(half_life_s=3600.0)
    planner = ShieldedPlanner(
        kernel, _BenignModel(), model_version="x", debt_tracker=tracker
    )
    res = planner.plan(demo_state, queued_workload_uid, 900, pol)
    assert not res.abstain
    # Fairness evidence is in the ledger.
    fair = [
        e for e in res.ledger if e.kind == "ranked" and "within_budget" in e.evidence
    ]
    assert fair, "fairness-on plan must record Eq11 fairness ledger entries"
    # Recommendation is still a member of the kernel-safe set (H10 / constructive).
    safe = {
        aid
        for aid, _ in enumerate_safe_candidates(
            kernel, demo_state, queued_workload_uid, pol
        )
    }
    assert res.recommended_action_id in safe
    # Ranks remain dense from 0.
    assert [c.rank for c in res.candidates] == list(range(len(res.candidates)))


# ──────────────────────────────────────────────────────────────────────────────
# Runner: slice-wise fairness aggregation (H9)
# ──────────────────────────────────────────────────────────────────────────────
def _fwins() -> list[FairnessEvalWindow]:
    return [
        FairnessEvalWindow(
            "w1",
            fd_h=0.2,
            benchmark_welfare=3.0,
            realized_welfare=2.5,
            per_tenant_debt={"tA": 5.0, "tB": -1.0},
            per_tenant_gaming={"tA": {"priority_inflation": 0.8}},
        ),
        FairnessEvalWindow(
            "w2",
            fd_h=0.4,
            benchmark_welfare=4.0,
            realized_welfare=4.0,
            per_tenant_debt={"tA": 3.0, "tB": 0.5},
            per_tenant_gaming={"tB": {"queue_hopping": 0.1}},
        ),
    ]


def test_evaluate_fairness_aggregates_fd_regret_debt_gaming():
    rep = evaluate_fairness(_fwins(), poa=1.3)
    assert rep.n_windows == 2
    assert abs(rep.fd_h_mean - 0.3) < 1e-9
    assert abs(rep.fd_h_max - 0.4) < 1e-9
    assert abs(rep.fair_regret_mean - 0.25) < 1e-9
    assert rep.max_positive_debt == 5.0
    assert rep.per_tenant_max_debt["tA"] == 5.0
    assert rep.n_flagged_gaming == 1  # tA flagged on priority_inflation > 0.5
    assert rep.poa == 1.3


def test_runner_reports_fairness_slice_wise():
    inputs = TrackInputs(
        fairness_windows=_fwins(),
        fairness_slice_labels={"w1": "topology_shift", "w2": "nominal"},
        fairness_poa=1.3,
    )
    assert run_track_fairness(inputs) is not None
    slices = compute_fairness_slices(inputs)
    assert "_overall" in slices
    assert set(slices["by_slice"].keys()) == {"topology_shift", "nominal"}
    report = run_benchmark(inputs)
    assert "_fairness" in report.slices
    assert report.metadata.get("fairness") is not None
    assert report.metadata["n_fairness_windows"] == 2


def test_fairness_windows_round_trip_through_json_ingestion():
    payload = {
        "fairness_windows": [
            {
                "window_id": "w1",
                "fd_h": 0.2,
                "benchmark_welfare": 3.0,
                "realized_welfare": 2.5,
                "per_tenant_debt": {"tA": 5.0},
                "per_tenant_gaming": {"tA": {"priority_inflation": 0.8}},
            }
        ],
        "fairness_slice_labels": {"w1": "s"},
        "fairness_poa": 1.3,
    }
    ti = track_inputs_from_records(payload)
    assert len(ti.fairness_windows) == 1
    assert ti.fairness_poa == 1.3
    assert ti.fairness_windows[0].per_tenant_debt["tA"] == 5.0


def test_fairness_windows_by_slice_buckets_correctly():
    buckets = fairness_windows_by_slice(
        _fwins(), {"w1": "a", "w2": "b"}
    )
    assert set(buckets) == {"a", "b"}
    assert len(buckets["a"]) == 1


# ══════════════════════════════════════════════════════════════════════════════
#  TEST-FAIRNESS-LANE — fair-division benchmark properties, drift/debt math,
#  gaming/SP/PoA diagnostics, axiom checks, and the CVaR-gate sanity (doc V
#  fairness_axioms_satisfied + per-benchmark property tables, H9 hooks; doc M E5).
#
#  These pin the *benchmark-level* properties of the fairness layer, distinct
#  from the planner-wiring tests above. Every assertion exercises a doc-V (or, for
#  the CVaR gate, doc-M) property against the real implemented behaviour.
# ══════════════════════════════════════════════════════════════════════════════

_CAP4 = np.array  # ordered (gpu, cpu, mem, net) capacity helper


def _ray(gpu=0.0, cpu=0.0, mem=0.0, net=0.0) -> np.ndarray:
    return np.array([gpu, cpu, mem, net], dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# DRF — equalized weighted dominant shares + Pareto on canonical instances
# (doc V D4/Eq4/Eq5, Th2 property table: pareto / envy-free / sharing-incentive)
# ──────────────────────────────────────────────────────────────────────────────
def test_drf_canonical_ghodsi_instance_equalizes_dominant_shares():
    # Classic Ghodsi DRF example mapped onto the (gpu, cpu) axes: tenant A demands
    # <1, 4>, tenant B demands <3, 1>, capacity <9, 18>. The textbook solution is
    # A=3 tasks, B=2 tasks, with both dominant shares equalized at 2/3.
    A = TenantDemand("A", _ray(gpu=1.0, cpu=4.0))
    B = TenantDemand("B", _ray(gpu=3.0, cpu=1.0))
    cap = _CAP4([9.0, 18.0, 0.0, 0.0])
    div = weighted_drf([A, B], cap)

    assert div.rule == "drf"
    np.testing.assert_allclose(div.service, [3.0, 2.0], atol=1e-6)
    # A's bundle is (3, 12); B's is (6, 2).
    np.testing.assert_allclose(div.bundles[0, :2], [3.0, 12.0], atol=1e-6)
    np.testing.assert_allclose(div.bundles[1, :2], [6.0, 2.0], atol=1e-6)
    # Eq5: dominant shares equalized at 2/3 (A bound by mem 12/18, B by cpu 6/9).
    np.testing.assert_allclose(div.dominant_share, [2.0 / 3.0, 2.0 / 3.0], atol=1e-6)
    # equal weights ⇒ weighted shares also equal.
    np.testing.assert_allclose(div.weighted_share, div.dominant_share, atol=1e-9)


def test_drf_weighted_equalizes_weighted_dominant_shares():
    # Doc V Eq5: at the progressive-filling optimum ds_t / w_t is equalized across
    # all simultaneously active unsatiated tenants. Give B twice the weight.
    A = TenantDemand("A", _ray(gpu=1.0, cpu=4.0), weight=1.0)
    B = TenantDemand("B", _ray(gpu=3.0, cpu=1.0), weight=2.0)
    cap = _CAP4([9.0, 18.0, 0.0, 0.0])
    div = weighted_drf([A, B], cap)
    # weighted dominant shares equalized.
    np.testing.assert_allclose(
        div.weighted_share[0], div.weighted_share[1], atol=1e-6
    )
    # B (heavier weight) gets a strictly larger raw dominant share.
    assert div.dominant_share[1] > div.dominant_share[0] + 1e-6


def test_drf_pareto_efficient_and_envy_free_on_canonical_instance():
    # Th2 property table: DRF is Pareto efficient, weighted-envy-free, and
    # sharing-incentive on the static divisible Leontief benchmark.
    A = TenantDemand("A", _ray(gpu=1.0, cpu=4.0))
    B = TenantDemand("B", _ray(gpu=3.0, cpu=1.0))
    cap = _CAP4([9.0, 18.0, 0.0, 0.0])
    div = weighted_drf([A, B], cap)
    rep = verify_axioms(div, [A, B], cap, probe_monotonicity=True)
    assert rep.pareto_efficient is True
    assert rep.weighted_envy_free is True
    assert rep.sharing_incentive is True
    assert rep.max_weighted_envy <= 1e-6


def test_drf_satiation_freezes_satiated_tenant_and_fills_others():
    # Satiation cap: A is satisfied after 1 task; the remaining capacity flows to
    # B (progressive filling continues on the unfrozen tenant).
    A = TenantDemand("A", _ray(gpu=1.0, cpu=4.0), max_service=1.0)
    B = TenantDemand("B", _ray(gpu=3.0, cpu=1.0))
    cap = _CAP4([9.0, 18.0, 0.0, 0.0])
    div = weighted_drf([A, B], cap)
    # A is held exactly at its satiation cap, never beyond it.
    assert abs(div.service[0] - 1.0) < 1e-6
    # B fills strictly past the unsatiated equalized point (gets more than 2).
    assert div.service[1] > 2.0 + 1e-6
    # No resource is overflowed.
    used = div.bundles.sum(axis=0)
    assert np.all(used <= cap + 1e-6)


def test_drf_staggered_freeze_respects_sequential_binding():
    # Staggered freeze: A demands gpu only; B demands gpu + cpu. The gpu resource
    # saturates first (freezing A's growth via that binding resource) while B
    # keeps filling against cpu. The result must still equalize weighted shares
    # among co-active tenants and never overflow any resource.
    A = TenantDemand("A", _ray(gpu=2.0))
    B = TenantDemand("B", _ray(gpu=1.0, cpu=3.0))
    cap = _CAP4([10.0, 9.0, 0.0, 0.0])
    div = weighted_drf([A, B], cap)
    used = div.bundles.sum(axis=0)
    assert np.all(used <= cap + 1e-6)
    # gpu is the first binding resource and is saturated.
    assert abs(used[0] - cap[0]) < 1e-6
    # both tenants are active and weighted shares are equalized at the optimum.
    assert div.service[0] > 0 and div.service[1] > 0
    np.testing.assert_allclose(
        div.weighted_share[0], div.weighted_share[1], atol=1e-6
    )


def test_drf_blocks_tenant_demanding_zero_capacity_resource():
    # A tenant that needs a resource with zero capacity is Leontief-infeasible and
    # gets zero service; the other tenant is unaffected.
    A = TenantDemand("A", _ray(gpu=1.0, net=1.0))  # net cap is 0 below
    B = TenantDemand("B", _ray(gpu=1.0))
    cap = _CAP4([8.0, 0.0, 0.0, 0.0])
    div = weighted_drf([A, B], cap)
    assert div.service[0] == 0.0
    assert div.service[1] > 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Weighted lexicographic max-min — egalitarian weighted-service equalization
# ──────────────────────────────────────────────────────────────────────────────
def test_maxmin_equalizes_weighted_service():
    # Doc V D6/Eq6: weighted max-min over Leontief service. With a single binding
    # resource and weights 1:2, weighted service u_t / w_t is equalized.
    A = TenantDemand("A", _ray(gpu=1.0), weight=1.0)
    B = TenantDemand("B", _ray(gpu=1.0), weight=2.0)
    cap = _CAP4([6.0, 0.0, 0.0, 0.0])
    div = weighted_lexi_max_min([A, B], cap)
    weighted_service = div.service / np.array([1.0, 2.0])
    np.testing.assert_allclose(weighted_service[0], weighted_service[1], atol=1e-6)
    # capacity is exactly consumed (egalitarian fill saturates the resource).
    assert abs(div.bundles.sum(axis=0)[0] - 6.0) < 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# PF / NBS — KKT/Pareto sanity + NBS individual-rationality d_t floors
# (doc V Eq7/Th4, Eq9/Th5)
# ──────────────────────────────────────────────────────────────────────────────
def test_pf_single_resource_splits_proportional_to_weights():
    # Weighted PF on a single binding resource is the analytic water-fill:
    # u_t ∝ w_t. Weights 1:3 over capacity 4 ⇒ services 1, 3.
    A = TenantDemand("A", _ray(gpu=1.0), weight=1.0)
    B = TenantDemand("B", _ray(gpu=1.0), weight=3.0)
    cap = _CAP4([4.0, 0.0, 0.0, 0.0])
    pf = weighted_proportional_fair([A, B], cap)
    np.testing.assert_allclose(pf.service, [1.0, 3.0], atol=1e-4)


def test_pf_kkt_certificate_certifies_optimum_and_pareto():
    A = TenantDemand("A", _ray(gpu=1.0), weight=1.0)
    B = TenantDemand("B", _ray(gpu=1.0), weight=3.0)
    cap = _CAP4([4.0, 0.0, 0.0, 0.0])
    pf = weighted_proportional_fair([A, B], cap)
    cert = optimality_certificate(pf, [A, B])
    assert cert.rule == "pf"
    assert cert.certified is True
    assert cert.stationarity_residual < 1e-4
    assert cert.primal_feasibility < 1e-6
    assert cert.dual_feasibility >= -1e-6  # dual feasibility λ ≥ 0
    # Pareto efficiency holds (Th4: positive concave utilities).
    rep = verify_axioms(pf, [A, B], cap)
    assert rep.pareto_efficient is True


def test_nbs_respects_individual_rationality_disagreement_floor():
    # Doc V Th5: NBS is individually rational — every tenant does at least as well
    # as its stand-alone disagreement (nominal-quota) service d_t. A's floor is 2.
    A = TenantDemand("A", _ray(gpu=1.0), nominal_service=2.0)
    B = TenantDemand("B", _ray(gpu=1.0), weight=3.0)
    cap = _CAP4([6.0, 0.0, 0.0, 0.0])
    nbs = weighted_nash_bargaining([A, B], cap)
    assert nbs.rule == "nbs"
    assert nbs.service[0] >= 2.0 - 1e-6  # IR: A never below its floor
    assert nbs.service[1] >= 0.0
    # disagreement vector is carried in meta for the debt tracker.
    assert nbs.meta["disagreement"][0] == pytest.approx(2.0)
    # KKT certificate certifies optimality above the disagreement points.
    cert = optimality_certificate(nbs, [A, B])
    assert cert.rule == "nbs"
    assert cert.certified is True


def test_nbs_reduces_to_pf_when_all_disagreements_zero():
    # Doc V Eq10: with all d_t = 0 the NBS welfare maximizer reduces to weighted
    # PF, so the allocations coincide.
    rays = [_ray(gpu=1.0), _ray(gpu=1.0)]
    pf = weighted_proportional_fair(
        [TenantDemand("A", rays[0], weight=1.0),
         TenantDemand("B", rays[1], weight=2.0)],
        _CAP4([6.0, 0.0, 0.0, 0.0]),
    )
    nbs = weighted_nash_bargaining(
        [TenantDemand("A", rays[0], weight=1.0, nominal_service=0.0),
         TenantDemand("B", rays[1], weight=2.0, nominal_service=0.0)],
        _CAP4([6.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(pf.service, nbs.service, atol=1e-4)


def test_optimality_certificate_trivial_for_progressive_filling_rules():
    # drf / maxmin are solved by exact water-filling; the KKT residual does not
    # apply, so the certificate is trivially certified.
    A = TenantDemand("A", _ray(gpu=1.0))
    B = TenantDemand("B", _ray(gpu=1.0))
    cap = _CAP4([4.0, 0.0, 0.0, 0.0])
    drf = weighted_drf([A, B], cap)
    cert = optimality_certificate(drf, [A, B])
    assert cert.rule == "drf"
    assert cert.certified is True


# ──────────────────────────────────────────────────────────────────────────────
# Axiom checks (doc V D12-D17 / fairness_axioms_satisfied)
# ──────────────────────────────────────────────────────────────────────────────
def test_axiom_report_fields_present_and_typed():
    A = TenantDemand("A", _ray(gpu=1.0, cpu=4.0))
    B = TenantDemand("B", _ray(gpu=3.0, cpu=1.0))
    cap = _CAP4([9.0, 18.0, 0.0, 0.0])
    div = weighted_drf([A, B], cap)
    rep = verify_axioms(div, [A, B], cap, probe_monotonicity=True)
    d = rep.as_dict()
    for key in (
        "rule",
        "pareto_efficient",
        "weighted_envy_free",
        "sharing_incentive",
        "population_monotone",
        "resource_monotone",
        "max_weighted_envy",
    ):
        assert key in d
    # the envy matrix is square over the tenant set.
    assert rep.envy_matrix.shape == (2, 2)
    # monotonicity probes ran (≥ 2 tenants) and returned a verdict, not None.
    assert rep.population_monotone is not None
    assert rep.resource_monotone is not None


def test_axiom_resource_monotonicity_holds_for_drf():
    # D17 probe: scaling capacity up by 10% must not make any tenant worse off.
    A = TenantDemand("A", _ray(gpu=1.0, cpu=4.0))
    B = TenantDemand("B", _ray(gpu=3.0, cpu=1.0))
    cap = _CAP4([9.0, 18.0, 0.0, 0.0])
    div = weighted_drf([A, B], cap)
    rep = verify_axioms(div, [A, B], cap, probe_monotonicity=True)
    assert rep.resource_monotone is True


def test_axiom_monotonicity_not_probed_for_single_tenant():
    A = TenantDemand("A", _ray(gpu=1.0))
    cap = _CAP4([4.0, 0.0, 0.0, 0.0])
    div = weighted_drf([A], cap)
    rep = verify_axioms(div, [A], cap, probe_monotonicity=True)
    # population monotonicity is not applicable with a single tenant.
    assert rep.population_monotone is None


def test_effective_bundle_utility_is_leontief_min():
    # Doc V D2/Eq2: u = min_r x_r / a_r over demanded resources; undemanded
    # resources (a_r == 0) impose no Leontief bottleneck.
    bundle = _ray(gpu=6.0, cpu=2.0, mem=100.0)
    ray = _ray(gpu=2.0, cpu=1.0)  # mem undemanded
    # min(6/2, 2/1) = min(3, 2) = 2.
    assert effective_bundle_utility(bundle, ray) == pytest.approx(2.0)
    # a tenant with no demand has zero utility.
    assert effective_bundle_utility(bundle, _ray()) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Fairness-drift KPI — Eq12/Eq13/Eq14 vs hand-computed spreads
# ──────────────────────────────────────────────────────────────────────────────
def test_cohort_drift_eq12_is_max_minus_min_spread():
    # Eq12: Delta_CQ_c = max_q s_q - min_q s_q over the cohort's contending CQs.
    shares = {"q1": 0.1, "q2": 0.5, "q3": 0.3, "q4": 0.9}
    cohort = {"q1": "c1", "q2": "c1", "q3": "c1", "q4": "c2"}
    cd = cohort_drift(shares, cohort)
    assert cd["c1"] == pytest.approx(0.5 - 0.1)  # 0.4
    assert cd["c2"] == 0.0  # single-member cohort ⇒ zero spread


def test_cohort_drift_restricts_to_contending_set():
    shares = {"q1": 0.1, "q2": 0.5, "q3": 0.9}
    cohort = {"q1": "c1", "q2": "c1", "q3": "c1"}
    cd = cohort_drift(shares, cohort, contending=["q1", "q2"])
    # q3 excluded ⇒ spread is over {0.1, 0.5} = 0.4, not 0.8.
    assert cd["c1"] == pytest.approx(0.4)


def test_local_queue_drift_eq13_is_max_minus_min_spread():
    # Eq13: Delta_LQ_k = max_l h_l - min_l h_l over contending LocalQueues.
    hist = {"l1": 0.2, "l2": 0.6, "l3": 0.6}
    cq_of = {"l1": "k", "l2": "k", "l3": "k"}
    lq = local_queue_drift(hist, cq_of)
    assert lq["k"] == pytest.approx(0.6 - 0.2)  # 0.4


def test_fairness_drift_kpi_eq14_time_average_and_blend():
    # Eq14: FD_H is the trapezoidal time-average of the blended spread. Linear
    # ramp 0.2→0.6 over the window averages to 0.4; max is 0.6.
    s1 = DriftSample(ts_s=0.0, cohort_spreads={"c": 0.2})
    s2 = DriftSample(ts_s=10.0, cohort_spreads={"c": 0.6})
    fd = fairness_drift_kpi([s1, s2], alpha=1.0)
    assert fd.mean == pytest.approx(0.4)
    assert fd.max == pytest.approx(0.6)
    assert fd.n_samples == 2
    # alpha blends the two layers: 0.25*cohort + 0.75*local.
    sb = DriftSample(
        ts_s=0.0, cohort_spreads={"c": 0.8}, cluster_queue_spreads={"k": 0.2}
    )
    assert sb.blended(0.25) == pytest.approx(0.25 * 0.8 + 0.75 * 0.2)


def test_fairness_drift_kpi_rejects_alpha_out_of_range():
    with pytest.raises(ValueError):
        fairness_drift_kpi([DriftSample(ts_s=0.0)], alpha=1.5)


def test_action_conditioned_drift_delta_d23():
    # D23: delta_FD(a|s) = E[FD_H|s,a] - E[FD_H|s,hold]; positive ⇒ worsens drift.
    fd_a = FairnessDrift(
        horizon_s=10, alpha=1.0, mean=0.5, p95=0.5, max=0.5, n_samples=1
    )
    fd_hold = FairnessDrift(
        horizon_s=10, alpha=1.0, mean=0.2, p95=0.2, max=0.2, n_samples=1
    )
    assert action_conditioned_drift_delta(fd_a, fd_hold) == pytest.approx(0.3)
    # an action that repairs drift yields a negative delta.
    assert action_conditioned_drift_delta(fd_hold, fd_a) == pytest.approx(-0.3)


# ──────────────────────────────────────────────────────────────────────────────
# Fairness debt — Eq17 recursive decay (doc V D29-D32 / H4)
# ──────────────────────────────────────────────────────────────────────────────
def test_decay_factor_eq17_half_life_semantics():
    # rho = exp(-ln2 * dt / T_half); at dt == T_half it is exactly 1/2.
    assert decay_factor(100.0, 100.0) == pytest.approx(0.5)
    assert decay_factor(200.0, 100.0) == pytest.approx(0.25)
    # half_life <= 0 disables decay (debt never forgives); dt <= 0 is also rho=1.
    assert decay_factor(100.0, 0.0) == 1.0
    assert decay_factor(0.0, 100.0) == 1.0
    assert decay_factor(50.0, 100.0) == pytest.approx(math.exp(-math.log(2.0) * 0.5))


def test_debt_tracker_eq17_recursive_update_and_decay():
    # Eq17: Debt_{t+1} = rho*Debt_t + (b_t - u_t). Window 0 with dt=0 ⇒ rho=1.
    tr = _DebtTracker(half_life_s=100.0)
    tr.update({"t": 0.0}, {"t": 10.0}, ts_s=0.0)
    assert tr.debt("t") == pytest.approx(10.0)  # 1*0 + (10 - 0)
    # Next window 100s later ⇒ rho=0.5, no new service: 0.5*10 + 0 = 5.
    tr.update({"t": 0.0}, {"t": 0.0}, ts_s=100.0)
    assert tr.debt("t") == pytest.approx(5.0)
    assert tr.max_positive_debt() == pytest.approx(5.0)


def test_debt_positive_part_clamps_overserved_tenant():
    # An over-served tenant carries negative debt; Debt_t^+ clamps it to 0 so it
    # never *credits* the worst-served-debt budget.
    tr = _DebtTracker(half_life_s=0.0)  # no decay
    tr.update({"t": 100.0}, {"t": 0.0}, ts_s=0.0)
    assert tr.debt("t") == pytest.approx(-100.0)
    assert tr.positive_debt("t") == 0.0
    assert tr.max_positive_debt() == 0.0


def test_debt_update_from_benchmark_pulls_disagreement_baseline():
    # update_from_benchmark refreshes each tenant's disagreement baseline from an
    # NBS FairDivision's meta["disagreement"].
    A = TenantDemand("A", _ray(gpu=1.0), nominal_service=2.0)
    B = TenantDemand("B", _ray(gpu=1.0))
    div = weighted_nash_bargaining([A, B], _CAP4([6.0, 0.0, 0.0, 0.0]))
    tr = _DebtTracker(half_life_s=0.0)
    tr.update_from_benchmark({"A": 0.0, "B": 0.0}, div, ts_s=0.0)
    snap = tr.snapshot()
    assert snap["A"]["disagreement_baseline"] == pytest.approx(2.0)


def test_fair_regret_and_welfare_swf():
    # D32: Reg_fair = M(B) - M(U). PF welfare is Σ w_t log(u_t + ε).
    assert fair_regret(5.0, 3.0) == pytest.approx(2.0)
    w_high = welfare_of_service({"a": 4.0, "b": 4.0}, rule="pf")
    w_low = welfare_of_service({"a": 1.0, "b": 1.0}, rule="pf")
    assert w_high > w_low  # more service ⇒ higher Nash-log welfare
    # maxmin SWF is the min weighted service.
    assert welfare_of_service(
        {"a": 2.0, "b": 6.0}, rule="maxmin", weights={"a": 1.0, "b": 1.0}
    ) == pytest.approx(2.0)


# ──────────────────────────────────────────────────────────────────────────────
# Gaming detectors (doc V D24 / M1-M6, H9 hooks)
# ──────────────────────────────────────────────────────────────────────────────
def test_gaming_detector_flags_all_surfaces_for_a_gaming_tenant():
    b = TenantBehavior(
        tenant_id="t",
        tenant_set_priority=True,  # M1
        priority_changes_while_pending=4,  # M1
        new_local_queues_in_window=5,  # M2
        requested_gpu=1.0,
        realized_peak_gpu=2.0,  # M4 under-request (2x)
        submit_timestamps_s=tuple(float(i) for i in range(11)),  # M6 burst
    )
    g = diagnose_gaming(b)
    assert g.priority_inflation == pytest.approx(1.0)
    assert g.under_requesting == pytest.approx(1.0)
    assert g.queue_hopping > 0.5
    assert g.burst_submission > 0.5
    assert set(g.flagged()) == {
        "M1_priority_inflation",
        "M2_queue_hopping",
        "M4_under_requesting",
        "M6_burst_submission",
    }


def test_gaming_detector_clean_tenant_flags_nothing():
    b = TenantBehavior(
        tenant_id="clean",
        requested_gpu=10.0,
        realized_peak_gpu=5.0,  # used less than requested ⇒ no under-request
        distinct_local_queues=1,
        submit_timestamps_s=(0.0,),
    )
    g = diagnose_gaming(b)
    assert g.under_requesting == 0.0
    assert g.priority_inflation == 0.0
    assert g.flagged() == []


def test_under_requesting_detector_handles_zero_request():
    # Requested ~nothing but realized something ⇒ strong under-request (score 1).
    b = TenantBehavior(
        tenant_id="t", requested_gpu=0.0, realized_peak_gpu=8.0
    )
    assert diagnose_gaming(b).under_requesting == pytest.approx(1.0)


def test_aggregate_gaming_summarises_replay_pressure():
    diags = [
        diagnose_gaming(
            TenantBehavior(tenant_id="t1", tenant_set_priority=True)
        ),
        diagnose_gaming(TenantBehavior(tenant_id="t2")),
    ]
    agg = aggregate_gaming(diags)
    # one of two tenants exceeds the 0.5 flag threshold (priority inflation 0.6).
    assert agg["n_flagged"] == 1.0
    assert agg["mean_priority_inflation"] == pytest.approx(0.3)  # (0.6 + 0) / 2
    # empty input is a benign all-zero summary.
    assert aggregate_gaming([])["n_flagged"] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Restricted strategy-proofness condition logic (doc V D25 / C1-C6 / Th6)
# ──────────────────────────────────────────────────────────────────────────────
def test_sp_holds_when_all_conditions_met():
    pol = PolicyConfig(
        priority_trust_mode="operator_only", tenant_scope="tenant_id"
    )
    rep = check_strategy_proofness(
        pol,
        requests_verified=True,
        demand_rays_verifiable=True,
        disagreement_operator_defined=True,
        fairness_over_safe_set=True,
    )
    assert rep.benchmark_sp_holds is True
    assert all(rep.conditions.values())


def test_sp_fails_on_tenant_writable_priority_and_local_queue_scope():
    # C1 violated (tenant can write priority) ⇒ priority inflation live exploit.
    # C2 violated (LocalQueue-scoped accounting) ⇒ queue hopping resets history.
    pol = PolicyConfig(
        priority_trust_mode="tenant_claim", tenant_scope="local_queue"
    )
    rep = check_strategy_proofness(pol, requests_verified=True)
    assert rep.benchmark_sp_holds is False
    assert rep.conditions["C1_operator_controlled_priority_and_weights"] is False
    assert rep.conditions["C2_tenant_scoped_queue_membership"] is False
    assert rep.notes  # human-readable exploit notes are recorded


def test_sp_fails_when_requests_unverified():
    # C3 violated: requests not verified against telemetry ⇒ under-requesting (M4).
    pol = PolicyConfig()
    rep = check_strategy_proofness(pol, requests_verified=False)
    assert rep.conditions["C3_verified_resource_requests"] is False
    assert rep.benchmark_sp_holds is False


# ──────────────────────────────────────────────────────────────────────────────
# Price-of-anarchy edge cases (doc V D26-D28 / Eq15-Eq16)
# ──────────────────────────────────────────────────────────────────────────────
def test_fairness_constrained_welfare_is_gate_not_penalty():
    # D27: W_H = throughput if BOTH budgets hold, -inf otherwise (a gate).
    assert fairness_constrained_welfare(
        100.0, 0.1, 0.0, drift_budget=1.0, debt_budget=1.0
    ) == 100.0
    # drift over budget ⇒ -inf.
    assert fairness_constrained_welfare(
        100.0, 5.0, 0.0, drift_budget=1.0, debt_budget=1.0
    ) == float("-inf")
    # debt over budget ⇒ -inf.
    assert fairness_constrained_welfare(
        100.0, 0.1, 9.0, drift_budget=1.0, debt_budget=1.0
    ) == float("-inf")


def test_poa_both_feasible_is_welfare_ratio():
    # Eq16: hat_PoA = W_centralized / W_decentralized when both are feasible.
    poa = empirical_price_of_anarchy(
        100.0, 0.1, 0.0,
        50.0, 0.2, 0.0,
        drift_budget=1.0, debt_budget=10.0,
    )
    assert poa.centralized_feasible and poa.decentralized_feasible
    assert poa.poa == pytest.approx(2.0)


def test_poa_decentralized_budget_violation_dominates():
    # When the decentralized baseline violates a budget, the advisor strictly
    # dominates ⇒ PoA → ∞.
    poa = empirical_price_of_anarchy(
        100.0, 0.1, 0.0,
        50.0, 5.0, 0.0,  # decentralized drift over budget
        drift_budget=1.0, debt_budget=10.0,
    )
    assert poa.decentralized_feasible is False
    assert poa.poa == float("inf")


def test_poa_degenerate_decentralized_welfare():
    # Zero decentralized welfare with positive centralized welfare ⇒ ∞.
    poa = empirical_price_of_anarchy(
        100.0, 0.1, 0.0,
        0.0, 0.1, 0.0,
        drift_budget=1.0, debt_budget=10.0,
    )
    assert poa.poa == float("inf")
    assert "degenerate" in poa.note


# ──────────────────────────────────────────────────────────────────────────────
# Epsilon-share-stability best-response checker (doc V D28)
# ──────────────────────────────────────────────────────────────────────────────
def test_share_stability_stable_when_no_profitable_deviation():
    base = {"a": 10.0, "b": 10.0}
    devs = {"a": [("inflate", 9.0)], "b": [("hop", 8.0)]}  # all worse
    rep = check_epsilon_share_stability(
        ["a", "b"], lambda t: base[t], lambda t: devs[t], epsilon=0.0
    )
    assert rep.is_stable is True
    assert rep.max_gain == pytest.approx(0.0)


def test_share_stability_unstable_when_a_tenant_can_gain():
    base = {"a": 10.0, "b": 10.0}
    devs = {"a": [("inflate", 15.0)], "b": [("hop", 8.0)]}
    rep = check_epsilon_share_stability(
        ["a", "b"], lambda t: base[t], lambda t: devs[t], epsilon=0.0
    )
    assert rep.is_stable is False
    assert rep.max_gain == pytest.approx(5.0)
    assert rep.deviations[0].best_deviation == "inflate"
    assert rep.deviations[0].profitable is True


def test_share_stability_epsilon_slack_absorbs_small_gains():
    base = {"a": 10.0}
    rep = check_epsilon_share_stability(
        ["a"], lambda t: base[t], lambda t: [("x", 10.5)], epsilon=1.0
    )
    # a 0.5 gain is within the 1.0 stability slack ⇒ still ε-share-stable.
    assert rep.is_stable is True


# ──────────────────────────────────────────────────────────────────────────────
# CVaR gate (doc M E5) — a USABLE cvar_budget does not reject all candidates.
# Depends on the wave-4a risk-gate fix: RiskThresholds.usable_cvar_budget().
# ──────────────────────────────────────────────────────────────────────────────
def _benign_bundle() -> KPIForecastBundle:
    # A candidate whose every hazard sits well within the per-channel budgets.
    return KPIForecastBundle(
        action_id="a",
        hazards=[
            HazardForecast(
                channel=ch, horizon_s=900, probability=0.02, conformal_upper=0.04
            )
            for ch in HazardChannel
        ],
    )


def test_usable_cvar_budget_resolves_default_when_unsized():
    # cvar_alpha > 0 with the broken 0.0 default budget would reject every
    # candidate; usable_cvar_budget() substitutes DEFAULT_CVAR_BUDGET instead.
    risk = RiskThresholds(cvar_alpha=0.9, cvar_budget=0.0)
    assert risk.usable_cvar_budget() == pytest.approx(DEFAULT_CVAR_BUDGET)
    # an explicitly sized positive budget is honoured verbatim.
    assert RiskThresholds(
        cvar_alpha=0.9, cvar_budget=0.25
    ).usable_cvar_budget() == pytest.approx(0.25)
    # when the tail gate is off the raw value is returned (moot).
    assert RiskThresholds(cvar_alpha=0.0, cvar_budget=0.0).usable_cvar_budget() == 0.0


def test_cvar_gate_with_usable_budget_does_not_reject_benign_candidate():
    # The crux of the wave-4a fix: enabling tail aversion without sizing a budget
    # must NOT empty the safe set (doc M E1: a budget ranks feasibility, never
    # gates the whole set away).
    bundle = _benign_bundle()
    risk = RiskThresholds(cvar_alpha=0.9, cvar_budget=0.0)  # unsized → usable default
    assessment = assess_risk(bundle, risk, 900)
    assert assessment.passes is True
    assert assessment.violated_channels == []
    # the CVaR checks ran (cvar_alpha > 0) and each is within the usable budget.
    assert assessment.cvar_checks
    for cv, budget, exceeds in assessment.cvar_checks.values():
        assert budget == pytest.approx(DEFAULT_CVAR_BUDGET)
        assert exceeds is False
        assert cv <= budget + 1e-9


def test_cvar_gate_tight_explicit_budget_still_rejects_tail():
    # Sanity counterpart: a deliberately tiny explicit budget DOES reject the tail
    # mass, proving the gate is live (not a no-op) and the default is what saves
    # the benign case, not a disabled gate.
    bundle = _benign_bundle()
    risk = RiskThresholds(cvar_alpha=0.9, cvar_budget=0.001)
    assert passes_risk_gate(bundle, risk, 900) is False


def test_cvar_gate_disabled_when_alpha_zero():
    # cvar_alpha == 0 disables the tail gate entirely (the v1 default).
    bundle = _benign_bundle()
    risk = RiskThresholds(cvar_alpha=0.0)
    assert passes_risk_gate(bundle, risk, 900) is True
    assert cvar_from_margin(bundle, HazardChannel.HOTSPOT, 900, 0.0) == 0.0


def test_cvar_var_le_cvar_relation_holds():
    # doc M E5: VaR_α ≤ CVaR_α for the non-negative violation margin, and
    # CVaR_1 collapses onto the worst atom (the max).
    dist = MarginDistribution(values=(0.1, 0.3, 0.9), weights=(0.5, 0.3, 0.2),
                              source="grid")
    for alpha in (0.5, 0.9, 0.99):
        assert dist.var(alpha) <= dist.cvar(alpha) + 1e-9
    assert dist.cvar(1.0 - 1e-12) == pytest.approx(0.9, abs=1e-3)
