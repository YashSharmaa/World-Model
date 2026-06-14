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
from gpu_center_world_model.fairness.debt import FairnessDebtTracker
from gpu_center_world_model.fairness.hooks import fairness_rank
from gpu_center_world_model.planner.contracts import ScoredAction
from gpu_center_world_model.schema.actions import AdmitAndPlace, Hold
from gpu_center_world_model.schema.entities import (
    LocalQueue,
    PodSet,
    TopologyDomain,
    TopologyLevel,
    Workload,
    WorkloadStatus,
)
from gpu_center_world_model.schema.kpis import KPIForecastBundle
from gpu_center_world_model.schema.policy import PolicyConfig
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
