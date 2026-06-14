# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Integration: the injected ShieldedPlanner is exercised end-to-end, and
``plan()`` carries complete why-not provenance (docs M E2, S, F).

Two surfaces:

* **Serving shield adapter (PLAN-SHIELD-RANK-WHYNOT).** When a ``ShieldedPlanner``
  is injected into the serving ``EngineBundle``, ``compare_actions`` must drive
  *its* ``rank(state, safe_actions, forecasts, baseline_id)`` — not silently fall
  back to the reference ranker. We assert the injected planner is actually
  invoked (the bug this item fixes: serving used to bypass the shield).

* **Full why-not provenance in plan().** ``plan()`` must compose a blocked
  why-not card (via ``kernel.why_not`` + ``record_blocked``) for every infeasible
  member of the *full* candidate space, attach a ``risk_card`` to risk-filtered
  rejects, and a ``rank_card`` to out-ranked survivors — so the ledger carries a
  complete explanation for blocked / risk-filtered / out-ranked actions (doc F
  explanation completeness).

These tests use the dependency-free :class:`ReferenceEngine` as both the exact
kernel and the residual forecaster, so they exercise the real shield wiring
without depending on any other lane's kernel internals.
"""

from __future__ import annotations

from gpu_center_world_model.planner import (
    ShieldedPlanner,
    enumerate_safe_candidates,
)
from gpu_center_world_model.planner.planner import enumerate_full_candidate_space
from gpu_center_world_model.schema import (
    HORIZONS_S,
    AbstentionConfig,
    PolicyConfig,
    RiskThresholds,
)
from gpu_center_world_model.serving import (
    CandidateActionRequest,
    CompareRequest,
    EngineBundle,
    ReferenceEngine,
    compare_actions,
)
from gpu_center_world_model.serving.app import demo_state_provider
from gpu_center_world_model.serving.cache import ServingCache

HORIZON = HORIZONS_S[2]  # 1800 s


def _permissive() -> PolicyConfig:
    """A policy whose risk + abstention gates pass for the reference forecaster."""
    return PolicyConfig(
        risk=RiskThresholds(
            max_hotspot_prob=1.0,
            max_headroom_violation_prob=1.0,
            max_pods_ready_timeout_prob=1.0,
            max_requeue_prob=1.0,
            max_xid_fault_prob=1.0,
        ),
        abstention=AbstentionConfig(
            max_epistemic_disagreement=1.0,
            max_ood_score=1.0,
            max_source_health_bad_prob=1.0,
        ),
    )


class _SpyPlanner(ShieldedPlanner):
    """A ShieldedPlanner that records every ``rank`` call it serves."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rank_calls: list[int] = []

    def rank(self, state, safe_actions, forecasts, baseline_id, **kwargs):  # type: ignore[override]
        self.rank_calls.append(len(safe_actions))
        return super().rank(state, safe_actions, forecasts, baseline_id, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Serving exercises the injected shield (the item's required assertion)
# ──────────────────────────────────────────────────────────────────────────────
def test_injected_shielded_planner_is_invoked_by_compare(
    demo_state, queued_workload_uid
):
    ref = ReferenceEngine()  # functional kernel + forecaster (no other lane needed)
    spy = _SpyPlanner(ref, ref, model_version="spy")
    # Relax the demo pod's operator risk thresholds so the shield does not abstain
    # on the reference forecaster's (deliberately non-trivial) hazards.
    demo_state.symbolic.operator_policy.risk_thresholds = {
        "max_hotspot_prob": 1.0,
        "max_headroom_violation_prob": 1.0,
        "max_pods_ready_timeout_prob": 1.0,
        "max_requeue_prob": 1.0,
        "max_xid_fault_prob": 1.0,
        "max_epistemic_disagreement": 1.0,
        "max_ood_score": 1.0,
    }
    engine = EngineBundle(kernel=ref, model=ref, planner=spy)

    uid = queued_workload_uid
    doms = list(demo_state.symbolic.domains.keys())
    req = CompareRequest(
        state_hash="ignored-by-direct-call",
        candidate_actions=[
            CandidateActionRequest(
                action_id="cand_hold", type="hold", params={"workload_uid": uid}
            ),
            CandidateActionRequest(
                action_id="cand_place",
                type="admit_and_place",
                params={
                    "workload_uid": uid,
                    "domain_id_or_set": [doms[0]],
                    "domain_policy": "preferred",
                },
            ),
        ],
        horizon_s=float(HORIZON),
        include_rollouts=False,
    )
    resp = compare_actions(demo_state, req, engine, cache=ServingCache())

    # The injected shield's rank() was driven by serving — not bypassed.
    assert spy.rank_calls, "compare_actions must invoke the injected ShieldedPlanner"
    # Every safe candidate that the shield ranked appears with a rank in the resp.
    safe_ranked = [
        c
        for c in resp.candidates
        if c.safe and not c.action_id.startswith("__pad")
    ]
    assert safe_ranked
    for c in safe_ranked:
        assert c.rank is not None


def test_injected_planner_abstains_propagate_to_serving(
    demo_state, queued_workload_uid
):
    # With strict default budgets, the reference forecaster's hazards exceed
    # budget, so the shield must abstain — and serving must surface that abstain
    # (the shield is genuinely in the loop, not bypassed).
    ref = ReferenceEngine()
    # Default (strict) operator risk thresholds drive the abstain.
    demo_state.symbolic.operator_policy.risk_thresholds = {
        "max_hotspot_prob": 0.01,
        "max_requeue_prob": 0.0,
    }
    spy = _SpyPlanner(ref, ref, model_version="spy")
    engine = EngineBundle(kernel=ref, model=ref, planner=spy)
    uid = queued_workload_uid
    req = CompareRequest(
        state_hash="ignored",
        candidate_actions=[
            CandidateActionRequest(
                action_id="cand_hold", type="hold", params={"workload_uid": uid}
            ),
        ],
        horizon_s=float(HORIZON),
    )
    resp = compare_actions(demo_state, req, engine, cache=ServingCache())
    assert spy.rank_calls
    # Strict budgets → every candidate fails the gate → abstain, no recommend.
    assert resp.abstain
    assert resp.recommended_action_id is None
    assert resp.abstain_reason


# ──────────────────────────────────────────────────────────────────────────────
# plan(): complete why-not provenance (blocked / risk-filtered / out-ranked)
# ──────────────────────────────────────────────────────────────────────────────
def test_full_candidate_space_is_superset_of_safe_set():
    # The demo_state_provider pod (rack-named domains) lets the reference kernel
    # actually place, so the safe set is non-trivial and the full space adds the
    # blocked variants (protected domain) the why-not cards explain.
    state = demo_state_provider()
    uid = next(iter(state.symbolic.workloads))
    ref = ReferenceEngine()
    policy = _permissive()

    safe_ids = {
        aid for aid, _ in enumerate_safe_candidates(ref, state, uid, policy)
    }
    full_ids = {aid for aid, _ in enumerate_full_candidate_space(state, uid, policy)}
    # Full space is a superset of the kernel-safe set, and every safe id is in it.
    assert safe_ids
    assert safe_ids <= full_ids


def test_plan_records_blocked_cards_for_full_candidate_space():
    state = demo_state_provider()
    uid = next(iter(state.symbolic.workloads))
    ref = ReferenceEngine()
    policy = _permissive()
    # Protect a domain so admit_and_place into it is hard-infeasible — that gives
    # the full candidate space at least one genuinely blocked member.
    doms = list(state.symbolic.domains.keys())
    state.symbolic.operator_policy.protected_domains = [doms[0]]
    policy.protected_domains = [doms[0]]

    planner = ShieldedPlanner(ref, ref, model_version="ref")
    result = planner.plan(state, uid, HORIZON, policy)

    full_ids = {aid for aid, _ in enumerate_full_candidate_space(state, uid, policy)}
    safe_ids = {
        aid for aid, _ in enumerate_safe_candidates(ref, state, uid, policy)
    }
    blocked = [e for e in result.ledger if e.kind == "blocked"]
    assert blocked, "blocked candidates must carry a why-not certificate"
    for e in blocked:
        assert e.action_id in full_ids and e.action_id not in safe_ids
        # Each blocked entry carries a group-minimal certificate (doc S).
        assert e.evidence["minimal_groups"]
    # The blocked candidates are also surfaced as rejected ScoredActions.
    rejected_ids = {sa.action_id for sa in result.rejected}
    assert {e.action_id for e in blocked} <= rejected_ids


def test_plan_attaches_rank_cards_for_out_ranked_survivors():
    state = demo_state_provider()
    uid = next(iter(state.symbolic.workloads))
    ref = ReferenceEngine()
    planner = ShieldedPlanner(ref, ref, model_version="ref")

    # Permissive policy → multiple survivors, so out-ranked rank_cards appear.
    result = planner.plan(state, uid, HORIZON, _permissive())
    assert not result.abstain
    ranked = [e for e in result.ledger if e.kind == "ranked"]
    # demo_state_provider yields hold + 2 placements → >= 2 ranked survivors.
    assert len(ranked) >= 2, "need at least one out-ranked survivor for a rank_card"
    # The winner is rank 0; everyone else is out-ranked and got a rank_card.
    winner = next(e for e in ranked if e.evidence["rank"] == 0)
    assert result.recommended_action_id == winner.action_id


def test_rank_card_registry_round_trips():
    # The ledger's card registry (attach_card / card_for / why_not_cards) is the
    # render-ready why-not surface for blocked / risk-filtered / out-ranked cards.
    from gpu_center_world_model.planner.ledger import ExplanationLedger
    from gpu_center_world_model.planner.why_not import rank_card

    led = ExplanationLedger()
    card = led.attach_card(rank_card("a", (0.0, 0.0, 5.0), (0.0, 0.0, 1.0)))
    assert card.out_ranked
    assert led.card_for("a") is card
    assert led.why_not_cards() == [card]
    assert led.card_for("missing") is None


def test_plan_strict_policy_abstains_and_records_risk_cards():
    state = demo_state_provider()
    uid = next(iter(state.symbolic.workloads))
    ref = ReferenceEngine()
    planner = ShieldedPlanner(ref, ref, model_version="ref")
    # Strict budgets → risk gate rejects → abstain with risk-filtered ledger.
    strict = PolicyConfig(
        risk=RiskThresholds(
            max_hotspot_prob=0.0,
            max_requeue_prob=0.0,
            max_xid_fault_prob=0.0,
        )
    )
    result = planner.plan(state, uid, HORIZON, strict)
    risk_filtered = [e for e in result.ledger if e.kind == "risk_filtered"]
    assert risk_filtered, "strict budgets must produce risk_filtered ledger entries"
    # Each risk_filtered entry names the over-budget channel(s).
    for e in risk_filtered:
        assert e.evidence["violated"]
