# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Unit tests for the shielded, lexicographic, risk-sensitive planner (docs M/S/F).

The planner is a *selector*, not a generator. Its constructive zero-violation
property: every action it can ever emit is drawn from the exact kernel's safe
candidate set, so a recommendation is hard-safe by construction. These tests pin:

* **Containment** — the recommended action id is always a member of the
  kernel-certified safe set (or a still-safe fallback).
* **Lex ordering** — surviving candidates are sorted by their lexicographic key
  (lex-argmin), lower is better.
* **Abstention** — when the abstention gate trips (high disagreement / OOD /
  degraded telemetry), the planner abstains and falls back to a *safe* action.
"""

from __future__ import annotations

from gpu_center_world_model.planner import (
    PlannerResult,
    ShieldedPlanner,
    enumerate_safe_candidates,
)
from gpu_center_world_model.schema import (
    HORIZONS_S,
    AbstentionConfig,
    HazardChannel,
    HazardForecast,
    KPIChannel,
    KPIForecastBundle,
    PolicyConfig,
    QuantileForecast,
    RiskThresholds,
)
from gpu_center_world_model.serving import ReferenceEngine

HORIZON = HORIZONS_S[1]  # 900 s


# ──────────────────────────────────────────────────────────────────────────────
# A tiny, fully-controllable benign world model (passes every gate by default).
# ──────────────────────────────────────────────────────────────────────────────
class _BenignModel:
    """A deterministic forecaster emitting all hazard channels at near-zero risk.

    It conforms to the ``WorldModel`` Protocol (``forecast`` / ``score_actions``)
    so the planner uses it directly. Every modelled hazard channel is populated
    (so the risk gate has no missing-channel worst-case), the conformal upper
    bounds are tiny, and epistemic/OOD are zero — so the *default* operator policy
    yields a clean, non-abstaining recommendation.

    A small per-action utility (lower queue-wait delta for a placement than for a
    hold) gives a deterministic, checkable lexicographic ordering.
    """

    def __init__(self, *, epistemic: float = 0.0, ood: float = 0.0):
        self.epistemic = epistemic
        self.ood = ood

    def encode(self, state):  # pragma: no cover - not used by the planner path
        return state.symbolic

    def forecast(self, state, action, horizon_s):
        is_placement = action.type.value != "hold"
        # placement reduces queue wait (good); hold leaves it unchanged.
        qwait = -300.0 if is_placement else 0.0
        gpu_hours = 4.0 if is_placement else 0.0
        quantiles = [
            QuantileForecast(
                channel=KPIChannel.QUEUE_WAIT_DELTA_S,
                horizon_s=horizon_s,
                quantiles={0.05: qwait - 30, 0.5: qwait, 0.95: qwait + 30},
            ),
            QuantileForecast(
                channel=KPIChannel.ADMITTED_GPU_HOURS_DELTA,
                horizon_s=horizon_s,
                quantiles={0.05: gpu_hours, 0.5: gpu_hours, 0.95: gpu_hours},
            ),
            QuantileForecast(
                channel=KPIChannel.CROSS_RACK_RATE,
                horizon_s=horizon_s,
                quantiles={0.05: 0.0, 0.5: 0.0, 0.95: 0.0},
            ),
            QuantileForecast(
                channel=KPIChannel.TOPOLOGY_FRAGMENTATION_DELTA,
                horizon_s=horizon_s,
                quantiles={0.05: 0.0, 0.5: 0.0, 0.95: 0.0},
            ),
        ]
        hazards = [
            HazardForecast(ch, horizon_s, 0.001, conformal_upper=0.002)
            for ch in HazardChannel
        ]
        return KPIForecastBundle(
            action_id=getattr(action, "workload_uid", "") or action.type.value,
            quantiles=quantiles,
            hazards=hazards,
            epistemic_score=self.epistemic,
            ood_score=self.ood,
        )

    def score_actions(self, state, actions, horizon_s):
        return [self.forecast(state, a, horizon_s) for a in actions]


# ──────────────────────────────────────────────────────────────────────────────
# Containment: emitted action is always in the kernel-safe set
# ──────────────────────────────────────────────────────────────────────────────
def test_recommendation_is_in_safe_set(
    kernel, demo_state, queued_workload_uid, default_policy
):
    planner = ShieldedPlanner(kernel, _BenignModel(), model_version="benign")
    res = planner.plan(demo_state, queued_workload_uid, HORIZON, default_policy)
    assert isinstance(res, PlannerResult)
    assert res.recommended_action_id is not None
    assert not res.abstain

    safe_ids = {
        aid
        for aid, _ in enumerate_safe_candidates(
            kernel, demo_state, queued_workload_uid, default_policy
        )
    }
    assert res.recommended_action_id in safe_ids


def test_fallback_recommendation_is_also_in_safe_set(
    kernel, demo_state, queued_workload_uid
):
    # Force abstention; the conservative fallback must still be a safe action.
    policy = PolicyConfig(
        abstention=AbstentionConfig(
            max_epistemic_disagreement=0.1, fallback="conservative_rule"
        )
    )
    planner = ShieldedPlanner(kernel, _BenignModel(epistemic=0.9), model_version="x")
    res = planner.plan(demo_state, queued_workload_uid, HORIZON, policy)
    assert res.abstain
    if res.recommended_action_id is not None:
        safe_ids = {
            aid
            for aid, _ in enumerate_safe_candidates(
                kernel, demo_state, queued_workload_uid, policy
            )
        }
        assert res.recommended_action_id in safe_ids


# ──────────────────────────────────────────────────────────────────────────────
# Lexicographic ordering
# ──────────────────────────────────────────────────────────────────────────────
def test_candidates_are_lexicographically_sorted(
    kernel, demo_state, queued_workload_uid, default_policy
):
    planner = ShieldedPlanner(kernel, _BenignModel(), model_version="benign")
    res = planner.plan(demo_state, queued_workload_uid, HORIZON, default_policy)
    keys = [c.lex_key for c in res.candidates]
    assert keys == sorted(keys), "survivors must be lex-argmin ordered"
    # Ranks are assigned densely from 0 in order.
    assert [c.rank for c in res.candidates] == list(range(len(res.candidates)))


def test_top_ranked_is_the_recommendation_when_not_abstaining(
    kernel, demo_state, queued_workload_uid, default_policy
):
    planner = ShieldedPlanner(kernel, _BenignModel(), model_version="benign")
    res = planner.plan(demo_state, queued_workload_uid, HORIZON, default_policy)
    assert not res.abstain
    assert res.candidates
    assert res.candidates[0].action_id == res.recommended_action_id


def test_placement_outranks_hold_when_utility_favors_it(
    kernel, demo_state, queued_workload_uid, default_policy
):
    # The benign model gives placements a strictly better queue-wait delta, so the
    # lex winner should be a placement, not hold (utility differentiates a tie in
    # the equal hard/statistical-safety prefixes).
    planner = ShieldedPlanner(kernel, _BenignModel(), model_version="benign")
    res = planner.plan(demo_state, queued_workload_uid, HORIZON, default_policy)
    winner = res.candidates[0]
    assert winner.action.type.value == "admit_and_place"


# ──────────────────────────────────────────────────────────────────────────────
# Abstention
# ──────────────────────────────────────────────────────────────────────────────
def test_abstains_on_high_epistemic_disagreement(
    kernel, demo_state, queued_workload_uid
):
    policy = PolicyConfig(
        abstention=AbstentionConfig(max_epistemic_disagreement=0.2, fallback="hold")
    )
    planner = ShieldedPlanner(kernel, _BenignModel(epistemic=0.95), model_version="x")
    res = planner.plan(demo_state, queued_workload_uid, HORIZON, policy)
    assert res.abstain
    assert res.abstain_reason


def test_abstains_when_no_candidate_passes_risk_gate(
    kernel, demo_state, queued_workload_uid
):
    # A model whose hazards blow past every (strict) budget leaves no survivor.
    class _RiskyModel(_BenignModel):
        def forecast(self, state, action, horizon_s):
            b = super().forecast(state, action, horizon_s)
            for hz in b.hazards:
                hz.probability = 0.99
                hz.conformal_upper = 0.99
            return b

    policy = PolicyConfig(risk=RiskThresholds())  # strict default budgets
    planner = ShieldedPlanner(kernel, _RiskyModel(), model_version="x")
    res = planner.plan(demo_state, queued_workload_uid, HORIZON, policy)
    assert res.abstain
    assert not res.candidates  # everything was risk-filtered


def test_reference_engine_as_model_is_safe_under_permissive_policy(
    kernel, demo_state, queued_workload_uid, permissive_policy
):
    # End-to-end with the dependency-free reference forecaster: with permissive
    # budgets it recommends a real, safe action (no abstention path needed).
    ref = ReferenceEngine()
    planner = ShieldedPlanner(kernel, ref, model_version="reference")
    res = planner.plan(demo_state, queued_workload_uid, HORIZON, permissive_policy)
    assert not res.abstain
    assert res.recommended_action_id is not None
    safe_ids = {
        aid
        for aid, _ in enumerate_safe_candidates(
            kernel, demo_state, queued_workload_uid, permissive_policy
        )
    }
    assert res.recommended_action_id in safe_ids


def test_ledger_records_a_recommendation(
    kernel, demo_state, queued_workload_uid, default_policy
):
    planner = ShieldedPlanner(kernel, _BenignModel(), model_version="benign")
    res = planner.plan(demo_state, queued_workload_uid, HORIZON, default_policy)
    kinds = {e.kind for e in res.ledger}
    assert "recommended" in kinds
    # Provenance travels with the result.
    assert res.model_version == "benign"
    assert res.state_hash
