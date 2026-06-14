# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Unit tests for amortization heads + bounded search (doc M D1-D4 / E1).

PLAN-AMORTIZATION-SEARCH pins:

* **Distilled heads.** Proposal/value/abstention heads trained from offline
  full-enumeration replay recover the oracle action (proposal argmax), regress
  the scalar lex score S_t (value argmin), and conform to the
  ``AmortizationHeads`` Protocol.
* **Guarded inference.** The guarded proposal falls back to exhaustive scoring
  when the candidate count is below threshold or u_ω flags OOD; otherwise it
  keeps the top-k (always retaining hold).
* **Bounded search.** CEM/MPPI/MCTS over multi-step sequences find the
  lexicographically-best safe sequence via the aggregated E_lex key, and the
  value head warm-starts the first step.
* **Planner integration.** ``plan_sequence`` emits a safe first action; the
  exact ``plan`` path is unchanged when no heads are injected.
"""

from __future__ import annotations

import numpy as np

from gpu_center_world_model.model.contracts import AmortizationHeads
from gpu_center_world_model.planner.enumerate import (
    AmortizationDataset,
    CEMSearch,
    LinearAmortizationHeads,
    MCTSSearch,
    MPPISearch,
    aggregate_sequence_lex_key,
    guarded_proposal,
    train_amortization_heads,
)
from gpu_center_world_model.schema.actions import AdmitAndPlace, Hold
from gpu_center_world_model.schema.entities import (
    PodSet,
    TopologyDomain,
    TopologyLevel,
    Workload,
    WorkloadStatus,
)
from gpu_center_world_model.schema.state import SymbolicState, WorldState


# ──────────────────────────────────────────────────────────────────────────────
# Minimal state + action helpers (no data engine).
# ──────────────────────────────────────────────────────────────────────────────
def _state() -> WorldState:
    wl = Workload(
        workload_uid="w",
        tenant_id="t",
        local_queue_id="lq",
        cluster_queue_id="cq",
        status=WorkloadStatus.PENDING,
        podsets=[
            PodSet(
                workload_uid="w", name="p", count=1, gpu_req=4, cpu_req=1, mem_req_gb=8
            )
        ],
    )
    dom = TopologyDomain(
        domain_id="rack0",
        level=TopologyLevel.RACK,
        free_gpu=16,
        free_cpu=64.0,
        free_mem_gb=256.0,
    )
    return WorldState(
        symbolic=SymbolicState(workloads={"w": wl}, domains={"rack0": dom})
    )


_HOLD = Hold(workload_uid="w")
_PLACE_A = AdmitAndPlace(
    workload_uid="w",
    cluster_queue_id="cq",
    resource_flavor_id="f",
    domain_policy="required",
    domain_id_or_set=("rack0",),
)
_PLACE_B = AdmitAndPlace(
    workload_uid="w",
    cluster_queue_id="cq",
    resource_flavor_id="f",
    domain_policy="preferred",
    domain_id_or_set=(),
)


def _replay_dataset(n: int = 40) -> AmortizationDataset:
    """A replay where PLACE_A is consistently the oracle (lowest lex score)."""
    state = _state()
    windows = []
    for _ in range(n):
        actions = [_HOLD, _PLACE_A, _PLACE_B]
        lex_scores = [10.0, 1.0, 5.0]  # PLACE_A best
        abstain = [0.0, 0.0, 0.0]
        windows.append((state, actions, lex_scores, abstain))
    return AmortizationDataset.from_replay(windows)


# ──────────────────────────────────────────────────────────────────────────────
# Head distillation
# ──────────────────────────────────────────────────────────────────────────────
def test_trained_heads_conform_to_protocol():
    heads = train_amortization_heads(_replay_dataset())
    assert isinstance(heads, AmortizationHeads)


def test_proposal_head_recovers_oracle():
    heads = train_amortization_heads(_replay_dataset())
    state = _state()
    scores = heads.proposal_scores(state, [_HOLD, _PLACE_A, _PLACE_B], 1800)
    # PLACE_A is the oracle, so it must have the highest proposal score.
    assert int(np.argmax(scores)) == 1


def test_value_head_regresses_scalar_lex_score():
    heads = train_amortization_heads(_replay_dataset())
    state = _state()
    vals = [heads.value(state, a, 1800) for a in (_HOLD, _PLACE_A, _PLACE_B)]
    # Lower value is better; PLACE_A had the lowest S_t target (1.0).
    assert int(np.argmin(vals)) == 1


def test_abstain_head_in_unit_interval():
    heads = train_amortization_heads(_replay_dataset())
    state = _state()
    for a in (_HOLD, _PLACE_A, _PLACE_B):
        s = heads.abstain_score(state, a, 1800)
        assert 0.0 <= s <= 1.0


def test_empty_dataset_yields_benign_zero_heads():
    heads = train_amortization_heads(AmortizationDataset())
    assert np.allclose(heads.w_pi, 0.0)
    # zero abstain logits → sigmoid(0) = 0.5 (never trips a 0.5+ guard strictly).
    assert heads.abstain_score(_state(), _HOLD, 1800) == 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Guarded proposal
# ──────────────────────────────────────────────────────────────────────────────
def test_guarded_proposal_below_threshold_falls_back():
    heads = train_amortization_heads(_replay_dataset())
    state = _state()
    safe = [("hold", _HOLD), ("a", _PLACE_A), ("b", _PLACE_B)]
    gp = guarded_proposal(heads, state, safe, 1800, enumeration_threshold=6)
    assert gp.used_fallback
    assert gp.reason == "below_threshold"
    assert len(gp.actions) == 3  # full set scored


def test_guarded_proposal_topk_keeps_hold():
    heads = train_amortization_heads(_replay_dataset())
    state = _state()
    big = [("hold", _HOLD)] + [
        (
            f"x{i}",
            AdmitAndPlace(
                workload_uid="w",
                cluster_queue_id="cq",
                resource_flavor_id=f"f{i}",
                domain_policy="required",
                domain_id_or_set=("rack0",),
            ),
        )
        for i in range(8)
    ]
    gp = guarded_proposal(heads, state, big, 1800, top_k=3, enumeration_threshold=6)
    assert not gp.used_fallback
    assert gp.reason == "amortized_topk"
    # hold survives the pruning regardless of its proposal score.
    assert any(a.type.value == "hold" for _aid, a in gp.actions)


def test_guarded_proposal_u_omega_flag_falls_back():
    state = _state()

    class _Flagging(LinearAmortizationHeads):
        def abstain_score(self, state, action, horizon_s):
            return 0.9  # always OOD

    base = train_amortization_heads(_replay_dataset())
    flag = _Flagging(w_pi=base.w_pi, w_v=base.w_v, w_u=base.w_u)
    big = [("hold", _HOLD)] + [
        (
            f"x{i}",
            AdmitAndPlace(
                workload_uid="w",
                cluster_queue_id="cq",
                resource_flavor_id=f"f{i}",
                domain_policy="required",
                domain_id_or_set=("rack0",),
            ),
        )
        for i in range(8)
    ]
    gp = guarded_proposal(flag, state, big, 1800, enumeration_threshold=6)
    assert gp.used_fallback
    assert gp.reason == "u_omega_flag"
    assert len(gp.actions) == len(big)  # exhaustive fallback


# ──────────────────────────────────────────────────────────────────────────────
# Sequence energy aggregation + bounded search
# ──────────────────────────────────────────────────────────────────────────────
def test_aggregate_sequence_lex_key_sums_fields():
    agg = aggregate_sequence_lex_key([(0.0, 1.0, 2.0), (0.0, 0.0, 3.0)])
    assert agg == (0.0, 1.0, 5.0)
    assert aggregate_sequence_lex_key([]) == ()


def _sampler(state, step):
    return [_HOLD, _PLACE_A, _PLACE_B]


def _scorer(state, seq):
    keys = []
    for a in seq:
        if a is _PLACE_A:
            keys.append((0.0, 0.0, 1.0))
        elif a is _PLACE_B:
            keys.append((0.0, 0.0, 5.0))
        else:
            keys.append((0.0, 0.0, 10.0))
    return aggregate_sequence_lex_key(keys)


def test_cem_finds_optimal_sequence():
    heads = train_amortization_heads(_replay_dataset())
    b = CEMSearch(
        _sampler, _scorer, horizon=3, value_head=heads, rng=np.random.default_rng(0)
    )
    r = b.search(_state(), 1800)
    # The all-PLACE_A sequence (key sum 3.0) is optimal.
    assert r.best_key == (0.0, 0.0, 3.0)
    assert r.emitted_action is _PLACE_A
    assert len(r.best_sequence) == 3
    assert r.n_rollouts > 0


def test_mppi_finds_optimal_sequence():
    b = MPPISearch(_sampler, _scorer, horizon=3, rng=np.random.default_rng(1))
    r = b.search(_state(), 1800)
    assert r.best_key == (0.0, 0.0, 3.0)
    assert r.backend == "mppi"


def test_mcts_finds_optimal_sequence():
    heads = train_amortization_heads(_replay_dataset())
    b = MCTSSearch(
        _sampler,
        _scorer,
        horizon=3,
        value_head=heads,
        rng=np.random.default_rng(2),
        n_simulations=120,
    )
    r = b.search(_state(), 1800)
    assert r.best_key == (0.0, 0.0, 3.0)
    assert r.emitted_action is _PLACE_A


def test_value_head_warm_start_orders_first_step():
    # The value head argsorts the first-step candidates (lower value first).
    heads = train_amortization_heads(_replay_dataset())
    b = CEMSearch(_sampler, _scorer, horizon=1, value_head=heads)
    order = b._warm_first_step(_state(), [_HOLD, _PLACE_A, _PLACE_B], 1800)
    # PLACE_A (lowest predicted value) should be warm-started first.
    assert order[0] == 1


# ──────────────────────────────────────────────────────────────────────────────
# Planner integration (uses the real demo kernel)
# ──────────────────────────────────────────────────────────────────────────────
def test_plan_with_amortization_is_safe_and_records_ledger(
    kernel, demo_state, queued_workload_uid, permissive_policy
):
    from test_planner import _BenignModel

    from gpu_center_world_model.planner import (
        ShieldedPlanner,
        build_lex_key,
        enumerate_safe_candidates,
        scalar_embedding,
    )

    model = _BenignModel()
    safe = enumerate_safe_candidates(
        kernel, demo_state, queued_workload_uid, permissive_policy
    )
    acts = [a for _, a in safe]
    bundles = model.score_actions(demo_state, acts, 900)
    lex_scores = [
        scalar_embedding(build_lex_key(b, 900, risk=permissive_policy.risk))
        for b in bundles
    ]
    ds = AmortizationDataset.from_replay(
        [(demo_state, acts, lex_scores, [0.0] * len(acts))] * 20
    )
    heads = train_amortization_heads(ds)

    planner = ShieldedPlanner(kernel, model, model_version="x", amortization=heads)
    res = planner.plan(demo_state, queued_workload_uid, 900, permissive_policy)
    safe_ids = {aid for aid, _ in safe}
    assert res.recommended_action_id in safe_ids
    # The guard decision is recorded for audit.
    amort = [e for e in res.ledger if e.action_id == "__amortization__"]
    assert amort and "amortization_reason" in amort[0].evidence


def test_plan_sequence_emits_safe_first_action(
    kernel, demo_state, queued_workload_uid, permissive_policy
):
    from test_planner import _BenignModel

    from gpu_center_world_model.planner import (
        ShieldedPlanner,
        enumerate_safe_candidates,
    )

    planner = ShieldedPlanner(kernel, _BenignModel(), model_version="x")
    safe_ids = {
        aid
        for aid, _ in enumerate_safe_candidates(
            kernel, demo_state, queued_workload_uid, permissive_policy
        )
    }
    for backend in ("cem", "mppi", "mcts"):
        sr = planner.plan_sequence(
            demo_state,
            queued_workload_uid,
            900,
            permissive_policy,
            backend=backend,
            sequence_horizon=2,
            rng_seed=3,
        )
        assert sr.emitted_action is not None
        # The emitted first action is drawn from the kernel-safe set.
        from gpu_center_world_model.planner import action_id_for

        assert action_id_for(sr.emitted_action) in safe_ids
        assert len(sr.best_sequence) == 2
