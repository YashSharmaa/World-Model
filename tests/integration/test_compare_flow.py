# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Integration test: kernel + planner + model end-to-end on a tiny pack (doc H).

This exercises the full spine on the canonical demo pod through two surfaces:

* the **planner path** — the real :class:`ExactSymbolicKernel` (safety shield)
  wired into :class:`ShieldedPlanner` with the dependency-free reference
  forecaster, producing a recommendation that is provably a member of the
  kernel-certified safe set; and
* the **serving compare path** — :func:`compare_actions` wiring the same exact
  kernel into the doc-G DTOs, returning per-candidate ``safe`` + ``exact_blockers``
  + ``rank`` over a shape-stable branch set.

The point is that *hard safety is a gate, not a weighted metric*: every emitted
recommendation is hard-safe by construction and every unsafe candidate carries an
exact blocker ledger.
"""

from __future__ import annotations

from gpu_center_world_model.planner import (
    ShieldedPlanner,
    enumerate_safe_candidates,
)
from gpu_center_world_model.schema import HORIZONS_S, world_state_hash
from gpu_center_world_model.serving import (
    CandidateActionRequest,
    CompareRequest,
    EngineBundle,
    ReferenceEngine,
    compare_actions,
    serialize_state,
    to_dict,
)
from gpu_center_world_model.serving.dto import DEFAULT_BRANCHES
from gpu_center_world_model.simulator import ExactSymbolicKernel

HORIZON = HORIZONS_S[2]  # 1800 s


# ──────────────────────────────────────────────────────────────────────────────
# Planner path: kernel + model end-to-end
# ──────────────────────────────────────────────────────────────────────────────
def test_planner_recommends_a_kernel_safe_action(
    kernel, demo_state, queued_workload_uid, permissive_policy
):
    model = ReferenceEngine()  # closed-form residual forecaster
    planner = ShieldedPlanner(kernel, model, model_version="reference-it")
    result = planner.plan(demo_state, queued_workload_uid, HORIZON, permissive_policy)

    assert not result.abstain
    assert result.recommended_action_id is not None
    assert result.candidates  # at least hold survived the risk gate

    # The recommendation must be a member of the kernel-certified safe set.
    safe_ids = {
        aid
        for aid, _ in enumerate_safe_candidates(
            kernel, demo_state, queued_workload_uid, permissive_policy
        )
    }
    assert result.recommended_action_id in safe_ids

    # Every candidate the planner ranked is itself hard-feasible.
    for sa in result.candidates:
        assert kernel.feasible(demo_state, sa.action).feasible

    # Provenance + state hash travel with the result.
    assert result.state_hash == world_state_hash(demo_state)
    assert result.model_version == "reference-it"


# ──────────────────────────────────────────────────────────────────────────────
# Serving compare path: DTOs + exact gate
# ──────────────────────────────────────────────────────────────────────────────
def _compare_request(state, uid):
    cq = state.symbolic.workloads[uid].cluster_queue_id
    flavor = next(iter(state.symbolic.flavors))
    state_h = world_state_hash(state)
    return CompareRequest(
        state_hash=state_h,
        candidate_actions=[
            CandidateActionRequest(
                action_id="cand_hold", type="hold", params={"workload_uid": uid}
            ),
            CandidateActionRequest(
                action_id="cand_place",
                type="admit_and_place",
                params={
                    "workload_uid": uid,
                    "cluster_queue_id": cq,
                    "resource_flavor_id": flavor,
                    "domain_policy": "preferred",
                },
            ),
            CandidateActionRequest(
                action_id="cand_bad",
                type="admit_and_place",
                params={
                    "workload_uid": uid,
                    "cluster_queue_id": cq,
                    "resource_flavor_id": flavor,
                    "domain_policy": "required",
                    "domain_id_or_set": ["nonexistent-domain"],
                },
            ),
        ],
        horizon_s=float(HORIZON),
        include_rollouts=True,
    )


def test_compare_actions_gates_on_exact_kernel(demo_state, queued_workload_uid):
    # Inject the real exact kernel; the reference closed-form model handles ranking.
    engine = EngineBundle(kernel=ExactSymbolicKernel())
    req = _compare_request(demo_state, queued_workload_uid)
    resp = compare_actions(demo_state, req, engine)

    by_id = {c.action_id: c for c in resp.candidates}
    # hold is always safe with no blockers.
    assert by_id["cand_hold"].safe
    assert by_id["cand_hold"].exact_blockers == []
    # the impossible-domain placement is unsafe and carries an exact blocker.
    assert not by_id["cand_bad"].safe
    assert by_id["cand_bad"].exact_blockers

    # The recommended action (if any) is one of the safe candidates.
    if resp.recommended_action_id is not None:
        assert by_id[resp.recommended_action_id].safe

    # Every safe candidate has a rank; unsafe ones are unranked.
    for c in resp.candidates:
        if c.safe and not c.action_id.startswith("__pad"):
            assert c.rank is not None
        if not c.safe:
            assert c.rank is None


def test_compare_response_is_shape_stable_and_serializable(
    demo_state, queued_workload_uid
):
    engine = EngineBundle(kernel=ExactSymbolicKernel())
    req = _compare_request(demo_state, queued_workload_uid)
    resp = compare_actions(demo_state, req, engine)

    # Shape-stable branch count for the compiled scoring graph (doc G).
    assert len(resp.candidates) >= DEFAULT_BRANCHES
    # Provenance carried alongside the recommendation.
    assert resp.state_hash == req.state_hash
    assert resp.model_version
    # Round-trips to a plain JSON-able dict.
    payload = to_dict(resp)
    assert isinstance(payload, dict)
    assert "candidates" in payload and isinstance(payload["candidates"], list)


def test_safe_candidate_gets_a_rollout(demo_state, queued_workload_uid):
    engine = EngineBundle(kernel=ExactSymbolicKernel())
    req = _compare_request(demo_state, queued_workload_uid)
    resp = compare_actions(demo_state, req, engine)
    safe = [
        c for c in resp.candidates if c.safe and not c.action_id.startswith("__pad")
    ]
    assert safe
    # Rollouts were requested; each safe candidate carries semantic points only.
    for c in safe:
        assert c.rollout is not None
        assert c.rollout.points  # at least the root + one horizon point
        # No geometry — only semantic entity summaries (doc G hard rule).
        for pt in c.rollout.points:
            assert not hasattr(pt, "mesh")
            assert hasattr(pt, "rack_state")


def test_serialize_state_emits_semantic_snapshot(demo_state):
    engine = EngineBundle()
    sr = serialize_state(demo_state, engine)
    assert sr.state_hash == world_state_hash(demo_state)
    assert sr.entities.racks  # racks present
    assert sr.entities.gpus  # gpus present
    assert sr.current_kpis["rack_count"] == len(demo_state.symbolic.racks)
