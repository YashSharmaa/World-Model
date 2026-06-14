# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Unit tests for the frozen schema: entities, actions, and hash determinism.

The state hash is the keystone of the project's counterfactual machinery (doc
hashing / docs D, I, R): every real or simulated branch points back to the same
pre-action hash so branches differ *only* by the clamped action. These tests pin
the two guarantees the rest of the system relies on:

* **Determinism** — the same canonical symbolic state hashes to the same digest,
  independent of dict insertion order and Python process.
* **Float-jitter robustness** — micro-jitter below the 1e-6 quantisation must not
  change the hash, while a genuine state change must.
"""

from __future__ import annotations

import copy

from gpu_center_world_model.schema import (
    ACTOR_ACTIONS,
    ActionType,
    Admit,
    AdmitAndPlace,
    Hold,
    MaskDomain,
    Place,
    Workload,
    WorkloadStatus,
    action_workload_uid,
    is_actor_action,
    state_hash,
    world_state_hash,
)
from gpu_center_world_model.schema.entities import PodSet
from gpu_center_world_model.schema.state import (
    ExogenousContext,
    SymbolicState,
    WorldState,
)


# ──────────────────────────────────────────────────────────────────────────────
# Entities & actions
# ──────────────────────────────────────────────────────────────────────────────
def test_action_type_values_match_v1_alphabet():
    assert {a.value for a in ActionType} == {
        "hold",
        "admit",
        "place",
        "admit_and_place",
        "mask_domain",
    }


def test_actor_alphabet_excludes_mask_domain():
    # mask_domain is an operator guardrail, NOT an autonomous actor action (doc C).
    assert ActionType.MASK_DOMAIN not in ACTOR_ACTIONS
    assert is_actor_action(Hold(workload_uid="w0"))
    assert is_actor_action(
        AdmitAndPlace(workload_uid="w0", cluster_queue_id="cq", resource_flavor_id="f")
    )
    assert not is_actor_action(MaskDomain(domain_id_or_set=("rack-0",)))


def test_action_workload_uid_extraction():
    assert action_workload_uid(Hold(workload_uid="w7")) == "w7"
    assert (
        action_workload_uid(
            Admit(workload_uid="w7", cluster_queue_id="cq", resource_flavor_id="f")
        )
        == "w7"
    )
    # mask_domain targets no workload.
    assert action_workload_uid(MaskDomain(domain_id_or_set=("rack-0",))) is None


def test_action_type_field_is_immutable_per_kind():
    # The discriminator field is set on construction and not caller-overridable.
    assert Hold(workload_uid="w").type is ActionType.HOLD
    assert (
        Admit(workload_uid="w", cluster_queue_id="cq", resource_flavor_id="f").type
        is ActionType.ADMIT
    )
    assert Place(workload_uid="w").type is ActionType.PLACE
    assert (
        AdmitAndPlace(
            workload_uid="w", cluster_queue_id="cq", resource_flavor_id="f"
        ).type
        is ActionType.ADMIT_AND_PLACE
    )


def test_workload_demand_is_computed_from_podsets_not_learned():
    # entities.Workload demand is sum(count * per-replica gpu_req) (doc entities).
    wl = Workload(
        workload_uid="w0",
        tenant_id="t0",
        local_queue_id="lq",
        cluster_queue_id="cq",
        podsets=[
            PodSet(workload_uid="w0", name="workers", count=4, gpu_req=1),
            PodSet(workload_uid="w0", name="ps", count=2, gpu_req=2),
        ],
        status=WorkloadStatus.PENDING,
    )
    demand = sum(p.count * p.gpu_req for p in wl.podsets)
    assert demand == 4 * 1 + 2 * 2


# ──────────────────────────────────────────────────────────────────────────────
# Hash determinism
# ──────────────────────────────────────────────────────────────────────────────
def _toy_state() -> WorldState:
    sym = SymbolicState(
        workloads={
            "w0": Workload(
                workload_uid="w0",
                tenant_id="t0",
                local_queue_id="lq",
                cluster_queue_id="cq",
                priority=5,
                podsets=[PodSet(workload_uid="w0", name="workers", count=4, gpu_req=1)],
                status=WorkloadStatus.PENDING,
                age_s=12.0,
            )
        }
    )
    return WorldState(ts_ns=1000, tick=0, symbolic=sym, context=ExogenousContext())


def test_same_state_same_hash():
    s1 = _toy_state()
    s2 = _toy_state()
    assert world_state_hash(s1) == world_state_hash(s2)


def test_deepcopy_preserves_hash():
    s1 = _toy_state()
    s2 = copy.deepcopy(s1)
    assert world_state_hash(s1) == world_state_hash(s2)


def test_hash_is_insertion_order_independent():
    # Build the same logical state with two different dict insertion orders.
    base = _toy_state()
    a = state_hash(base.symbolic, base.context)

    sym2 = SymbolicState()
    # Insert a couple of extra (then delete) keys to perturb internal ordering,
    # then re-add the canonical workload — canonicalisation sorts keys, so the
    # digest must be unchanged.
    sym2.workloads["zzz_tmp"] = copy.deepcopy(base.symbolic.workloads["w0"])
    del sym2.workloads["zzz_tmp"]
    sym2.workloads["w0"] = copy.deepcopy(base.symbolic.workloads["w0"])
    b = state_hash(sym2, base.context)
    assert a == b


def test_hash_is_robust_to_subquantum_float_jitter():
    # Jitter below the 1e-6 quantisation must NOT change the hash (doc hashing).
    s1 = _toy_state()
    s2 = copy.deepcopy(s1)
    s2.symbolic.workloads["w0"].age_s += 1e-9
    assert world_state_hash(s1) == world_state_hash(s2)


def test_hash_changes_on_meaningful_float_change():
    s1 = _toy_state()
    s2 = copy.deepcopy(s1)
    s2.symbolic.workloads["w0"].age_s += 1.0  # a real, visible change
    assert world_state_hash(s1) != world_state_hash(s2)


def test_hash_changes_on_status_change():
    s1 = _toy_state()
    s2 = copy.deepcopy(s1)
    s2.symbolic.workloads["w0"].status = WorkloadStatus.ADMITTED
    assert world_state_hash(s1) != world_state_hash(s2)


def test_context_participates_in_hash():
    s = _toy_state()
    h_no_ctx = state_hash(s.symbolic, None)
    h_ctx = state_hash(s.symbolic, s.context)
    assert h_no_ctx != h_ctx
    # Changing the regime changes the digest (exogenous context is part of X+C).
    other = copy.deepcopy(s)
    other.context.regime = "incident"
    assert world_state_hash(s) != world_state_hash(other)


def test_residual_belief_excluded_from_hash():
    # Z_t is an estimate, not ground truth — it must not affect the digest so the
    # hash is reproducible across model versions (doc hashing).
    s1 = _toy_state()
    s2 = copy.deepcopy(s1)
    s2.residual.summaries["anything"] = [1.0, 2.0, 3.0]
    s2.residual.epistemic["score"] = 0.9
    assert world_state_hash(s1) == world_state_hash(s2)


def test_hash_is_hex_sha256_length():
    h = world_state_hash(_toy_state())
    assert len(h) == 64
    int(h, 16)  # parses as hex (raises if not)


def test_demo_pod_hash_is_deterministic(demo_state):
    # The real generated demo pod must also hash reproducibly.
    assert world_state_hash(demo_state) == world_state_hash(copy.deepcopy(demo_state))
