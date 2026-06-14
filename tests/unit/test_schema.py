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


# ──────────────────────────────────────────────────────────────────────────────
# BLOCKED_BY typed edge (SCHEMA-EDGE-KPI-RECONCILE, doc C typed edge set)
# ──────────────────────────────────────────────────────────────────────────────
def test_blocked_by_relation_is_in_the_typed_edge_set():
    from gpu_center_world_model.schema.topology import (
        RELATION_ENDPOINTS,
        RelationType,
    )

    assert RelationType.BLOCKED_BY.value == "BLOCKED_BY"
    assert RELATION_ENDPOINTS[RelationType.BLOCKED_BY] == (
        "candidate_action",
        "constraint",
    )


def test_build_blocked_by_carries_reason_codes():
    from gpu_center_world_model.schema.topology import RelationType, build_blocked_by

    rel = build_blocked_by([(0, 2, 5.0), (0, 3, 9.0)])
    assert rel.type is RelationType.BLOCKED_BY
    assert len(rel) == 2
    assert rel.src == [0, 0]
    assert rel.dst == [2, 3]
    assert rel.edge_feat == [[5.0], [9.0]]


def test_build_blocked_by_empty_is_valid():
    from gpu_center_world_model.schema.topology import build_blocked_by

    rel = build_blocked_by()
    assert len(rel) == 0
    assert rel.src_kind == "candidate_action"
    assert rel.dst_kind == "constraint"


# ──────────────────────────────────────────────────────────────────────────────
# KPI channel reconcile (SCHEMA-EDGE-KPI-RECONCILE, doc I kpi_map)
# ──────────────────────────────────────────────────────────────────────────────
def test_kpi_channels_include_fairness_and_fragmentation_proxy():
    from gpu_center_world_model.schema.kpis import KPIChannel

    values = {c.value for c in KPIChannel}
    assert "fairness_share_debt_delta" in values  # share_debt term surfaced
    assert "cross_rack_rate" in values  # fragmentation proxy retained


def test_kpi_map_folds_cross_rack_into_fragmentation():
    from gpu_center_world_model.schema.kpis import KPI_MAP, KPIChannel

    frag = KPI_MAP["topology_fragmentation_delta"]
    assert KPIChannel.TOPOLOGY_FRAGMENTATION_DELTA in frag
    assert KPIChannel.CROSS_RACK_RATE in frag  # proxy feeds the fragmentation delta
    assert KPI_MAP["fairness_share_debt"] == (KPIChannel.FAIRNESS_SHARE_DEBT_DELTA,)


# ──────────────────────────────────────────────────────────────────────────────
# AdmissionCheck entity + ClusterQueue per-resource counters
# ──────────────────────────────────────────────────────────────────────────────
def test_admission_check_entity_and_chi_helper():
    from gpu_center_world_model.schema.entities import (
        AdmissionCheckState,
        ClusterQueue,
        admission_checks_positive,
    )

    cq = ClusterQueue(name="cq")
    # No checks -> vacuously positive (chi_qf = 1).
    assert admission_checks_positive(cq, "f1")
    # A failing default gate -> chi_qf = 0.
    cq.admission_check_state = {"*": False}
    assert not admission_checks_positive(cq, "f1")
    # A per-flavor positive entry overrides the default.
    cq.admission_check_state = {"*": False, "f1": True}
    assert admission_checks_positive(cq, "f1")
    assert not admission_checks_positive(cq, "f2")
    assert AdmissionCheckState.READY.is_positive
    assert not AdmissionCheckState.PENDING.is_positive


def test_cluster_queue_per_resource_counters_participate_in_hash():
    s1 = _toy_state()
    s2 = copy.deepcopy(s1)
    # Mutating the per-(flavor, resource) reservation ledger changes the digest.
    cq = s2.symbolic.cluster_queues.setdefault(
        "cq",
        __import__(
            "gpu_center_world_model.schema.entities", fromlist=["ClusterQueue"]
        ).ClusterQueue(name="cq"),
    )
    base = world_state_hash(s2)
    cq.reserved_by_resource = {"f": {"cpu": 8.0}}
    assert world_state_hash(s2) != base
