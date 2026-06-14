# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The synthetic causal-SCM generator — the heart of the data engine.

This package implements the generative structural causal model of design docs D,
R, and N: latent parents (workload phase, device health, cooling-zone state,
operator-policy mode) emit observables; children are never sampled independently;
temperature lags power through a shared anchored-RC thermal state; arrivals are
Markov-modulated Hawkes/Cox bursts; faults are coherent parent-offspring clusters;
rare events are importance-sampled with likelihood-ratio accounting; and every
counterfactual pack is a set of ≥ 3 branches sharing one pre-action state hash and
one exogenous seed bundle.

The public entrypoint is :func:`generate_pack`, which builds a demo pod, injects a
pending workload, enumerates a feasible action catalog, and emits a paired
counterfactual pack ready for the windows builder and tensorizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...schema.entities import (
    CommClass,
    PodSet,
    TopologyPolicy,
    Workload,
    WorkloadStatus,
)
from ...schema.state import WorldState
from ...schema.tensors import ShapeCaps
from ...schema.topology import RELATION_ENDPOINTS, Relation, RelationType
from ...utils.seeding import SeedBundle
from .arrivals import (
    Arrival,
    RegimeChain,
    TenantClass,
    simulate_arrivals,
)
from .counterfactual import (
    BranchOutcome,
    CounterfactualPack,
    build_pack,
    default_action_catalog,
    feasible,
    step,
)
from .faults import (
    FaultParams,
    sample_fault_cluster,
)
from .fidelity import (
    FidelityReport,
    TestResult,
    classifier_two_sample_test,
    energy_distance_test,
    exact_kernel_conformance,
    incident_bundle_test,
    lag_structure_test,
    mmd2_test,
    predecision_leakage_test,
)
from .network import Commodity, Link, allocate
from .power import PowerParams, init_power_state, rack_power_kw, step_power
from .rare_events import (
    RARE_EVENT_FAMILIES,
    RareEventEstimate,
    build_tilt,
    importance_estimate,
    point_process_log_lr,
)
from .scm import (
    OPERATOR_MODES,
    WORKLOAD_PHASES,
    PodSpec,
    PodTopology,
    build_pod,
)
from .thermal import ThermalState, step_thermal


@dataclass(slots=True)
class ScenarioConfig:
    """A scenario family (doc D scenario_packs): pod shape + rare-event tilts."""

    name: str = "normal"
    pod: PodSpec = None  # type: ignore[assignment]
    horizon_s: int = 1800
    workload_gpu: int = 16
    comm_class: CommClass = CommClass.ALLREDUCE_HEAVY
    fault_tilt: float = 1.0  # rare-event importance tilt on fault hazards
    rare_event: str | None = None  # one of rare_events.RARE_EVENT_FAMILIES

    def __post_init__(self) -> None:
        if self.pod is None:
            self.pod = PodSpec()


#: Canonical scenario packs shipped in v1 (doc D: normal/thermal/sched/fault).
SCENARIO_PACKS: dict[str, ScenarioConfig] = {
    "normal": ScenarioConfig(name="normal", fault_tilt=1.0),
    "thermal_stress": ScenarioConfig(
        name="thermal_stress", fault_tilt=1.0, rare_event="cooling_derate"
    ),
    "scheduling_stress": ScenarioConfig(
        name="scheduling_stress", workload_gpu=48, rare_event="queue_burst"
    ),
    "fault_stress": ScenarioConfig(
        name="fault_stress", fault_tilt=20.0, rare_event="gpu_fault_burst"
    ),
}


def _inject_pending_workload(
    pod: PodTopology, gpu_demand: int, comm_class: CommClass, submit_ts: float
) -> tuple[WorldState, str]:
    """Add one PENDING workload to the pod and wire its workload-side relations.

    The workload demand is computed from PodSets (``count * gpu_req``) — never
    re-learned (doc entities.Workload). Returns the pre-action world state and the
    new workload uid; the kernel will decide feasibility per candidate action.
    """
    import copy

    sym = copy.deepcopy(pod.symbolic)
    uid = "wl-demo-0"
    lq = next(iter(sym.local_queues), "lq-tenant-0")
    cq = next(iter(sym.cluster_queues), "cq-main")
    podset = PodSet(
        workload_uid=uid,
        name="workers",
        role="worker",
        count=gpu_demand,
        gpu_req=1,
        cpu_req=8.0,
        mem_req_gb=64.0,
    )
    wl = Workload(
        workload_uid=uid,
        tenant_id="tenant-0",
        local_queue_id=lq,
        cluster_queue_id=cq,
        priority=0,
        submit_ts=submit_ts,
        max_exec_s=3600.0,
        topology_policy=TopologyPolicy.PREFERRED,
        comm_class=comm_class,
        podsets=[podset],
        status=WorkloadStatus.PENDING,
    )
    sym.workloads[uid] = wl
    sym.podsets[f"{uid}/workers"] = podset

    # workload-side relations (SUBMITTED_TO, TARGETS, CONTAINS_PODSET)
    wl_idx = {w: i for i, w in enumerate(sym.workloads)}
    lq_idx = {q: i for i, q in enumerate(sym.local_queues)}
    cq_idx = {c: i for i, c in enumerate(sym.cluster_queues)}
    ps_idx = {p: i for i, p in enumerate(sym.podsets)}

    def rel(rt: RelationType) -> Relation:
        sk, dk = RELATION_ENDPOINTS[rt]
        return Relation(type=rt, src_kind=sk, dst_kind=dk)

    sub = rel(RelationType.SUBMITTED_TO)
    sub.src.append(wl_idx[uid])
    sub.dst.append(lq_idx.get(lq, 0))
    sym.relations[RelationType.SUBMITTED_TO.value] = sub

    tgt = rel(RelationType.TARGETS)
    tgt.src.append(lq_idx.get(lq, 0))
    tgt.dst.append(cq_idx.get(cq, 0))
    sym.relations[RelationType.TARGETS.value] = tgt

    cps = rel(RelationType.CONTAINS_PODSET)
    cps.src.append(wl_idx[uid])
    cps.dst.append(ps_idx.get(f"{uid}/workers", 0))
    sym.relations[RelationType.CONTAINS_PODSET.value] = cps

    state = WorldState(ts_ns=int(submit_ts * 1e9), tick=0, symbolic=sym)
    return state, uid


def generate_pack(
    caps: ShapeCaps | None = None,
    seed: int = 0,
    scenario: str | ScenarioConfig = "normal",
) -> CounterfactualPack:
    """Generate one paired counterfactual pack (the synthetic entrypoint).

    Steps (doc R counterfactual_branch_construction):

    1. Build the demo pod (respecting ``caps`` if given) and inject a pending
       workload at a deterministic submit time.
    2. Freeze the pre-action state hash; derive the shared exogenous seed bundle.
    3. Enumerate ≥ 3 feasible candidate actions (hold + placements).
    4. Roll every branch forward under common random numbers and emit the pack.

    Parameters
    ----------
    caps:
        Padded shape caps; only the pod-size fields constrain the generated pod so
        the resulting tensor batch fits a single compile bucket.
    seed:
        Master seed for the shared bundle (CRN across branches).
    scenario:
        A scenario name (key of :data:`SCENARIO_PACKS`) or a :class:`ScenarioConfig`.
    """
    cfg = (
        scenario
        if isinstance(scenario, ScenarioConfig)
        else SCENARIO_PACKS.get(scenario, SCENARIO_PACKS["normal"])
    )
    pod_spec = cfg.pod
    if caps is not None:
        # shrink the pod so G ≤ caps.G_max, N ≤ caps.N_max, R ≤ caps.R_max
        pod_spec = _fit_pod_to_caps(pod_spec, caps)
    pod = build_pod(pod_spec)
    state, uid = _inject_pending_workload(
        pod, cfg.workload_gpu, cfg.comm_class, submit_ts=1000.0
    )
    actions = default_action_catalog(state, uid)
    pack = build_pack(
        pod,
        state,
        actions,
        seed=seed,
        horizon_s=cfg.horizon_s,
        baseline_action_id=actions[0].type.value,
        fault_tilt=cfg.fault_tilt,
    )
    # tag baseline id to the concrete hold action id for delta computation
    pack.baseline_action_id = pack.branches[0].action_id
    return pack


def _fit_pod_to_caps(spec: PodSpec, caps: ShapeCaps) -> PodSpec:
    """Shrink a pod spec so it fits inside a compile bucket (doc tensors)."""
    n_racks = min(spec.n_racks, max(caps.R_max, 1))
    gpus_per_rack = spec.nodes_per_rack * spec.gpus_per_node
    while n_racks * gpus_per_rack > caps.G_max and spec.nodes_per_rack > 1:
        spec = PodSpec(
            n_racks=n_racks,
            nodes_per_rack=spec.nodes_per_rack - 1,
            gpus_per_node=spec.gpus_per_node,
            n_tenants=spec.n_tenants,
            ambient_temp_c=spec.ambient_temp_c,
            flavor_name=spec.flavor_name,
        )
        gpus_per_rack = spec.nodes_per_rack * spec.gpus_per_node
    return PodSpec(
        n_racks=n_racks,
        nodes_per_rack=spec.nodes_per_rack,
        gpus_per_node=spec.gpus_per_node,
        n_tenants=spec.n_tenants,
        ambient_temp_c=spec.ambient_temp_c,
        flavor_name=spec.flavor_name,
    )


def generate_pod_state(
    spec: PodSpec | None = None, seed: int = 0, gpu_demand: int = 16
) -> tuple[PodTopology, WorldState, str]:
    """Build a pod + pending workload (used by windows/normalize/tensorize)."""
    pod = build_pod(spec or PodSpec())
    state, uid = _inject_pending_workload(
        pod, gpu_demand, CommClass.ALLREDUCE_HEAVY, submit_ts=1000.0
    )
    return pod, state, uid


__all__ = [
    # entrypoint
    "generate_pack",
    "generate_pod_state",
    "ScenarioConfig",
    "SCENARIO_PACKS",
    # scm
    "WORKLOAD_PHASES",
    "OPERATOR_MODES",
    "PodSpec",
    "PodTopology",
    "build_pod",
    # arrivals
    "TenantClass",
    "Arrival",
    "RegimeChain",
    "simulate_arrivals",
    # thermal / power / network
    "ThermalState",
    "step_thermal",
    "PowerParams",
    "init_power_state",
    "step_power",
    "rack_power_kw",
    "Link",
    "Commodity",
    "allocate",
    # faults / rare events
    "FaultParams",
    "sample_fault_cluster",
    "RARE_EVENT_FAMILIES",
    "RareEventEstimate",
    "build_tilt",
    "importance_estimate",
    "point_process_log_lr",
    # counterfactual
    "BranchOutcome",
    "CounterfactualPack",
    "build_pack",
    "default_action_catalog",
    "feasible",
    "step",
    # fidelity
    "FidelityReport",
    "TestResult",
    "exact_kernel_conformance",
    "mmd2_test",
    "energy_distance_test",
    "classifier_two_sample_test",
    "lag_structure_test",
    "incident_bundle_test",
    "predecision_leakage_test",
]
