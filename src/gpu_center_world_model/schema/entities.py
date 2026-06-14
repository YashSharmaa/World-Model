# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Typed relational entities for the GPU-center world state.

The world is modelled as a *typed dynamical graph*, not as video or pixels.
Entities are jobs, queues, GPUs, nodes, switches, racks, power domains, cooling
zones, and operator policy; they are connected by topology and containment
edges. This module defines the canonical entity dataclasses — the single source
of truth shared by ingest, the exact kernel, the data engine, the learned
model, the planner, serving, and evaluation.

Each entity separates **static** fields (slowly/never changing) from **dynamic**
fields (evolving telemetry / scheduler state). Fields map directly to current
open scheduler and telemetry surfaces (Kueue, Kubernetes DRA, NVIDIA DCGM,
node_exporter, Redfish) so the representation stays auditable rather than
re-learned.

Authoritative reference: research design doc C (world representation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────
class WorkloadStatus(str, Enum):
    """Lifecycle status of a workload (Kueue ``Workload`` semantics)."""

    PENDING = "pending"
    RESERVING = "reserving"
    ADMITTED = "admitted"
    RUNNING = "running"
    FINISHED = "finished"
    EVICTED = "evicted"
    REQUEUED = "requeued"


class TopologyLevel(str, Enum):
    """Hierarchical data-center topology levels (Kueue TAS domains)."""

    BLOCK = "block"
    RACK = "rack"
    NODE = "node"
    COMPUTE_DOMAIN = "compute_domain"  # optional NVL/MNNVL future entity


class TopologyPolicy(str, Enum):
    """Locality intent attached to a workload's placement request."""

    REQUIRED = "required"  # must fit inside one domain
    PREFERRED = "preferred"  # prefer one domain, fall back if needed
    UNCONSTRAINED = "unconstrained"  # fill small gaps, reduce fragmentation


class CommClass(str, Enum):
    """Communication intensity class — drives the network surrogate."""

    NONE = "none"
    LIGHT = "light"
    ALLREDUCE_HEAVY = "allreduce_heavy"
    PIPELINE = "pipeline"


class NodeHealth(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    UNSCHEDULABLE = "unschedulable"
    DRAINING = "draining"
    DOWN = "down"


class DeviceHealth(str, Enum):
    """Latent device health states (doc U hidden semi-Markov degradation)."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    INTERMITTENTLY_FAULTY = "intermittently_faulty"
    RESET_RECOVERY = "reset_recovery"
    MASKED = "masked"


# ──────────────────────────────────────────────────────────────────────────────
# Workload-side entities
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class PodSet:
    """A homogeneous group of pods inside a workload (the gang building block)."""

    workload_uid: str
    name: str
    role: str = "worker"
    count: int = 1
    cpu_req: float = 0.0
    mem_req_gb: float = 0.0
    gpu_req: int = 0
    gpu_mem_class: str = "default"
    nic_bw_class: str = "default"
    # group of podsets that must share a topology domain (Kueue podset-group)
    podset_group: str | None = None
    # dynamic
    ready_count: int = 0
    running_count: int = 0
    pending_count: int = 0


@dataclass(slots=True)
class Workload:
    """The unit of admission and the primary planning object.

    A workload's resource demand is computed from its PodSets as
    ``sum(count * per-replica request)`` — never re-learned statistically.
    """

    workload_uid: str
    tenant_id: str
    local_queue_id: str
    cluster_queue_id: str
    priority: int = 0
    submit_ts: float = 0.0
    max_exec_s: float = 0.0
    topology_policy: TopologyPolicy = TopologyPolicy.PREFERRED
    comm_class: CommClass = CommClass.NONE
    labels: dict[str, str] = field(default_factory=dict)
    podsets: list[PodSet] = field(default_factory=list)
    # dynamic
    status: WorkloadStatus = WorkloadStatus.PENDING
    admitted_flavor_id: str | None = None
    admitted_domain_id: str | None = None
    age_s: float = 0.0
    elapsed_run_s: float = 0.0
    est_remaining_s: float = 0.0
    ready_fraction: float = 0.0
    requeue_count: int = 0


@dataclass(slots=True)
class LocalQueue:
    """Tenant-facing submission queue, mapped to a ClusterQueue."""

    name: str
    namespace: str
    cluster_queue_ref: str
    tenant_id: str
    # dynamic
    pending_ct: int = 0
    pending_gpu: int = 0
    historical_usage: float = 0.0
    wait_time_p50_s: float = 0.0
    wait_time_p95_s: float = 0.0
    # Kueue Admission Fair Sharing observable (doc V)
    fairshare_weighted_share: float = 0.0


@dataclass(slots=True)
class ClusterQueue:
    """Quota, borrowing, fair-sharing, and preemption domain."""

    name: str
    cohort_id: str | None = None
    fairshare_weight: float = 1.0
    preemption_policy: str = "never"
    flavor_order: list[str] = field(default_factory=list)
    # quota / borrowing accounting is keyed (flavor, resource)
    nominal_quota: dict[str, dict[str, float]] = field(default_factory=dict)
    borrowing_limit: dict[str, dict[str, float]] = field(default_factory=dict)
    lending_limit: dict[str, dict[str, float]] = field(default_factory=dict)
    # dynamic
    reserved_gpu: float = 0.0
    borrowed_gpu: float = 0.0
    pending_gpu: float = 0.0
    weighted_share: float = 0.0
    active_workloads: int = 0


@dataclass(slots=True)
class ResourceFlavor:
    """Maps workloads to a heterogeneous node subset by labels/taints."""

    name: str
    node_label_constraints: dict[str, str] = field(default_factory=dict)
    tolerations: list[str] = field(default_factory=list)
    topology_name: str | None = None
    nominal_quota_by_resource: dict[str, float] = field(default_factory=dict)
    borrow_limit_by_resource: dict[str, float] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Topology / infrastructure entities
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class TopologyDomain:
    """Block / rack / node / compute-domain abstraction (TAS hierarchy)."""

    domain_id: str
    level: TopologyLevel
    parent_domain_id: str | None = None
    # dynamic free capacity (Kueue TAS convention: ready & schedulable only)
    free_gpu: int = 0
    free_cpu: float = 0.0
    free_mem_gb: float = 0.0
    headroom_score: float = 1.0
    hotspot_score: float = 0.0


@dataclass(slots=True)
class Node:
    """A schedulable compute host."""

    node_uid: str
    rack_id: str
    block_id: str
    switch_id: str
    flavor_id: str
    labels: dict[str, str] = field(default_factory=dict)
    cpu_cap: float = 0.0
    mem_cap_gb: float = 0.0
    gpu_cap: int = 0
    # dynamic
    ready: bool = True
    schedulable: bool = True
    cpu_alloc: float = 0.0
    mem_alloc_gb: float = 0.0
    running_pods: int = 0
    node_health: NodeHealth = NodeHealth.READY


@dataclass(slots=True)
class GPU:
    """An allocatable accelerator resource."""

    gpu_uuid: str
    node_uid: str
    mig_profile: str = "full"
    memory_total_mb: int = 0
    interconnect_class: str = "nvlink"
    # dynamic telemetry (DCGM field semantics)
    allocated_to: str | None = None  # workload_uid
    util: float = 0.0
    mem_util: float = 0.0
    temp_c: float = 0.0
    memory_temp_c: float = 0.0
    power_w: float = 0.0
    energy_mj: float = 0.0
    xid_code: int = 0
    recovery_action: str = "none"
    # latent (filtered) device-health belief — populated by reliability module
    health_state: DeviceHealth = DeviceHealth.HEALTHY


@dataclass(slots=True)
class Switch:
    """Network bottleneck and topology-boundary object (e.g. ToR)."""

    switch_id: str
    tier: str = "tor"
    rack_scope: str = ""
    # dynamic
    port_util_p95: float = 0.0
    egress_hotspot_score: float = 0.0


@dataclass(slots=True)
class Rack:
    """Primary visual and operational grouping unit."""

    rack_id: str
    block_id: str
    power_domain_id: str
    cooling_zone_id: str
    designed_gpu_cap: int = 0
    # dynamic
    rack_power_kw: float = 0.0
    rack_headroom_kw: float = 0.0
    inferred_thermal_margin_c: float = 0.0


@dataclass(slots=True)
class PowerDomain:
    """Power-budget constraint aggregation point."""

    power_domain_id: str
    pdu_class: str = "default"
    cap_kw: float = 0.0
    # dynamic
    used_kw: float = 0.0
    reserve_kw: float = 0.0
    violation_risk: float = 0.0


@dataclass(slots=True)
class CoolingZone:
    """Cooling-coupling aggregation point (shared thermal state)."""

    cooling_zone_id: str
    loop_type: str = "liquid"
    zone_cap_kw: float = 0.0
    # dynamic
    inlet_temp_c: float = 0.0
    return_temp_c: float = 0.0
    cooling_margin: float = 1.0


@dataclass(slots=True)
class OperatorPolicy:
    """Human override and risk-configuration surface (guardrails)."""

    policy_id: str = "default"
    allowed_actions: list[str] = field(
        default_factory=lambda: ["hold", "admit", "place", "admit_and_place"]
    )
    protected_domains: list[str] = field(default_factory=list)
    # configurable risk thresholds consumed by the planner / shield
    risk_thresholds: dict[str, float] = field(default_factory=dict)
    active_since_ts: float = 0.0


# Convenient registry of all entity types (used by tensorizer & docs generation).
ENTITY_TYPES: tuple[str, ...] = (
    "workload",
    "podset",
    "local_queue",
    "cluster_queue",
    "resource_flavor",
    "topology_domain",
    "node",
    "gpu",
    "switch",
    "rack",
    "power_domain",
    "cooling_zone",
    "operator_policy",
)

__all__ = [
    "ENTITY_TYPES",
    "GPU",
    "ClusterQueue",
    "CommClass",
    "CoolingZone",
    "DeviceHealth",
    "LocalQueue",
    "Node",
    "NodeHealth",
    "OperatorPolicy",
    "PodSet",
    "PowerDomain",
    "Rack",
    "ResourceFlavor",
    "Switch",
    "TopologyDomain",
    "TopologyLevel",
    "TopologyPolicy",
    "Workload",
    "WorkloadStatus",
]
