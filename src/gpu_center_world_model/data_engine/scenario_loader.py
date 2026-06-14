# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Load a human-authored scenario YAML into a :class:`WorldState` + job queue.

A *scenario pack* is a small, human-editable description of one demo pod (2-4
racks) plus a queue of mixed gang jobs and an optional rare-event injection
(doc D rare-event library). Authors can add new packs without touching Python:
the YAML describes topology, queues/quota, jobs, operator policy, and the
exogenous context, and this loader compiles it into the exact, typed symbolic
state the kernel and data engine consume.

Design rationale (doc D / C):

* The scenario file is the *invariant world definition* — topology graph,
  quota arithmetic, gang structure — plus a *seed* and a single rare-event
  injection. Topology structure, quota arithmetic, and gang atomicity are kept
  invariant; only arrivals, fault onset, and operator policy regimes vary
  (doc D "keep invariant" vs "randomize aggressively").
* Resource demand is **computed** from podsets as ``sum(count * gpu_req)``,
  never re-learned (Kueue ``Workload`` semantics, doc C).
* Topology free capacity is materialised from ready & schedulable nodes minus
  admitted usage (Kueue TAS convention, doc C / entities ``TopologyDomain``).
* Rare-event injection is recorded into the :class:`ExogenousContext` and the
  :class:`SourceState`/entity dynamic fields so the same pre-action state hash
  reproduces it exactly; the heavy stochastic dynamics live in the simulator,
  not here (this module is a *deterministic compiler*, not a generator).

The loader degrades gracefully without PyYAML installed only insofar as
:func:`gpu_center_world_model.utils.config.load_config` does (it raises a clear
error). All schema construction below is pure-stdlib and import-clean.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schema.entities import (
    GPU,
    ClusterQueue,
    CommClass,
    CoolingZone,
    DeviceHealth,
    LocalQueue,
    Node,
    NodeHealth,
    OperatorPolicy,
    PodSet,
    PowerDomain,
    Rack,
    ResourceFlavor,
    Switch,
    TopologyDomain,
    TopologyLevel,
    TopologyPolicy,
    Workload,
    WorkloadStatus,
)
from ..schema.state import (
    ExogenousContext,
    SourceState,
    SymbolicState,
    WorldState,
)
from ..schema.tensors import DEFAULT_CAPS, ShapeCaps, pick_bucket
from ..schema.topology import Relation, RelationType
from ..utils.config import load_config
from ..utils.seeding import SeedBundle

# Canonical rare-event family identifiers (doc D rare_event_library). These are
# the only injection ``kind`` values the loader recognises; an unknown kind is a
# hard error so scenario typos never silently degrade to "normal".
RARE_EVENT_KINDS: tuple[str, ...] = (
    "gpu_fault_burst",
    "pods_ready_timeout",
    "cooling_derate",
    "power_derate",
    "queue_burst",
    "maintenance_mask",
    "telemetry_degradation",
)

# Scenario families used for split/provenance tagging (matches directory layout).
SCENARIO_FAMILIES: tuple[str, ...] = ("normal", "stress", "rare_events")


# ──────────────────────────────────────────────────────────────────────────────
# Public result container
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class LoadedScenario:
    """The compiled output of one scenario pack.

    ``state`` is the full hybrid :class:`WorldState` with an authoritative
    :class:`SymbolicState`. ``pending_queue`` is the ordered list of *pending*
    workload uids (the admission/placement decision queue), sorted by priority
    then submit time so the planner/simulator process them deterministically.
    """

    scenario_id: str
    scenario_family: str
    state: WorldState
    pending_queue: list[str] = field(default_factory=list)
    seed_bundle: SeedBundle = field(default_factory=lambda: SeedBundle(0))
    caps: ShapeCaps = field(default_factory=lambda: DEFAULT_CAPS)
    rare_event: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def n_workloads(self) -> int:
        return len(self.state.symbolic.workloads)

    @property
    def n_nodes(self) -> int:
        return len(self.state.symbolic.nodes)

    @property
    def n_gpus(self) -> int:
        return len(self.state.symbolic.gpus)


# ──────────────────────────────────────────────────────────────────────────────
# Small typed-coercion helpers (YAML is permissive; the schema is not)
# ──────────────────────────────────────────────────────────────────────────────
def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _enum(enum_cls: type, value: Any, default: Any) -> Any:
    """Coerce a YAML string into an enum member, falling back to ``default``."""
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value))
    except ValueError as exc:  # pragma: no cover - author error path
        allowed = ", ".join(m.value for m in enum_cls)  # type: ignore[attr-defined]
        raise ValueError(
            f"invalid {enum_cls.__name__} value {value!r}; allowed: {allowed}"
        ) from exc


def _req(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ValueError(f"scenario {ctx}: missing required field {key!r}")
    return d[key]


# ──────────────────────────────────────────────────────────────────────────────
# Topology / infrastructure builders
# ──────────────────────────────────────────────────────────────────────────────
def _build_infrastructure(pod: dict[str, Any], symbolic: SymbolicState) -> None:
    """Materialise racks, nodes, gpus, switches, power & cooling from ``pod``.

    The pod block uses a compact, repetition-friendly schema: a rack template
    declares how many nodes it holds and how many GPUs each node holds, so a
    4-rack demo pod stays a few lines of YAML rather than 256 GPU records.
    """
    gpus_per_node_default = int(pod.get("gpus_per_node", 8))
    gpu_mem_mb = int(pod.get("gpu_memory_mb", 81920))  # ~80 GiB class default
    interconnect = str(pod.get("interconnect_class", "nvlink"))
    cpu_per_node = float(pod.get("node_cpu", 128.0))
    mem_per_node_gb = float(pod.get("node_mem_gb", 2048.0))
    flavor_id = str(pod.get("flavor", "gpu-default"))

    # A single block domain parents the rack domains (TAS hierarchy).
    block_id = str(pod.get("block_id", "block-0"))
    symbolic.domains[block_id] = TopologyDomain(
        domain_id=block_id, level=TopologyLevel.BLOCK
    )

    racks = _as_list(_req(pod, "racks", "pod"))
    for ri, rack_cfg in enumerate(racks):
        rack_id = str(rack_cfg.get("id", f"rack-{ri}"))
        power_id = str(rack_cfg.get("power_domain", f"pdu-{ri}"))
        cooling_id = str(rack_cfg.get("cooling_zone", f"zone-{ri}"))
        switch_id = str(rack_cfg.get("switch", f"tor-{ri}"))
        n_nodes = int(rack_cfg.get("nodes", 4))
        gpus_per_node = int(rack_cfg.get("gpus_per_node", gpus_per_node_default))
        designed_cap = n_nodes * gpus_per_node

        # Power domain (one per rack unless shared via explicit id reuse).
        if power_id not in symbolic.power_domains:
            symbolic.power_domains[power_id] = PowerDomain(
                power_domain_id=power_id,
                pdu_class=str(rack_cfg.get("pdu_class", "default")),
                cap_kw=float(rack_cfg.get("power_cap_kw", 120.0)),
            )
        # Cooling zone (often shared across racks in a row).
        if cooling_id not in symbolic.cooling_zones:
            symbolic.cooling_zones[cooling_id] = CoolingZone(
                cooling_zone_id=cooling_id,
                loop_type=str(rack_cfg.get("loop_type", "liquid")),
                zone_cap_kw=float(rack_cfg.get("cooling_cap_kw", 150.0)),
                inlet_temp_c=float(rack_cfg.get("inlet_temp_c", 24.0)),
                cooling_margin=float(rack_cfg.get("cooling_margin", 1.0)),
            )
        symbolic.switches[switch_id] = Switch(
            switch_id=switch_id,
            tier=str(rack_cfg.get("switch_tier", "tor")),
            rack_scope=rack_id,
        )
        symbolic.racks[rack_id] = Rack(
            rack_id=rack_id,
            block_id=block_id,
            power_domain_id=power_id,
            cooling_zone_id=cooling_id,
            designed_gpu_cap=designed_cap,
            rack_headroom_kw=float(rack_cfg.get("power_cap_kw", 120.0)),
            inferred_thermal_margin_c=float(rack_cfg.get("thermal_margin_c", 12.0)),
        )

        # Each rack is a topology domain at RACK level, parented by the block.
        rack_domain_id = str(rack_cfg.get("domain_id", rack_id))
        rack_domain = TopologyDomain(
            domain_id=rack_domain_id,
            level=TopologyLevel.RACK,
            parent_domain_id=block_id,
        )
        symbolic.domains[rack_domain_id] = rack_domain

        for ni in range(n_nodes):
            node_uid = f"{rack_id}-node-{ni}"
            health = NodeHealth.READY
            ready = True
            schedulable = True
            node = Node(
                node_uid=node_uid,
                rack_id=rack_id,
                block_id=block_id,
                switch_id=switch_id,
                flavor_id=flavor_id,
                cpu_cap=cpu_per_node,
                mem_cap_gb=mem_per_node_gb,
                gpu_cap=gpus_per_node,
                ready=ready,
                schedulable=schedulable,
                node_health=health,
            )
            symbolic.nodes[node_uid] = node
            for gi in range(gpus_per_node):
                gpu_uuid = f"{node_uid}-gpu-{gi}"
                symbolic.gpus[gpu_uuid] = GPU(
                    gpu_uuid=gpu_uuid,
                    node_uid=node_uid,
                    memory_total_mb=gpu_mem_mb,
                    interconnect_class=interconnect,
                    health_state=DeviceHealth.HEALTHY,
                )

    # Default flavor mapping every node onto the demo flavor.
    if flavor_id not in symbolic.flavors:
        symbolic.flavors[flavor_id] = ResourceFlavor(
            name=flavor_id,
            topology_name=block_id,
        )

    _recompute_domain_capacity(symbolic)


def _recompute_domain_capacity(symbolic: SymbolicState) -> None:
    """Set per-domain free capacity from ready & schedulable nodes (Kueue TAS).

    Free capacity is allocatable capacity on ready & schedulable nodes minus
    already-allocated usage — exactly the entities ``TopologyDomain`` contract.
    Rack domains aggregate their nodes; the block domain aggregates the racks.
    """
    # node -> rack domain id (rack_id is the domain id by construction).
    rack_free: dict[str, dict[str, float]] = {}
    for node in symbolic.nodes.values():
        dom = node.rack_id
        agg = rack_free.setdefault(dom, {"gpu": 0.0, "cpu": 0.0, "mem": 0.0})
        if (
            node.ready
            and node.schedulable
            and node.node_health
            in (
                NodeHealth.READY,
                NodeHealth.DEGRADED,
            )
        ):
            agg["gpu"] += node.gpu_cap
            agg["cpu"] += node.cpu_cap - node.cpu_alloc
            agg["mem"] += node.mem_cap_gb - node.mem_alloc_gb

    # Subtract GPUs already allocated to running/admitted workloads.
    alloc_by_rack: dict[str, int] = {}
    for gpu in symbolic.gpus.values():
        if gpu.allocated_to is not None:
            node = symbolic.nodes.get(gpu.node_uid)
            if node is not None:
                alloc_by_rack[node.rack_id] = alloc_by_rack.get(node.rack_id, 0) + 1

    block_free = {"gpu": 0.0, "cpu": 0.0, "mem": 0.0}
    for dom_id, dom in symbolic.domains.items():
        if dom.level is TopologyLevel.RACK:
            agg = rack_free.get(dom_id, {"gpu": 0.0, "cpu": 0.0, "mem": 0.0})
            free_gpu = int(agg["gpu"]) - alloc_by_rack.get(dom_id, 0)
            dom.free_gpu = max(0, free_gpu)
            dom.free_cpu = max(0.0, agg["cpu"])
            dom.free_mem_gb = max(0.0, agg["mem"])
            block_free["gpu"] += dom.free_gpu
            block_free["cpu"] += dom.free_cpu
            block_free["mem"] += dom.free_mem_gb
    for dom in symbolic.domains.values():
        if dom.level is TopologyLevel.BLOCK:
            dom.free_gpu = int(block_free["gpu"])
            dom.free_cpu = block_free["cpu"]
            dom.free_mem_gb = block_free["mem"]


# ──────────────────────────────────────────────────────────────────────────────
# Queue / quota builders
# ──────────────────────────────────────────────────────────────────────────────
def _build_queues(scenario: dict[str, Any], symbolic: SymbolicState) -> None:
    """Build ClusterQueues, LocalQueues, and ResourceFlavors from the YAML."""
    for cq_cfg in _as_list(scenario.get("cluster_queues")):
        name = str(_req(cq_cfg, "name", "cluster_queue"))
        nominal = cq_cfg.get("nominal_quota", {}) or {}
        borrowing = cq_cfg.get("borrowing_limit", {}) or {}
        lending = cq_cfg.get("lending_limit", {}) or {}
        symbolic.cluster_queues[name] = ClusterQueue(
            name=name,
            cohort_id=cq_cfg.get("cohort"),
            fairshare_weight=float(cq_cfg.get("fairshare_weight", 1.0)),
            preemption_policy=str(cq_cfg.get("preemption_policy", "never")),
            flavor_order=[str(f) for f in _as_list(cq_cfg.get("flavor_order"))],
            nominal_quota={k: dict(v) for k, v in nominal.items()},
            borrowing_limit={k: dict(v) for k, v in borrowing.items()},
            lending_limit={k: dict(v) for k, v in lending.items()},
        )

    for lq_cfg in _as_list(scenario.get("local_queues")):
        name = str(_req(lq_cfg, "name", "local_queue"))
        symbolic.local_queues[name] = LocalQueue(
            name=name,
            namespace=str(lq_cfg.get("namespace", "default")),
            cluster_queue_ref=str(_req(lq_cfg, "cluster_queue", "local_queue")),
            tenant_id=str(lq_cfg.get("tenant_id", name)),
        )

    for fl_cfg in _as_list(scenario.get("flavors")):
        name = str(_req(fl_cfg, "name", "flavor"))
        symbolic.flavors[name] = ResourceFlavor(
            name=name,
            node_label_constraints=dict(fl_cfg.get("node_labels", {}) or {}),
            tolerations=[str(t) for t in _as_list(fl_cfg.get("tolerations"))],
            topology_name=fl_cfg.get("topology"),
            nominal_quota_by_resource=dict(fl_cfg.get("nominal_quota", {}) or {}),
            borrow_limit_by_resource=dict(fl_cfg.get("borrow_limit", {}) or {}),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Workload / gang-job builders
# ──────────────────────────────────────────────────────────────────────────────
def _build_workloads(
    scenario: dict[str, Any], symbolic: SymbolicState, base_ts: float
) -> list[str]:
    """Build Workloads (with PodSets) and return the ordered pending queue uids.

    Each job is a *gang*: a workload whose total GPU demand is the sum over its
    podsets of ``count * gpu_req`` (Kueue arithmetic, never re-learned). The
    canonical demo mix is 1-GPU inference, 4-GPU fine-tune, and 8/16-GPU
    training jobs (task spec / doc D workload mix).
    """
    pending: list[tuple[int, float, str]] = []
    for ji, job_cfg in enumerate(_as_list(scenario.get("jobs"))):
        uid = str(job_cfg.get("uid", f"wl-{ji:03d}"))
        submit_offset = float(job_cfg.get("submit_offset_s", 0.0))
        submit_ts = base_ts + submit_offset
        priority = int(job_cfg.get("priority", 0))
        status = _enum(WorkloadStatus, job_cfg.get("status"), WorkloadStatus.PENDING)

        # PodSets: either an explicit list or a compact (gpus, replicas) form.
        podsets: list[PodSet] = []
        if "podsets" in job_cfg:
            for psi, ps_cfg in enumerate(_as_list(job_cfg["podsets"])):
                ps_name = str(ps_cfg.get("name", f"ps-{psi}"))
                podsets.append(
                    PodSet(
                        workload_uid=uid,
                        name=ps_name,
                        role=str(ps_cfg.get("role", "worker")),
                        count=int(ps_cfg.get("count", 1)),
                        cpu_req=float(ps_cfg.get("cpu_req", 16.0)),
                        mem_req_gb=float(ps_cfg.get("mem_req_gb", 128.0)),
                        gpu_req=int(ps_cfg.get("gpu_req", 1)),
                        gpu_mem_class=str(ps_cfg.get("gpu_mem_class", "default")),
                        nic_bw_class=str(ps_cfg.get("nic_bw_class", "default")),
                        podset_group=ps_cfg.get("podset_group"),
                    )
                )
        else:
            # Compact form: total GPUs spread over `replicas` worker pods.
            total_gpu = int(_req(job_cfg, "gpus", f"job {uid}"))
            replicas = int(job_cfg.get("replicas", 1))
            if replicas <= 0 or total_gpu % replicas != 0:
                raise ValueError(
                    f"job {uid}: gpus={total_gpu} not divisible by replicas={replicas}"
                )
            per_replica = total_gpu // replicas
            podsets.append(
                PodSet(
                    workload_uid=uid,
                    name="worker",
                    role="worker",
                    count=replicas,
                    cpu_req=float(job_cfg.get("cpu_req", 16.0)),
                    mem_req_gb=float(job_cfg.get("mem_req_gb", 128.0)),
                    gpu_req=per_replica,
                    podset_group=job_cfg.get("podset_group"),
                )
            )

        local_queue_id = str(job_cfg.get("local_queue", _default_lq(symbolic)))
        lq = symbolic.local_queues.get(local_queue_id)
        cluster_queue_id = str(
            job_cfg.get(
                "cluster_queue",
                lq.cluster_queue_ref if lq is not None else "cq-default",
            )
        )
        tenant_id = str(
            job_cfg.get("tenant_id", lq.tenant_id if lq is not None else "tenant-a")
        )

        wl = Workload(
            workload_uid=uid,
            tenant_id=tenant_id,
            local_queue_id=local_queue_id,
            cluster_queue_id=cluster_queue_id,
            priority=priority,
            submit_ts=submit_ts,
            max_exec_s=float(job_cfg.get("max_exec_s", 3600.0)),
            topology_policy=_enum(
                TopologyPolicy,
                job_cfg.get("topology_policy"),
                TopologyPolicy.PREFERRED,
            ),
            comm_class=_enum(CommClass, job_cfg.get("comm_class"), CommClass.NONE),
            labels=dict(job_cfg.get("labels", {}) or {}),
            podsets=podsets,
            status=status,
            est_remaining_s=float(job_cfg.get("est_remaining_s", 0.0)),
        )
        symbolic.workloads[uid] = wl
        for ps in podsets:
            symbolic.podsets[f"{uid}/{ps.name}"] = ps

        # Pre-place already running/admitted gangs so free capacity is faithful.
        # A scenario expresses this with status: running|admitted plus an optional
        # `place_on` rack/domain hint; the loader fills concrete GPUs greedily.
        if status in (WorkloadStatus.RUNNING, WorkloadStatus.ADMITTED):
            place_on = job_cfg.get("place_on")
            total_gpu = sum(ps.count * ps.gpu_req for ps in podsets)
            _place_running(symbolic, wl, total_gpu, place_on)

        if status is WorkloadStatus.PENDING:
            pending.append((-priority, submit_ts, uid))

    # Deterministic decision order: higher priority first, then earlier submit.
    pending.sort()
    return [uid for _, _, uid in pending]


def _place_running(
    symbolic: SymbolicState,
    wl: Workload,
    total_gpu: int,
    place_on: Any,
) -> None:
    """Greedily allocate ``total_gpu`` free GPUs to an already-running gang.

    Used to express the *initial* fleet occupancy of a scenario so domain free
    capacity reflects running jobs (Kueue TAS: free = ready capacity minus
    admitted usage). Prefers GPUs inside the requested rack/domain so the
    fragmentation a scenario intends is reproduced exactly.
    """
    if total_gpu <= 0:
        return
    target = str(place_on) if place_on else None
    allocated = 0
    placed_nodes: set[str] = set()
    # First pass: GPUs inside the preferred rack/domain (if any).
    for prefer in (True, False):
        if allocated >= total_gpu:
            break
        for gpu in symbolic.gpus.values():
            if allocated >= total_gpu:
                break
            if gpu.allocated_to is not None:
                continue
            node = symbolic.nodes.get(gpu.node_uid)
            if node is None or not (node.ready and node.schedulable):
                continue
            in_target = target is None or node.rack_id == target
            if prefer and not in_target:
                continue
            gpu.allocated_to = wl.workload_uid
            gpu.util = 0.85
            gpu.mem_util = 0.7
            placed_nodes.add(gpu.node_uid)
            allocated += 1
    # Record the admitted domain on the workload for downstream consumers.
    if placed_nodes:
        first_node = symbolic.nodes.get(next(iter(placed_nodes)))
        if first_node is not None:
            wl.admitted_domain_id = first_node.rack_id
            wl.ready_fraction = 1.0
    for node_uid in placed_nodes:
        node = symbolic.nodes.get(node_uid)
        if node is not None:
            node.running_pods += 1


def _default_lq(symbolic: SymbolicState) -> str:
    if symbolic.local_queues:
        return next(iter(symbolic.local_queues))
    return "lq-default"


# ──────────────────────────────────────────────────────────────────────────────
# Operator policy + context + rare-event injection
# ──────────────────────────────────────────────────────────────────────────────
def _build_policy(scenario: dict[str, Any]) -> OperatorPolicy:
    pol = scenario.get("operator_policy", {}) or {}
    return OperatorPolicy(
        policy_id=str(pol.get("policy_id", "default")),
        allowed_actions=[
            str(a)
            for a in _as_list(
                pol.get(
                    "allowed_actions",
                    ["hold", "admit", "place", "admit_and_place"],
                )
            )
        ],
        protected_domains=[str(d) for d in _as_list(pol.get("protected_domains"))],
        risk_thresholds={
            str(k): float(v) for k, v in (pol.get("risk_thresholds", {}) or {}).items()
        },
    )


def _build_context(scenario: dict[str, Any]) -> ExogenousContext:
    ctx = scenario.get("context", {}) or {}
    windows = [
        (float(w[0]), float(w[1]))
        for w in _as_list(ctx.get("maintenance_windows"))
        if isinstance(w, (list, tuple)) and len(w) == 2
    ]
    return ExogenousContext(
        regime=str(ctx.get("regime", "normal")),
        maintenance_windows=windows,
        manual_drains=[str(d) for d in _as_list(ctx.get("manual_drains"))],
        ambient_temp_c=float(ctx.get("ambient_temp_c", 22.0)),
        carbon_intensity=(
            float(ctx["carbon_intensity"])
            if ctx.get("carbon_intensity") is not None
            else None
        ),
        arrival_burst_regime=str(ctx.get("arrival_burst_regime", "steady")),
        workload_mix_regime=str(ctx.get("workload_mix_regime", "mixed")),
    )


def _apply_rare_event(
    rare: dict[str, Any],
    symbolic: SymbolicState,
    context: ExogenousContext,
    source: SourceState,
) -> dict[str, Any]:
    """Stamp a single rare-event injection into the deterministic pre-state.

    This deterministically perturbs the *initial conditions* (masks, derates,
    health flags, source-health) so the simulator can evolve them stochastically
    from a reproducible starting point. Onset/severity remain in ``rare`` for the
    simulator to schedule; here we only set what must be true at ``t0`` so the
    state hash captures the regime (doc D: missingness/derates are state).
    """
    kind = str(_req(rare, "kind", "rare_event"))
    if kind not in RARE_EVENT_KINDS:
        allowed = ", ".join(RARE_EVENT_KINDS)
        raise ValueError(f"unknown rare_event kind {kind!r}; allowed: {allowed}")
    severity = float(rare.get("severity", 0.5))
    onset_s = float(rare.get("onset_s", 0.0))
    targets = [str(t) for t in _as_list(rare.get("targets"))]
    pre_onset = onset_s > 0.0  # if it starts later, t0 stays nominal

    if kind == "maintenance_mask":
        # Operator drain: mask domains immediately (a guardrail at t0).
        context.regime = "maintenance"
        for dom_id in targets:
            if dom_id not in context.manual_drains:
                context.manual_drains.append(dom_id)
            dom = symbolic.domains.get(dom_id)
            if dom is not None:
                dom.free_gpu = 0
                dom.headroom_score = 0.0
            # Mark member nodes unschedulable so capacity recompute agrees.
            for node in symbolic.nodes.values():
                if node.rack_id == dom_id:
                    node.schedulable = False
                    node.node_health = NodeHealth.DRAINING
        _recompute_domain_capacity(symbolic)

    elif kind == "cooling_derate":
        context.regime = "incident" if not pre_onset else context.regime
        for zone_id in targets or list(symbolic.cooling_zones):
            zone = symbolic.cooling_zones.get(zone_id)
            if zone is not None:
                zone.loop_type = zone.loop_type
                zone.cooling_margin = max(0.0, zone.cooling_margin * (1.0 - severity))
                zone.inlet_temp_c += 6.0 * severity
                if not pre_onset:
                    continue
        # Couple derate into rack thermal margin (shared zone state, doc D).
        for rack in symbolic.racks.values():
            if not targets or rack.cooling_zone_id in targets:
                rack.inferred_thermal_margin_c *= max(0.0, 1.0 - 0.5 * severity)

    elif kind == "power_derate":
        for pd_id in targets or list(symbolic.power_domains):
            pd = symbolic.power_domains.get(pd_id)
            if pd is not None:
                pd.cap_kw *= max(0.0, 1.0 - severity)
                pd.violation_risk = min(1.0, severity)
        for rack in symbolic.racks.values():
            pd = symbolic.power_domains.get(rack.power_domain_id)
            if pd is not None and (not targets or rack.power_domain_id in targets):
                rack.rack_headroom_kw = max(0.0, pd.cap_kw - pd.used_kw)

    elif kind == "gpu_fault_burst":
        # Mark seed GPUs as intermittently faulty + record an Xid code so the
        # health mask and reliability filter have a reproducible start.
        xid = int(rare.get("xid_code", 79))  # 79 = GPU has fallen off the bus
        affected = targets or _first_n_gpus(symbolic, int(rare.get("count", 2)))
        for gpu_uuid in affected:
            gpu = symbolic.gpus.get(gpu_uuid)
            if gpu is not None:
                gpu.health_state = DeviceHealth.INTERMITTENTLY_FAULTY
                if not pre_onset:
                    gpu.xid_code = xid
                    gpu.recovery_action = "reset_required"

    elif kind == "pods_ready_timeout":
        # The decision-time state is nominal; the simulator induces the timeout.
        # We tag context so windows/eval slice this family correctly.
        context.regime = "normal"

    elif kind == "queue_burst":
        context.arrival_burst_regime = str(rare.get("burst_regime", "campaign"))

    elif kind == "telemetry_degradation":
        # Degrade source health for chosen channels/families at t0.
        channels = [str(c) for c in _as_list(rare.get("channels"))] or ["gpu_telemetry"]
        for ch in channels:
            source.collector_health[ch] = "degraded"
            source.channel_missing_prob[ch] = min(1.0, severity)
            source.channel_age_s[ch] = 30.0 + 90.0 * severity

    return {
        "kind": kind,
        "severity": severity,
        "onset_s": onset_s,
        "targets": targets,
        "params": {
            k: v
            for k, v in rare.items()
            if k not in ("kind", "severity", "onset_s", "targets")
        },
    }


def _first_n_gpus(symbolic: SymbolicState, n: int) -> list[str]:
    return list(symbolic.gpus)[: max(0, n)]


# ──────────────────────────────────────────────────────────────────────────────
# Relation (typed-edge) materialisation
# ──────────────────────────────────────────────────────────────────────────────
def _build_relations(symbolic: SymbolicState) -> None:
    """Compile the explicit typed edges the tensorizer turns into incidence.

    We build index-based relations using stable insertion order of each entity
    table (the same order the tensorizer pads), so a downstream consumer can map
    ``Relation.src/dst`` directly onto array rows (doc Q fixed incidence).
    """
    wl_index = {uid: i for i, uid in enumerate(symbolic.workloads)}
    ps_index = {key: i for i, key in enumerate(symbolic.podsets)}
    lq_index = {name: i for i, name in enumerate(symbolic.local_queues)}
    cq_index = {name: i for i, name in enumerate(symbolic.cluster_queues)}
    fl_index = {name: i for i, name in enumerate(symbolic.flavors)}
    node_index = {uid: i for i, uid in enumerate(symbolic.nodes)}
    gpu_index = {uid: i for i, uid in enumerate(symbolic.gpus)}
    sw_index = {sid: i for i, sid in enumerate(symbolic.switches)}
    rack_index = {rid: i for i, rid in enumerate(symbolic.racks)}
    pd_index = {pid: i for i, pid in enumerate(symbolic.power_domains)}
    cz_index = {cid: i for i, cid in enumerate(symbolic.cooling_zones)}
    dom_index = {did: i for i, did in enumerate(symbolic.domains)}

    def _rel(rt: RelationType, src_kind: str, dst_kind: str) -> Relation:
        rel = Relation(type=rt, src_kind=src_kind, dst_kind=dst_kind)
        symbolic.relations[rt.value] = rel
        return rel

    submitted = _rel(RelationType.SUBMITTED_TO, "workload", "local_queue")
    targets = _rel(RelationType.TARGETS, "local_queue", "cluster_queue")
    contains_ps = _rel(RelationType.CONTAINS_PODSET, "workload", "podset")
    requests_fl = _rel(RelationType.REQUESTS_FLAVOR, "podset", "resource_flavor")
    in_domain = _rel(RelationType.IN_DOMAIN, "node", "topology_domain")
    hosts = _rel(RelationType.HOSTS, "node", "gpu")
    connected = _rel(RelationType.CONNECTED_TO, "node", "switch")
    contains_node = _rel(RelationType.CONTAINS_NODE, "rack", "node")
    powered = _rel(RelationType.POWERED_BY, "rack", "power_domain")
    cooled = _rel(RelationType.COOLED_BY, "rack", "cooling_zone")
    runs_on = _rel(RelationType.RUNS_ON, "workload", "node")
    parent_of = _rel(RelationType.PARENT_OF, "topology_domain", "topology_domain")

    default_fl_idx = next(iter(fl_index.values()), None)

    for uid, wl in symbolic.workloads.items():
        si = wl_index[uid]
        if wl.local_queue_id in lq_index:
            submitted.src.append(si)
            submitted.dst.append(lq_index[wl.local_queue_id])
        for ps in wl.podsets:
            key = f"{uid}/{ps.name}"
            if key in ps_index:
                contains_ps.src.append(si)
                contains_ps.dst.append(ps_index[key])
                if default_fl_idx is not None:
                    requests_fl.src.append(ps_index[key])
                    requests_fl.dst.append(default_fl_idx)
        # RUNS_ON edge for already-placed workloads (admitted/running).
        node_uids = _placed_nodes(symbolic, uid)
        for n_uid in node_uids:
            runs_on.src.append(si)
            runs_on.dst.append(node_index[n_uid])

    for lq_name, lq in symbolic.local_queues.items():
        if lq.cluster_queue_ref in cq_index:
            targets.src.append(lq_index[lq_name])
            targets.dst.append(cq_index[lq.cluster_queue_ref])

    for n_uid, node in symbolic.nodes.items():
        ni = node_index[n_uid]
        if node.rack_id in dom_index:
            in_domain.src.append(ni)
            in_domain.dst.append(dom_index[node.rack_id])
        if node.switch_id in sw_index:
            connected.src.append(ni)
            connected.dst.append(sw_index[node.switch_id])
        if node.rack_id in rack_index:
            contains_node.src.append(rack_index[node.rack_id])
            contains_node.dst.append(ni)

    for g_uid, gpu in symbolic.gpus.items():
        if gpu.node_uid in node_index:
            hosts.src.append(node_index[gpu.node_uid])
            hosts.dst.append(gpu_index[g_uid])

    for r_id, rack in symbolic.racks.items():
        ri = rack_index[r_id]
        if rack.power_domain_id in pd_index:
            powered.src.append(ri)
            powered.dst.append(pd_index[rack.power_domain_id])
        if rack.cooling_zone_id in cz_index:
            cooled.src.append(ri)
            cooled.dst.append(cz_index[rack.cooling_zone_id])

    for d_id, dom in symbolic.domains.items():
        if dom.parent_domain_id and dom.parent_domain_id in dom_index:
            parent_of.src.append(dom_index[dom.parent_domain_id])
            parent_of.dst.append(dom_index[d_id])


def _placed_nodes(symbolic: SymbolicState, workload_uid: str) -> list[str]:
    nodes: list[str] = []
    seen: set[str] = set()
    for gpu in symbolic.gpus.values():
        if gpu.allocated_to == workload_uid and gpu.node_uid not in seen:
            seen.add(gpu.node_uid)
            nodes.append(gpu.node_uid)
    return nodes


# ──────────────────────────────────────────────────────────────────────────────
# Public entry points
# ──────────────────────────────────────────────────────────────────────────────
def load_scenario_dict(
    scenario: dict[str, Any], *, scenario_id: str | None = None
) -> LoadedScenario:
    """Compile an already-parsed scenario mapping into a :class:`LoadedScenario`.

    Useful for tests and programmatic scenario synthesis. See
    :func:`load_scenario` for the file-based entry point.
    """
    sid = str(scenario_id or scenario.get("scenario_id", "scenario"))
    family = str(scenario.get("scenario_family", "normal"))
    if family not in SCENARIO_FAMILIES:
        # Tolerate custom families but keep the canonical set discoverable.
        family = family or "normal"
    seed = int(scenario.get("seed", 0))
    base_ts = float(scenario.get("base_ts", 0.0))
    ts_ns = int(scenario.get("ts_ns", int(base_ts * 1e9)))
    tick = int(scenario.get("tick", 0))

    symbolic = SymbolicState()
    pod = _req(scenario, "pod", sid)
    _build_infrastructure(pod, symbolic)
    _build_queues(scenario, symbolic)
    pending = _build_workloads(scenario, symbolic, base_ts)
    # Recompute free capacity now that running gangs have claimed GPUs.
    _recompute_domain_capacity(symbolic)
    symbolic.operator_policy = _build_policy(scenario)

    context = _build_context(scenario)
    source = SourceState()

    rare_event_meta: dict[str, Any] | None = None
    rare = scenario.get("rare_event")
    if rare:
        rare_event_meta = _apply_rare_event(rare, symbolic, context, source)

    # Apply masked domains declared in policy *after* infra so capacity agrees.
    for dom_id in symbolic.operator_policy.protected_domains:
        dom = symbolic.domains.get(dom_id)
        if dom is not None and dom_id in symbolic.operator_policy.protected_domains:
            # Protected != removed; capacity stays but the kernel forbids use.
            dom.headroom_score = min(dom.headroom_score, 1.0)

    _build_relations(symbolic)

    state = WorldState(
        ts_ns=ts_ns,
        tick=tick,
        symbolic=symbolic,
        context=context,
        source=source,
    )

    caps = pick_bucket(
        n_workloads=len(symbolic.workloads),
        n_nodes=len(symbolic.nodes),
        n_gpus=len(symbolic.gpus),
    )

    return LoadedScenario(
        scenario_id=sid,
        scenario_family=family,
        state=state,
        pending_queue=pending,
        seed_bundle=SeedBundle(seed),
        caps=caps,
        rare_event=rare_event_meta,
        raw=scenario,
    )


def load_scenario(path: str | os.PathLike[str]) -> LoadedScenario:
    """Load a scenario YAML file into a :class:`LoadedScenario`.

    The returned object holds the full :class:`WorldState` (authoritative
    symbolic block, exogenous context, and source-state) plus the ordered
    pending-workload decision queue and the reproducible seed bundle.
    """
    p = Path(path)
    raw = load_config(p)
    if not isinstance(raw, dict):  # pragma: no cover - malformed file
        raise ValueError(f"scenario {p} did not parse to a mapping")
    scenario_id = str(raw.get("scenario_id", p.stem))
    return load_scenario_dict(raw, scenario_id=scenario_id)


__all__ = [
    "RARE_EVENT_KINDS",
    "SCENARIO_FAMILIES",
    "LoadedScenario",
    "load_scenario",
    "load_scenario_dict",
]
