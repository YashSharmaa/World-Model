# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The generative structural causal model (design docs D, R, N).

This is the causal *skeleton*. A small set of **latent parents** emit the
observables; children are **never sampled independently** (doc D
correlation_rules; doc R parent_to_children_map). The latent parents are:

* ``workload_phase`` Φ — a **semi-Markov** chain over
  ``{staging, warmup, steady_compute, collective_sync, checkpoint, eval,
  teardown}`` with phase-specific dwell distributions (doc N, ASSUMPTION A2);
  parent of GPU utilisation mix, communication activity, power draw, runtime
  progress, and pods-ready latency.
* ``device_health_state`` H — a hidden semi-Markov degradation chain
  (``healthy → degraded → intermittently_faulty → reset_recovery → masked``);
  parent of Xid/ECC/PCIe marks, degraded availability, slowdown, and telemetry
  degradation.
* ``cooling_zone_state`` C — ``normal | derated | recovering``; parent of inlet/
  return temperature, thermal margin, and delayed hotspot emergence.
* ``operator_policy_mode`` O — ``normal | conservative | maintenance |
  incident``; parent of observational action propensity, protected-domain masks,
  and admission-check disposition.

The SCM is wired so that the structural equations of doc R (Eq2-Eq9) are realised
in code: the exact symbolic block ``X_t`` is owned by the kernel; this module owns
the *stochastic residual* parents and the emission ``Y ~ p_θ(· | parents)``. The
factual/counterfactual split (doc R, [A4]) is honoured by the
:mod:`counterfactual` module, which holds every parent's noise stream fixed and
changes only the action structural equation.

Building a pod (2-4 racks) here produces a canonical
:class:`~...schema.state.SymbolicState` plus the relational structure, so the
kernel, tensorizer, and windows all consume one schema.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...schema.entities import (
    GPU,
    ClusterQueue,
    CoolingZone,
    DeviceHealth,
    LocalQueue,
    Node,
    OperatorPolicy,
    PowerDomain,
    Rack,
    ResourceFlavor,
    Switch,
    TopologyDomain,
    TopologyLevel,
)
from ...schema.observations import (
    Channel,
    Observation,
    ObservationFamily,
    SourceHealth,
)
from ...schema.state import (
    ExogenousContext,
    SourceState,
    SymbolicState,
)
from ...schema.topology import RELATION_ENDPOINTS, Relation, RelationType
from .thermal import (
    ThermalEdgeSpec,
    ThermalGraph,
    ThermalNodeSpec,
    build_thermal_graph,
)

# ──────────────────────────────────────────────────────────────────────────────
# Latent parent state spaces
# ──────────────────────────────────────────────────────────────────────────────
#: Workload-phase semi-Markov state set (doc D / doc N).
WORKLOAD_PHASES: tuple[str, ...] = (
    "staging",
    "warmup",
    "steady_compute",
    "collective_sync",
    "checkpoint",
    "eval",
    "teardown",
)

# ──────────────────────────────────────────────────────────────────────────────
# UNCALIBRATED PRIORS (doc D randomization_rules ``calibrated_bounds_only`` tier)
# ──────────────────────────────────────────────────────────────────────────────
# The semi-Markov dwell means, jump-chain probabilities, util/power emission
# tables, and health-transition rates below are *uncalibrated priors*: they encode
# the qualitative shape of public H100 / DCGM / Kueue behaviour but have NOT been
# fit to a real trace. Per doc D they belong to the calibrated-bounds tier — fit
# from traces first, then randomize from the fitted posterior. The system-
# identification module (:mod:`...synthetic.sysid`) fits the power/RC/hazard blocks
# and exposes a joint posterior; :func:`sysid.sample_tier` draws calibrated
# randomization. Treat every constant here as a prior centre, never ground truth.

#: Mean phase dwell time (s) — semi-Markov (different per phase, doc N A2).
#: **Uncalibrated prior** (see the banner above).
_PHASE_DWELL_S: dict[str, float] = {
    "staging": 90.0,
    "warmup": 120.0,
    "steady_compute": 480.0,
    "collective_sync": 60.0,
    "checkpoint": 90.0,
    "eval": 150.0,
    "teardown": 60.0,
}

#: Phase successor distribution (semi-Markov jump chain). steady_compute and
#: collective_sync alternate (the training inner loop) before checkpoint/eval.
_PHASE_NEXT: dict[str, dict[str, float]] = {
    "staging": {"warmup": 1.0},
    "warmup": {"steady_compute": 1.0},
    "steady_compute": {"collective_sync": 0.6, "checkpoint": 0.25, "eval": 0.15},
    "collective_sync": {"steady_compute": 0.85, "checkpoint": 0.15},
    "checkpoint": {"steady_compute": 0.7, "eval": 0.2, "teardown": 0.1},
    "eval": {"steady_compute": 0.5, "teardown": 0.5},
    "teardown": {"teardown": 1.0},  # absorbing
}

#: GPU utilisation / mem-util means per phase (the emission of Φ → util mix).
_PHASE_UTIL: dict[str, tuple[float, float, float]] = {
    # (sm_util, mem_util, comm_activity)
    "staging": (0.05, 0.10, 0.02),
    "warmup": (0.45, 0.35, 0.10),
    "steady_compute": (0.94, 0.70, 0.15),
    "collective_sync": (0.55, 0.60, 0.85),
    "checkpoint": (0.20, 0.30, 0.40),
    "eval": (0.65, 0.55, 0.10),
    "teardown": (0.05, 0.08, 0.02),
}

#: Device-health hidden semi-Markov chain (doc U semantics, used here as parent).
HEALTH_STATES: tuple[str, ...] = tuple(d.value for d in DeviceHealth)

# Per-step (30 s) transition probabilities. Kept small so that under nominal
# conditions a 30-60 min rollout over the demo pod sees only occasional
# degradation, while a fault burst (fault_pressure / rate_tilt) sharply
# accelerates the healthy→degraded→intermittently_faulty cascade.
_HEALTH_NEXT: dict[str, dict[str, float]] = {
    "healthy": {"healthy": 0.9997, "degraded": 0.0003},
    "degraded": {"degraded": 0.995, "intermittently_faulty": 0.003, "healthy": 0.002},
    "intermittently_faulty": {
        "intermittently_faulty": 0.9,
        "reset_recovery": 0.08,
        "masked": 0.02,
    },
    "reset_recovery": {"reset_recovery": 0.6, "healthy": 0.35, "masked": 0.05},
    "masked": {"masked": 1.0},
}

#: Cooling-zone state set (doc D latent parent cooling_zone_state).
COOLING_STATES: tuple[str, ...] = ("normal", "derated", "recovering")

#: Operator-policy modes (doc D latent parent operator_policy_mode).
OPERATOR_MODES: tuple[str, ...] = ("normal", "conservative", "maintenance", "incident")


# ──────────────────────────────────────────────────────────────────────────────
# Pod specification + construction
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class PodSpec:
    """The demo-pod shape (2-4 racks; doc C recommended_demo_caps)."""

    n_racks: int = 2
    nodes_per_rack: int = 4
    gpus_per_node: int = 8
    n_tenants: int = 3
    ambient_temp_c: float = 22.0
    flavor_name: str = "h100-sxm"


@dataclass(slots=True)
class PodTopology:
    """Index maps + the thermal graph for a constructed pod (reused everywhere).

    Holds the canonical :class:`SymbolicState` plus the dense index arrays the
    tensorizer/thermal/power surrogates need. Construction is deterministic given
    a :class:`PodSpec` so the pre-action ``state_hash`` is reproducible.
    """

    spec: PodSpec
    symbolic: SymbolicState
    gpu_ids: list[str]
    node_ids: list[str]
    rack_ids: list[str]
    gpu_node_idx: np.ndarray  # (G,) node index per gpu
    gpu_rack_idx: np.ndarray  # (G,) rack index per gpu
    node_rack_idx: np.ndarray  # (N,) rack index per node
    thermal: ThermalGraph
    gpu_thermal_idx: np.ndarray  # (G,) row in thermal graph for each gpu


def build_pod(spec: PodSpec) -> PodTopology:
    """Construct a canonical pod: entities, relations, and the thermal graph.

    The pod is a small typed dynamical graph (doc C): racks contain nodes, nodes
    host GPUs and connect to a ToR switch, racks are powered/cooled by domains.
    The thermal graph mirrors the containment so temperature lags power through
    shared node/rack/zone state (doc N, EQUATIONS E3-E6).
    """
    sym = SymbolicState()
    flavor = ResourceFlavor(name=spec.flavor_name)
    sym.flavors[flavor.name] = flavor

    cq = ClusterQueue(
        name="cq-main",
        nominal_quota={spec.flavor_name: {"gpu": float(_total_gpus(spec))}},
    )
    sym.cluster_queues[cq.name] = cq

    # one cooling zone + one power domain shared by the pod (the shared state)
    zone = CoolingZone(cooling_zone_id="zone-0", zone_cap_kw=200.0, inlet_temp_c=0.0)
    pdom = PowerDomain(power_domain_id="pd-0", cap_kw=300.0)
    sym.cooling_zones[zone.cooling_zone_id] = zone
    sym.power_domains[pdom.power_domain_id] = pdom

    block = TopologyDomain(domain_id="block-0", level=TopologyLevel.BLOCK)
    sym.domains[block.domain_id] = block

    gpu_ids: list[str] = []
    node_ids: list[str] = []
    rack_ids: list[str] = []
    gpu_node_idx: list[int] = []
    gpu_rack_idx: list[int] = []
    node_rack_idx: list[int] = []

    # Thermal compartments: one per gpu, node, rack, plus a single zone node.
    # Conductances are calibrated so that under full load (per-GPU heat ≈ 6.5
    # source units, i.e. power_w/100) the GPU package sits ~25-35 K above inlet
    # and the rack plenum ~8-12 K above inlet — the realistic operating band for
    # liquid-cooled H100 racks. The bulk of the heat is rejected directly at each
    # GPU's cold plate (large per-GPU g_env); only a fraction couples upward to
    # warm the node/rack/zone, which is what gives temperature its inertia and
    # the shared-state coupling the planner needs (doc N, EQUATIONS E3-E6).
    t_nodes: list[ThermalNodeSpec] = []
    t_edges: list[ThermalEdgeSpec] = []
    zone_tid = "T:zone-0"
    t_nodes.append(
        ThermalNodeSpec(zone_tid, "zone", capacitance=8000.0, g_env=6.0, w_gain=1.0)
    )
    gpu_thermal_idx: list[int] = []

    node_counter = 0
    gpu_counter = 0
    for r in range(spec.n_racks):
        rid = f"rack-{r}"
        rack_ids.append(rid)
        rack = Rack(
            rack_id=rid,
            block_id=block.domain_id,
            power_domain_id=pdom.power_domain_id,
            cooling_zone_id=zone.cooling_zone_id,
            designed_gpu_cap=spec.nodes_per_rack * spec.gpus_per_node,
        )
        sym.racks[rid] = rack
        switch = Switch(switch_id=f"tor-{r}", tier="tor", rack_scope=rid)
        sym.switches[switch.switch_id] = switch
        rack_dom = TopologyDomain(
            domain_id=f"dom-rack-{r}",
            level=TopologyLevel.RACK,
            parent_domain_id=block.domain_id,
            free_gpu=spec.nodes_per_rack * spec.gpus_per_node,
        )
        sym.domains[rack_dom.domain_id] = rack_dom

        rack_tid = f"T:{rid}"
        t_nodes.append(ThermalNodeSpec(rack_tid, "rack", capacitance=2000.0, g_env=0.4))
        t_edges.append(ThermalEdgeSpec(rack_tid, zone_tid, conductance=1.2))

        for nn in range(spec.nodes_per_rack):
            nid = f"node-{r}-{nn}"
            node_ids.append(nid)
            node_rack_idx.append(r)
            node = Node(
                node_uid=nid,
                rack_id=rid,
                block_id=block.domain_id,
                switch_id=switch.switch_id,
                flavor_id=flavor.name,
                cpu_cap=128.0,
                mem_cap_gb=2048.0,
                gpu_cap=spec.gpus_per_node,
            )
            sym.nodes[nid] = node
            node_tid = f"T:{nid}"
            t_nodes.append(
                ThermalNodeSpec(node_tid, "node", capacitance=500.0, g_env=0.15)
            )
            t_edges.append(ThermalEdgeSpec(node_tid, rack_tid, conductance=0.6))

            for gg in range(spec.gpus_per_node):
                gid = f"gpu-{r}-{nn}-{gg}"
                gpu_ids.append(gid)
                gpu_node_idx.append(node_counter)
                gpu_rack_idx.append(r)
                gpu = GPU(
                    gpu_uuid=gid,
                    node_uid=nid,
                    memory_total_mb=80000,
                    interconnect_class="nvlink",
                )
                sym.gpus[gid] = gpu
                gpu_tid = f"T:{gid}"
                t_nodes.append(
                    ThermalNodeSpec(
                        gpu_tid, "gpu", capacitance=40.0, g_env=0.18, heat_gain=1.0
                    )
                )
                t_edges.append(ThermalEdgeSpec(gpu_tid, node_tid, conductance=0.25))
                gpu_thermal_idx.append(len(t_nodes) - 1)
                gpu_counter += 1
            node_counter += 1

    # tenants → local queues
    for tt in range(spec.n_tenants):
        lq = LocalQueue(
            name=f"lq-tenant-{tt}",
            namespace=f"tenant-{tt}",
            cluster_queue_ref=cq.name,
            tenant_id=f"tenant-{tt}",
        )
        sym.local_queues[lq.name] = lq

    sym.operator_policy = OperatorPolicy(policy_id="default")
    thermal = build_thermal_graph(t_nodes, t_edges)
    _build_relations(sym)

    return PodTopology(
        spec=spec,
        symbolic=sym,
        gpu_ids=gpu_ids,
        node_ids=node_ids,
        rack_ids=rack_ids,
        gpu_node_idx=np.array(gpu_node_idx, dtype=int),
        gpu_rack_idx=np.array(gpu_rack_idx, dtype=int),
        node_rack_idx=np.array(node_rack_idx, dtype=int),
        thermal=thermal,
        gpu_thermal_idx=np.array(gpu_thermal_idx, dtype=int),
    )


def _total_gpus(spec: PodSpec) -> int:
    return spec.n_racks * spec.nodes_per_rack * spec.gpus_per_node


def _build_relations(sym: SymbolicState) -> None:
    """Populate the canonical typed relations from the constructed entities.

    The tensorizer compiles these into one-hot incidence matrices ``S_r``/``T_r``
    (doc Q). We build the structural (topology/containment) edges here; workload-
    side edges (SUBMITTED_TO, CONTAINS_PODSET, RUNS_ON) are added when workloads
    are injected.
    """
    node_idx = {nid: i for i, nid in enumerate(sym.nodes)}
    gpu_idx = {gid: i for i, gid in enumerate(sym.gpus)}
    rack_idx = {rid: i for i, rid in enumerate(sym.racks)}
    dom_idx = {d: i for i, d in enumerate(sym.domains)}
    sw_idx = {s: i for i, s in enumerate(sym.switches)}
    pd_idx = {p: i for i, p in enumerate(sym.power_domains)}
    cz_idx = {c: i for i, c in enumerate(sym.cooling_zones)}

    def rel(rt: RelationType) -> Relation:
        src_kind, dst_kind = RELATION_ENDPOINTS[rt]
        return Relation(type=rt, src_kind=src_kind, dst_kind=dst_kind)

    hosts = rel(RelationType.HOSTS)
    for gid, gpu in sym.gpus.items():
        hosts.src.append(node_idx[gpu.node_uid])
        hosts.dst.append(gpu_idx[gid])
    sym.relations[RelationType.HOSTS.value] = hosts

    contains_node = rel(RelationType.CONTAINS_NODE)
    connected = rel(RelationType.CONNECTED_TO)
    for nid, node in sym.nodes.items():
        contains_node.src.append(rack_idx[node.rack_id])
        contains_node.dst.append(node_idx[nid])
        connected.src.append(node_idx[nid])
        connected.dst.append(sw_idx[node.switch_id])
    sym.relations[RelationType.CONTAINS_NODE.value] = contains_node
    sym.relations[RelationType.CONNECTED_TO.value] = connected

    powered = rel(RelationType.POWERED_BY)
    cooled = rel(RelationType.COOLED_BY)
    for rid, rack in sym.racks.items():
        powered.src.append(rack_idx[rid])
        powered.dst.append(pd_idx[rack.power_domain_id])
        cooled.src.append(rack_idx[rid])
        cooled.dst.append(cz_idx[rack.cooling_zone_id])
    sym.relations[RelationType.POWERED_BY.value] = powered
    sym.relations[RelationType.COOLED_BY.value] = cooled

    # rack-level topology domains parented to the block (PARENT_OF)
    parent_of = rel(RelationType.PARENT_OF)
    for did, dom in sym.domains.items():
        if dom.parent_domain_id is not None and dom.parent_domain_id in dom_idx:
            parent_of.src.append(dom_idx[dom.parent_domain_id])
            parent_of.dst.append(dom_idx[did])
    sym.relations[RelationType.PARENT_OF.value] = parent_of


# ──────────────────────────────────────────────────────────────────────────────
# Latent-parent semi-Markov samplers
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class PhaseProgram:
    """A sampled semi-Markov phase trajectory for one workload (doc N, EQ P1)."""

    changepoints_s: list[float]  # entry time of each segment
    phases: list[str]  # phase name per segment

    def at(self, t: float) -> str:
        cur = self.phases[0]
        for cp, ph in zip(self.changepoints_s, self.phases, strict=False):
            if cp <= t:
                cur = ph
            else:
                break
        return cur

    def index_at(self, t: float) -> int:
        return WORKLOAD_PHASES.index(self.at(t))


def sample_phase_program(
    horizon_s: float, rng: np.random.Generator, start: str = "staging"
) -> PhaseProgram:
    """Sample a semi-Markov phase program (doc N, ASSUMPTION A2 / EQUATION P1).

    Dwell times are gamma-distributed around the phase mean (shape 4) so dwell
    variance is realistic but bounded; the jump chain follows ``_PHASE_NEXT``.
    """
    cps = [0.0]
    phs = [start]
    t = 0.0
    cur = start
    while t < horizon_s:
        mean = _PHASE_DWELL_S[cur]
        dwell = float(rng.gamma(shape=4.0, scale=mean / 4.0))
        t += dwell
        if t >= horizon_s or cur == "teardown":
            break
        nxt = _PHASE_NEXT[cur]
        keys = list(nxt)
        probs = np.array([nxt[k] for k in keys], dtype=float)
        probs = probs / probs.sum()
        cur = str(rng.choice(keys, p=probs))
        cps.append(t)
        phs.append(cur)
    return PhaseProgram(changepoints_s=cps, phases=phs)


def step_health(
    state: str, rng: np.random.Generator, fault_pressure: float = 0.0
) -> str:
    """One step of the device-health hidden semi-Markov chain (doc U parent).

    ``fault_pressure`` ∈ [0, 1] tilts the degradation transitions upward (a fault
    burst or thermal stress accelerates degradation); used by :mod:`faults` and
    :mod:`rare_events`.
    """
    nxt = dict(_HEALTH_NEXT[state])
    if fault_pressure > 0 and state in ("healthy", "degraded"):
        bad = "degraded" if state == "healthy" else "intermittently_faulty"
        boost = min(fault_pressure, 0.5)
        nxt[bad] = nxt.get(bad, 0.0) + boost
        nxt[state] = max(nxt.get(state, 0.0) - boost, 0.0)
    keys = list(nxt)
    probs = np.array([nxt[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(keys, p=probs))


def step_cooling(
    state: str, rng: np.random.Generator, derate_event: bool = False
) -> str:
    """One step of the cooling-zone state (doc D parent cooling_zone_state)."""
    if derate_event:
        return "derated"
    table = {
        "normal": {"normal": 0.99, "derated": 0.01},
        "derated": {"derated": 0.85, "recovering": 0.15},
        "recovering": {"recovering": 0.6, "normal": 0.4},
    }[state]
    keys = list(table)
    probs = np.array([table[k] for k in keys], dtype=float)
    return str(rng.choice(keys, p=probs / probs.sum()))


# ──────────────────────────────────────────────────────────────────────────────
# Emission: latent parents → observation channels (Y ~ p_θ(·|parents))
# ──────────────────────────────────────────────────────────────────────────────
def emit_gpu_telemetry(
    gpu_id: str,
    phase: str,
    health: str,
    temp_c: float,
    power_w: float,
    rng: np.random.Generator,
    *,
    ts_ns: int,
    source_health: SourceHealth = SourceHealth.OBSERVED,
) -> Observation:
    """Emit a GPU telemetry :class:`Observation` from the latent parents.

    Utilisation mix, power, and temperature are *children* of (phase, health,
    thermal state) — never sampled independently (doc R parent_to_children_map).
    Under degraded source health the channels are marked unavailable so
    missingness travels as state, never zero-filled (doc D correlation_rules #5).
    """
    sm, mem, _comm = _PHASE_UTIL.get(phase, _PHASE_UTIL["staging"])
    # device health degrades utilisation and can mark telemetry stale/dropped
    deg = {
        "healthy": 1.0,
        "degraded": 0.85,
        "intermittently_faulty": 0.5,
        "reset_recovery": 0.0,
        "masked": 0.0,
    }[health]
    available = source_health in (SourceHealth.OBSERVED, SourceHealth.DELAYED)
    age = 0.0 if source_health == SourceHealth.OBSERVED else 30.0
    util = float(np.clip(sm * deg + rng.normal(0, 0.03), 0.0, 1.0))
    memutil = float(np.clip(mem * deg + rng.normal(0, 0.03), 0.0, 1.0))
    xid = 0
    if health == "intermittently_faulty" and rng.random() < 0.2:
        xid = int(rng.choice([48, 63, 79, 94]))  # ECC / NVLink / Xid-class codes

    def ch(name: str, value: float) -> Channel:
        return Channel(
            name=name,
            value=value,
            available=available,
            age_s=age,
            source_health=source_health,
        )

    channels = {
        "DCGM_FI_DEV_GPU_UTIL": ch("DCGM_FI_DEV_GPU_UTIL", util * 100.0),
        "DCGM_FI_DEV_MEM_COPY_UTIL": ch("DCGM_FI_DEV_MEM_COPY_UTIL", memutil * 100.0),
        "DCGM_FI_DEV_POWER_USAGE": ch("DCGM_FI_DEV_POWER_USAGE", power_w),
        "DCGM_FI_DEV_GPU_TEMP": ch("DCGM_FI_DEV_GPU_TEMP", temp_c),
        "DCGM_FI_DEV_XID_ERRORS": ch("DCGM_FI_DEV_XID_ERRORS", float(xid)),
    }
    return Observation(
        entity_kind="gpu",
        entity_id=gpu_id,
        family=ObservationFamily.GPU_TELEMETRY,
        ts_event_ns=ts_ns,
        channels=channels,
    )


def context_from_latents(
    regime: str, operator_mode: str, cooling_state: str, ambient_c: float
) -> ExogenousContext:
    """Map latent parents into the canonical exogenous context block ``C_t``."""
    regime_map = {
        "normal": "normal",
        "conservative": "conservative",
        "maintenance": "maintenance",
        "incident": "incident",
    }
    return ExogenousContext(
        regime=regime_map.get(operator_mode, "normal"),
        ambient_temp_c=ambient_c,
        arrival_burst_regime=regime,
        workload_mix_regime="mixed",
    )


def source_state_from_health(
    degraded_gpu_ids: list[str], stale_age_s: float = 30.0
) -> SourceState:
    """Build the source-health/missingness state ``M_t`` (doc D #5, doc T)."""
    return SourceState(
        collector_health={g: "stale" for g in degraded_gpu_ids},
        channel_age_s={g: stale_age_s for g in degraded_gpu_ids},
        channel_missing_prob={g: 0.5 for g in degraded_gpu_ids},
    )


def calibrated_posterior(
    pod: PodTopology,
    *,
    power_traces: list | None = None,
    thermal_trace: object | None = None,
    fault_events: int = 0,
    fault_exposure_gpu_s: float = 0.0,
    cross_corr: float = 0.0,
):
    """Fit the SCM's calibrated parameter block and return the joint posterior.

    Ties the SCM's residual parents (power source, thermal RC, fault hazard) to the
    system-identification module: it fits whatever traces are supplied (any subset
    of power / thermal / fault), centring un-fit blocks on their uncalibrated
    priors, and returns the joint :class:`sysid.JointPosterior` from which
    :func:`sysid.sample_tier` draws calibrated randomization (doc N Q2/Q3; doc D
    calibrated_bounds_only tier). When *no* traces are supplied the posterior is
    the prior block (mean = prior centre, prior-width covariance) so a generator
    can still sample tiers before any real trace exists.
    """
    from . import sysid
    from .faults import FaultParams
    from .power import PowerParams

    power_blocks = (
        sysid.fit_power(power_traces, PowerParams()) if power_traces else None
    )
    thermal_block = (
        sysid.fit_thermal(thermal_trace, pod.thermal)  # type: ignore[arg-type]
        if thermal_trace is not None
        else None
    )
    hazard_block = sysid.fit_hazard(
        n_events=fault_events,
        exposure_gpu_s=fault_exposure_gpu_s,
        prior=FaultParams(),
    )
    return sysid.build_joint_posterior(
        power_blocks=power_blocks,
        thermal_block=thermal_block,
        hazard_block=hazard_block,
        graph=pod.thermal,
        cross_corr=cross_corr,
    )


__all__ = [
    "COOLING_STATES",
    "HEALTH_STATES",
    "OPERATOR_MODES",
    "WORKLOAD_PHASES",
    "PhaseProgram",
    "PodSpec",
    "PodTopology",
    "build_pod",
    "calibrated_posterior",
    "context_from_latents",
    "emit_gpu_telemetry",
    "sample_phase_program",
    "source_state_from_health",
    "step_cooling",
    "step_health",
]
