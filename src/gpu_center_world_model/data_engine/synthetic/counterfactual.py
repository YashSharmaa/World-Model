# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Paired counterfactual branch construction (design docs D, R).

The training/evaluation unit is a *decision window* with a frozen pre-action
state hash and a shared exogenous seed bundle (doc R, counterfactual_branch_
construction). Branches differ **only** by the clamped action — never by drifting
exogenous noise — which is the single biggest defence against optimism bias in
learned world models (doc D, Inference on the state hash).

The construction follows doc R verbatim:

1. **Abduction.** Fix the exact pre-action canonical state ``X_τ`` and read (or
   sample) the exogenous seed bundle ``Û`` from provenance.
2. **Branch root.** ``B0 = (h, S_τ, Û, action_catalog)`` — every descendant
   shares ``S_τ`` and ``Û``.
3. **Action substitution.** For each candidate ``a_b``, replace the observational
   action law with ``do(A_τ = a_b)``.
4. **Deterministic exact update.** Run the exact kernel once (feasibility +
   blockers + ``X_τ⁺``). Infeasible branches terminate with a why-not ledger.
5. **Prediction.** Feasible branches propagate the stochastic residual SCM under
   the **same** ``Û`` (common random numbers).
6. **Paired pack.** Emit **≥ 3 actions** from the same hash + seed bundle so
   branch deltas estimate intervention effects, not intervention-plus-noise
   (doc D, splits.counterfactual_harness; doc R, Eq11/Eq12).

The exact-kernel logic here is a *self-contained, deterministic* implementation of
the demo-pod feasibility/transition consistent with
:mod:`...simulator.contracts` (``FeasibilityResult``, ``Blocker``,
``BlockerGroup``). It owns hard feasibility (quota, gang atomicity, topology fit,
health masks, operator whitelist) — it is the shield and is never learned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...schema.actions import (
    Action,
    ActionType,
    Admit,
    AdmitAndPlace,
    Hold,
    Place,
    action_workload_uid,
)
from ...schema.hashing import state_hash
from ...schema.kpis import HazardChannel, KPIChannel
from ...schema.state import SymbolicState, WorldState
from ...simulator.contracts import (
    Blocker,
    BlockerGroup,
    FeasibilityResult,
)
from ...utils.seeding import SeedBundle
from .faults import (
    FaultParams,
    fault_pressure_at,
    sample_fault_cluster,
)
from .power import (
    PowerParams,
    init_power_state,
    power_cap_array,
    rack_power_kw,
    step_power,
)
from .scm import (
    WORKLOAD_PHASES,
    PodTopology,
    sample_phase_program,
    step_cooling,
    step_health,
)
from .thermal import ThermalState, step_thermal


# ──────────────────────────────────────────────────────────────────────────────
# Exact kernel (self-contained demo-pod feasibility + transition)
# ──────────────────────────────────────────────────────────────────────────────
def _workload_gpu_demand(state: SymbolicState, uid: str) -> int:
    wl = state.workloads.get(uid)
    if wl is None:
        return 0
    return int(sum(ps.count * ps.gpu_req for ps in wl.podsets))


def feasible(state: WorldState, action: Action) -> FeasibilityResult:
    """Exact feasibility for one demo action (the shield; doc R, [D3]/[Eq1]).

    Checks, in semantic blocker order (doc S): action whitelist → status
    precondition → quota → topology fit (gang atomicity inside a domain) →
    health/schedulability. Returns the integer assignment when feasible.
    """
    sym = state.symbolic
    policy = sym.operator_policy
    blockers: list[Blocker] = []

    if action.type == ActionType.HOLD:
        return FeasibilityResult(feasible=True)  # hold is always feasible

    if action.type.value not in policy.allowed_actions:
        blockers.append(
            Blocker(BlockerGroup.ACTION_WHITELIST, f"{action.type.value} not allowed")
        )
        return FeasibilityResult(feasible=False, blockers=blockers)

    uid = action_workload_uid(action)
    wl = sym.workloads.get(uid) if uid else None
    if wl is None:
        blockers.append(Blocker(BlockerGroup.STATUS_PRECONDITION, "unknown workload"))
        return FeasibilityResult(feasible=False, blockers=blockers)

    demand = _workload_gpu_demand(sym, uid)

    # Quota: cluster-queue reserved + demand must fit nominal quota.
    if action.type in (ActionType.ADMIT, ActionType.ADMIT_AND_PLACE):
        cq_id = getattr(action, "cluster_queue_id", None)
        cq = sym.cluster_queues.get(cq_id) if cq_id else None
        if cq is None:
            blockers.append(
                Blocker(BlockerGroup.QUEUE_FLAVOR_ELIGIBILITY, "unknown cluster queue")
            )
            return FeasibilityResult(feasible=False, blockers=blockers)
        flavor_id = getattr(action, "resource_flavor_id", "")
        nominal = cq.nominal_quota.get(flavor_id, {}).get("gpu", 0.0)
        if cq.reserved_gpu + demand > nominal + 1e-9:
            deficit = cq.reserved_gpu + demand - nominal
            blockers.append(
                Blocker(BlockerGroup.QUOTA, "gpu quota exceeded", {"gpu": deficit})
            )

    # Topology fit: gang atomicity — the whole gang must fit one feasible domain.
    feasible_domains: list[str] = []
    assignment: dict[str, dict[str, int]] = {}
    if action.type in (ActionType.PLACE, ActionType.ADMIT_AND_PLACE):
        masked = set(policy.protected_domains)
        cand_domains = getattr(action, "domain_id_or_set", ()) or tuple(
            d for d, dom in sym.domains.items() if dom.level.value == "rack"
        )
        for dom_id in cand_domains:
            if dom_id in masked:
                continue
            free = _domain_free_gpu(sym, dom_id)
            if free >= demand:
                feasible_domains.append(dom_id)
        if not feasible_domains:
            blockers.append(
                Blocker(
                    BlockerGroup.GANG_ATOMICITY,
                    "gang does not fit any feasible domain",
                    {"gpu": float(demand)},
                )
            )
        else:
            assignment = _assign_gang(sym, feasible_domains[0], demand)
            if not assignment:
                blockers.append(
                    Blocker(BlockerGroup.HEALTH_SCHEDULABILITY, "no healthy gpus")
                )

    ok = len(blockers) == 0
    return FeasibilityResult(
        feasible=ok,
        blockers=blockers,
        feasible_domains=feasible_domains,
        assignment=assignment if ok else {},
    )


def _domain_free_gpu(sym: SymbolicState, dom_id: str) -> int:
    """Free schedulable GPU in a rack domain (ready & unallocated; doc TAS rule)."""
    dom = sym.domains.get(dom_id)
    if dom is None:
        return 0
    # rack domain id convention: "dom-rack-{r}" ↔ rack "rack-{r}"
    rack_id = dom_id.replace("dom-rack-", "rack-")
    free = 0
    for gpu in sym.gpus.values():
        node = sym.nodes.get(gpu.node_uid)
        if node is None or node.rack_id != rack_id:
            continue
        if not node.ready or not node.schedulable:
            continue
        if node.node_health.value not in ("ready", "degraded"):
            continue
        if gpu.allocated_to is None and gpu.health_state.value in (
            "healthy",
            "degraded",
        ):
            free += 1
    return free


def _assign_gang(
    sym: SymbolicState, dom_id: str, demand: int
) -> dict[str, dict[str, int]]:
    """Produce a concrete integer placement (podset → {node: replicas})."""
    rack_id = dom_id.replace("dom-rack-", "rack-")
    by_node: dict[str, int] = {}
    remaining = demand
    for gpu in sym.gpus.values():
        if remaining <= 0:
            break
        node = sym.nodes.get(gpu.node_uid)
        if node is None or node.rack_id != rack_id:
            continue
        if gpu.allocated_to is None and gpu.health_state.value in (
            "healthy",
            "degraded",
        ):
            by_node[gpu.node_uid] = by_node.get(gpu.node_uid, 0) + 1
            remaining -= 1
    if remaining > 0:
        return {}
    return {"gang": by_node}


def step(state: WorldState, action: Action) -> WorldState:
    """Apply the exact symbolic transition ``X_{t+1} = K(X_t, a_t)`` (doc R, Eq1).

    Deterministic. Infeasible actions are a *reject / no-op*, never a teleport
    into an infeasible allocation (doc actions.py interventional semantics).
    """
    import copy

    res = feasible(state, action)
    new_sym = copy.deepcopy(state.symbolic)
    if not res.feasible:
        return WorldState(
            ts_ns=state.ts_ns,
            tick=state.tick + 1,
            symbolic=new_sym,
            context=state.context,
            source=state.source,
        )
    uid = action_workload_uid(action)
    wl = new_sym.workloads.get(uid) if uid else None
    if action.type in (ActionType.ADMIT, ActionType.ADMIT_AND_PLACE) and wl is not None:
        cq_id = getattr(action, "cluster_queue_id", None)
        cq = new_sym.cluster_queues.get(cq_id)
        demand = _workload_gpu_demand(new_sym, uid)
        if cq is not None:
            cq.reserved_gpu += demand
        wl.status = (
            new_sym.workloads[uid].status.ADMITTED
            if action.type == ActionType.ADMIT
            else new_sym.workloads[uid].status.RUNNING
        )
        wl.admitted_flavor_id = getattr(action, "resource_flavor_id", None)
    if action.type in (ActionType.PLACE, ActionType.ADMIT_AND_PLACE) and wl is not None:
        if res.feasible_domains:
            wl.admitted_domain_id = res.feasible_domains[0]
        # commit the integer assignment onto the GPUs
        for _ps, by_node in res.assignment.items():
            for node_id, k in by_node.items():
                assigned = 0
                for gpu in new_sym.gpus.values():
                    if assigned >= k:
                        break
                    if gpu.node_uid == node_id and gpu.allocated_to is None:
                        gpu.allocated_to = uid
                        assigned += 1
    return WorldState(
        ts_ns=state.ts_ns,
        tick=state.tick + 1,
        symbolic=new_sym,
        context=state.context,
        source=state.source,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Branch root + common-random-number rollout
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class BranchOutcome:
    """The realised KPI/hazard summary for one action branch (doc R, Eq11)."""

    action_id: str
    action: Action
    feasible: bool
    blockers: list[str] = field(default_factory=list)
    kpis: dict[str, float] = field(default_factory=dict)  # KPIChannel -> value
    hazards: dict[str, float] = field(default_factory=dict)  # HazardChannel -> prob
    placed_domain: str | None = None


@dataclass(slots=True)
class CounterfactualPack:
    """A paired counterfactual pack: ≥ 3 branches from one hash + seed bundle."""

    state_hash: str
    seed_bundle: int
    horizon_s: int
    baseline_action_id: str
    branches: list[BranchOutcome] = field(default_factory=list)

    def deltas(self, baseline_id: str | None = None) -> dict[str, dict[str, float]]:
        """Branch KPI deltas vs. a baseline action (doc R, Eq12; doc kpis deltas).

        Returns ``{action_id: {channel: Δ}}`` so labels can be expressed as
        deltas relative to ``hold`` (the canonical baseline, doc kpis docstring).
        """
        base_id = baseline_id or self.baseline_action_id
        base = next((b for b in self.branches if b.action_id == base_id), None)
        out: dict[str, dict[str, float]] = {}
        if base is None:
            return out
        for b in self.branches:
            if b.action_id == base_id:
                continue
            out[b.action_id] = {
                k: b.kpis.get(k, 0.0) - base.kpis.get(k, 0.0) for k in b.kpis
            }
        return out


def build_pack(
    pod: PodTopology,
    base_state: WorldState,
    actions: list[Action],
    seed: int,
    horizon_s: int,
    *,
    baseline_action_id: str = "hold",
    fault_tilt: float = 1.0,
) -> CounterfactualPack:
    """Build a counterfactual pack: roll each action under common random numbers.

    Every branch shares the exogenous seed bundle ``Û`` (doc R, [A5]); only the
    clamped action changes (doc R, [A4]). The pre-action hash anchors the pack
    (doc R, abduction step 1/2). At least three actions are required.
    """
    if len(actions) < 3:
        raise ValueError("counterfactual pack requires >= 3 actions (doc D harness)")
    bundle = SeedBundle(master=seed)
    h = state_hash(base_state.symbolic, base_state.context)
    pack = CounterfactualPack(
        state_hash=h,
        seed_bundle=seed,
        horizon_s=horizon_s,
        baseline_action_id=baseline_action_id,
    )
    for i, action in enumerate(actions):
        aid = _action_id(action, i)
        outcome = _rollout_branch(
            pod, base_state, action, aid, bundle, horizon_s, fault_tilt
        )
        pack.branches.append(outcome)
    return pack


def _action_id(action: Action, idx: int) -> str:
    uid = action_workload_uid(action) or "-"
    dom = getattr(action, "domain_id_or_set", ())
    dom_s = ",".join(dom) if dom else ""
    return f"{idx}:{action.type.value}:{uid}:{dom_s}"


def _rollout_branch(
    pod: PodTopology,
    base_state: WorldState,
    action: Action,
    action_id: str,
    bundle: SeedBundle,
    horizon_s: int,
    fault_tilt: float,
) -> BranchOutcome:
    """Roll one action branch forward deterministically under shared seeds.

    Step 4 (exact update) then step 5 (stochastic residual SCM) of doc R. The
    residual rollout draws from per-stream generators derived from the *shared*
    bundle, so two branches see identical phase/health/cooling/fault noise — the
    common-random-number property (doc R, Eq11/Eq12).
    """
    res = feasible(base_state, action)
    if not res.feasible:
        return BranchOutcome(
            action_id=action_id,
            action=action,
            feasible=False,
            blockers=[b.group.value for b in res.blockers],
        )
    post = step(base_state, action)
    placed = None
    uid = action_workload_uid(action)
    if uid is not None:
        wl = post.symbolic.workloads.get(uid)
        placed = wl.admitted_domain_id if wl else None

    # ----- stochastic residual rollout under common random numbers -----------
    rng_phase = np.random.default_rng(bundle.stream("phase"))
    rng_health = np.random.default_rng(bundle.stream("health"))
    rng_cooling = np.random.default_rng(bundle.stream("cooling"))
    rng_tel = np.random.default_rng(bundle.stream("telemetry"))
    rng_fault = np.random.default_rng(bundle.stream("health"))

    n_gpu = len(pod.gpu_ids)
    n_rack = len(pod.rack_ids)
    dt_s = 30.0
    n_steps = max(1, int(horizon_s / dt_s))

    # which GPUs are active under this branch (allocated_to set ⇒ running a job)
    active_mask = np.array(
        [post.symbolic.gpus[g].allocated_to is not None for g in pod.gpu_ids]
    )
    phase_prog = sample_phase_program(horizon_s, rng_phase)
    health = {g: post.symbolic.gpus[g].health_state.value for g in pod.gpu_ids}
    cooling_state = "normal"

    faults = sample_fault_cluster(
        [g for g, a in zip(pod.gpu_ids, active_mask, strict=False) if a],
        horizon_s,
        rng_fault,
        FaultParams(),
        health_state=health,
        rate_tilt=fault_tilt,
    )

    pwr_params = PowerParams()
    pwr = init_power_state(n_gpu, n_rack)
    thermal = ThermalState(theta=np.zeros(pod.thermal.n))
    ambient = base_state.context.ambient_temp_c

    max_rack_power = 0.0
    min_headroom = float("inf")
    hotspot_steps = 0
    requeue_flag = 0.0
    xid_flag = 0.0
    pods_ready_timeout = 0.0
    completion_progress = 0.0

    for k in range(n_steps):
        t = k * dt_s
        phase_idx = phase_prog.index_at(t)
        gpu_phase = np.full(n_gpu, phase_idx, dtype=int)
        # synchronised transitions per rack: gang ⇒ all active gpus transition
        prev_idx = phase_prog.index_at(max(t - dt_s, 0.0))
        transitioned = int(phase_idx != prev_idx)
        trans_per_rack = np.zeros(n_rack)
        for r in range(n_rack):
            n_active_in_rack = int(np.sum(active_mask & (pod.gpu_rack_idx == r)))
            trans_per_rack[r] = transitioned * (n_active_in_rack > 0)

        caps = power_cap_array(n_gpu, pwr_params)
        # idle GPUs draw near-zero: encode as a "staging"-like phase
        eff_phase = gpu_phase.copy()
        eff_phase[~active_mask] = WORKLOAD_PHASES.index("staging")
        pwr = step_power(
            pwr,
            eff_phase,
            list(WORKLOAD_PHASES),
            pod.gpu_rack_idx,
            trans_per_rack,
            caps,
            pwr_params,
            rng_tel,
            dt_s,
        )
        # thermal: map per-gpu power into the gpu thermal compartments. Power (W)
        # is scaled to heat-source units so a full-load GPU sits ~30 K above the
        # inlet reference (matched to the conductances in scm.build_pod).
        heat = np.zeros(pod.thermal.n)
        for gi, tidx in enumerate(pod.gpu_thermal_idx):
            heat[tidx] = pwr.p_gpu_w[gi] / 100.0
        boundary = ambient + (3.0 if cooling_state == "derated" else 0.0)
        thermal = step_thermal(pod.thermal, thermal, heat, boundary, dt_s)

        racks_kw = rack_power_kw(pwr, pod.gpu_rack_idx, n_rack, pwr_params)
        max_rack_power = max(max_rack_power, float(racks_kw.max()))

        # Rack thermal headroom = design_limit − rack temperature deviation. The
        # 25 K design margin above the inlet reference leaves positive headroom at
        # full load (~18 K rack deviation) but tightens toward zero — and into
        # violation — under a cooling derate, exactly the delayed-hotspot pathway
        # of doc D (cooling_derate rare event).
        rack_temp = np.array(
            [thermal.theta[pod.thermal.index[f"T:{rid}"]] for rid in pod.rack_ids]
        )
        headroom = 25.0 - rack_temp
        min_headroom = min(min_headroom, float(headroom.min()))
        if float(headroom.min()) < 4.0:
            hotspot_steps += 1

        # health chain under fault pressure (common random numbers)
        for gi, g in enumerate(pod.gpu_ids):
            if not active_mask[gi]:
                continue
            fp = fault_pressure_at(faults, g, t)
            health[g] = step_health(health[g], rng_health, fault_pressure=fp)
            if health[g] in ("intermittently_faulty", "masked"):
                xid_flag = 1.0
        cooling_state = step_cooling(cooling_state, rng_cooling)
        completion_progress += float(np.mean(active_mask)) * dt_s

    # requeue / pods-ready timeout hazards from realised faults
    requeue_flag = (
        1.0
        if any(
            any(d[0] == "requeue" for d in f.descendants) and f.t_parent_s < horizon_s
            for f in faults
        )
        else 0.0
    )
    if action.type == ActionType.HOLD:
        pods_ready_timeout = 0.0
    else:
        # if the gang barely fit, readiness can time out under fault pressure
        pods_ready_timeout = 0.3 * xid_flag

    kpis = {
        KPIChannel.RACK_POWER_KW.value: max_rack_power,
        KPIChannel.RACK_THERMAL_HEADROOM_C.value: (
            min_headroom if np.isfinite(min_headroom) else 15.0
        ),
        KPIChannel.CROSS_RACK_RATE.value: _branch_cross_rack(pod, post, uid),
        KPIChannel.ADMITTED_GPU_HOURS_DELTA.value: completion_progress / 3600.0,
        KPIChannel.COMPLETION_TIME_DELTA_S.value: -completion_progress * 0.1,
    }
    hazards = {
        HazardChannel.HOTSPOT.value: hotspot_steps / max(n_steps, 1),
        HazardChannel.XID_FAULT.value: xid_flag,
        HazardChannel.REQUEUE.value: requeue_flag,
        HazardChannel.PODS_READY_TIMEOUT.value: pods_ready_timeout,
        HazardChannel.HEADROOM_VIOLATION.value: 1.0 if min_headroom < 0 else 0.0,
    }
    return BranchOutcome(
        action_id=action_id,
        action=action,
        feasible=True,
        kpis=kpis,
        hazards=hazards,
        placed_domain=placed,
    )


def _branch_cross_rack(pod: PodTopology, post: WorldState, uid: str | None) -> float:
    """Cross-rack communication rate for the placed gang (locality KPI)."""
    if uid is None:
        return 0.0
    gpus = [g for g, gp in post.symbolic.gpus.items() if gp.allocated_to == uid]
    if len(gpus) < 2:
        return 0.0
    racks = {pod.symbolic.nodes[post.symbolic.gpus[g].node_uid].rack_id for g in gpus}
    return 0.0 if len(racks) <= 1 else 1.0


def default_action_catalog(state: WorldState, workload_uid: str) -> list[Action]:
    """Enumerate ≥ 3 candidate actions for a workload (doc D harness).

    hold, admit+place-in-preferred-domain, admit+place-in-alternative-domain
    (doc D, splits.counterfactual_harness). Domains are read from the symbolic
    state so the catalog is always feasible-shaped.
    """
    sym = state.symbolic
    cq_id = next(iter(sym.cluster_queues), "cq-main")
    flavor_id = next(iter(sym.flavors), "h100-sxm")
    rack_domains = [d for d, dom in sym.domains.items() if dom.level.value == "rack"]
    actions: list[Action] = [Hold(workload_uid=workload_uid)]
    if rack_domains:
        actions.append(
            AdmitAndPlace(
                workload_uid=workload_uid,
                cluster_queue_id=cq_id,
                resource_flavor_id=flavor_id,
                domain_policy="preferred",
                domain_id_or_set=(rack_domains[0],),
            )
        )
    if len(rack_domains) > 1:
        actions.append(
            AdmitAndPlace(
                workload_uid=workload_uid,
                cluster_queue_id=cq_id,
                resource_flavor_id=flavor_id,
                domain_policy="preferred",
                domain_id_or_set=(rack_domains[1],),
            )
        )
    else:
        actions.append(
            Admit(
                workload_uid=workload_uid,
                cluster_queue_id=cq_id,
                resource_flavor_id=flavor_id,
            )
        )
    # spread across two domains (unconstrained) as a third placement variant
    if len(rack_domains) > 1:
        actions.append(
            Place(
                workload_uid=workload_uid,
                domain_policy="unconstrained",
                domain_id_or_set=tuple(rack_domains[:2]),
            )
        )
    return actions


__all__ = [
    "BranchOutcome",
    "CounterfactualPack",
    "build_pack",
    "default_action_catalog",
    "feasible",
    "step",
]
