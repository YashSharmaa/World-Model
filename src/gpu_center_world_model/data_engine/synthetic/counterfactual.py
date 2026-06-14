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

KERNEL UNIFICATION (doc R, [D3]/[Eq1]/[T2]): the generator and the safety shield
**must be byte-identical** so that "same (X_τ, a_τ, M_τ) ⇒ same feasibility,
blocker set, feasible domains, X_τ⁺" (doc R, T2). To guarantee that, the
feasibility/transition used for branch construction is the **real**
:class:`...simulator.exact_kernel.ExactSymbolicKernel` — there is no second
hand-rolled copy to drift out of sync. :func:`feasible` and :func:`step` below are
**thin module-level adapters** over a shared kernel instance (read-only use of the
kernel's stable public API ``feasible`` / ``safe_actions`` / ``step`` / ``why_not``).
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
from ...schema.state import WorldState
from ...simulator.contracts import FeasibilityResult
from ...simulator.exact_kernel import ExactSymbolicKernel
from ...utils.seeding import SeedBundle
from .faults import (
    FaultParams,
    HazardResidual,
    fault_pressure_at,
    sample_fault_cluster,
)
from .power import (
    PowerParams,
    PowerResidual,
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
from .thermal import ThermalResidual, ThermalState, step_thermal

# ──────────────────────────────────────────────────────────────────────────────
# Exact kernel adapters — the generator uses the REAL shield (doc R, [D3]/[T2])
# ──────────────────────────────────────────────────────────────────────────────
#: One shared, stateless kernel instance. The kernel is deterministic and owns no
#: mutable per-call state, so a module-level singleton is safe and keeps the
#: generator's feasibility byte-identical to the deployed shield (doc R, T2).
_KERNEL = ExactSymbolicKernel()


def feasible(state: WorldState, action: Action) -> FeasibilityResult:
    """Exact feasibility for one action — thin adapter over the real shield.

    Delegates verbatim to :meth:`ExactSymbolicKernel.feasible` so the generator
    and the deployed safety gate certify the *same* safe set (doc R, T2). The
    return type is the shared :class:`...simulator.contracts.FeasibilityResult`
    (blockers, feasible-domain selection labels, integer assignment).
    """
    return _KERNEL.feasible(state, action)


def step(state: WorldState, action: Action) -> WorldState:
    """Apply the exact symbolic transition ``X_{t+1} = K(X_t, a_t)`` (doc R, Eq1).

    Thin adapter over :meth:`ExactSymbolicKernel.step`: deterministic, with
    interventional reject-as-no-op semantics. Using the kernel here means a branch
    transition can never diverge from the shield's transition (doc R, T2).
    """
    return _KERNEL.step(state, action)


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


@dataclass(slots=True)
class RareEventSpec:
    """A single scheduled rare-event onset (the ONE unified injection path).

    The data engine ships exactly one rare-event application path (doc D
    rare_event_library; doc R rare_event_simulation). ``cfg.rare_event`` (or a
    scenario_loader injection) is translated into this spec, which the rollout
    applies **at a scheduled onset step** through the simulator's state-mutating
    library (:func:`...simulator.rare_events.apply_event`). The *same* library the
    branch runner and exact kernel re-read is used here, so a fault burst masks
    GPUs, a cooling derate lowers the shared zone margin/coupling before any
    delayed thermal effect, and a queue burst bursts pending counts — all under
    common random numbers so paired branches see the identical exogenous event
    (doc R causal-consistency rules: perturb the shared parent state first).

    ``name`` is a member of :data:`...simulator.rare_events.RARE_EVENTS`.
    ``onset_s`` schedules the mutation: ``0`` applies it at ``t0`` (reconciling the
    scenario_loader pre-state path), ``>0`` applies it mid-rollout at the first
    step ``k`` with ``k·dt ≥ onset_s`` (the "fault onset time" the doc randomizes
    aggressively). ``kwargs`` carry the event's severity-shaped parameters.
    """

    name: str
    onset_s: float = 0.0
    kwargs: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ResidualBundle:
    """The optional learned residuals applied during a branch rollout (doc N H1).

    Defaults to ``None`` everywhere ⇒ identity ⇒ the rollout is the calibrated
    pure-physics generator (TP1). A trained model injects modulating residuals to
    couple physics + learned dynamics without breaking passivity (doc N H2/H3).
    """

    thermal: ThermalResidual | None = None
    power: PowerResidual | None = None
    hazard: HazardResidual | None = None


def build_pack(
    pod: PodTopology,
    base_state: WorldState,
    actions: list[Action],
    seed: int,
    horizon_s: int,
    *,
    baseline_action_id: str = "hold",
    fault_tilt: float = 1.0,
    rare_event: RareEventSpec | None = None,
    residuals: ResidualBundle | None = None,
) -> CounterfactualPack:
    """Build a counterfactual pack: roll each action under common random numbers.

    Every branch shares the exogenous seed bundle ``Û`` (doc R, [A5]); only the
    clamped action changes (doc R, [A4]). The pre-action hash anchors the pack
    (doc R, abduction step 1/2). At least three actions are required.

    ``rare_event`` is the single unified rare-event onset applied identically to
    every branch via the simulator library (doc D/R); ``residuals`` are optional
    learned physics residuals (doc N H1, identity by default).
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
            pod,
            base_state,
            action,
            aid,
            bundle,
            horizon_s,
            fault_tilt,
            rare_event=rare_event,
            residuals=residuals,
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
    *,
    rare_event: RareEventSpec | None = None,
    residuals: ResidualBundle | None = None,
) -> BranchOutcome:
    """Roll one action branch forward deterministically under shared seeds.

    Step 4 (exact update) then step 5 (stochastic residual SCM) of doc R. The
    residual rollout draws from per-stream generators derived from the *shared*
    bundle, so two branches see identical phase/health/cooling/fault noise — the
    common-random-number property (doc R, Eq11/Eq12).

    The unified rare-event onset (``rare_event``) is applied **once**, at its
    scheduled step, through the simulator's state-mutating library. Its symbolic
    effect (masked GPUs, derated zone, burst pending) then *propagates* through
    the rest of the rollout — the kernel sees a smaller healthy pool, the thermal
    boundary rises, etc. — exactly the doc-R causal-consistency requirement that a
    rare event perturbs the shared parent state first and induces downstream
    effects only afterwards.
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

    residuals = residuals or ResidualBundle()

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

    # Onset bookkeeping: the rare event fires once, at the first step whose end of
    # interval reaches ``onset_s`` (``onset_s == 0`` fires at t0, before step 0).
    onset_step = _onset_step(rare_event, dt_s, n_steps) if rare_event else None
    rare_applied = False
    cur_state = post  # the symbolic world the rare event mutates in place

    # If the onset is at t0, apply it before the dynamics so the very first step
    # already runs on the perturbed parent state (reconciles scenario_loader t0).
    if rare_event is not None and onset_step == 0:
        cur_state = _apply_rare_onset(rare_event, cur_state, bundle, step_i=0)
        rare_applied = True

    # which GPUs are active under this branch (allocated_to set ⇒ running a job).
    # Recomputed lazily from ``cur_state`` after every onset so a fault burst that
    # masks a placed GPU drops it from the active set immediately.
    def _active_from(state: WorldState) -> np.ndarray:
        return np.array(
            [
                state.symbolic.gpus[g].allocated_to is not None
                and state.symbolic.gpus[g].health_state.value
                in ("healthy", "degraded")
                for g in pod.gpu_ids
            ]
        )

    active_mask = _active_from(cur_state)
    phase_prog = sample_phase_program(horizon_s, rng_phase)
    health = {g: cur_state.symbolic.gpus[g].health_state.value for g in pod.gpu_ids}
    cooling_state = "derated" if _zone_derated(cur_state) else "normal"

    faults = sample_fault_cluster(
        [g for g, a in zip(pod.gpu_ids, active_mask, strict=False) if a],
        horizon_s,
        rng_fault,
        FaultParams(),
        health_state=health,
        rate_tilt=fault_tilt,
        residual=residuals.hazard,
    )

    pwr_params = PowerParams()
    pwr = init_power_state(n_gpu, n_rack)
    thermal = ThermalState(theta=np.zeros(pod.thermal.n))
    ambient = base_state.context.ambient_temp_c

    max_rack_power = 0.0
    min_headroom = float("inf")
    hotspot_steps = 0
    requeue_flag = 0.0
    xid_flag = 1.0 if _zone_fault_present(cur_state, pod) else 0.0
    pods_ready_timeout = 0.0
    completion_progress = 0.0
    queue_burst_active = _queue_bursting(cur_state)

    for k in range(n_steps):
        t = k * dt_s
        # ---- mid-rollout rare-event onset (the single unified injection path) --
        if (
            rare_event is not None
            and not rare_applied
            and onset_step is not None
            and k == onset_step
        ):
            cur_state = _apply_rare_onset(rare_event, cur_state, bundle, step_i=k)
            rare_applied = True
            # Propagate the perturbed parent state into the rollout immediately.
            active_mask = _active_from(cur_state)
            for g in pod.gpu_ids:
                health[g] = cur_state.symbolic.gpus[g].health_state.value
            if _zone_derated(cur_state):
                cooling_state = "derated"
            if _zone_fault_present(cur_state, pod):
                xid_flag = 1.0
            if _queue_bursting(cur_state):
                queue_burst_active = True

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
            residual=residuals.power,
        )
        # thermal: map per-gpu power into the gpu thermal compartments. Power (W)
        # is scaled to heat-source units so a full-load GPU sits ~30 K above the
        # inlet reference (matched to the conductances in scm.build_pod).
        heat = np.zeros(pod.thermal.n)
        for gi, tidx in enumerate(pod.gpu_thermal_idx):
            heat[tidx] = pwr.p_gpu_w[gi] / 100.0
        # Boundary inlet rise from an active cooling derate, read from the mutated
        # zone state so the *magnitude* tracks the event severity (a 30 % derate
        # raises inlet ~6 K — the simulator library's zone perturbation) rather
        # than a flat penalty (doc D: shared zone-level perturbation, then delayed
        # hotspot via thermal inertia).
        derate_inlet = (
            _zone_inlet_rise(cur_state) if cooling_state == "derated" else 0.0
        )
        boundary = ambient + derate_inlet
        thermal = step_thermal(
            pod.thermal, thermal, heat, boundary, dt_s, residual=residuals.thermal
        )

        racks_kw = rack_power_kw(pwr, pod.gpu_rack_idx, n_rack, pwr_params)
        max_rack_power = max(max_rack_power, float(racks_kw.max()))

        # Rack thermal headroom = design_limit − rack temperature deviation. The
        # 25 K design margin above the inlet reference leaves positive headroom at
        # full load but tightens toward zero — and into violation — under a cooling
        # derate. A derate shrinks BOTH sides of the budget: it raises the inlet
        # boundary (above) AND lowers the usable design limit, because a zone that
        # has lost cooling capacity can safely reject less heat (doc D: shared
        # zone-level perturbation → delayed hotspot via thermal inertia).
        design_limit = 25.0 - (
            _zone_margin_loss(cur_state) * 18.0 if cooling_state == "derated" else 0.0
        )
        rack_temp = np.array(
            [thermal.theta[pod.thermal.index[f"T:{rid}"]] for rid in pod.rack_ids]
        )
        headroom = design_limit - rack_temp
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
        # if the gang barely fit, readiness can time out under fault pressure; a
        # queue burst raises the pods-ready timeout pressure regardless of faults.
        pods_ready_timeout = max(0.3 * xid_flag, 0.5 if queue_burst_active else 0.0)

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


# ──────────────────────────────────────────────────────────────────────────────
# Unified rare-event onset application (THE single injection path)
# ──────────────────────────────────────────────────────────────────────────────
def _onset_step(spec: RareEventSpec, dt_s: float, n_steps: int) -> int:
    """First rollout step at which the scheduled onset fires (``onset_s == 0 → 0``).

    The onset fires at the first step ``k`` whose interval *end* ``(k+1)·dt`` has
    reached ``onset_s`` — i.e. ``k = floor(onset_s / dt)`` — clamped into range so
    a late onset still lands on the last step rather than being silently dropped.
    """
    if spec.onset_s <= 0.0:
        return 0
    return int(min(max(0, int(spec.onset_s // dt_s)), n_steps - 1))


def _apply_rare_onset(
    spec: RareEventSpec, state: WorldState, bundle: SeedBundle, *, step_i: int
) -> WorldState:
    """Apply ONE rare-event onset through the simulator state-mutating library.

    This is the single point where a named rare event mutates the world during a
    rollout. It dispatches to :func:`...simulator.rare_events.apply_event` — the
    *same* library the branch runner uses and the exact kernel re-reads — with a
    seed derived from the event's natural noise stream (so a fault burst draws from
    ``health``, a cooling derate from ``cooling``, a queue burst from ``arrivals``)
    XOR-mixed with the onset step for determinism under common random numbers
    across paired branches (doc seeding STREAMS; doc R causal-consistency rules).
    """
    from ...simulator.rare_events import apply_event

    stream_of = {
        "gpu_fault_burst": "health",
        "pods_ready_timeout": "phase",
        "cooling_derate": "cooling",
        "power_derate": "cooling",
        "queue_burst": "arrivals",
        "maintenance_mask": "operator",
        "telemetry_degradation": "telemetry",
    }
    base = bundle.stream(stream_of.get(spec.name, "regime"))
    seed = base ^ (step_i * 0x85EBCA6B)
    return apply_event(spec.name, state, seed, **spec.kwargs)


def _zone_derated(state: WorldState) -> bool:
    """True iff any cooling zone has lost margin (a derate is active; doc D)."""
    return any(
        cz.cooling_margin < 0.999 or cz.inlet_temp_c > 0.0 + 1e-6
        for cz in state.symbolic.cooling_zones.values()
    )


def _zone_inlet_rise(state: WorldState) -> float:
    """Max inlet-temperature rise across cooling zones (K) from an active derate.

    The simulator ``cooling_derate`` event raises each zone's ``inlet_temp_c``;
    the rollout reads the realised rise so the thermal boundary tracks the event's
    severity rather than a flat penalty (doc D shared-zone perturbation).
    """
    rises = [cz.inlet_temp_c for cz in state.symbolic.cooling_zones.values()]
    return float(max(rises)) if rises else 0.0


def _zone_margin_loss(state: WorldState) -> float:
    """Fractional cooling-margin loss across zones in ``[0, 1]`` (derate depth).

    ``cooling_margin`` starts at ~1.0 and is multiplied down by a derate; the loss
    ``1 − margin`` measures how much usable thermal headroom the zone forfeits.
    """
    losses = [
        max(0.0, 1.0 - cz.cooling_margin)
        for cz in state.symbolic.cooling_zones.values()
    ]
    return float(max(losses)) if losses else 0.0


def _zone_fault_present(state: WorldState, pod: PodTopology) -> bool:
    """True iff any GPU is in a fault/masked device-health state (burst onset)."""
    return any(
        state.symbolic.gpus[g].health_state.value
        in ("intermittently_faulty", "masked", "reset_recovery")
        for g in pod.gpu_ids
    )


def _queue_bursting(state: WorldState) -> bool:
    """True iff the context records an active arrival burst (queue_burst onset)."""
    return state.context.arrival_burst_regime in ("burst", "campaign")


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
    "RareEventSpec",
    "ResidualBundle",
    "build_pack",
    "default_action_catalog",
    "feasible",
    "step",
]
