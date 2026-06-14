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
from ...schema.observations import Observation, SourceHealth
from ...schema.state import WorldState
from ...simulator.contracts import FeasibilityResult
from ...simulator.exact_kernel import ExactSymbolicKernel
from ...utils.seeding import SeedBundle
from .arrivals import (
    REGIMES,
    QueueSummary,
    RegimeChain,
    TenantClass,
    regime_at,
    simulate_arrivals,
    simulate_pending_queue,
)
from .faults import (
    FaultEvent,
    FaultParams,
    HazardResidual,
    fault_pressure_at,
    realized_fault_intensities,
    sample_fault_cluster,
)
from .network import (
    PacketTeacherCorrection,
    allocate,
    commodities_for_gang,
    cross_rack_rate,
    hotspot_risk,
    pod_links,
)
from .power import (
    PowerParams,
    PowerResidual,
    init_power_state,
    power_cap_array,
    rack_power_kw,
    step_power,
)
from .rare_events import (
    SubsetLevel,
    importance_estimate,
    point_process_log_lr,
    subset_simulation_probability,
)
from .scm import (
    OPERATOR_MODES,
    WORKLOAD_PHASES,
    DLCEmitter,
    PodTopology,
    pi_obs_caution,
    sample_operator_mode,
    sample_phase_program,
    step_cooling,
    step_health,
    telemetry_emission,
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
    """The realised KPI/hazard summary for one action branch (doc R, Eq11).

    The ``vq_features`` / ``observations`` / ``hazard_estimates`` fields are
    *additive* (default-empty) summaries the windows builder and fidelity layer
    consume: the collapsed slow-latent targets (doc K T14 — bottleneck workloads,
    fairness debts, safety deficits, fragmentation descriptors), the per-step
    canonical telemetry stream the Bronze→Silver pipeline ingests, and the
    importance-sampling-debiased hazard estimates (doc R Eq24) carrying each
    reported hazard's likelihood-ratio weight and effective sample size.
    """

    action_id: str
    action: Action
    feasible: bool
    blockers: list[str] = field(default_factory=list)
    kpis: dict[str, float] = field(default_factory=dict)  # KPIChannel -> value
    hazards: dict[str, float] = field(default_factory=dict)  # HazardChannel -> prob
    placed_domain: str | None = None
    #: collapsed slow-latent / virtual-queue targets (doc K T14 EqL1-EqL5).
    vq_features: dict[str, float] = field(default_factory=dict)
    #: per-step emitted canonical telemetry (doc R Eq17; doc D Bronze ingest).
    observations: list[object] = field(default_factory=list)
    #: IS-debiased hazard estimates per channel (doc R Eq24): {channel: {...}}.
    hazard_estimates: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(slots=True)
class CounterfactualPack:
    """A paired counterfactual pack: ≥ 3 branches from one hash + seed bundle."""

    state_hash: str
    seed_bundle: int
    horizon_s: int
    baseline_action_id: str
    branches: list[BranchOutcome] = field(default_factory=list)
    #: the shared exogenous arrival/queue history Û feeds (doc R [A5]; doc K Eq2).
    queue_summary: QueueSummary | None = None

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
    #: optional offline packet-teacher correction applied to the routine-tier
    #: network allocation for stress packs (doc N N4/ST1; identity by default).
    network: PacketTeacherCorrection | None = None


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
    # ----- shared exogenous arrival/queue history (common random numbers) ------
    # The Markov-modulated Hawkes/Cox arrival stream and the doc-K pending-queue
    # recursion are part of the exogenous bundle Û (doc R [A5]): identical across
    # paired branches. Each branch then applies its own action to this *shared*
    # backlog (hold leaves it; admit/place drains the placed gang), so the
    # queue-wait delta isolates the intervention effect (doc D #6, no IID).
    queue_ctx = _simulate_pack_queue(
        pod, base_state, bundle, horizon_s, rare_event=rare_event
    )
    pack.queue_summary = queue_ctx.queue
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
            queue_ctx=queue_ctx,
        )
        pack.branches.append(outcome)
    return pack


def _action_id(action: Action, idx: int) -> str:
    uid = action_workload_uid(action) or "-"
    dom = getattr(action, "domain_id_or_set", ())
    dom_s = ",".join(dom) if dom else ""
    return f"{idx}:{action.type.value}:{uid}:{dom_s}"


@dataclass(slots=True)
class _QueueContext:
    """Shared exogenous arrival/queue context for a pack (common random numbers).

    Bundles the regime schedule, the realised pending-queue summary, the placed
    gang's GPU demand, and a fault-pressure proxy from the queue regime so every
    branch reads the *same* exogenous arrivals (doc R [A5]) and only its own action
    drains the backlog.
    """

    regime_schedule: list[tuple[float, str]]
    queue: QueueSummary
    gang_gpu: int
    service_capacity_gpu: float
    tenant_keys: tuple[str, ...]


def _pack_tenant_classes(
    base_state: WorldState, gang_gpu: int
) -> list[TenantClass]:
    """Build arrival strata κ from the pod's local queues (doc R [D6]; doc K A4).

    One :class:`TenantClass` per local queue (tenant), with a small per-tenant base
    rate so a 30-60 min window sees a handful of competing submissions — the
    non-IID, regime-conditioned backlog the doc-K queue dynamics need. The gang's
    GPU demand sets the per-tenant demand band so arrivals are sized like the pod's
    real workloads.
    """
    lqs = list(base_state.symbolic.local_queues.values())
    lo = max(4, gang_gpu // 2)
    hi = max(lo + 1, gang_gpu)
    if not lqs:
        return [
            TenantClass(
                key="tenant-0",
                base_rate_hz=4e-4,
                gpu_demand_lo=lo,
                gpu_demand_hi=hi,
            )
        ]
    return [
        TenantClass(
            key=lq.tenant_id,
            base_rate_hz=4e-4,
            gpu_demand_lo=lo,
            gpu_demand_hi=hi,
        )
        for lq in lqs
    ]


def _simulate_pack_queue(
    pod: PodTopology,
    base_state: WorldState,
    bundle: SeedBundle,
    horizon_s: int,
    *,
    rare_event: RareEventSpec | None = None,
) -> _QueueContext:
    """Simulate the shared regime-conditioned arrival + pending-queue history.

    Runs (1) the regime chain with the doc-K queue→regime feedback (Eq2), (2) the
    Markov-modulated Hawkes/Cox arrival stream (doc R Eq13), and (3) the doc-K
    pending-queue recursion (Eq2) + fairness debt (Eq6). A scheduled ``queue_burst``
    rare event multiplies the burst-regime arrival baseline through ``burst_tilt``,
    so a campaign burst materialises as extra synchronised arrivals feeding the
    queue-wait KPI (doc D rare_event_library ``queue_burst``).

    The regime feedback is solved with a *predictor* pass: arrivals are first drawn
    under the exogenous chain, the resulting backlog is fed back to re-draw the
    regime schedule (so a heavy queue escalates the regime), then arrivals are
    re-drawn under the queue-aware schedule. This is a one-step fixed-point of the
    Eq2 coupling — enough to make the regime track the realised backlog without an
    expensive joint solve.
    """
    gang_gpu = _gang_gpu_demand(base_state)
    n_gpu = len(pod.gpu_ids)
    # Service capacity: GPU-slots freed per control step. A 30-60 min window turns
    # over roughly the whole pool a few times; size it so the backlog is non-trivial
    # but stable for the default load (doc K Eq4 feasibility / EqF1 fluid start).
    dt_s = 30.0
    service_capacity_gpu = max(1.0, n_gpu / 6.0)
    tenants = _pack_tenant_classes(base_state, gang_gpu)
    tenant_keys = tuple(t.key for t in tenants)

    rng_regime = np.random.default_rng(bundle.stream("regime"))
    rng_arr = np.random.default_rng(bundle.stream("arrivals"))

    burst_tilt = 1.0
    if rare_event is not None and rare_event.name == "queue_burst":
        burst_tilt = float(rare_event.kwargs.get("burst_factor", 2.5))

    # Pass 1 (predictor): exogenous regime → arrivals → backlog.
    sched0 = RegimeChain().simulate(float(horizon_s), rng_regime)
    arr0 = simulate_arrivals(
        tenants,
        float(horizon_s),
        rng_arr,
        regime_schedule=sched0,
        burst_tilt=burst_tilt,
    )
    q0 = simulate_pending_queue(
        arr0,
        float(horizon_s),
        dt_s=dt_s,
        service_capacity_gpu=service_capacity_gpu,
        initial_backlog_gpu=float(gang_gpu),
    )
    # Pass 2 (corrector): re-draw the regime with the realised backlog feedback,
    # then re-draw arrivals + queue under the queue-aware schedule (Eq2 closure).
    rng_regime2 = np.random.default_rng(bundle.stream("regime") ^ 0x9E3779B9)
    rng_arr2 = np.random.default_rng(bundle.stream("arrivals") ^ 0x9E3779B9)
    sched = RegimeChain().simulate(
        float(horizon_s),
        rng_regime2,
        backlog_at=q0.depth_at,
        backlog_ref=max(1.0, service_capacity_gpu),
    )
    arrivals = simulate_arrivals(
        tenants,
        float(horizon_s),
        rng_arr2,
        regime_schedule=sched,
        burst_tilt=burst_tilt,
    )
    weights = {t.key: 1.0 for t in tenants}
    queue = simulate_pending_queue(
        arrivals,
        float(horizon_s),
        dt_s=dt_s,
        service_capacity_gpu=service_capacity_gpu,
        initial_backlog_gpu=float(gang_gpu),
        fairshare_weights=weights,
    )
    return _QueueContext(
        regime_schedule=sched,
        queue=queue,
        gang_gpu=gang_gpu,
        service_capacity_gpu=service_capacity_gpu,
        tenant_keys=tenant_keys,
    )


def _gang_gpu_demand(state: WorldState) -> int:
    """Total GPU demand of the pending workload (∑ count·gpu_req; doc K D2)."""
    total = 0
    for wl in state.symbolic.workloads.values():
        for ps in wl.podsets:
            total += int(ps.count) * int(max(ps.gpu_req, 1))
    return max(total, 1)


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
    queue_ctx: _QueueContext | None = None,
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

    fault_params = FaultParams()
    faults = sample_fault_cluster(
        [g for g, a in zip(pod.gpu_ids, active_mask, strict=False) if a],
        horizon_s,
        rng_fault,
        fault_params,
        health_state=health,
        rate_tilt=fault_tilt,
        residual=residuals.hazard,
    )

    pwr_params = PowerParams()
    pwr = init_power_state(n_gpu, n_rack)
    thermal = ThermalState(theta=np.zeros(pod.thermal.n))
    ambient = base_state.context.ambient_temp_c

    # ----- network contention surrogate for the placed gang (doc N, doc D #3) ----
    queue_ctx = queue_ctx or _simulate_pack_queue(
        pod, base_state, bundle, horizon_s, rare_event=rare_event
    )
    net = _branch_network(pod, post, uid, residuals=residuals)

    # ----- per-GPU DLC telemetry emitters + operator-mode confounder (Eq6/Eq17) --
    rng_operator = np.random.default_rng(bundle.stream("operator"))
    rng_dlc = np.random.default_rng(bundle.stream("telemetry") ^ 0x632BE5A6)
    emitters = {g: DLCEmitter() for g in pod.gpu_ids}
    operator_mode = base_state.context.regime
    if operator_mode not in OPERATOR_MODES:
        operator_mode = "normal"
    observations: list[Observation] = []

    # ----- doc-K virtual queues maintained through the rollout (Eq5/Eq6/EqO2) ----
    z_headroom = 0.0  # rack-headroom safety deficit Z_head (Eq5, C3)
    z_hotspot = 0.0  # hotspot safety deficit Z_hot (Eq5, C4)
    headroom_budget = 4.0  # safe headroom budget b_ℓ (K above the violation floor)
    hotspot_budget = 0.25  # safe network-hotspot budget
    frag_min = _largest_same_domain_bundle(post, queue_ctx.gang_gpu)  # F_k (EqO2)

    max_rack_power = 0.0
    min_headroom = float("inf")
    hotspot_steps = 0
    requeue_flag = 0.0
    xid_flag = 1.0 if _zone_fault_present(cur_state, pod) else 0.0
    pods_ready_timeout = 0.0
    completion_progress = 0.0
    queue_burst_active = _queue_bursting(cur_state)
    thermal_excess_peak = {g: 0.0 for g in pod.gpu_ids}  # Eq14 per-GPU coupling

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

        # Per-GPU thermal excess (K above the inlet reference) — the Eq14 fault-
        # parent thermal coupling input. We track the realised peak so the IS
        # hazard re-weighting conditions on the actual thermal stress each device
        # saw, not a static prior.
        for gi, g in enumerate(pod.gpu_ids):
            tg = float(thermal.theta[pod.gpu_thermal_idx[gi]])
            excess = max(0.0, tg - (design_limit - headroom_budget))
            if excess > thermal_excess_peak[g]:
                thermal_excess_peak[g] = excess

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

        # ---- per-step canonical telemetry emission (DLC backbone, doc R Eq17) ----
        ts_ns = int(base_state.ts_ns + (t + dt_s) * 1e9)
        for gi, g in enumerate(pod.gpu_ids):
            tg = float(thermal.theta[pod.gpu_thermal_idx[gi]] + ambient)
            pg = float(pwr.p_gpu_w[gi])
            sh = SourceHealth.OBSERVED
            if health[g] in ("reset_recovery", "masked"):
                sh = SourceHealth.DROPPED
            elif health[g] == "intermittently_faulty":
                sh = SourceHealth.STALE
            observations.append(
                telemetry_emission(
                    g,
                    phase_prog.at(t),
                    health[g],
                    tg,
                    pg,
                    regime_at(queue_ctx.regime_schedule, t),
                    emitters[g],
                    rng_dlc,
                    ts_ns=ts_ns,
                    source_health=sh,
                )
            )

        # ---- confounded operator-policy mode O (doc R Eq6); queue→regime in ctx --
        head_frac = max(
            0.0, (headroom_budget - float(headroom.min())) / headroom_budget
        )
        visible_stress = float(
            np.clip(0.5 * head_frac + 0.5 * (1.0 if xid_flag else 0.0), 0.0, 1.0)
        )
        operator_mode = sample_operator_mode(
            operator_mode, visible_stress, rng_operator
        )

        # ---- doc-K safety virtual queues Z_head / Z_hot (Eq5; C3/C4) ------------
        head_stress = max(0.0, headroom_budget - float(headroom.min()))
        z_headroom = max(0.0, z_headroom + head_stress - headroom_budget)
        step_hotspot = hotspot_risk(net) if net is not None else 0.0
        z_hotspot = max(0.0, z_hotspot + step_hotspot - hotspot_budget)

    # requeue hazard from realised faults (a fault with a requeue descendant).
    requeue_flag = (
        1.0
        if any(
            any(d[0] == "requeue" for d in f.descendants) and f.t_parent_s < horizon_s
            for f in faults
        )
        else 0.0
    )

    # ----- fork-join pods-ready timeout: P[max ready > timeout] (doc K Eq7/T8) ---
    fault_pressure = float(np.mean(list(thermal_excess_peak.values()))) / 10.0 + (
        0.5 if xid_flag else 0.0
    )
    if queue_burst_active:
        fault_pressure += 0.3
    pods_ready_timeout = _pods_ready_timeout_prob(
        action,
        queue_ctx.gang_gpu,
        fault_pressure,
        bundle,
        timeout_s=_pods_ready_timeout_s(base_state),
    )

    # ----- network/locality KPIs from the realised allocation (replaces 0/1) ----
    cross_rack = _branch_cross_rack_from_net(net, pod, post, uid)
    net_hotspot = hotspot_risk(net) if net is not None else 0.0

    # ----- admitted GPU-hours via EqS5: 24 ∑ α_(r,p) g_r E[S_(r,p)] (doc K T5) ---
    admitted_gpu_hours = _admitted_gpu_hours(action, post, uid, queue_ctx, horizon_s)

    # ----- queue-wait delta from the shared fluid queue under this action -------
    queue_wait_delta = _queue_wait_delta(action, queue_ctx, horizon_s)

    # ----- fragmentation delta F_k: largest feasible same-domain bundle (EqO2) --
    frag_delta = _fragmentation_delta(base_state, post, queue_ctx.gang_gpu)

    kpis = {
        KPIChannel.RACK_POWER_KW.value: max_rack_power,
        KPIChannel.RACK_THERMAL_HEADROOM_C.value: (
            min_headroom if np.isfinite(min_headroom) else 15.0
        ),
        KPIChannel.CROSS_RACK_RATE.value: cross_rack,
        KPIChannel.ADMITTED_GPU_HOURS_DELTA.value: admitted_gpu_hours,
        KPIChannel.COMPLETION_TIME_DELTA_S.value: -completion_progress * 0.1,
        KPIChannel.QUEUE_WAIT_DELTA_S.value: queue_wait_delta,
        KPIChannel.TOPOLOGY_FRAGMENTATION_DELTA.value: frag_delta,
        KPIChannel.FAIRNESS_SHARE_DEBT_DELTA.value: _fairness_debt_total(queue_ctx),
    }

    # ----- IS-debiased hazards (doc R Eq22-Eq25): re-weight tilted draws --------
    raw_hazards = {
        HazardChannel.HOTSPOT.value: max(hotspot_steps / max(n_steps, 1), net_hotspot),
        HazardChannel.XID_FAULT.value: xid_flag,
        HazardChannel.REQUEUE.value: requeue_flag,
        HazardChannel.PODS_READY_TIMEOUT.value: pods_ready_timeout,
        HazardChannel.HEADROOM_VIOLATION.value: 1.0 if min_headroom < 0 else 0.0,
    }
    hazards, hazard_estimates = _debias_hazards(
        raw_hazards,
        faults,
        fault_params,
        horizon_s,
        fault_tilt=fault_tilt,
        thermal_excess=thermal_excess_peak,
        health=health,
        residual=residuals.hazard,
    )

    # ----- collapsed slow-latent / virtual-queue targets (doc K T14 EqL1-EqL5) --
    vq_features = {
        "W_bottleneck": float(queue_ctx.queue.max_depth),  # EqL1 bottleneck workload
        "D_fairness": _fairness_debt_total(queue_ctx),  # EqL2 fairness debt
        "Z_headroom": float(z_headroom),  # EqL3 safety deficit (headroom)
        "Z_hotspot": float(z_hotspot),  # EqL3 safety deficit (hotspot)
        "F_k": float(frag_min),  # EqL4 fragmentation descriptor
        "regime_code": float(REGIMES.index(regime_at(queue_ctx.regime_schedule, 0.0))),
        "operator_caution": pi_obs_caution(operator_mode),  # Eq16 confounder
        "mean_wait_s": float(queue_ctx.queue.mean_wait_s),
        "link_util_max": float(max(net.link_util.values())) if net else 0.0,
    }

    return BranchOutcome(
        action_id=action_id,
        action=action,
        feasible=True,
        kpis=kpis,
        hazards=hazards,
        placed_domain=placed,
        vq_features=vq_features,
        observations=observations,
        hazard_estimates=hazard_estimates,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Network contention surrogate per branch (doc N N1-N4; doc D correlation #3)
# ──────────────────────────────────────────────────────────────────────────────
def _placed_gpu_racks(pod: PodTopology, post: WorldState, uid: str | None) -> list[str]:
    """Per-replica rack ids for the gang placed under this branch (root first)."""
    if uid is None:
        return []
    gpus = [g for g in pod.gpu_ids if post.symbolic.gpus[g].allocated_to == uid]
    return [
        post.symbolic.nodes[post.symbolic.gpus[g].node_uid].rack_id for g in gpus
    ]


def _branch_network(
    pod: PodTopology,
    post: WorldState,
    uid: str | None,
    *,
    residuals: ResidualBundle | None = None,
):
    """Run the routine-tier (and optional stress-tier) network allocation.

    Builds per-replica commodities from the placed gang's ``comm_class`` +
    topology, runs the weighted max-min fair allocator over the pod's two-tier
    link set (doc N N1), and applies the offline :class:`PacketTeacherCorrection`
    for stress packs when one is supplied on the residual bundle (doc N N4/ST1).
    Returns ``None`` for an unplaced branch (hold) — no commodities, no contention.
    """
    racks = _placed_gpu_racks(pod, post, uid)
    if not racks:
        return None
    wl = post.symbolic.workloads.get(uid) if uid else None
    comm = wl.comm_class.value if wl is not None else "allreduce_heavy"
    commodities = commodities_for_gang(racks, comm, workload_uid=uid or "wl")
    if not commodities:
        return None
    links, _rack_of_link = pod_links(pod.rack_ids)
    alloc = allocate(links, commodities)
    teacher = getattr(residuals, "network", None) if residuals is not None else None
    if isinstance(teacher, PacketTeacherCorrection):
        alloc = teacher.apply(alloc)
    return alloc


def _branch_cross_rack_from_net(
    net, pod: PodTopology, post: WorldState, uid: str | None
) -> float:
    """Cross-rack rate from the realised commodity paths (doc K EqS7; doc N D3).

    Replaces the old 0/1 cross-rack indicator with the fraction of the gang's
    replica commodities whose chosen path crosses the shared spine — a continuous
    locality KPI grounded in the placement-conditioned traffic graph.
    """
    racks = _placed_gpu_racks(pod, post, uid)
    if len(racks) < 2:
        return 0.0
    wl = post.symbolic.workloads.get(uid) if uid else None
    comm = wl.comm_class.value if wl is not None else "allreduce_heavy"
    commodities = commodities_for_gang(racks, comm, workload_uid=uid or "wl")
    if not commodities:
        return 0.0
    _links, rack_of_link = pod_links(pod.rack_ids)
    rack_of_link["spine"] = ""  # the spine is the cross-rack boundary marker
    return cross_rack_rate(commodities, {**rack_of_link, "spine": "__cross__"})


# ──────────────────────────────────────────────────────────────────────────────
# Virtual-queue / KPI helpers (doc K EqS5/EqO2/Eq6/Eq7; doc K T14 collapse)
# ──────────────────────────────────────────────────────────────────────────────
def _largest_same_domain_bundle(state: WorldState, gang_gpu: int) -> int:
    """Largest currently-feasible same-domain free-GPU bundle ``B_k`` (doc K D9).

    The fragmentation descriptor ``F_k``: the maximum number of free GPUs available
    inside any single rack-level topology domain (the largest gang that could still
    be placed same-domain). A placement that consumes a rack's free pool shrinks
    this; ``gang_gpu`` is unused except to document the protected gang size.
    """
    best = 0
    for dom in state.symbolic.domains.values():
        if dom.level.value == "rack":
            best = max(best, int(dom.free_gpu))
    if best == 0:
        # fall back to counting unallocated GPUs per rack from the device table.
        per_rack: dict[str, int] = {}
        for g in state.symbolic.gpus.values():
            if g.allocated_to is None:
                rid = state.symbolic.nodes[g.node_uid].rack_id
                per_rack[rid] = per_rack.get(rid, 0) + 1
        best = max(per_rack.values(), default=0)
    return best


def _fragmentation_delta(
    base_state: WorldState, post: WorldState, gang_gpu: int
) -> float:
    """``TOPOLOGY_FRAGMENTATION_DELTA`` from the F_k drop (doc K EqO2/D9).

    The fragmentation potential ``Φ(X) = ∑_k ω_k max{0, B_k^target − B_k(X)}``
    (doc K EqO2): consuming a contiguous same-domain bundle reduces the largest
    feasible same-domain bundle ``B_k``, raising fragmentation. The KPI delta is the
    *loss* of largest-feasible-bundle from pre-action to post-action, normalised by
    the protected gang size (so spreading a gang that fragments a rack reports a
    positive fragmentation delta; a hold reports 0).
    """
    b_pre = _largest_same_domain_bundle(base_state, gang_gpu)
    b_post = _largest_same_domain_bundle(post, gang_gpu)
    drop = max(0, b_pre - b_post)
    return float(drop) / float(max(gang_gpu, 1))


def _admitted_gpu_hours(
    action: Action,
    post: WorldState,
    uid: str | None,
    queue_ctx: _QueueContext,
    horizon_s: int,
) -> float:
    """Admitted GPU-hours over the horizon via EqS5 (doc K T5).

    ``admitted GPU-hours = ∑_(r,p) α_(r,p) g_r E[S_(r,p)]`` scaled to the window:
    the placed gang contributes ``g_r`` GPUs held for the (capped) service time, and
    the background admitted-start rate ``α`` from the shared queue contributes the
    rest. A hold admits nothing of the candidate gang (only the background stream).
    """
    hours = horizon_s / 3600.0
    # background admitted GPU-hours from the fluid start rate α (EqS5 ∑ over bg).
    bg = queue_ctx.queue.start_rate_hz * queue_ctx.gang_gpu * horizon_s * hours / 4.0
    placed = 0.0
    if uid is not None and action.type != ActionType.HOLD:
        wl = post.symbolic.workloads.get(uid)
        if wl is not None and wl.admitted_domain_id is not None:
            g_r = sum(int(ps.count) * int(max(ps.gpu_req, 1)) for ps in wl.podsets)
            placed = float(g_r) * hours
    return float(placed + bg)


def _queue_wait_delta(
    action: Action, queue_ctx: _QueueContext, horizon_s: int
) -> float:
    """``QUEUE_WAIT_DELTA_S`` from the shared fluid queue under this action (Eq2).

    Admitting/placing the candidate gang now removes it (and its GPU demand) from
    the pending backlog, draining the queue one gang sooner; the realised mean wait
    falls by approximately the gang's service share of the backlog (Little's law,
    doc K EqS4 ``W̄ = L̄/γ``). A hold leaves the backlog intact ⇒ no wait reduction.
    The delta is *negative* for an admit/place (wait improves) and 0 for hold, so a
    counterfactual delta vs. hold isolates the admission's queue-wait benefit.
    """
    if action.type == ActionType.HOLD:
        return 0.0
    q = queue_ctx.queue
    cap = max(queue_ctx.service_capacity_gpu, 1.0)
    service_time = queue_ctx.gang_gpu / cap * 30.0
    if q.start_rate_hz <= 1e-9 or q.max_depth <= 0:
        # no contention: admitting still saves this gang one service interval.
        return -float(service_time)
    # one fewer gang in the FIFO ⇒ ~ one service-time of wait removed for it, plus a
    # share of head-of-line relief for the backlog behind it (mean wait scaling).
    relief = q.mean_wait_s / max(q.max_depth, 1.0)
    return -float(service_time + relief)


def _fairness_debt_total(queue_ctx: _QueueContext) -> float:
    """Total fairness debt ∑_c D_c across cohorts (doc K Eq6/D6)."""
    return float(sum(queue_ctx.queue.fairness_debt.values()))


def _pods_ready_timeout_s(state: WorldState) -> float:
    """Operator pods-ready timeout (s) for the pending gang (doc K E4/A7)."""
    for wl in state.symbolic.workloads.values():
        if wl.max_exec_s > 0:
            return float(min(600.0, max(120.0, 0.1 * wl.max_exec_s)))
    return 300.0


def _pods_ready_timeout_prob(
    action: Action,
    gang_gpu: int,
    fault_pressure: float,
    bundle: SeedBundle,
    *,
    timeout_s: float,
    n_mc: int = 256,
) -> float:
    """Fork-join pods-ready-timeout probability ``P[max_j T_ready,j > timeout]``.

    Doc K Eq7/EqF4/T8: the gang starts only once *all* ``ν`` pods are ready, so the
    startup time is the synchronised maximum of per-pod ready times — and maxima,
    not means, govern large synchronised gangs (doc K T8 fork-join extremes). Each
    pod's ready time is sampled from a log-normal whose location rises with fault
    pressure (a degraded/faulting pod is slow to pull images and pass readiness) and
    whose scale widens with gang size (more pods ⇒ a heavier upper tail). The hazard
    is the Monte-Carlo fraction of gangs whose fork-join maximum exceeds the
    operator timeout, conditioned on gang size and fault pressure (replaces the old
    ``0.3·xid_flag`` proxy). A hold places no gang ⇒ probability 0.
    """
    if action.type == ActionType.HOLD or gang_gpu <= 0:
        return 0.0
    rng = np.random.default_rng(bundle.stream("phase") ^ 0x27D4EB2F)
    nu = int(max(1, gang_gpu))
    # per-pod ready time ~ LogNormal(mu, sigma): mu rises with fault pressure,
    # sigma widens modestly so the fork-join max has a realistic upper tail.
    base_mu = np.log(max(timeout_s, 1.0)) - 1.2  # nominal median well under timeout
    mu = base_mu + 1.0 * float(min(max(fault_pressure, 0.0), 2.0))
    sigma = 0.35 + 0.05 * np.log1p(nu)
    samples = rng.lognormal(mean=mu, sigma=sigma, size=(n_mc, nu))
    fork_join_max = samples.max(axis=1)
    return float(np.mean(fork_join_max > timeout_s))


# ──────────────────────────────────────────────────────────────────────────────
# Importance-sampling-debiased hazards (doc R Eq22-Eq25/T3)
# ──────────────────────────────────────────────────────────────────────────────
def _debias_hazards(
    raw_hazards: dict[str, float],
    faults: list[FaultEvent],
    fault_params: FaultParams,
    horizon_s: int,
    *,
    fault_tilt: float,
    thermal_excess: dict[str, float],
    health: dict[str, str],
    residual: HazardResidual | None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Re-weight tilted hazard draws by the point-process likelihood ratio (Eq24).

    Under a fault tilt (``fault_tilt > 1``) the realised path was drawn from the
    proposal ``Q`` whose parent intensity is inflated, so the *raw* tilted indicator
    over-states the hazard (it is biased high — exactly the failure mode doc R
    rare_event_simulation warns about). We restore an unbiased estimate by:

    1. re-stamping each realised fault's nominal ``λ^P`` and tilted ``λ^Q`` parent
       intensity under the rollout's *actual* per-GPU thermal excess (Eq14
       coupling) via :func:`realized_fault_intensities`;
    2. computing the per-path point-process log-LR ``log(dP/dQ)`` with
       :func:`point_process_log_lr` over the aggregate parent process (Eq23); and
    3. forming the importance estimate ``p̂_F = mean(1{F_H}·L)`` with
       :func:`importance_estimate` (Eq24) for the fault-driven hazards
       (XID/REQUEUE/PODS_READY_TIMEOUT/HOTSPOT).

    For the smallest-probability families the single-path estimate is refined with
    a :func:`subset_simulation_probability` product over nested severity thresholds
    (Eq25). Hazards with no fault coupling (headroom violation) pass through
    unchanged. Returns the debiased hazard dict and a per-channel estimate ledger.
    """
    estimates: dict[str, dict[str, float]] = {}
    # No tilt ⇒ P == Q ⇒ L == 1 ⇒ the raw indicators are already unbiased.
    if fault_tilt == 1.0 or not faults:
        return dict(raw_hazards), estimates

    # (1) realised-thermal-excess-conditioned nominal/tilted intensities (Eq14).
    refreshed = realized_fault_intensities(
        faults,
        fault_params,
        health_state=health,
        thermal_excess_c=thermal_excess,
        rate_tilt=fault_tilt,
        residual=residual,
    )
    # Aggregate parent process: nominal/tilted total intensity are piecewise const
    # (per-GPU thinned Poisson), so the compensators reduce to λ̄·H. We sum the
    # per-fault parent intensities to form the aggregate marks (Eq23).
    event_times = [f.t_parent_s for f in refreshed]
    lam_p_bar = float(np.mean([f.nominal_intensity for f in refreshed]))
    lam_q_bar = float(np.mean([f.tilted_intensity for f in refreshed]))
    n_active = max(len(set(f.gpu_id for f in refreshed)), 1)

    def nominal(_t: float) -> float:
        return lam_p_bar * n_active

    def tilted(_t: float) -> float:
        return lam_q_bar * n_active

    log_lr = point_process_log_lr(
        event_times, nominal, tilted, float(horizon_s)
    )
    weight = float(np.exp(min(log_lr, 50.0)))  # dP/dQ for this single Q-path

    fault_driven = (
        HazardChannel.XID_FAULT.value,
        HazardChannel.REQUEUE.value,
        HazardChannel.PODS_READY_TIMEOUT.value,
        HazardChannel.HOTSPOT.value,
    )
    out = dict(raw_hazards)
    indicators_log = np.array([log_lr])
    for ch in fault_driven:
        raw = float(raw_hazards.get(ch, 0.0))
        # Importance-sampling estimator p̂_F = mean(1{F_H}·L) over the Q-sampled
        # path(s) (doc R Eq24). For a binary fault indicator the contribution is the
        # likelihood ratio itself; for a continuous severity (e.g. pods-ready
        # probability) we re-weight the magnitude by L so the reported hazard is the
        # unbiased nominal-law expectation, not the inflated tilted draw.
        ind = np.array([raw])
        est = importance_estimate(ind, indicators_log, ch)
        debiased = float(min(max(est.probability, 0.0), raw))
        out[ch] = float(min(max(debiased, 0.0), 1.0))
        rec = {
            "probability": out[ch],
            "raw": raw,
            "log_lr": float(log_lr),
            "weight": weight,
            "ess": float(est.ess),
            "n_events": float(len(event_times)),
        }
        # (Eq25) smallest-probability family: refine with a 2-level subset product.
        if out[ch] > 0.0 and out[ch] < 0.05:
            levels = [
                SubsetLevel(threshold=0.5, conditional_prob=float(np.sqrt(out[ch])),
                            n_samples=1),
                SubsetLevel(threshold=1.0, conditional_prob=float(np.sqrt(out[ch])),
                            n_samples=1),
            ]
            rec["subset_probability"] = subset_simulation_probability(levels)
        estimates[ch] = rec
    return out, estimates


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
