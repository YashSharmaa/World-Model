# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""DE-WIRE-ARRIVALS-NETWORK / DE-LABELS-VIRTUAL-QUEUES / DE-TELEMETRY-WINDOWS-SCM /
DE-IS-DEBIASED-HAZARDS — the arrival/queue/network/telemetry/IS-hazard wiring.

Verifies that pack generation now drives the doc-K/doc-R machinery end-to-end:

* the Markov-modulated Hawkes/Cox arrival stream is finite (sub-critical Hawkes,
  no explosion) and regime-conditioned, and the pending-queue recursion (doc K
  Eq2/Eq6) produces a non-IID backlog + fairness debt;
* the placement-conditioned network surrogate (doc N N1; doc D #3) produces a
  continuous cross-rack rate + hotspot hazard from the weighted-max-min-fair
  allocation (replacing the 0/1 indicator), and the packet-teacher correction
  hooks in for stress packs;
* the rollout emits the doc-K virtual queues (Z_headroom/Z_hotspot/D_c/F_k) and
  the doc-K KPIs (queue-wait, fragmentation EqO2, admitted GPU-hours EqS5) as
  window features/labels, plus a fork-join pods-ready-timeout (Eq7/EqF4/T8);
* the DLC telemetry backbone (doc R Eq17) emits coherent cross-channel telemetry
  routed through Bronze→Silver into the window's observational_context /
  realized_horizon, with the three Gold URIs populated and no leakage; and
* hazards sampled under a fault tilt are likelihood-ratio-corrected (doc R
  Eq22-Eq25/T3) so they are unbiased (lower) rather than biased high.
"""

from __future__ import annotations

import tempfile

import numpy as np

import gpu_center_world_model.data_engine.synthetic as S  # noqa: N812
from gpu_center_world_model.data_engine.synthetic.arrivals import (
    RegimeChain,
    TenantClass,
    simulate_arrivals,
    simulate_pending_queue,
)
from gpu_center_world_model.data_engine.synthetic.counterfactual import (
    _debias_hazards,
    build_pack,
)
from gpu_center_world_model.data_engine.synthetic.faults import (
    FaultParams,
    sample_fault_cluster,
)
from gpu_center_world_model.data_engine.synthetic.network import (
    Link,
    PacketTeacherCorrection,
    allocate,
    commodities_for_gang,
    hotspot_risk,
    pod_links,
)
from gpu_center_world_model.data_engine.synthetic.scm import (
    OPERATOR_MODES,
    DLCEmitter,
    PodSpec,
    sample_operator_mode,
    telemetry_emission,
)
from gpu_center_world_model.data_engine.windows.builder import (
    build_window,
    persist_window_artifacts,
    realized_horizon_frames,
    silver_from_pack,
)
from gpu_center_world_model.schema.kpis import HazardChannel, KPIChannel
from gpu_center_world_model.schema.observations import SourceHealth
from gpu_center_world_model.simulator.exact_kernel import ExactSymbolicKernel


# ── helpers ──────────────────────────────────────────────────────────────────
def _feasible_placement_pack(tilt: float = 1.0, seed: int = 7):
    """A pack with a *feasible* placement so placement-conditioned features fire.

    The default scenarios are hold-only-feasible for a 16-GPU gang; here we use the
    kernel's own ``safe_candidates`` to obtain a feasible admit_and_place for a
    small gang so cross-rack/network/IS-hazard code paths are exercised.
    """
    pod, state, uid = S.generate_pod_state(
        PodSpec(n_racks=2, nodes_per_rack=2, gpus_per_node=4), seed=seed, gpu_demand=4
    )
    kernel = ExactSymbolicKernel()
    safe = kernel.safe_candidates(state, uid)
    placements = [a for a in safe if a.type.value != "hold"][:2]
    from gpu_center_world_model.schema.actions import Hold

    actions = [Hold(workload_uid=uid), *placements]
    while len(actions) < 3:  # guarantee the ≥3-branch harness contract
        actions.append(Hold(workload_uid=uid))
    pack = build_pack(pod, state, actions, seed=seed, horizon_s=900, fault_tilt=tilt)
    return pod, state, uid, pack


# ── DE-WIRE-ARRIVALS-NETWORK: arrivals + pending queue ──────────────────────
def test_hawkes_arrivals_are_finite_and_regime_conditioned():
    # Sub-critical Hawkes ⇒ the cluster process terminates (no explosion); a burst
    # regime produces materially more (and campaign-tagged) arrivals than a drain.
    tenants = [TenantClass(key=f"t{i}", base_rate_hz=4e-4) for i in range(3)]
    rng = np.random.default_rng(0)
    burst = simulate_arrivals(
        tenants, 1800.0, rng, regime_schedule=[(0.0, "burst")], burst_tilt=2.5
    )
    drain = simulate_arrivals(
        tenants, 1800.0, np.random.default_rng(0), regime_schedule=[(0.0, "drain")]
    )
    assert len(burst) > len(drain)
    assert any(a.is_campaign for a in burst)  # self-excited campaign launches


def test_pending_queue_recursion_and_fairness_debt():
    # Eq2 backlog + Eq6 fairness debt: an unbalanced arrival mix accrues debt.
    arrivals = simulate_arrivals(
        [
            TenantClass(key="t0", base_rate_hz=4e-3),
            TenantClass(key="t1", base_rate_hz=4e-3),
        ],
        1800.0,
        np.random.default_rng(1),
        regime_schedule=[(0.0, "ramping")],
    )
    q = simulate_pending_queue(
        arrivals,
        1800.0,
        dt_s=30.0,
        service_capacity_gpu=4.0,  # tight capacity ⇒ a real backlog forms
        initial_backlog_gpu=16.0,
        fairshare_weights={"t0": 1.0, "t1": 1.0},
    )
    assert q.max_depth >= 1.0  # a non-trivial pending backlog (doc K D4)
    assert q.arrival_rate_hz > 0.0 and q.mean_wait_s >= 0.0  # Little's law grounded
    assert len(q.depth_series) == 60  # one entry per control step


def test_regime_chain_backlog_feedback_escalates(  # Eq2 queue→regime closure
):
    # A heavy persistent backlog tilts the regime away from 'drain' toward busy
    # regimes relative to the pure exogenous chain (doc R Eq2 queue_summary).
    chain = RegimeChain()
    heavy = chain.simulate(
        7200.0, np.random.default_rng(3), backlog_at=lambda t: 200.0, backlog_ref=4.0
    )
    none = chain.simulate(7200.0, np.random.default_rng(3))
    drain_heavy = sum(1 for _, r in heavy if r == "drain")
    drain_none = sum(1 for _, r in none if r == "drain")
    # backlog feedback should not *increase* drain occupancy (it sheds load less).
    assert drain_heavy <= drain_none + 1


def test_pack_carries_regime_conditioned_queue_summary():
    _pod, _s, _u, pack = _feasible_placement_pack()
    assert pack.queue_summary is not None
    qs = pack.queue_summary  # the public QueueSummary (doc K Eq2/Eq6)
    assert qs.times_s and len(qs.depth_series) == len(qs.times_s)
    # the queue summary is shared (CRN) across branches: every branch reads it.
    for b in pack.branches:
        if b.feasible:
            assert "mean_wait_s" in b.vq_features


# ── DE-WIRE-ARRIVALS-NETWORK: network contention surrogate ──────────────────
def test_spread_gang_loads_spine_and_raises_hotspot():
    # A gang spread across two racks crosses the shared (scarce) spine ⇒ a higher
    # hotspot risk than a rack-local gang that only loads its fat ToR (doc N N1;
    # doc D #3). A light comm class keeps the ToR-only case below saturation so the
    # spine-crossing case is the distinguishing hotspot.
    links, _ = pod_links(["rack-0", "rack-1"])
    local = commodities_for_gang(["rack-0"] * 4, "light", workload_uid="w")
    spread = commodities_for_gang(
        ["rack-0", "rack-0", "rack-1", "rack-1"], "allreduce_heavy", workload_uid="w"
    )
    a_local = allocate(links, local)
    a_spread = allocate(links, spread)
    assert hotspot_risk(a_spread) > hotspot_risk(a_local)
    assert "spine" in a_spread.bottleneck_links  # the spread gang jams the spine


def test_hotspot_risk_is_continuous_not_binary():
    links = [Link("tor:rack-0", capacity=400.0), Link("spine", capacity=100.0)]
    comms = commodities_for_gang(
        ["rack-0", "rack-1", "rack-1"], "allreduce_heavy", workload_uid="w"
    )
    # add the second rack's ToR link so the allocation is well-defined
    links.append(Link("tor:rack-1", capacity=400.0))
    risk = hotspot_risk(allocate(links, comms))
    assert 0.0 < risk <= 1.0  # a continuous hazard, not a 0/1 indicator


def test_packet_teacher_correction_applies_for_stress():
    links, _ = pod_links(["rack-0", "rack-1"])
    comms = commodities_for_gang(
        ["rack-0", "rack-1"], "allreduce_heavy", workload_uid="w"
    )
    base = allocate(links, comms)
    cid = comms[0].commodity_id
    teacher = PacketTeacherCorrection(
        rate_factor={cid: 0.5}, delay_add_s={cid: 0.01}
    )
    corrected = teacher.apply(base)
    assert corrected.rate[cid] < base.rate[cid]
    assert corrected.transfer_time_s[cid] > base.transfer_time_s[cid]


# ── DE-LABELS-VIRTUAL-QUEUES: KPIs + virtual queues + fork-join ─────────────
def test_branch_emits_virtual_queue_and_kpi_labels():
    _pod, _s, _u, pack = _feasible_placement_pack()
    placed = next(
        (b for b in pack.branches if b.feasible and b.action.type.value != "hold"),
        None,
    )
    assert placed is not None
    # collapsed slow-latent targets (doc K T14 EqL1-EqL5)
    for key in ("W_bottleneck", "D_fairness", "Z_headroom", "Z_hotspot", "F_k"):
        assert key in placed.vq_features
    # doc-K KPIs present
    for ch in (
        KPIChannel.QUEUE_WAIT_DELTA_S.value,
        KPIChannel.TOPOLOGY_FRAGMENTATION_DELTA.value,
        KPIChannel.ADMITTED_GPU_HOURS_DELTA.value,
    ):
        assert ch in placed.kpis


def test_admit_reduces_queue_wait_vs_hold():
    # QUEUE_WAIT_DELTA_S: admitting the gang drains the backlog sooner ⇒ a negative
    # (improving) delta; a hold leaves the backlog intact ⇒ 0 (doc K Eq2/EqS4).
    _pod, _s, _u, pack = _feasible_placement_pack()
    hold = next(b for b in pack.branches if b.action.type.value == "hold")
    placed = next(
        b for b in pack.branches if b.feasible and b.action.type.value != "hold"
    )
    assert hold.kpis[KPIChannel.QUEUE_WAIT_DELTA_S.value] == 0.0
    assert placed.kpis[KPIChannel.QUEUE_WAIT_DELTA_S.value] < 0.0


def test_admitted_gpu_hours_uses_eqs5_not_zero():
    _pod, _s, _u, pack = _feasible_placement_pack()
    placed = next(
        b for b in pack.branches if b.feasible and b.action.type.value != "hold"
    )
    # placing a 4-GPU gang for a 900 s window contributes real GPU-hours (EqS5).
    assert placed.kpis[KPIChannel.ADMITTED_GPU_HOURS_DELTA.value] > 0.0


def test_fork_join_pods_ready_scales_with_fault_pressure():
    # P[max ready > timeout] rises with fault pressure and gang size (doc K T8);
    # a hold places no gang ⇒ 0. (replaces the old 0.3*xid_flag proxy)
    from gpu_center_world_model.data_engine.synthetic.counterfactual import (
        _pods_ready_timeout_prob,
    )
    from gpu_center_world_model.schema.actions import AdmitAndPlace, Hold
    from gpu_center_world_model.utils.seeding import SeedBundle

    bundle = SeedBundle(7)
    place = AdmitAndPlace(
        workload_uid="w",
        cluster_queue_id="cq",
        resource_flavor_id="f",
        domain_id_or_set=("d",),
    )
    p_low = _pods_ready_timeout_prob(place, 8, 0.0, bundle, timeout_s=300.0)
    p_high = _pods_ready_timeout_prob(place, 8, 1.5, bundle, timeout_s=300.0)
    p_hold = _pods_ready_timeout_prob(
        Hold(workload_uid="w"), 8, 1.5, bundle, timeout_s=300.0
    )
    assert p_hold == 0.0
    assert p_high > p_low
    assert 0.0 <= p_high <= 1.0


# ── DE-TELEMETRY-WINDOWS-SCM: DLC telemetry + Bronze→Silver + URIs ──────────
def test_dlc_emission_is_cross_channel_coherent():
    # The DLC backbone shares one latent factor across channels, so util/power/temp
    # offsets are correlated across steps — never independent (doc R Eq17).
    emitter = DLCEmitter()
    rng = np.random.default_rng(0)
    util, power = [], []
    for _ in range(400):
        o = telemetry_emission(
            "g0", "steady_compute", "healthy", 30.0, 500.0, "burst", emitter, rng,
            ts_ns=0,
        )
        util.append(o.channels["DCGM_FI_DEV_GPU_UTIL"].value)
        power.append(o.channels["DCGM_FI_DEV_POWER_USAGE"].value)
    corr = np.corrcoef(util, power)[0, 1]
    # the shared factor induces a real cross-channel correlation (not ~0).
    assert abs(corr) > 0.2


def test_operator_mode_confounder_responds_to_stress():
    # high visible stress tilts the operator toward conservative/incident (Eq6).
    rng = np.random.default_rng(0)
    modes = [sample_operator_mode("normal", 1.0, rng) for _ in range(200)]
    assert any(m in ("conservative", "incident") for m in modes)
    assert all(m in OPERATOR_MODES for m in modes)


def test_rollout_emits_telemetry_through_bronze_to_silver():
    _pod, pre, _u, pack = _feasible_placement_pack()
    # every feasible branch emits per-step canonical observations
    feas = next(b for b in pack.branches if b.feasible)
    assert feas.observations
    silver = silver_from_pack(pack, pre_action_state=pre)
    assert silver  # three-lane Silver tables built from the emitted telemetry
    fast = silver.get(next(iter(silver.keys())))
    assert fast is not None


def test_window_sets_three_uris_and_runs_leakage_guard():
    _pod, pre, _u, pack = _feasible_placement_pack()
    with tempfile.TemporaryDirectory() as d:
        ctx_uri, hor_uri, cf_uri = persist_window_artifacts(d, pack, pre)
        window, labels = build_window(
            pre,
            pack,
            attach_pack_telemetry=True,  # forces assert_no_leakage to run
            observational_context_uri=ctx_uri,
            realized_horizon_uri=hor_uri,
            counterfactual_pack_uri=cf_uri,
            generator_commit="t",
        )
    assert window.observational_context_uri == ctx_uri
    assert window.realized_horizon_uri == hor_uri
    assert window.counterfactual_pack_uri == cf_uri
    # labels carry the collapsed-coordinate vq:* channels (doc K T14)
    chans = set(labels[0].kpi_quantiles)
    assert any(c.startswith("vq:") for c in chans)


def test_incident_telemetry_round_trips_as_degraded():
    # A faulting GPU's telemetry is emitted STALE; through Bronze→Silver it must
    # surface as degraded (missingness as state), not a clean observation (doc D #5).
    from gpu_center_world_model.data_engine.normalize.canonical import (
        observations_to_metric_samples,
    )

    emitter = DLCEmitter()
    # DELAYED telemetry is *available* (a late-but-present sample) yet not OBSERVED;
    # under degrade_nonobserved it must surface as degraded in Bronze.
    obs_delayed = telemetry_emission(
        "g0", "steady_compute", "degraded", 40.0, 400.0, "burst",
        emitter, np.random.default_rng(0), ts_ns=0,
        source_health=SourceHealth.DELAYED,
    )
    rows = observations_to_metric_samples([obs_delayed], degrade_nonobserved=True)
    assert any(r.quality_mask == 0 for r in rows)  # incident staleness as state
    # the default (byte-stable) path keeps a DELAYED-but-available channel good.
    rows_default = observations_to_metric_samples([obs_delayed])
    assert all(r.quality_mask == 1 for r in rows_default)
    # a hard DROP is masked on both paths (always missingness-as-state).
    obs_drop = telemetry_emission(
        "g0", "steady_compute", "masked", 40.0, 400.0, "burst",
        emitter, np.random.default_rng(0), ts_ns=0,
        source_health=SourceHealth.DROPPED,
    )
    assert all(r.quality_mask == 0 for r in observations_to_metric_samples([obs_drop]))


def test_realized_horizon_is_after_t0_and_context_before():
    _pod, pre, _u, pack = _feasible_placement_pack()
    silver = silver_from_pack(pack, pre_action_state=pre)
    t0 = pre.ts_ns
    horizon = realized_horizon_frames(silver, t0)
    assert horizon  # the realised future exists
    assert all(fr.ts_ns >= t0 for fr in horizon)
    # the pre-action snapshot is strictly before t0 (the leak-free context).
    from gpu_center_world_model.data_engine.windows.builder import assemble_features

    feats = assemble_features(silver, t0)
    assert feats.all_observations()
    assert all(o.ts_event_ns < t0 for o in feats.all_observations())


# ── DE-IS-DEBIASED-HAZARDS: likelihood-ratio correction ─────────────────────
def test_tilted_hazards_are_debiased_below_raw():
    gpus = [f"g{i}" for i in range(8)]
    faults = sample_fault_cluster(
        gpus, 3600.0, np.random.default_rng(3), FaultParams(), rate_tilt=50.0
    )
    assert faults  # the inflated proposal produced parent faults
    raw = {
        HazardChannel.XID_FAULT.value: 1.0,
        HazardChannel.REQUEUE.value: 1.0,
        HazardChannel.PODS_READY_TIMEOUT.value: 0.5,
        HazardChannel.HOTSPOT.value: 0.2,
        HazardChannel.HEADROOM_VIOLATION.value: 0.0,
    }
    out, est = _debias_hazards(
        raw,
        faults,
        FaultParams(),
        3600,
        fault_tilt=50.0,
        thermal_excess={g: 2.0 for g in gpus},
        health={g: "healthy" for g in gpus},
        residual=None,
    )
    # fault-driven hazards are corrected DOWN (unbiased, not biased high).
    assert out[HazardChannel.XID_FAULT.value] < raw[HazardChannel.XID_FAULT.value]
    assert out[HazardChannel.REQUEUE.value] < raw[HazardChannel.REQUEUE.value]
    # the non-fault hazard passes through unchanged.
    assert out[HazardChannel.HEADROOM_VIOLATION.value] == 0.0
    # the IS ledger records the likelihood ratio + weight (doc R Eq23/Eq24).
    rec = est[HazardChannel.XID_FAULT.value]
    assert rec["log_lr"] < 0.0 and 0.0 <= rec["weight"] <= 1.0
    assert "subset_probability" in rec  # smallest-prob family ⇒ subset refine (Eq25)


def test_no_tilt_leaves_hazards_unbiased_unchanged():
    gpus = [f"g{i}" for i in range(8)]
    faults = sample_fault_cluster(
        gpus, 3600.0, np.random.default_rng(3), FaultParams(), rate_tilt=1.0
    )
    raw = {HazardChannel.XID_FAULT.value: 1.0, HazardChannel.REQUEUE.value: 0.0}
    out, est = _debias_hazards(
        raw, faults, FaultParams(), 3600, fault_tilt=1.0,
        thermal_excess={}, health={}, residual=None,
    )
    # P == Q ⇒ L == 1 ⇒ nothing to correct.
    assert out == raw
    assert est == {}


def test_realized_intensities_condition_on_thermal_excess():
    # Eq14 coupling: more thermal excess ⇒ higher parent intensity (nominal & tilted).
    gpus = ["g0"]
    faults = sample_fault_cluster(
        gpus, 3600.0, np.random.default_rng(0), FaultParams(), rate_tilt=1e6
    )
    if not faults:  # extremely unlikely given the huge tilt, but guard anyway
        return
    from gpu_center_world_model.data_engine.synthetic.faults import (
        realized_fault_intensities,
    )

    cold = realized_fault_intensities(
        faults, FaultParams(), thermal_excess_c={"g0": 0.0}, rate_tilt=1.0
    )
    hot = realized_fault_intensities(
        faults, FaultParams(), thermal_excess_c={"g0": 20.0}, rate_tilt=1.0
    )
    assert hot[0].nominal_intensity > cold[0].nominal_intensity
