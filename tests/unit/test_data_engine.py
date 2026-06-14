# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""TEST-DATAENGINE-LANE — the consolidated data-engine certification test lane.

One end-to-end lane over the doc-D/doc-R synthetic data engine, exercising the
load-bearing contracts of every stage of the Bronze→Silver→Gold→release pipeline:

* **CRN pack equivalence** (doc R counterfactual_branch_construction step 6 / Eq11):
  :func:`build_pack` with the same seed produces byte-identical branch summaries
  (KPIs, hazards, virtual-queue targets, and emitted telemetry), because paired
  branches share one exogenous seed bundle Û. A different seed differs.
* **≥ 3-action-per-window requirement** (doc D splits.counterfactual_harness):
  :func:`build_pack` raises below three actions, and the default catalog ships
  ≥ 3 feasible-shaped candidate actions.
* **Importance-sampling unbiasedness** (doc R Eq24 / T3): on a tractable
  exponential-tail problem the IS estimator ``p̂_F=mean(1{F}·L)`` recovers the
  analytic nominal probability (≈ a naive Monte-Carlo run under P), and the
  tilted-hazard debiasing corrects fault-driven hazards *down* from the inflated
  raw indicator.
* **Each fidelity test on a known-good vs known-bad corpus** (doc R
  fidelity_certification_suite + poisoning_detection_tests): MMD², energy
  distance, the origin C2ST, lag-structure, incident-bundle coherence, and the
  pre-decision-leakage test each *pass* a faithful corpus and *fail* a poisoned
  one.
* **bronze_to_silver missingness preservation** (doc D correlation_rules #5):
  a ``quality_mask == 0`` row survives as an *unavailable* channel carrying its
  age-of-information, never dropped or zero-filled; staleness surfaces as
  ``SourceHealth.STALE``.
* **tensorize mask invariants** (doc Q hierarchical pooling): per-entity boolean
  masks mark real rows ``True`` / padding ``False``; every padded row is exactly
  zero and the incidence matrices are one-hot.
* **assemble_release certification gating** (doc R Eq27 c_g gate): a release is
  refused when the provenance manifest is invalid or — under ``require_certified``
  — when the fidelity report is uncertified, and ships when certified.

Split disjointness across the four test suites lives in the property file.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from gpu_center_world_model.data_engine.contracts import (
    DecisionWindow,
    Origin,
    ProvenanceManifest,
)
from gpu_center_world_model.data_engine.normalize.canonical import (
    Lane,
    MetricSample,
    bronze_to_silver,
    source_state_from_silver,
)
from gpu_center_world_model.data_engine.provenance.manifest import (
    build_manifest,
    validate_manifest,
)
from gpu_center_world_model.data_engine.release_manifest import assemble_release
from gpu_center_world_model.data_engine.synthetic import (
    build_pack,
    default_action_catalog,
    generate_pack,
    generate_pod_state,
)
from gpu_center_world_model.data_engine.synthetic.counterfactual import _debias_hazards
from gpu_center_world_model.data_engine.synthetic.faults import (
    FaultParams,
    sample_fault_cluster,
)
from gpu_center_world_model.data_engine.synthetic.fidelity import (
    FidelityReport,
    classifier_two_sample_test,
    energy_distance_test,
    incident_bundle_test,
    lag_structure_test,
    mmd2_test,
    predecision_leakage_test,
)
from gpu_center_world_model.data_engine.synthetic.fidelity import (
    TestResult as FidTestResult,  # aliased so pytest does not collect the dataclass
)
from gpu_center_world_model.data_engine.synthetic.rare_events import (
    importance_estimate,
    point_process_log_lr,
)
from gpu_center_world_model.data_engine.tensorize.tensorizer import (
    GPU_FEATURES,
    WORKLOAD_FEATURES,
    tensorize,
)
from gpu_center_world_model.schema.kpis import HazardChannel
from gpu_center_world_model.schema.observations import SourceHealth


# ──────────────────────────────────────────────────────────────────────────────
# build_pack — common-random-number equivalence (doc R Eq11; step 6)
# ──────────────────────────────────────────────────────────────────────────────
def _pack(seed: int, horizon_s: int = 600):
    pod, state, uid = generate_pod_state(seed=0, gpu_demand=8)
    actions = default_action_catalog(state, uid)
    return build_pack(pod, state, actions, seed=seed, horizon_s=horizon_s)


def test_build_pack_same_seed_is_byte_identical():
    """Same seed ⇒ byte-identical branches (shared Û, common random numbers)."""
    p1 = _pack(seed=7)
    p2 = _pack(seed=7)
    assert p1.state_hash == p2.state_hash
    assert p1.seed_bundle == p2.seed_bundle
    assert [b.action_id for b in p1.branches] == [b.action_id for b in p2.branches]
    for b1, b2 in zip(p1.branches, p2.branches, strict=True):
        assert b1.feasible == b2.feasible
        # KPI / hazard / slow-latent summaries are bit-for-bit equal
        assert b1.kpis == b2.kpis
        assert b1.hazards == b2.hazards
        assert b1.vq_features == b2.vq_features
        assert b1.hazard_estimates == b2.hazard_estimates
        # the emitted per-step telemetry stream is reproduced exactly
        assert len(b1.observations) == len(b2.observations)
        for o1, o2 in zip(b1.observations, b2.observations, strict=True):
            assert o1.ts_event_ns == o2.ts_event_ns
            assert o1.entity_id == o2.entity_id
            v1 = {k: c.value for k, c in o1.channels.items()}
            v2 = {k: c.value for k, c in o2.channels.items()}
            assert v1 == v2


def test_build_pack_different_seed_differs():
    """A different master seed perturbs Û ⇒ the realised branches differ."""
    p1 = _pack(seed=7)
    p2 = _pack(seed=123456)
    assert p1.state_hash == p2.state_hash  # same pre-action state
    differs = any(
        b1.kpis != b2.kpis
        or b1.hazards != b2.hazards
        or b1.vq_features != b2.vq_features
        for b1, b2 in zip(p1.branches, p2.branches, strict=True)
    )
    assert differs


def test_build_pack_requires_three_actions():
    """< 3 actions is rejected (doc D splits.counterfactual_harness)."""
    pod, state, uid = generate_pod_state(seed=0, gpu_demand=8)
    actions = default_action_catalog(state, uid)
    with pytest.raises(ValueError, match=r">= 3 actions"):
        build_pack(pod, state, actions[:2], seed=0, horizon_s=600)


def test_default_action_catalog_has_at_least_three_actions():
    """The default catalog enumerates ≥ 3 candidate actions (hold + placements)."""
    _pod, state, uid = generate_pod_state(seed=0, gpu_demand=8)
    actions = default_action_catalog(state, uid)
    assert len(actions) >= 3
    # the canonical baseline (hold) is the first action
    assert actions[0].type.value == "hold"


def test_generate_pack_tags_baseline_to_concrete_hold_id():
    """The pack's baseline id resolves to the first (hold) branch's concrete id."""
    pack = generate_pack(seed=0, scenario="normal")
    assert len(pack.branches) >= 3
    assert pack.baseline_action_id == pack.branches[0].action_id
    # deltas vs. baseline are zero on the baseline and defined on the rest
    deltas = pack.deltas()
    assert pack.baseline_action_id not in deltas
    assert len(deltas) == len(pack.branches) - 1


# ──────────────────────────────────────────────────────────────────────────────
# Importance-sampling unbiasedness (doc R Eq24 / T3)
# ──────────────────────────────────────────────────────────────────────────────
def test_importance_estimate_matches_naive_mc_on_tractable_case():
    """IS estimate ≈ analytic nominal probability ≈ naive MC (doc R Eq24, T3).

    Tractable case: estimate ``p = P_P(X > thr)`` for ``X ~ Exp(rate=lam_p)``
    (nominal law P) while drawing from a *tilted* proposal ``Q = Exp(rate=lam_q)``
    with a lower rate (a heavier tail ⇒ many more hits in the rare set). The exact
    point likelihood ratio ``dP/dQ`` re-weights every Q-sample, so the unbiased
    estimator recovers the nominal probability — within a few MC standard errors of
    both the closed form and an independent naive Monte-Carlo run under P.
    """
    rng = np.random.default_rng(0)
    n = 200_000
    lam_p, lam_q, thr = 2.0, 0.5, 3.0
    x = rng.exponential(1.0 / lam_q, size=n)  # ω_i ~ Q
    indicators = (x > thr).astype(float)  # 1{ω ∈ F_H}
    # dP/dQ for the exponential: (lam_p/lam_q)·exp(-(lam_p-lam_q)·x)
    log_w = np.log(lam_p / lam_q) - (lam_p - lam_q) * x
    est = importance_estimate(indicators, log_w, "tail")

    analytic = float(np.exp(-lam_p * thr))  # P_P(X > thr)
    # naive MC under the nominal law P (an independent reference)
    naive = float((rng.exponential(1.0 / lam_p, size=n) > thr).mean())

    assert est.std_error > 0.0
    assert abs(est.probability - analytic) <= 5.0 * est.std_error
    # the IS run buys variance reduction: it hits the rare set far more often than
    # the naive run, so the two point estimates still agree to a few %.
    assert abs(est.probability - naive) <= 0.0005
    # the proposal dominates P on F_H ⇒ ESS is a large fraction of N (not degenerate)
    assert est.ess > 0.1 * n
    # hit-rate under Q is materially above the nominal rare probability (tilt works)
    assert est.hit_rate_q > analytic


def test_point_process_log_lr_is_zero_when_p_equals_q():
    """``log(dP/dQ) == 0`` when the proposal equals the nominal law (Eq23)."""
    log_lr = point_process_log_lr([1.0, 2.0, 3.0], lambda _t: 0.7, lambda _t: 0.7, 10.0)
    assert log_lr == pytest.approx(0.0, abs=1e-9)


def test_point_process_log_lr_matches_closed_form():
    """``log(dP/dQ)`` equals the Eq23 closed form: Σ log(λ^P/λ^Q) − ∫(Λ^P−Λ^Q).

    For constant intensities ``λ^P, λ^Q`` over ``[0, H]`` with ``m`` realised marks
    the formula collapses to ``m·log(λ^P/λ^Q) − (λ^P−λ^Q)·H``. A proposal that
    inflates intensity and *also* realises many marks (more than the nominal mean
    ``λ^P·H``) drives ``log(dP/dQ) < 0`` (weight < 1), the correcting down-weight of
    an over-sampled rare path.
    """
    lam_p, lam_q, horizon = 0.5, 2.0, 10.0
    # a path with MANY marks under the up-tilt (16 ≫ λ^P·H = 5)
    times = list(np.linspace(0.2, 9.8, 16))
    log_lr = point_process_log_lr(times, lambda _t: lam_p, lambda _t: lam_q, horizon)
    expected = len(times) * np.log(lam_p / lam_q) - (lam_p - lam_q) * horizon
    assert log_lr == pytest.approx(expected, rel=1e-6, abs=1e-6)
    # an over-sampled tilted path is corrected DOWN
    assert log_lr < 0.0
    assert 0.0 < float(np.exp(log_lr)) < 1.0


def test_debias_hazards_corrects_tilted_fault_hazards_down():
    """Tilted fault hazards are debiased *below* the inflated raw draw (Eq24).

    Under a fault tilt the realised path comes from the inflated proposal Q, so the
    raw tilted indicator over-states the hazard. Re-weighting by the point-process
    likelihood ratio restores the unbiased nominal-law estimate, which must not
    exceed the raw value; an untilted (non-fault-coupled) hazard passes through.
    """
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
    # fault-driven hazards corrected DOWN (unbiased, not biased high)
    assert out[HazardChannel.XID_FAULT.value] <= raw[HazardChannel.XID_FAULT.value]
    assert out[HazardChannel.REQUEUE.value] <= raw[HazardChannel.REQUEUE.value]
    # the un-coupled headroom hazard passes through unchanged
    assert out[HazardChannel.HEADROOM_VIOLATION.value] == 0.0
    # the IS ledger records the exact log-LR (≤ 0 under the up-tilt) and weight ≤ 1
    rec = est[HazardChannel.XID_FAULT.value]
    assert rec["log_lr"] <= 0.0
    assert 0.0 <= rec["weight"] <= 1.0


def test_debias_hazards_identity_without_tilt():
    """No tilt ⇒ P == Q ⇒ L == 1 ⇒ the raw indicators pass through unbiased."""
    gpus = [f"g{i}" for i in range(8)]
    faults = sample_fault_cluster(
        gpus, 3600.0, np.random.default_rng(3), FaultParams(), rate_tilt=1.0
    )
    raw = {HazardChannel.XID_FAULT.value: 1.0, HazardChannel.REQUEUE.value: 0.0}
    out, est = _debias_hazards(
        raw,
        faults,
        FaultParams(),
        3600,
        fault_tilt=1.0,
        thermal_excess={},
        health={},
        residual=None,
    )
    assert out == raw
    assert est == {}


# ──────────────────────────────────────────────────────────────────────────────
# Each fidelity test on a known-good vs known-bad corpus (doc R suite)
# ──────────────────────────────────────────────────────────────────────────────
def _good_bad_embeddings():
    rng = np.random.default_rng(0)
    real = rng.normal(0.0, 1.0, size=(60, 5))
    good_syn = rng.normal(0.0, 1.0, size=(60, 5))  # same law ⇒ faithful
    bad_syn = rng.normal(5.0, 1.0, size=(60, 5))  # location-shifted ⇒ poisoned
    return real, good_syn, bad_syn


def test_mmd2_test_separates_good_from_bad():
    real, good, bad = _good_bad_embeddings()
    assert mmd2_test(real, good).passed  # fail to reject P_real == P_syn
    assert not mmd2_test(real, bad).passed  # detect the shift


def test_energy_distance_test_separates_good_from_bad():
    real, good, bad = _good_bad_embeddings()
    assert energy_distance_test(real, good).passed
    assert not energy_distance_test(real, bad).passed


def test_classifier_two_sample_test_separates_good_from_bad():
    real, good, bad = _good_bad_embeddings()
    g = classifier_two_sample_test(real, good)
    b = classifier_two_sample_test(real, bad)
    assert g.passed and g.statistic <= 0.65  # AUC reverts toward chance
    assert not b.passed and b.statistic > 0.65  # separable ⇒ high AUC


def test_lag_structure_test_flags_reversed_lag():
    """Good corpus preserves the parent→child causal delay; bad reverses it."""
    t = np.arange(200)
    parent = np.sin(0.3 * t)
    child_good = np.concatenate([np.zeros(3), parent[:-3]])  # child lags by 3
    child_bad = np.concatenate([parent[3:], np.zeros(3)])  # child LEADS (reversed)
    good = lag_structure_test([(parent, child_good)], [(parent, child_good)])
    bad = lag_structure_test([(parent, child_good)], [(parent, child_bad)])
    assert good.passed and good.detail["peak_lag_syn"] == 3.0
    assert not bad.passed  # peak lag collapsed to 0 ⇒ causal order broken


def test_incident_bundle_test_flags_missing_descendant():
    """A coherent bundle co-occurs required descendants; an isolated spike fails."""
    required = ["xid", "requeue"]
    good = [{"xid": 1.0, "requeue": 1.0} for _ in range(10)]
    bad = [{"xid": 1.0, "requeue": 0.0} for _ in range(10)]  # requeue never present
    assert incident_bundle_test(good, good, required).passed
    res_bad = incident_bundle_test(good, bad, required)
    assert not res_bad.passed
    assert res_bad.detail["worst_cooccurrence"] == 0.0


def test_predecision_leakage_test_flags_post_action_leak():
    """Augmenting pre-features with the leaked label boosts AUC ⇒ leakage flagged."""
    rng = np.random.default_rng(0)
    n = 120
    y = np.tile([0.0, 1.0], n // 2)  # perfectly balanced labels
    # pre-features carry a weak signal about y (auc_pre is well-defined, not chance)
    pre = rng.normal(0.0, 1.0, size=(n, 4)) + 0.3 * y[:, None]
    # benign post: duplicate the SAME pre columns ⇒ deterministically no new
    # information ⇒ the augmented predictor cannot beat the pre-only one.
    post_good = np.concatenate([pre, pre[:, :2]], axis=1)
    # poisoned post: the label is copied straight in ⇒ a large AUC gain.
    leak = np.stack([y, y], axis=1) * 5.0
    post_bad = np.concatenate([pre, leak], axis=1)
    good = predecision_leakage_test(pre, post_good, y)
    bad = predecision_leakage_test(pre, post_bad, y)
    assert good.passed and good.statistic <= 0.05  # no leak ⇒ AUC gain ≈ 0
    assert not bad.passed and bad.statistic > 0.05  # leaked label ⇒ AUC jumps


# ──────────────────────────────────────────────────────────────────────────────
# bronze_to_silver missingness preservation (doc D correlation_rules #5)
# ──────────────────────────────────────────────────────────────────────────────
def test_bronze_to_silver_preserves_missingness_as_state():
    """A degraded (quality_mask=0) row survives as an unavailable channel."""
    rows = [
        MetricSample(
            ts_event_ns=1_000_000_000,
            origin="sim",
            source="dcgm",
            entity_kind="gpu",
            entity_id="g0",
            metric_name="gpu_util",
            value=50.0,
            quality_mask=1,
        ),
        MetricSample(
            ts_event_ns=1_000_000_000,
            origin="sim",
            source="dcgm",
            entity_kind="gpu",
            entity_id="g0",
            metric_name="power_w",
            value=0.0,
            quality_mask=0,  # degraded/missing — must NOT be dropped or zero-filled
        ),
    ]
    silver = bronze_to_silver(rows)
    frames = silver[Lane.FAST]
    assert len(frames) == 1
    obs = frames[0].observations[0]
    # both channels are present in the frame (the missing one is kept as state)
    util = obs.channels["DCGM_FI_DEV_GPU_UTIL"]
    power = obs.channels["DCGM_FI_DEV_POWER_USAGE"]
    assert util.available and util.source_health == SourceHealth.OBSERVED
    assert not power.available
    assert power.source_health == SourceHealth.DROPPED
    # the derived source state carries the missingness fraction (½ channels down)
    ss = source_state_from_silver(frames)
    assert ss.channel_missing_prob["gpu:g0"] == pytest.approx(0.5)


def test_bronze_to_silver_marks_stale_when_aged_beyond_two_cadences():
    """An available-but-old channel is flagged STALE, carrying its age-of-info."""
    rows = [
        MetricSample(
            ts_event_ns=0,
            origin="sim",
            source="dcgm",
            entity_kind="gpu",
            entity_id="g0",
            metric_name="gpu_util",
            value=50.0,
            quality_mask=1,
        )
    ]
    # observed 1000 s in the past ⇒ well beyond 2× the fast-lane cadence ⇒ stale
    silver = bronze_to_silver(rows, now_ns=1_000_000_000_000)
    ch = silver[Lane.FAST][0].observations[0].channels["DCGM_FI_DEV_GPU_UTIL"]
    assert ch.available  # value still present
    assert ch.source_health == SourceHealth.STALE
    assert ch.age_s == pytest.approx(1000.0)


# ──────────────────────────────────────────────────────────────────────────────
# tensorize mask invariants (doc Q typed message passing)
# ──────────────────────────────────────────────────────────────────────────────
def test_tensorize_masks_cover_real_entities_only():
    """Entity masks mark real rows True / padding False; mask sum == entity count."""
    _pod, state, _uid = generate_pod_state(seed=0, gpu_demand=8)
    batch = tensorize(state)
    n_gpu = len(state.symbolic.gpus)
    n_wl = len(state.symbolic.workloads)
    gpu_mask = batch.masks["gpu"]
    wl_mask = batch.masks["workload"]
    assert gpu_mask.dtype == bool and wl_mask.dtype == bool
    assert int(gpu_mask.sum()) == n_gpu
    assert int(wl_mask.sum()) == n_wl
    # padded rows beyond the real count are masked off
    assert gpu_mask.shape[0] == batch.caps.G_max
    assert not gpu_mask[n_gpu:].any()


def test_tensorize_padded_rows_are_zero_and_columns_fixed():
    """Padded (masked-off) feature rows are exactly zero; column order is frozen."""
    _pod, state, _uid = generate_pod_state(seed=0, gpu_demand=8)
    batch = tensorize(state)
    gpu_arr = batch.arrays["gpu"]
    wl_arr = batch.arrays["workload"]
    assert gpu_arr.dtype == np.float32
    # every masked-off (padding) row is identically zero
    assert np.all(gpu_arr[~batch.masks["gpu"]] == 0.0)
    assert np.all(wl_arr[~batch.masks["workload"]] == 0.0)
    # the feature dimension matches the auditable column lists
    assert gpu_arr.shape[1] == len(GPU_FEATURES)
    assert wl_arr.shape[1] == len(WORKLOAD_FEATURES)


def test_tensorize_incidence_matrices_are_one_hot():
    """Each relation's S_r / T_r are one-hot incidence matrices (≤ 1 per row)."""
    _pod, state, _uid = generate_pod_state(seed=0, gpu_demand=8)
    batch = tensorize(state)
    any_edges = False
    for _rt, (s_mat, t_mat) in batch.incidence.items():
        # S and T share the edge dimension (rows)
        assert s_mat.shape[0] == t_mat.shape[0]
        if s_mat.shape[0] == 0:
            continue
        any_edges = True
        # one-hot: each edge selects at most one source / dest row, entries in {0,1}
        assert np.all(np.isin(s_mat, (0.0, 1.0)))
        assert np.all(np.isin(t_mat, (0.0, 1.0)))
        assert np.all(s_mat.sum(axis=1) <= 1.0 + 1e-6)
        assert np.all(t_mat.sum(axis=1) <= 1.0 + 1e-6)
    assert any_edges  # the demo pod has wired relations (workload/podset/queue …)


# ──────────────────────────────────────────────────────────────────────────────
# assemble_release certification gating (doc R Eq27 c_g gate)
# ──────────────────────────────────────────────────────────────────────────────
def _demo_window() -> DecisionWindow:
    return DecisionWindow(
        decision_id="d0",
        t0_ns=0,
        state_hash="d0",
        seed_bundle=0,
        candidate_action_ids=["0:hold::"],
        source_mix=Origin.SIM_COUNTERFACTUAL,
    )


def _valid_manifest():
    return build_manifest(
        generator_commit="abc123",
        calibration_window="demo",
        real_fraction=0.0,
    )


def test_assemble_release_refuses_uncertified_when_required():
    """An uncertified fidelity report blocks the release under require_certified."""
    manifest = _valid_manifest()
    assert validate_manifest(manifest) == []
    uncertified = FidelityReport(results=[FidTestResult("mmd2", False, 9.0)])
    assert not uncertified.certified
    with (
        tempfile.TemporaryDirectory() as d,
        pytest.raises(ValueError, match="uncertified release"),
    ):
        assemble_release(
            d,
            [_demo_window()],
            manifest,
            fidelity=uncertified,
            require_certified=True,
        )


def test_assemble_release_ships_certified_release():
    """A certified report ships and the data card records the certification."""
    manifest = _valid_manifest()
    certified = FidelityReport(results=[FidTestResult("mmd2", True, 0.0)])
    assert certified.certified and certified.gate() == 1.0
    with tempfile.TemporaryDirectory() as d:
        res = assemble_release(
            d,
            [_demo_window()],
            manifest,
            fidelity=certified,
            require_certified=True,
        )
    assert res["data_card"]["fidelity_certified"] is True
    assert res["data_card"]["fidelity_gate_c_g"] == 1.0


def test_assemble_release_gate_off_records_status_without_refusing():
    """With require_certified=False an uncertified release ships, marked so."""
    manifest = _valid_manifest()
    uncertified = FidelityReport(results=[FidTestResult("mmd2", False, 9.0)])
    with tempfile.TemporaryDirectory() as d:
        res = assemble_release(
            d,
            [_demo_window()],
            manifest,
            fidelity=uncertified,
            require_certified=False,
        )
    assert res["data_card"]["fidelity_certified"] is False


def test_assemble_release_refuses_invalid_manifest():
    """An incomplete provenance manifest is rejected before anything is written."""
    bad = ProvenanceManifest(
        schema_version="", generator_commit="", simulator_version=""
    )
    with (
        tempfile.TemporaryDirectory() as d,
        pytest.raises(ValueError, match="invalid provenance manifest"),
    ):
        assemble_release(d, [_demo_window()], bad)
