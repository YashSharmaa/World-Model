# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Changepoint, HSMM degradation, and quiet-failure validity lane (doc U §2/§3).

Three doc-U guarantees:

* **Quickest change** (§3 T1): CUSUM is Lorden-minimax — a larger threshold ``h``
  trades a *longer* average run length to false alarm (ARL0) for a *longer*
  detection delay, and the Siegmund threshold-for-ARL inverts that relationship.
  Shiryaev-Roberts / SPRT confirmation behave analogously on residual streams.
* **Graph-coupled HSMM** (§2 Eq5): the exact forward (state, dwell) filter is a
  valid belief — its state marginal is a simplex, a persistently-bad residualised
  stream with degraded telemetry raises ``P(Z ∈ {degraded, intermittent})``, and
  the MNAR coupling raises *expected* dropout as latent health degrades.
* **Quiet failure** (§2 I3 / §QF): when latent-fault mass rises *while* telemetry
  is bad/weak, the conjunction triggers the abstention posterior — a clean fresh
  channel never abstains; a quiet failure reliably does.

Operating-point stages 2-3 (budget threshold + uncertainty gate, doc U OI1/OT1)
are exercised here too, since they consume the quiet-failure posterior.
"""

from __future__ import annotations

import numpy as np

from gpu_center_world_model.reliability import (
    BOCPD,
    CUSUM,
    SPRT,
    SPRT_ACCEPT_H0,
    SPRT_ACCEPT_H1,
    DegradationBelief,
    DegradationFilter,
    HierarchicalScan,
    QuietFailureModel,
    ShiryaevRoberts,
    ThreeStageConfig,
    budget_threshold,
    detect_confirm_isolate,
    expected_bad_regime_prob,
    gaussian_shift_llr,
    integrity_ok,
    null_budget_threshold,
    three_stage_decision,
)
from gpu_center_world_model.reliability.degradation import (
    N_STATES,
    STATE_INDEX,
    exit_hazard,
    regime_probs,
)
from gpu_center_world_model.reliability.degradation import (
    DegradationParams as _Params,
)
from gpu_center_world_model.schema.entities import DeviceHealth
from gpu_center_world_model.schema.observations import SourceHealth


def _rng(seed: int = 20260614):
    return np.random.default_rng(seed)


def _empirical_arl(h: float, n_trials: int, T: int, rng) -> float:
    """Mean run length to a false alarm under the pre-change null N(0,1)."""
    runs = []
    for _ in range(n_trials):
        c = CUSUM(h=h, delta=1.0, sigma=1.0)
        t = c.run(rng.normal(size=T))
        runs.append(t if t > 0 else T)
    return float(np.mean(runs))


def _empirical_delay(h: float, n_trials: int, pre: int, rng) -> float:
    """Mean detection delay after a mean-shift changepoint at index `pre`."""
    delays = []
    for _ in range(n_trials):
        c = CUSUM(h=h, delta=1.0, sigma=1.0)
        x = np.concatenate([rng.normal(0.0, 1.0, pre), rng.normal(1.5, 1.0, 200)])
        t = c.run(x)
        if t > pre:
            delays.append(t - pre)
    return float(np.mean(delays)) if delays else float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# CUSUM ARL / detection-delay tradeoff (doc U §3 T1, Lorden-minimax)
# ──────────────────────────────────────────────────────────────────────────────
def test_cusum_arl_grows_with_threshold():
    rng = _rng(1)
    arl_low = _empirical_arl(4.0, 300, 4000, rng)
    arl_high = _empirical_arl(8.0, 300, 4000, rng)
    # A higher threshold yields a *longer* mean run length to false alarm.
    assert arl_high > arl_low
    # And it is genuinely large (false alarms are rare) for a Lorden-minimax CUSUM.
    assert arl_high > 1000.0


def test_cusum_delay_grows_with_threshold():
    rng = _rng(2)
    delay_low = _empirical_delay(4.0, 400, 200, rng)
    delay_high = _empirical_delay(8.0, 400, 200, rng)
    # The ARL/delay tradeoff: a higher threshold detects a real change *later*.
    assert delay_high > delay_low
    # Both detect the post-change shift reasonably quickly (not stuck).
    assert delay_low < 30.0


def test_threshold_for_arl_is_monotone_and_inverts_siegmund():
    c = CUSUM(delta=1.0, sigma=1.0)
    h100 = c.threshold_for_arl(100.0)
    h1000 = c.threshold_for_arl(1000.0)
    # Targeting a larger ARL0 demands a larger threshold (Siegmund inversion).
    assert h1000 > h100 > 0.0


def test_threshold_for_arl_actually_calibrates_false_alarm_rate():
    """A threshold sized for a *larger* ARL0 yields a *longer* empirical run length.

    Doc U §3 T1 calibration: ``threshold_for_arl(target)`` must move the empirical
    ARL0 in the right direction and by a meaningful margin, so the operator's
    false-alarm budget actually maps onto a threshold. We invert for two targets
    and confirm the empirical ARL0 under the null is ordered and grows materially.
    """
    rng = _rng(20)
    c = CUSUM(delta=1.0, sigma=1.0)
    h_small = c.threshold_for_arl(10.0)
    h_large = c.threshold_for_arl(20.0)
    assert h_large > h_small > 0.0
    arl_small = _empirical_arl(h_small, 400, 6000, rng)
    arl_large = _empirical_arl(h_large, 400, 6000, rng)
    # the larger target gives a meaningfully longer run length to false alarm.
    assert arl_large > 2.0 * arl_small
    # and the smaller-target threshold fires well inside the window (not vacuous).
    assert arl_small < 2000.0


def test_cusum_detects_a_real_shift_and_resets_clean():
    rng = _rng(3)
    c = CUSUM(h=5.0, delta=1.0, sigma=1.0)
    shifted = np.concatenate([rng.normal(0, 1, 50), rng.normal(2.0, 1, 100)])
    alarm = c.run(shifted)
    assert alarm > 50  # fired after the changepoint
    c.reset()
    assert c.stat == 0.0 and not c.alarmed


def test_gaussian_shift_llr_sign_matches_drift_direction():
    # Positive residuals score positive LLR for an upward-shift detector.
    assert gaussian_shift_llr(3.0, delta=1.0) > 0.0
    assert gaussian_shift_llr(-3.0, delta=1.0) < 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Shiryaev-Roberts + SPRT confirmation (doc U §3)
# ──────────────────────────────────────────────────────────────────────────────
def test_shiryaev_roberts_fires_faster_on_a_real_change():
    rng = _rng(4)
    pre = rng.normal(0, 1, 40)
    post = rng.normal(2.0, 1, 200)
    sr = ShiryaevRoberts(threshold=100.0, delta=1.0, sigma=1.0)
    alarm = sr.run(np.concatenate([pre, post]))
    assert alarm > 40  # detects the post-change regime


def test_shiryaev_roberts_arl_grows_with_threshold():
    """A larger Shiryaev-Roberts threshold trades a longer ARL0 for slower detection.

    Doc U §3: like CUSUM, the SR detector's average run length to false alarm is
    monotone in the alarm threshold ``A`` — the always-on surveillance ARL/delay
    knob. We measure empirical ARL0 under the null at two thresholds.
    """
    rng = _rng(40)

    def sr_arl(thr, n_trials, T):
        runs = []
        for _ in range(n_trials):
            sr = ShiryaevRoberts(threshold=thr, delta=1.0, sigma=1.0)
            t = sr.run(rng.normal(size=T))
            runs.append(t if t > 0 else T)
        return float(np.mean(runs))

    arl_low = sr_arl(50.0, 300, 3000)
    arl_high = sr_arl(5000.0, 300, 3000)
    assert arl_high > arl_low  # higher threshold => rarer false alarms


def test_sprt_confirms_h1_under_change_and_clears_under_null():
    rng = _rng(5)
    confirm = SPRT(mean0=0.0, mean1=2.0, sigma=1.0)
    assert confirm.run(rng.normal(2.0, 1.0, 200)) == SPRT_ACCEPT_H1
    clear = SPRT(mean0=0.0, mean1=2.0, sigma=1.0)
    assert clear.run(rng.normal(0.0, 1.0, 200)) == SPRT_ACCEPT_H0


def test_bocpd_run_length_mass_spikes_at_a_changepoint():
    rng = _rng(6)
    b = BOCPD(hazard=1.0 / 100.0)
    stream = np.concatenate([rng.normal(0, 0.5, 80), rng.normal(4.0, 0.5, 40)])
    cp = b.run(stream)
    # Recent-changepoint mass is highest right after the regime switch (index ~80).
    post = cp[80:95]
    pre = cp[40:60]
    assert post.max() > pre.max()


# ──────────────────────────────────────────────────────────────────────────────
# Hierarchical / detect-confirm-isolate chain (doc U §3 Eq1-Eq3)
# ──────────────────────────────────────────────────────────────────────────────
def test_detect_confirm_isolate_abstains_on_confirmed_common_mode():
    rng = _rng(7)
    t, d = 60, 4
    stream = rng.normal(0.0, 1.0, (t, d))
    # rack-level common-mode shift on all streams after t=20.
    stream[20:, :] += 3.0
    groups = {"rack": [[0, 1, 2, 3]]}
    verdict = detect_confirm_isolate(
        stream, groups=groups, h_device=5.0, h_group=8.0, sprt_mean1=2.0
    )
    assert verdict.primary_alarm
    assert verdict.sprt_confirmed  # Wald confirmation stage escalates the change
    assert verdict.hierarchical_alarm  # the rack-level common-mode scan fires
    assert "rack" in verdict.fired_levels
    assert verdict.abstain  # SPRT-confirmed or hierarchical alarm -> abstain


def test_detect_confirm_isolate_isolates_the_anomalous_stream():
    rng = _rng(11)
    t, d = 80, 4
    stream = rng.normal(0.0, 1.0, (t, d))
    stream[15:, 1] += 8.0  # only stream 1 carries a strong shift
    verdict = detect_confirm_isolate(stream, fdr=0.3, h_device=4.0)
    # BH-FDR over the per-stream CUSUM statistics isolates the true offender.
    assert 1 in verdict.isolated_streams
    # ... and does not falsely isolate the three clean streams.
    assert set(verdict.isolated_streams) == {1}


def test_hierarchical_scan_requires_correct_stream_shape():
    import pytest

    scan = HierarchicalScan(n_streams=3)
    with pytest.raises(ValueError):
        scan.step(np.zeros(2))


# ──────────────────────────────────────────────────────────────────────────────
# HSMM degradation belief correctness (doc U §2 Eq5)
# ──────────────────────────────────────────────────────────────────────────────
def test_state_marginal_is_a_simplex_each_step():
    filt = DegradationFilter()
    beliefs = filt.filter_stream(
        [SourceHealth.OBSERVED] * 8, [0.0, 0.2, 1.0, 2.0, 0.5, 1.5, 2.5, 0.1]
    )
    for b in beliefs:
        m = b.state_marginal()
        assert abs(float(m.sum()) - 1.0) < 1e-9
        assert np.all(m >= -1e-12) and m.shape == (N_STATES,)


def test_persistent_bad_residuals_raise_degraded_mass():
    filt = DegradationFilter()
    healthy = filt.filter_stream([SourceHealth.OBSERVED] * 10, [0.0] * 10)
    worsening = filt.filter_stream(
        [SourceHealth.OBSERVED] * 5 + [SourceHealth.DROPPED] * 5,
        [2.5] * 5 + [None] * 5,
    )
    # A clean stream keeps the device firmly healthy.
    assert healthy[-1].map_state() == DeviceHealth.HEALTHY
    assert healthy[-1].degraded_mass() < 0.05
    # A high-health-index + dropped-telemetry stream raises latent-fault mass.
    assert worsening[-1].degraded_mass() > 0.2


def test_hsmm_belief_sharpens_toward_the_bad_regime_with_evidence():
    """The exact forward filter *concentrates* off healthy as bad evidence grows.

    Doc U §2 Eq5: the exact (state, dwell) forward recursion is a valid belief that
    accumulates evidence. With more consecutive bad residuals the *healthy* mass
    collapses monotonically (the belief sharpens away from healthy) and the MAP
    state settles on a fault regime, while a long clean run keeps healthy mass near
    one. We assert these directional properties of the exact belief.
    """
    filt = DegradationFilter()
    healthy_idx = STATE_INDEX[DeviceHealth.HEALTHY]
    one_bad = filt.filter_stream([SourceHealth.OBSERVED] * 1, [2.5])[-1]
    few_bad = filt.filter_stream([SourceHealth.OBSERVED] * 3, [2.5] * 3)[-1]
    many_bad = filt.filter_stream([SourceHealth.OBSERVED] * 8, [2.5] * 8)[-1]
    h1 = float(one_bad.state_marginal()[healthy_idx])
    h3 = float(few_bad.state_marginal()[healthy_idx])
    h8 = float(many_bad.state_marginal()[healthy_idx])
    # healthy mass collapses sharply as bad evidence accrues and stays negligible.
    assert h1 > 10.0 * h3  # one ambiguous step -> three: an order-of-magnitude drop
    assert h3 < 0.01 and h8 < 0.01  # belief has firmly left the healthy regime
    # the sharpened belief sits firmly on a fault regime (degraded-mass dominates).
    assert many_bad.map_state() == DeviceHealth.DEGRADED
    assert many_bad.degraded_mass() > 0.5
    # a long clean run keeps the device confidently healthy.
    long_clean = filt.filter_stream([SourceHealth.OBSERVED] * 12, [0.0] * 12)[-1]
    assert long_clean.map_state() == DeviceHealth.HEALTHY
    assert long_clean.degraded_mass() < many_bad.degraded_mass()
    assert float(long_clean.state_marginal()[healthy_idx]) > 0.9


def test_mnar_coupling_raises_expected_dropout_with_degradation():
    filt = DegradationFilter()
    healthy = filt.filter_stream([SourceHealth.OBSERVED] * 8, [0.0] * 8)
    degraded = filt.filter_stream(
        [SourceHealth.OBSERVED] * 8, [2.5] * 8  # persistently bad health index
    )
    e_healthy = expected_bad_regime_prob(healthy[-1])
    e_degraded = expected_bad_regime_prob(degraded[-1])
    # MNAR: a more-degraded latent belief implies higher *expected* dropout.
    assert e_degraded > e_healthy


def test_force_masked_collapses_belief_onto_masked():
    filt = DegradationFilter()
    b0 = filt.init_belief()
    masked = filt.force_masked(b0)
    assert masked.map_state() == DeviceHealth.MASKED
    assert abs(masked.state_marginal()[STATE_INDEX[DeviceHealth.MASKED]] - 1.0) < 1e-9


def test_exit_hazard_increases_with_dwell_for_shortlived_states():
    p = _Params()
    intermittent = STATE_INDEX[DeviceHealth.INTERMITTENTLY_FAULTY]
    h_early = exit_hazard(p, np.zeros(N_STATES))[intermittent]
    h_late = exit_hazard(p, np.full(N_STATES, 10.0))[intermittent]
    # Short-lived states are self-limiting: the exit hazard rises with dwell (T1).
    assert h_late > h_early


def test_neighbor_fault_raises_healthy_exit_hazard():
    p = _Params()
    healthy = STATE_INDEX[DeviceHealth.HEALTHY]
    quiet = exit_hazard(p, np.zeros(N_STATES), neighbor_fault=0.0)[healthy]
    loud = exit_hazard(p, np.zeros(N_STATES), neighbor_fault=1.0)[healthy]
    # Graph coupling: neighbour faults push a healthy device to exit sooner (I2).
    assert loud > quiet


def test_regime_probs_rows_are_simplices():
    pr = regime_probs(_Params())
    assert pr.shape[0] == N_STATES
    assert np.allclose(pr.sum(axis=1), 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Quiet-failure trigger (doc U §2 I3 / §QF)
# ──────────────────────────────────────────────────────────────────────────────
def _belief_with_fault(fault_mass: float) -> DegradationBelief:
    joint = np.zeros((N_STATES, 64))
    joint[STATE_INDEX[DeviceHealth.INTERMITTENTLY_FAULTY], 0] = fault_mass * 0.75
    joint[STATE_INDEX[DeviceHealth.DEGRADED], 0] = fault_mass * 0.25
    joint[STATE_INDEX[DeviceHealth.HEALTHY], 0] = 1.0 - fault_mass
    return DegradationBelief(joint=joint)


def test_quiet_failure_triggers_when_fault_rises_under_bad_telemetry():
    qf = QuietFailureModel()
    bad = _belief_with_fault(0.95)
    ip = qf.assess(
        bad,
        regime=SourceHealth.DROPPED,
        telemetry_likelihood=0.0,
        source_health_bad_prob=1.0,
    )
    # The conjunction (rising fault AND degraded observation) trips the trigger.
    assert ip.is_quiet_failure
    assert ip.quiet_failure_score >= 0.5
    assert ip.u > 0.5


def test_clean_fresh_channel_does_not_abstain():
    qf = QuietFailureModel()
    healthy = _belief_with_fault(0.0)
    ip = qf.assess(healthy, regime=SourceHealth.OBSERVED, telemetry_likelihood=1.0)
    # A clean, fresh, healthy channel: low integrity posterior, no quiet failure.
    assert not ip.is_quiet_failure
    assert ip.u < 0.1
    assert integrity_ok(ip, kappa_obs=0.5)


def test_high_fault_but_fresh_telemetry_does_not_trip_quiet_failure_alone():
    # Quiet failure is the *conjunction* — a rising fault with perfectly fresh,
    # well-explained telemetry is not (yet) a quiet failure.
    qf = QuietFailureModel()
    ip = qf.assess(
        _belief_with_fault(0.9),
        regime=SourceHealth.OBSERVED,
        telemetry_likelihood=1.0,
        source_health_bad_prob=0.0,
    )
    assert not ip.is_quiet_failure


# ──────────────────────────────────────────────────────────────────────────────
# Operating-point stages 2-3 (doc U OI1 / OT1)
# ──────────────────────────────────────────────────────────────────────────────
def test_budget_threshold_respects_the_alert_budget():
    rng = _rng(8)
    scores = rng.uniform(0.0, 1.0, 5000)
    u = budget_threshold(scores, budget=50, n_windows=1000)  # 5% alert budget
    alert_rate = float(np.mean(scores >= u))
    assert alert_rate <= 0.05 + 0.01
    # Tightening the budget can only raise the threshold (fewer alerts).
    u_tight = budget_threshold(scores, budget=10, n_windows=1000)
    assert u_tight >= u


def test_null_budget_threshold_calibrates_on_incident_free_scores():
    rng = _rng(9)
    null_scores = rng.normal(0.0, 1.0, 5000)  # incident-free slice
    u = null_budget_threshold(null_scores, budget=50, n_windows=1000)
    assert float(np.mean(null_scores >= u)) <= 0.05 + 0.01


def test_three_stage_gate_abstains_on_quiet_failure():
    qf = QuietFailureModel()
    ip = qf.assess(
        _belief_with_fault(0.95),
        regime=SourceHealth.DROPPED,
        telemetry_likelihood=0.0,
        source_health_bad_prob=1.0,
    )
    cfg = ThreeStageConfig(score_threshold=0.5, kappa_obs=0.5, kappa_epistemic=0.5)
    dec = three_stage_decision(
        score=0.9, integrity=ip, epistemic_score=0.1, feasible=True, config=cfg
    )
    # High integrity posterior -> abstain (and labelled as a quiet failure).
    assert dec.abstain and not dec.emit
    assert dec.reason == "abstain_quiet_failure"


def test_three_stage_gate_emits_when_clean_and_over_budget():
    qf = QuietFailureModel()
    ip = qf.assess(_belief_with_fault(0.0), regime=SourceHealth.OBSERVED)
    cfg = ThreeStageConfig(score_threshold=0.5, kappa_obs=0.5, kappa_epistemic=0.5)
    emit = three_stage_decision(
        score=0.9, integrity=ip, epistemic_score=0.1, feasible=True, config=cfg
    )
    assert emit.emit and not emit.abstain
    # Below the budget threshold -> not emitted, but not an abstention either.
    below = three_stage_decision(
        score=0.1, integrity=ip, epistemic_score=0.1, feasible=True, config=cfg
    )
    assert not below.emit and not below.abstain


def test_three_stage_gate_never_emits_an_infeasible_candidate():
    qf = QuietFailureModel()
    ip = qf.assess(_belief_with_fault(0.0), regime=SourceHealth.OBSERVED)
    cfg = ThreeStageConfig()
    dec = three_stage_decision(
        score=10.0, integrity=ip, epistemic_score=0.0, feasible=False, config=cfg
    )
    # The learned head can never expand the exact-kernel safe set (doc U OI1).
    assert not dec.emit
    assert dec.reason == "exact_kernel_infeasible"
