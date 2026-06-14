# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""UNC-FILTER-COMPLETENESS: square-root EKF, smoothers, amortized variational
filter, EnKF localization, observability flooring, CVaR/OCE conformal risk
control, BC-ACI one-sided, and the selective m==n_cert / P3 fixes (docs T, L)."""

from __future__ import annotations

import numpy as np

from gpu_center_world_model.uncertainty import (
    AmortizedFilterParams,
    AmortizedVariationalFilter,
    BCACICalibrator,
    BundleResidualRiskHead,
    ConformalRiskController,
    EnsembleKalmanFilter,
    SquareRootEKF,
    certify,
    cvar_from_losses,
    ensemble_kalman_smoother,
    gaspari_cohn,
    oce_from_losses,
    rbpf_smooth_modes,
    severe_shift_certificate,
)
from gpu_center_world_model.uncertainty.diagnostics import ObservabilityMonitor
from gpu_center_world_model.uncertainty.filters import _qr_factor, _safe_cholesky


def _rng():
    return np.random.default_rng(20260614)


def _ekf(P0):
    F = np.eye(3)
    H = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    e = SquareRootEKF(
        f=lambda x: F @ x,
        h=lambda x: H @ x,
        Q=0.01 * np.eye(3),
        R=0.1 * np.eye(2),
        x=np.array([0.5, -0.3, 0.2]),
        F_jac=lambda x: F,
        H_jac=lambda x: H,
    )
    e.S = _qr_factor(_safe_cholesky(P0).T)
    return e, F, H


# ──────────────────────────────────────────────────────────────────────────────
# True square-root EKF measurement update (array + Potter) — no P inversion
# ──────────────────────────────────────────────────────────────────────────────
def test_sqrt_array_update_matches_textbook_kf():
    P0 = np.diag([1.0, 2.0, 0.5])
    e, _F, H = _ekf(P0)
    x0 = e.x.copy()
    y = np.array([1.0, -1.0])
    S_innov = H @ P0 @ H.T + 0.1 * np.eye(2)
    K = P0 @ H.T @ np.linalg.inv(S_innov)
    x_ref = x0 + K @ (y - H @ x0)
    P_ref = (np.eye(3) - K @ H) @ P0
    e.update(y)
    assert np.allclose(e.x, x_ref, atol=1e-8)
    assert np.allclose(e.P, P_ref, atol=1e-6)
    # the square-root form keeps P symmetric positive-definite.
    assert np.all(np.linalg.eigvalsh(e.P) > 0)


def test_potter_sequential_update_matches_textbook_kf():
    P0 = np.diag([1.0, 2.0, 0.5])
    e, _F, H = _ekf(P0)
    x0 = e.x.copy()
    y = np.array([1.0, -1.0])
    S_innov = H @ P0 @ H.T + 0.1 * np.eye(2)
    K = P0 @ H.T @ np.linalg.inv(S_innov)
    x_ref = x0 + K @ (y - H @ x0)
    P_ref = (np.eye(3) - K @ H) @ P0
    e.potter_update(y)
    assert np.allclose(e.x, x_ref, atol=1e-6)
    assert np.allclose(e.P, P_ref, atol=1e-6)


def test_missing_measurement_does_not_contract_covariance():
    P0 = np.diag([1.0, 2.0, 0.5])
    e, _F, _H = _ekf(P0)
    p_before = e.P.copy()
    out = e.update(np.zeros(0), idx=np.array([], int))
    assert out.size == 0
    assert np.allclose(e.P, p_before)  # E24: no observation -> no contraction


# ──────────────────────────────────────────────────────────────────────────────
# Observability flooring + bounded gain (C11/C12)
# ──────────────────────────────────────────────────────────────────────────────
def test_observability_floor_raises_weak_direction_variance():
    P0 = np.diag([1.0, 1.0, 1e-6])  # 3rd direction barely observed
    e, _F, H = _ekf(P0)
    mon = ObservabilityMonitor(tol=1e-3)
    report = mon.analyze_matrix(H)  # H observes only dims 0,1
    assert report.weak_directions.shape[1] >= 1
    floor = 0.5
    e.floor_weak_directions(report.weak_directions, floor)
    # variance along the weak (3rd) axis is now at least the floor.
    d = np.array([0.0, 0.0, 1.0])
    assert float(d @ e.P @ d) >= floor - 1e-9


def test_bounded_gain_inflation_grows_with_gap():
    mon = ObservabilityMonitor()
    assert mon.bounded_gain_scale(0) == 1.0
    assert mon.bounded_gain_scale(3) > mon.bounded_gain_scale(1)
    assert mon.bounded_gain_scale(100) <= 2.0  # saturating cap


# ──────────────────────────────────────────────────────────────────────────────
# EnKF full state x observation Gaspari-Cohn localization
# ──────────────────────────────────────────────────────────────────────────────
def test_gaspari_cohn_is_compactly_supported():
    d = np.array([0.0, 1.0, 2.0, 3.0])
    g = gaspari_cohn(d, length_scale=1.0)
    assert g[0] == 1.0
    assert g[-1] == 0.0  # zero beyond 2*length_scale
    assert np.all((g >= 0.0) & (g <= 1.0))


def test_enkf_full_state_obs_localization_tapers_cross_cov():
    rng = _rng()
    n, m, M = 4, 2, 60
    ens = rng.normal(size=(M, n))
    # localization_xy: only state dims 0,1 correlate with the 2 observations.
    loc_xy = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    enkf = EnsembleKalmanFilter(
        f=lambda x: x,
        h=lambda x: x[:2],
        Q_sqrt=0.1 * np.eye(n),
        R=0.1 * np.eye(m),
        ensemble=ens.copy(),
        localization_xy=loc_xy,
    )
    before = enkf.ensemble.copy()
    enkf.update(np.array([0.5, -0.5]))
    delta = enkf.ensemble - before
    # state dims 2,3 are fully tapered out -> the analysis cannot move them.
    assert np.allclose(delta[:, 2], 0.0, atol=1e-9)
    assert np.allclose(delta[:, 3], 0.0, atol=1e-9)
    # the observed dims do move.
    assert np.abs(delta[:, 0]).sum() > 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# RTS / EnKS / two-filter smoothers expose b_{t|T}
# ──────────────────────────────────────────────────────────────────────────────
def test_rts_smoother_reduces_variance_vs_filter():
    P0 = np.diag([1.0, 1.0, 1.0])
    e, F, _H = _ekf(P0)
    means, covs, pmeans, pcovs, Fs = [], [], [], [], []
    rng = _rng()
    for _ in range(6):
        e.predict()
        pmeans.append(e.x.copy())
        pcovs.append(e.P.copy())
        Fs.append(F)
        e.update(e.x[:2] + rng.normal(scale=0.1, size=2))
        means.append(e.x.copy())
        covs.append(e.P.copy())
    _xs, Ps = e.rts_smooth(means, covs, pmeans, pcovs, Fs)
    # the smoother variance at an early step is no larger than the filter's.
    assert np.trace(Ps[0]) <= np.trace(covs[0]) + 1e-9


def test_enks_returns_smoothed_ensembles():
    rng = _rng()
    T, M, n = 4, 30, 2
    filt = [rng.normal(size=(M, n)) for _ in range(T)]
    pred = [rng.normal(size=(M, n)) for _ in range(T)]
    sm = ensemble_kalman_smoother(filt, pred)
    assert len(sm) == T
    assert sm[-1].shape == (M, n)


def test_rbpf_two_filter_mode_smoother_is_a_simplex():
    A = np.array([[0.8, 0.2], [0.3, 0.7]])
    filtered = [np.array([0.9, 0.1]), np.array([0.4, 0.6]), np.array([0.2, 0.8])]
    sm = rbpf_smooth_modes(filtered, A)
    assert len(sm) == 3
    for g in sm:
        assert abs(float(g.sum()) - 1.0) < 1e-9
        assert np.all(g >= 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Amortized variational filter: ELBO + smoother distillation, delta/tau explicit
# ──────────────────────────────────────────────────────────────────────────────
def test_amortized_filter_emits_full_posterior_and_feeds_delta_tau():
    p = AmortizedFilterParams.build(in_dim=4, d=2, n_modes=4, n_haz=6, seed=0)
    f = AmortizedVariationalFilter(p)
    ys = [np.array([0.1, 0.2]), np.array([0.0, 0.0]), np.array([0.5, -0.3])]
    posts = f.filter_stream(ys, deltas=[1.0, 0.0, 1.0], taus=[0.0, 30.0, 0.0])
    assert len(posts) == 3
    q = posts[0]
    assert abs(float(q.mode_probs.sum()) - 1.0) < 1e-9
    assert q.hazard_probs.shape == (6,)
    assert 0.0 <= q.abstain_prob <= 1.0
    # a missing step (delta=0) yields a non-degenerate (positive-variance) mean.
    assert np.all(np.isfinite(posts[1].mean()))


def test_amortized_filter_elbo_and_distillation_are_finite():
    p = AmortizedFilterParams.build(in_dim=4, d=2, seed=1)
    f = AmortizedVariationalFilter(p)
    ys = [np.array([0.1, 0.2]), np.array([0.3, -0.1]), np.array([0.5, -0.3])]
    elbo = f.filtering_elbo(ys, deltas=[1.0, 1.0, 0.0])
    assert np.isfinite(elbo)
    posts = f.filter_stream(ys)
    targets = [posts[t].mean() + 0.01 for t in range(3)]
    loss = f.smoother_distillation_loss(ys, targets)
    assert np.isfinite(loss) and loss >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# CVaR / OCE conformal risk control + one-sided bundle residual-risk head
# ──────────────────────────────────────────────────────────────────────────────
def test_conformal_risk_controller_controls_mean_loss():
    rng = _rng()
    lambdas = np.linspace(0.0, 1.0, 20)
    # loss non-increasing in lambda (larger threshold = more conservative).
    base = rng.random((200, 1))
    losses = np.clip(base - lambdas[None, :], 0.0, 1.0)
    crc = ConformalRiskController(alpha=0.1, loss_bound=1.0).fit(lambdas, losses)
    lam = crc.threshold()
    # deployed mean loss at lambda_hat is at/below the target (up to fs slack).
    j = int(np.argmin(np.abs(lambdas - lam)))
    assert float(losses[:, j].mean()) <= 0.1 + 1e-6


def test_cvar_dominates_mean_and_oce_is_finite():
    x = _rng().random(500)
    assert cvar_from_losses(x, 0.9) >= float(x.mean())
    assert np.isfinite(oce_from_losses(x, 0.9))


def test_bundle_residual_risk_head_is_one_sided_upper():
    rng = _rng()
    rhat = rng.uniform(0.0, 0.3, size=500)
    rtrue = rhat + np.abs(rng.normal(scale=0.05, size=500))
    head = BundleResidualRiskHead(alpha=0.05).fit(rtrue, rhat)
    ub = head.upper(0.2)
    assert ub >= 0.2  # one-sided: bound never below the point estimate


# ──────────────────────────────────────────────────────────────────────────────
# BC-ACI one-sided / floored + coverage coupling
# ──────────────────────────────────────────────────────────────────────────────
def test_bcaci_one_sided_bias_is_floored_and_widens():
    cal = BCACICalibrator(one_sided=True, bias_floor=0.0)
    scores = np.linspace(0.0, 1.0, 50)
    r0 = cal.radius(scores)
    for _ in range(30):
        cal.update(covered=False, residual=0.4)  # persistent positive drift
    assert cal.bias >= 0.0  # floored one-sided
    r1 = cal.radius(scores)
    # the one-sided de-bias widens (never tightens) the safe radius under drift.
    assert r1 >= r0


# ──────────────────────────────────────────────────────────────────────────────
# Selective: m==n_cert coverage edge + P3 severe-shift certificate
# ──────────────────────────────────────────────────────────────────────────────
def test_coverage_lower_bound_is_finite_when_all_accepted():
    # every certification point accepted (m == n_cert): the old code degenerated.
    scores = np.ones(30)
    failures = np.zeros(30)
    cert = certify(scores, failures, 0.0, coverage_target=0.5)
    assert cert.n_accepted == cert.n_cert
    assert np.isfinite(cert.coverage_lower) and cert.coverage_lower > 0.0


def test_severe_shift_certificate_inflates_and_flags():
    # 0 failures over a large accept set -> a tight base accepted-risk upper.
    scores = _rng().random(400)
    cert = certify(scores, np.zeros(400), 0.0, risk_target=0.05, coverage_target=0.5)
    target = float(cert.accepted_risk_upper) + 0.05  # a target the base clears
    ok = severe_shift_certificate(cert, tv_bound=0.0, risk_target=target)
    bad = severe_shift_certificate(cert, tv_bound=0.5, risk_target=target)
    assert bad.shifted_risk_upper >= ok.shifted_risk_upper
    # zero shift holds (base clears target); a severe TV shift inflates past it.
    assert ok.holds_under_shift and not bad.holds_under_shift
