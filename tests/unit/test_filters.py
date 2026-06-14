# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Filter-consistency validity lane (doc T, UNC-FILTER-COMPLETENESS).

A Bayesian filter is *consistent* when the covariance it reports matches the
errors it makes (doc T E26/E27): the per-step Normalised Innovation Squared
``ν_t^T S_t^{-1} ν_t`` is ``χ²(dim_obs)`` and the Normalised Estimation Error
Squared ``e_t^T P_t^{-1} e_t`` is ``χ²(dim_state)``. A filter that invents
observability collapses ``P`` and becomes overconfident — its NIS/NEES then sit
*above* the consistency band (doc T C11). These tests build a correctly-specified
linear-Gaussian system, run the square-root EKF / UKF / EnKF, and assert:

* a correctly-specified filter lands *inside* the χ² consistency band;
* an over-confident filter (deflated ``R``) lands *above* it (caught);
* the EnKF Gaspari-Cohn cross-covariance localization is correct (state×obs);
* observability flooring lifts variance in weakly-observable directions only.

The square-root array / Potter measurement updates themselves are pinned against
the textbook KF in ``test_filters_completeness.py``; here we exercise the
*statistical* consistency the planner actually relies on.
"""

from __future__ import annotations

import numpy as np

from gpu_center_world_model.uncertainty import (
    EnsembleKalmanFilter,
    SquareRootEKF,
    SquareRootUKF,
    average_nees,
    average_nis,
    gaspari_cohn,
    nees,
    rank_histogram,
    spread_skill_ratio,
)
from gpu_center_world_model.uncertainty.diagnostics import ObservabilityMonitor
from gpu_center_world_model.uncertainty.filters import _qr_factor, _safe_cholesky

# A simple stable linear-Gaussian system: 3 state dims, 2 observed.
_F = np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.1], [0.0, 0.0, 0.95]])
_H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_Q = 0.01 * np.eye(3)
_R = 0.05 * np.eye(2)
_N, _M = 3, 2


def _rng(seed: int = 20260614):
    return np.random.default_rng(seed)


def _fresh_ekf(R: np.ndarray) -> SquareRootEKF:
    e = SquareRootEKF(
        f=lambda z: _F @ z,
        h=lambda z: _H @ z,
        Q=_Q,
        R=R,
        x=np.zeros(_N),
        F_jac=lambda z: _F,
        H_jac=lambda z: _H,
    )
    e.S = _qr_factor(_safe_cholesky(0.5 * np.eye(_N)).T)
    return e


def _consistency_samples(filt_R: np.ndarray, seed: int):
    """Run many *independent* trajectories; take one (NIS, NEES) per run.

    The data is always generated with the *true* measurement noise ``_R``; the
    filter may use a different ``filt_R`` (a misspecification). One sample per
    trajectory keeps the samples independent so the χ² band — which assumes
    independence — is well calibrated (within-trajectory NIS are serially
    correlated and would spuriously narrow the band).
    """
    rng = _rng(seed)
    sqQ = np.linalg.cholesky(_Q)
    sqR_true = np.linalg.cholesky(_R)  # ground-truth measurement noise
    nis_vals: list[float] = []
    nees_vals: list[float] = []
    for _ in range(600):
        x = rng.normal(0.0, 0.5, _N)
        e = _fresh_ekf(filt_R)
        last_nis = 0.0
        last_nees = 0.0
        for _t in range(25):
            x = _F @ x + sqQ @ rng.normal(size=_N)
            e.predict()
            # Pre-update innovation covariance S = H P H^T + R (doc T E26), using
            # the *filter's* believed R (deflated when overconfident).
            s_innov = _H @ e.P @ _H.T + filt_R
            y = _H @ x + sqR_true @ rng.normal(size=_M)
            nu = y - _H @ e.x
            last_nis = float(nu @ np.linalg.solve(s_innov, nu))
            e.update(y)
            last_nees = nees(x - e.x, e.P)
        nis_vals.append(last_nis)
        nees_vals.append(last_nees)
    return nis_vals, nees_vals


# ──────────────────────────────────────────────────────────────────────────────
# NIS / NEES consistency bands (doc T E26 / E27)
# ──────────────────────────────────────────────────────────────────────────────
def test_wellspecified_ekf_is_nis_and_nees_consistent():
    nis_vals, nees_vals = _consistency_samples(_R, seed=1)
    band_nis = average_nis(nis_vals, dim=_M)
    band_nees = average_nees(nees_vals, dim=_N)
    # A correctly-specified filter's averaged NIS/NEES sit inside the χ² band.
    assert band_nis.consistent, (band_nis.mean_stat, band_nis.lower, band_nis.upper)
    assert band_nees.consistent, (band_nees.mean_stat, band_nees.lower, band_nees.upper)
    # Per-step NIS ~ chi2(dim_obs) so the mean is ~ dim; NEES ~ dim_state.
    assert abs(band_nis.mean_stat - _M) < 0.4
    assert abs(band_nees.mean_stat - _N) < 0.5


def test_overconfident_filter_violates_the_nis_band_from_above():
    # Deflated R makes the filter trust measurements too much -> innovations are
    # large relative to the reported S, so NIS sits *above* the consistency band.
    nis_bad, _ = _consistency_samples(0.001 * np.eye(_M), seed=2)
    band = average_nis(nis_bad, dim=_M)
    assert not band.consistent
    assert band.mean_stat > band.upper  # over-confidence => over-band, not under


def test_consistency_band_is_two_sided_and_brackets_dimension():
    # The χ² band for an averaged statistic must bracket the per-step dof mean.
    band = average_nis([float(_M)] * 500, dim=_M)
    assert band.lower < _M < band.upper
    assert band.consistent  # a stream pinned exactly at the dof is consistent


# ──────────────────────────────────────────────────────────────────────────────
# Square-root EKF keeps P SPD across a full filter run (no collapse)
# ──────────────────────────────────────────────────────────────────────────────
def test_sqrt_ekf_keeps_covariance_spd_over_a_run():
    rng = _rng(3)
    sqQ = np.linalg.cholesky(_Q)
    sqR = np.linalg.cholesky(_R)
    e = _fresh_ekf(_R)
    x = rng.normal(0.0, 0.5, _N)
    for _t in range(40):
        x = _F @ x + sqQ @ rng.normal(size=_N)
        e.predict()
        y = _H @ x + sqR @ rng.normal(size=_M)
        e.update(y)
        eig = np.linalg.eigvalsh(e.P)
        assert np.all(eig > 0.0)  # square-root form is SPD by construction
        assert np.allclose(e.P, e.P.T)


# ──────────────────────────────────────────────────────────────────────────────
# Square-root UKF: a consistent unscented filter on the same system
# ──────────────────────────────────────────────────────────────────────────────
def test_sqrt_ukf_tracks_and_stays_consistent_on_linear_system():
    rng = _rng(4)
    sqQ = np.linalg.cholesky(_Q)
    sqR = np.linalg.cholesky(_R)
    nis_vals: list[float] = []
    for _ in range(400):
        x = rng.normal(0.0, 0.5, _N)
        u = SquareRootUKF(
            f=lambda z: _F @ z,
            h=lambda z: _H @ z,
            Q=_Q,
            R=_R,
            x=np.zeros(_N),
            P=0.5 * np.eye(_N),
        )
        last = 0.0
        for _t in range(25):
            x = _F @ x + sqQ @ rng.normal(size=_N)
            u.predict()
            s_innov = _H @ u.P @ _H.T + _R
            y = _H @ x + sqR @ rng.normal(size=_M)
            nu = y - _H @ u.x
            last = float(nu @ np.linalg.solve(s_innov, nu))
            u.update(y)
        nis_vals.append(last)
    band = average_nis(nis_vals, dim=_M)
    # On a linear-Gaussian system the UKF is exact, so it is NIS-consistent.
    assert band.consistent, (band.mean_stat, band.lower, band.upper)


# ──────────────────────────────────────────────────────────────────────────────
# EnKF Gaspari-Cohn cross-covariance localization correctness (doc T A9)
# ──────────────────────────────────────────────────────────────────────────────
def test_gaspari_cohn_taper_from_topology_distance_is_correct():
    # Built from a topology graph-distance row: identity at 0, compact support.
    d = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    g = gaspari_cohn(d, length_scale=1.0)
    assert g[0] == 1.0
    assert np.all(np.diff(g) <= 1e-12)  # monotone non-increasing in distance
    assert g[-1] == 0.0  # zero beyond 2*length_scale
    assert np.all((g >= 0.0) & (g <= 1.0))


def test_enkf_localization_zeroes_far_cross_covariance_updates():
    rng = _rng(5)
    n, m, members = 4, 2, 80
    ens = rng.normal(size=(members, n))
    # Only state dims 0,1 are local to the 2 observations; dims 2,3 fully tapered.
    loc_xy = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    enkf = EnsembleKalmanFilter(
        f=lambda z: z,
        h=lambda z: z[:2],
        Q_sqrt=0.1 * np.eye(n),
        R=0.1 * np.eye(m),
        ensemble=ens.copy(),
        localization_xy=loc_xy,
        rng=_rng(6),
    )
    before = enkf.ensemble.copy()
    enkf.update(np.array([0.8, -0.8]))
    delta = enkf.ensemble - before
    # Tapered-out state dims cannot be moved by the analysis (no spurious update).
    assert np.allclose(delta[:, 2], 0.0, atol=1e-9)
    assert np.allclose(delta[:, 3], 0.0, atol=1e-9)
    # The local (observed) dims are updated.
    assert np.abs(delta[:, 0]).sum() > 1e-6


def test_enkf_inflation_widens_spread_each_predict():
    rng = _rng(7)
    n, members = 3, 60
    enkf = EnsembleKalmanFilter(
        f=lambda z: z,  # static dynamics so growth is pure inflation
        h=lambda z: z,
        Q_sqrt=np.zeros((n, n)),  # no process noise -> isolate inflation
        R=0.1 * np.eye(n),
        ensemble=rng.normal(size=(members, n)),
        inflation=1.05,
        rng=_rng(8),
    )
    s0 = enkf.spread()
    enkf.predict()
    # Multiplicative inflation about the mean strictly grows the spread (A9).
    assert enkf.spread() > s0


# ──────────────────────────────────────────────────────────────────────────────
# Observability flooring lifts variance only in weak directions (doc T C11)
# ──────────────────────────────────────────────────────────────────────────────
def test_observability_floor_raises_only_weak_direction_variance():
    e = _fresh_ekf(_R)
    e.S = _qr_factor(_safe_cholesky(np.diag([1.0, 1.0, 1e-6])).T)
    mon = ObservabilityMonitor(tol=1e-3)
    report = mon.analyze_matrix(_H)  # H observes dims 0,1; dim 2 is weak
    assert report.rank == 2 and not report.full_rank
    assert report.weak_directions.shape[1] >= 1
    floor = 0.5
    var_obs_before = float(np.array([1, 0, 0]) @ e.P @ np.array([1, 0, 0]))
    e.floor_weak_directions(report.weak_directions, floor)
    weak = np.array([0.0, 0.0, 1.0])
    # The weakly-observed (3rd) direction is floored up to at least `floor`.
    assert float(weak @ e.P @ weak) >= floor - 1e-9
    # A well-observed direction is not inflated by the weak-direction floor.
    var_obs_after = float(np.array([1, 0, 0]) @ e.P @ np.array([1, 0, 0]))
    assert var_obs_after <= var_obs_before + 1e-6


def test_observability_floor_scales_inversely_with_observability():
    mon = ObservabilityMonitor(tol=1e-6)
    rep_strong = mon.analyze_gramian(np.diag([10.0, 10.0, 1.0]))
    rep_weak = mon.analyze_gramian(np.diag([10.0, 10.0, 1e-3]))
    # Weaker smallest-direction observability => larger required covariance floor.
    assert mon.covariance_floor(rep_weak) > mon.covariance_floor(rep_strong)


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble degeneration diagnostics react to spread (doc T)
# ──────────────────────────────────────────────────────────────────────────────
def test_rank_histogram_flat_for_calibrated_ensemble():
    rng = _rng(9)
    T, members = 4000, 9
    ensemble = rng.normal(size=(T, members))
    truth = rng.normal(size=T)  # truth drawn from the same law -> calibrated
    hist = rank_histogram(ensemble, truth)
    assert hist.sum() == T
    # A calibrated ensemble gives a ~flat rank histogram (no rank under-populated
    # to zero); the spread-skill ratio is ~1.
    assert np.all(hist > 0)
    ssr = spread_skill_ratio(ensemble, truth)
    assert 0.8 < ssr < 1.25


def test_spread_skill_flags_underspread_ensemble():
    rng = _rng(10)
    T, members = 2000, 9
    # Under-spread: ensemble too tight relative to its error from truth.
    ensemble = 0.2 * rng.normal(size=(T, members))
    truth = rng.normal(size=T)
    assert spread_skill_ratio(ensemble, truth) < 0.8  # overconfident


# ──────────────────────────────────────────────────────────────────────────────
# Square-root EKF Potter update == array update == textbook KF (doc T E26)
# ──────────────────────────────────────────────────────────────────────────────
def test_potter_and_array_updates_agree_with_each_other():
    """The Potter sequential update and the array update produce the same posterior.

    Both are square-root realisations of the *same* measurement update, so on the
    same prior + observation they must agree (and equal the textbook Joseph-form KF).
    Pinning them against each other guards against a regression in either path.
    """
    P0 = np.diag([1.0, 2.0, 0.5])
    y = np.array([0.7, -0.4])
    e_arr = _fresh_ekf(_R)
    e_arr.S = _qr_factor(_safe_cholesky(P0).T)
    e_pot = _fresh_ekf(_R)
    e_pot.S = _qr_factor(_safe_cholesky(P0).T)
    x0 = e_arr.x.copy()

    e_arr.update(y)
    e_pot.potter_update(y)
    assert np.allclose(e_arr.x, e_pot.x, atol=1e-6)
    assert np.allclose(e_arr.P, e_pot.P, atol=1e-6)

    # ... and both equal the textbook KF posterior.
    S_innov = _H @ P0 @ _H.T + _R
    K = P0 @ _H.T @ np.linalg.inv(S_innov)
    x_ref = x0 + K @ (y - _H @ x0)
    P_ref = (np.eye(_N) - K @ _H) @ P0
    assert np.allclose(e_arr.x, x_ref, atol=1e-6)
    assert np.allclose(e_arr.P, P_ref, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# EnKF analysis is statistically consistent on a linear-Gaussian system (doc T)
# ──────────────────────────────────────────────────────────────────────────────
def test_enkf_is_nis_consistent_on_linear_system():
    """The stochastic EnKF analysis is NIS-consistent on a linear-Gaussian model.

    The EnKF reports a *sample* innovation covariance; on a correctly-specified
    linear-Gaussian system with a large-enough ensemble its averaged NIS lands
    inside the χ² consistency band (doc T E26) — the same statistical validity the
    EKF/UKF satisfy. A large member count keeps the Monte-Carlo sampling error from
    inflating the innovation statistic out of band.
    """
    rng = _rng(11)
    sqQ = np.linalg.cholesky(_Q)
    sqR = np.linalg.cholesky(_R)
    members = 400
    nis_vals: list[float] = []
    for trial in range(150):
        x = rng.normal(0.0, 0.5, _N)
        ens = x + rng.normal(scale=0.5, size=(members, _N))
        enkf = EnsembleKalmanFilter(
            f=lambda z: (_F @ z),
            h=lambda z: (_H @ z),
            Q_sqrt=sqQ,
            R=_R,
            ensemble=ens,
            inflation=1.0,
            rng=_rng(1000 + trial),
        )
        last = 0.0
        for _t in range(20):
            x = _F @ x + sqQ @ rng.normal(size=_N)
            enkf.predict()
            # innovation covariance S = H P_f H^T + R from the *forecast* ensemble.
            P_f = enkf.covariance()
            s_innov = _H @ P_f @ _H.T + _R
            y = _H @ x + sqR @ rng.normal(size=_M)
            nu = y - _H @ enkf.mean
            last = float(nu @ np.linalg.solve(s_innov, nu))
            enkf.update(y)
        nis_vals.append(last)
    band = average_nis(nis_vals, dim=_M)
    # a well-specified, well-sampled EnKF is NIS-consistent (inside the χ² band).
    assert band.consistent, (band.mean_stat, band.lower, band.upper)
