# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Unit tests for the conformal calibration stack (doc L).

Each calibrator turns a base predictor + an exchangeable calibration set into a
*validity-restored* predictor with a finite-sample coverage guarantee. Under
exchangeability the empirical coverage on a held-out set should sit close to the
nominal ``1 - alpha`` level — these tests check exactly that, with generous bands
to stay robust to sampling noise while still catching a broken calibrator (which
would produce coverage far from nominal).
"""

from __future__ import annotations

import numpy as np

from gpu_center_world_model.uncertainty import (
    ConformalizedQuantileRegressor,
    OneSidedUpperConformal,
    SplitConformalRegressor,
    conformal_rank,
    empirical_coverage,
    one_sided_coverage,
    split_conformal_quantile,
)

N = 4000
ALPHA = 0.1


def _rng():
    return np.random.default_rng(20260614)


# ──────────────────────────────────────────────────────────────────────────────
# Finite-sample primitives
# ──────────────────────────────────────────────────────────────────────────────
def test_conformal_rank_formula():
    # ceil((n+1)(1-alpha)).
    assert conformal_rank(99, 0.1) == 90
    assert conformal_rank(100, 0.1) == 91


def test_split_quantile_infinite_when_undersized():
    # alpha smaller than 1/(n+1) cannot be certified with a finite radius.
    assert np.isinf(split_conformal_quantile(np.array([0.0, 1.0]), alpha=0.01))


def test_split_quantile_is_an_order_statistic():
    scores = np.arange(1.0, 101.0)  # 1..100
    q = split_conformal_quantile(scores, alpha=0.1)
    assert q == 91.0  # the 91st order statistic


# ──────────────────────────────────────────────────────────────────────────────
# Split conformal regression: ~ nominal two-sided coverage
# ──────────────────────────────────────────────────────────────────────────────
def test_split_conformal_marginal_coverage_near_nominal():
    rng = _rng()
    mu_cal = rng.normal(size=N)
    y_cal = mu_cal + rng.normal(scale=1.0, size=N)
    cal = SplitConformalRegressor(alpha=ALPHA).fit(y_cal, mu_cal)

    mu_test = rng.normal(size=N)
    y_test = mu_test + rng.normal(scale=1.0, size=N)
    lo, hi = cal.intervals(mu_test)
    cov = empirical_coverage(y_test, lo, hi)
    assert abs(cov - (1.0 - ALPHA)) < 0.03


def test_split_conformal_coverage_is_at_least_nominal_minus_slack():
    # Conformal guarantees coverage >= 1 - alpha marginally; allow small slack.
    rng = _rng()
    mu_cal = rng.normal(size=N)
    y_cal = mu_cal + rng.standard_t(df=5, size=N)  # heavier tails
    cal = SplitConformalRegressor(alpha=ALPHA).fit(y_cal, mu_cal)
    mu_test = rng.normal(size=N)
    y_test = mu_test + rng.standard_t(df=5, size=N)
    lo, hi = cal.intervals(mu_test)
    assert empirical_coverage(y_test, lo, hi) >= (1.0 - ALPHA) - 0.03


# ──────────────────────────────────────────────────────────────────────────────
# CQR: ~ nominal coverage with an adaptive base quantile model
# ──────────────────────────────────────────────────────────────────────────────
def test_cqr_marginal_coverage_near_nominal():
    rng = _rng()
    # Heteroscedastic truth; the "base" quantile model is a fixed-width band that
    # CQR then conformalizes to restore validity.
    x_cal = rng.uniform(0.0, 1.0, size=N)
    sigma_cal = 0.2 + x_cal
    y_cal = rng.normal(scale=sigma_cal)
    # naive base 80% band assuming sigma=1 (deliberately miscalibrated)
    q_lo_cal = -1.28 * np.ones(N)
    q_hi_cal = 1.28 * np.ones(N)
    cqr = ConformalizedQuantileRegressor(alpha=ALPHA).fit(y_cal, q_lo_cal, q_hi_cal)

    x_t = rng.uniform(0.0, 1.0, size=N)
    sigma_t = 0.2 + x_t
    y_t = rng.normal(scale=sigma_t)
    lo, hi = cqr.intervals(-1.28 * np.ones(N), 1.28 * np.ones(N))
    cov = empirical_coverage(y_t, lo, hi)
    assert abs(cov - (1.0 - ALPHA)) < 0.03


# ──────────────────────────────────────────────────────────────────────────────
# One-sided upper conformal: the safety-gate primitive
# ──────────────────────────────────────────────────────────────────────────────
def test_one_sided_upper_coverage_near_nominal():
    alpha = 0.05
    rng = _rng()
    r_hat_cal = rng.uniform(0.0, 1.0, size=N)
    r_true_cal = r_hat_cal + rng.normal(scale=0.1, size=N)
    cal = OneSidedUpperConformal(alpha=alpha).fit(r_true_cal, r_hat_cal)

    r_hat_t = rng.uniform(0.0, 1.0, size=N)
    r_true_t = r_hat_t + rng.normal(scale=0.1, size=N)
    upper = cal.uppers(r_hat_t)
    cov = one_sided_coverage(r_true_t, upper)
    assert cov >= (1.0 - alpha) - 0.02
    assert abs(cov - (1.0 - alpha)) < 0.03


def test_one_sided_gate_is_monotone_in_epsilon():
    # passes() must be monotone: a looser epsilon never rejects what a tighter one
    # accepts (the chance-constraint gate semantics, doc L E17).
    rng = _rng()
    r_hat_cal = rng.uniform(size=N)
    r_true_cal = r_hat_cal + rng.normal(scale=0.05, size=N)
    cal = OneSidedUpperConformal(alpha=0.05).fit(r_true_cal, r_hat_cal)
    r_hat = 0.4
    tight = cal.passes(r_hat, epsilon=0.3)
    loose = cal.passes(r_hat, epsilon=0.9)
    assert (not tight) or loose


def test_empty_calibration_yields_infinite_radius():
    # Honest answer with no data: no finite interval carries the guarantee.
    cal = SplitConformalRegressor(alpha=ALPHA).fit(np.zeros(0), np.zeros(0))
    lo, hi = cal.interval(0.0)
    assert np.isinf(hi - lo)
