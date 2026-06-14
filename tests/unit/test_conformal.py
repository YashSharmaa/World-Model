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
    ACICalibrator,
    ConformalizedQuantileRegressor,
    ConformalRiskController,
    MondrianConformal,
    OneSidedUpperConformal,
    SplitConformalRegressor,
    WeightedConformal,
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


# ──────────────────────────────────────────────────────────────────────────────
# UNC-CALIBRATION-FIX: consolidated one-sided calibrator (no divergent clamp)
# ──────────────────────────────────────────────────────────────────────────────
def test_model_calibration_onesided_has_no_divergent_clamp():
    """The model-side OneSidedUpperConformal must not fabricate ``1 - pred``.

    Before the fix, an *unfitted* calibrator returned ``pred + (1 - pred)`` (i.e.
    it pinned the upper bound to 1.0 regardless of the calibration data). After
    consolidating on the canonical uncertainty/conformal.py, an unfitted bound is
    just the (clipped) point estimate, and a *fitted* bound is ``pred + q⁺`` from
    the real conformal quantile.
    """
    from gpu_center_world_model.model.calibration.conformal import (
        OneSidedUpperConformal as ModelOneSided,
    )

    cal = ModelOneSided(alpha=0.05)
    # unfitted: never the old 1.0 clamp; the bound equals the (clipped) pred.
    assert cal.upper(0.2) == 0.2
    # fitted on calibration data: a real additive conformal margin.
    rng = np.random.default_rng(0)
    r_hat = rng.uniform(0.0, 0.5, size=2000)
    r_true = r_hat + np.abs(rng.normal(scale=0.1, size=2000))
    cal.fit(r_hat, r_true)
    ub = cal.upper(0.2)
    assert 0.2 < ub <= 1.0  # widened by the conformal quantile, clipped to [0,1]
    # one-sided coverage on a fresh draw is at/above nominal.
    r_hat_t = rng.uniform(0.0, 0.5, size=2000)
    r_true_t = r_hat_t + np.abs(rng.normal(scale=0.1, size=2000))
    uppers = np.array([cal.upper(float(p)) for p in r_hat_t])
    assert one_sided_coverage(r_true_t, uppers) >= 0.93


def test_calibration_target_is_branch_delta_not_predicted_median():
    """``_fit_calibration`` must calibrate against the realized branch KPI delta.

    The old code used the model's own predicted median as the conformal target
    (self-referential ⇒ vacuous coverage). The fixed path reads the labelled
    branch-simulated counterfactual delta from ``_kpi_delta_labels``. We assert
    that with labelled futures the calibrator becomes fitted (a finite CQR
    correction exists), which can only happen if real targets — not the model's
    own median (whose residual is identically zero) — were used.
    """
    from gpu_center_world_model.model.train import (
        TrainConfig,
        _fit_calibration,
        make_smoke_windows,
    )
    from gpu_center_world_model.model.world_model import ResidualWorldModel

    model = ResidualWorldModel.build(n_ensemble=1)
    windows = make_smoke_windows(16, seed=0)
    _fit_calibration(model, windows, TrainConfig())
    cal = model.calibrator
    # at least one CQR calibrator fitted to a finite correction.
    fitted = [c for c in cal.cqr.values() if np.isfinite(c.correction_)]
    assert fitted, "calibration produced no fitted calibrators from labelled futures"
    # The CQR score is max(q_lo - y, y - q_hi) on the *realized branch delta* y.
    # If y were the model's own median (the old self-referential target), the
    # score would collapse to exactly -(band half-width) = -1.64·spread for EVERY
    # window of a channel (symmetric band), giving one identical correction per
    # channel. A real branch delta makes the per-window scores vary, so distinct
    # channels with different realized deltas get *different* corrections.
    corr = {k: c.correction_ for k, c in cal.cqr.items() if np.isfinite(c.correction_)}
    assert len(set(round(v, 6) for v in corr.values())) > 1, (
        "all corrections identical — calibration looks self-referential"
    )


def test_calibration_fits_per_slice_mondrian_with_backoff():
    """The doc-L 5-tuple per-slice Mondrian must be fitted with back-off."""
    from gpu_center_world_model.model.train import (
        TrainConfig,
        _calibration_slice_key,
        _fit_calibration,
        make_smoke_windows,
    )
    from gpu_center_world_model.model.world_model import ResidualWorldModel

    windows = make_smoke_windows(20, seed=1)
    # the slice key is the doc-L 5-tuple (5 pipe-separated facets).
    key = _calibration_slice_key(windows[0], 1800)
    assert key.count("|") == 4
    model = ResidualWorldModel.build(n_ensemble=1)
    _fit_calibration(model, windows, TrainConfig())
    # a Mondrian map is attached and resolves a group-or-pooled width without
    # KeyError on an unseen slice (hierarchical back-off, doc L T3).
    assert model.calibrator.mondrian, "no per-slice Mondrian calibrators fitted"
    mond = next(iter(model.calibrator.mondrian.values()))
    lo, hi = mond.interval("unseen|h1800|prioX|none|clean", -1.0, 1.0)
    # back-off resolves a finite interval from the pooled conformal quantile
    # (CQR may *tighten* an over-wide base band, so lo/hi ordering is not pinned).
    assert np.isfinite(lo) and np.isfinite(hi)
    # the same fitted quantile applies to every group (back-off is consistent).
    lo2, hi2 = mond.interval("another|h1800|fooX|none|clean", -1.0, 1.0)
    assert (lo2, hi2) == (lo, hi)


# ──────────────────────────────────────────────────────────────────────────────
# ACI long-run coverage under drift (doc L T temporal_drift_and_aci / W2)
# ──────────────────────────────────────────────────────────────────────────────
def test_aci_long_run_coverage_tracks_target_under_drift():
    """ACI gives *long-run* (not one-shot) coverage near ``1 - alpha`` under drift.

    Doc L (E15, W2): ``alpha_{t+1} = clip(alpha_t + gamma(alpha_target - err_t))``
    drives the empirical long-run miss-rate toward ``alpha_target`` even when the
    base scale drifts. We stream a non-stationary process: the residual scale ramps
    up over time (so a fixed conformal radius would progressively under-cover). The
    ACI level adapts and the *realised* long-run coverage stays close to nominal.
    """
    rng = _rng()
    cal = ACICalibrator(alpha_target=ALPHA, gamma=0.02)
    base_scores = np.abs(rng.normal(scale=1.0, size=2000))  # absolute-residual scores
    covered_hist: list[bool] = []
    for t in range(6000):
        scale = 1.0 + 3.0 * (t / 6000.0)  # the residual scale drifts upward
        radius = cal.radius(base_scores)
        y_err = abs(rng.normal(scale=scale))  # this step's realised |residual|
        covered = y_err <= radius
        covered_hist.append(covered)
        cal.update(covered)
        # refresh the calibration scores to the current drifted scale (the base
        # split-conformal radius the wrapper sits on tracks the live distribution).
        base_scores = np.abs(rng.normal(scale=scale, size=2000))
    long_run_cov = float(np.mean(covered_hist[1000:]))  # discard burn-in
    assert abs(long_run_cov - (1.0 - ALPHA)) < 0.03
    # the adapted miscoverage level stayed inside its projection box.
    assert cal.alpha_min <= cal.alpha_t <= cal.alpha_max


def test_aci_widens_level_when_persistently_missing():
    # A wrapper that keeps missing must *raise* its acceptance (lower alpha_t ->
    # larger radius) to chase coverage back up (E15 negative-feedback direction).
    cal = ACICalibrator(alpha_target=0.1, gamma=0.05)
    a0 = cal.alpha_t
    for _ in range(50):
        cal.update(covered=False)  # persistent miss
    assert cal.alpha_t < a0  # lower miscoverage target => wider interval
    cal2 = ACICalibrator(alpha_target=0.1, gamma=0.05)
    for _ in range(50):
        cal2.update(covered=True)  # always covered => can tighten
    assert cal2.alpha_t > cal.alpha_t


# ──────────────────────────────────────────────────────────────────────────────
# Weighted conformal: target-domain marginal coverage under covariate shift (T5)
# ──────────────────────────────────────────────────────────────────────────────
def test_weighted_conformal_attains_target_coverage_under_covariate_shift():
    """Weighted conformal restores target-domain coverage; unweighted under-covers.

    Doc L T5 / E14: within a fixed action bucket the conditional law is unchanged
    (``P_t(Y|X) = P_c(Y|X)``) but the covariate marginal shifts. We build a
    heteroscedastic one-sided risk problem where the test distribution concentrates
    on the high-variance region. The IPW likelihood-ratio weights restore
    ``P_{P_t}{R <= U(X)} >= 1 - alpha``; an *unweighted* split conformal under the
    same shift under-covers (the validity actually being tested).
    """
    rng = _rng()
    alpha = 0.1
    n = 6000

    def draw(x):  # one-sided risk: R = r_hat(x) + heteroscedastic positive noise
        r_hat = 0.2 + 0.0 * x
        r_true = r_hat + np.abs(rng.normal(scale=0.1 + x, size=x.size))
        return r_hat, r_true

    # calibration: x ~ Uniform(0,1) (the source covariate law).
    x_cal = rng.uniform(0.0, 1.0, size=n)
    r_hat_cal, r_true_cal = draw(x_cal)
    # target law: x ~ Beta(5,1) concentrates on large x (the high-variance tail).
    # likelihood ratio w(x) = p_t(x) / p_c(x) = Beta(5,1) density = 5 x^4.
    weights = 5.0 * x_cal**4
    wc = WeightedConformal(alpha=alpha, mode="one_sided_upper").fit(
        r_true_cal, {"r_hat": r_hat_cal}, weights
    )

    # fresh *target-distributed* test points.
    x_t = rng.beta(5.0, 1.0, size=20000)
    r_hat_t, r_true_t = draw(x_t)
    q = wc._weighted_quantile(float(weights.mean()))
    upper_w = r_hat_t + q
    cov_w = one_sided_coverage(r_true_t, upper_w)
    assert cov_w >= (1.0 - alpha) - 0.02  # target-domain marginal coverage restored

    # the *unweighted* split-conformal quantile (equal weights) under-covers on the
    # shifted target law — this is exactly the failure weighting fixes.
    unweighted_q = split_conformal_quantile(r_true_cal - r_hat_cal, alpha)
    cov_unw = one_sided_coverage(r_true_t, r_hat_t + unweighted_q)
    assert cov_unw < cov_w  # weighting genuinely lifts coverage on the target law


def test_weighted_conformal_recovers_split_conformal_at_equal_weights():
    # With all weights equal, weighted conformal must reduce to plain split
    # conformal (no shift correction), up to the +inf test-point augmentation.
    rng = _rng()
    r_hat = rng.uniform(0.0, 0.5, size=2000)
    r_true = r_hat + np.abs(rng.normal(scale=0.1, size=2000))
    wc = WeightedConformal(alpha=ALPHA, mode="one_sided_upper").fit(
        r_true, {"r_hat": r_hat}, np.ones(2000)
    )
    q_w = wc._weighted_quantile(1.0)
    q_plain = split_conformal_quantile(r_true - r_hat, ALPHA)
    assert abs(q_w - q_plain) <= 0.02


# ──────────────────────────────────────────────────────────────────────────────
# Mondrian group-conditional coverage: per-group >= 1 - alpha (doc L T3 / P2)
# ──────────────────────────────────────────────────────────────────────────────
def test_mondrian_each_group_meets_nominal_coverage():
    """Each prespecified slice attains its own ``1 - alpha`` (group-conditional).

    Doc L T3 / P2: a *single pooled* quantile under-covers a high-variance group;
    the per-group Mondrian quantile restores per-group validity. We build three
    groups with very different residual scales and assert (a) the Mondrian per-group
    upper covers every group at ~nominal, and (b) the pooled quantile under-covers
    the highest-variance group (the failure Mondrian fixes).
    """
    rng = _rng()
    alpha = 0.1
    n_per = 4000
    scales = {"lo": 0.2, "mid": 1.0, "hi": 4.0}
    groups, r_hat, r_true = [], [], []
    for g, s in scales.items():
        rh = rng.uniform(0.0, 0.2, size=n_per)
        rt = rh + np.abs(rng.normal(scale=s, size=n_per))
        groups += [g] * n_per
        r_hat.append(rh)
        r_true.append(rt)
    groups = np.array(groups, dtype=object)
    r_hat = np.concatenate(r_hat)
    r_true = np.concatenate(r_true)

    mond = MondrianConformal(alpha=alpha, mode="one_sided_upper").fit(
        groups, r_true, {"r_hat": r_hat}
    )

    # fresh test draws per group; every group meets its own nominal coverage.
    for g, s in scales.items():
        rh_t = rng.uniform(0.0, 0.2, size=8000)
        rt_t = rh_t + np.abs(rng.normal(scale=s, size=8000))
        uppers = np.array([mond.upper(g, float(p)) for p in rh_t])
        cov_g = one_sided_coverage(rt_t, uppers)
        assert cov_g >= (1.0 - alpha) - 0.02, (g, cov_g)

    # a single pooled quantile under-covers the highest-variance "hi" group.
    rh_hi = rng.uniform(0.0, 0.2, size=8000)
    rt_hi = rh_hi + np.abs(rng.normal(scale=scales["hi"], size=8000))
    pooled_upper = rh_hi + mond.pooled_q
    assert one_sided_coverage(rt_hi, pooled_upper) < (1.0 - alpha)


def test_mondrian_backoff_uses_pooled_quantile_for_small_groups():
    # A group with < min_group calibration points is not fitted and backs off to
    # the pooled quantile (T3 hierarchical back-off), never KeyError.
    rng = _rng()
    big = np.array(["big"] * 200, dtype=object)
    tiny = np.array(["tiny"] * 5, dtype=object)
    groups = np.concatenate([big, tiny])
    r_hat = rng.uniform(0.0, 0.2, size=205)
    r_true = r_hat + np.abs(rng.normal(scale=0.3, size=205))
    mond = MondrianConformal(alpha=0.1, mode="one_sided_upper", min_group=30).fit(
        groups, r_true, {"r_hat": r_hat}
    )
    assert "big" in mond.quantiles_by_group
    assert "tiny" not in mond.quantiles_by_group  # too small -> not fitted
    # backs off to the pooled quantile (finite, no KeyError).
    assert mond.quantile_for("tiny") == mond.pooled_q
    assert np.isfinite(mond.quantile_for("never_seen_label"))


# ──────────────────────────────────────────────────────────────────────────────
# Conformal risk control: deployed mean loss controlled at the target (E18)
# ──────────────────────────────────────────────────────────────────────────────
def test_conformal_risk_controller_controls_long_run_mean_loss():
    rng = _rng()
    lambdas = np.linspace(0.0, 1.0, 25)
    # a bounded loss that is monotone non-increasing in the threshold lambda.
    base = rng.random((400, 1))
    losses = np.clip(base - lambdas[None, :], 0.0, 1.0)
    crc = ConformalRiskController(alpha=0.1, loss_bound=1.0).fit(lambdas, losses)
    lam = crc.threshold()
    # the deployed mean loss at lambda_hat respects the target risk (E18).
    j = int(np.argmin(np.abs(lambdas - lam)))
    assert float(losses[:, j].mean()) <= 0.1 + 1e-6
    # picking a *smaller* (less conservative) lambda would exceed the budget, so the
    # controller is at the binding edge (monotone RCPS selection).
    if j > 0:
        assert float(losses[:, j - 1].mean()) >= float(losses[:, j].mean()) - 1e-9
