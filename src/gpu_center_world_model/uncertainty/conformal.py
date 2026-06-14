# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Conformal calibrators with finite-sample coverage guarantees (doc L).

Every calibrator here turns a *base* model (point, quantile, or one-sided risk
predictor) plus an exchangeable calibration set into a *validity-restored*
predictor. The guarantee each one carries is stated in its docstring, exactly as
derived in doc L (theorems T1-T6). The shared backbone is the conformal
quantile of nonconformity scores at the finite-sample level
``ceil((n + 1)(1 - alpha)) / n``.

Critical project rule (doc L handoff): **calibrate only on action-conditioned
decision windows anchored by a pre-action state hash**; never pool observational
and interventional targets. The Mondrian / weighted / ACI wrappers exist so the
bucketing (action family x horizon x slice) is explicit and auditable.

Implemented:

* :func:`split_conformal_quantile` — the finite-sample level (E11).
* :class:`SplitConformalRegressor` — symmetric residual intervals (T1).
* :class:`ConformalizedQuantileRegressor` — CQR, the default for KPI deltas (T2).
* :class:`MondrianConformal` — group-conditional validity over prespecified
  slices, with hierarchical back-off for small groups (T3).
* :class:`OneSidedUpperConformal` — upper bounds for chance constraints, the
  primitive behind the safety gate and optimism-bias control (T4, T6).
* :func:`beta_corrected_level` — small-sample beta correction (P6).
* :class:`WeightedConformal` — covariate-shift coverage within a fixed action
  bucket (T5).
* :class:`ACICalibrator` / :class:`BCACICalibrator` — online ACI / bias-corrected
  ACI maintenance wrappers (E15).

All are pure NumPy. Distribution-free theory gives *marginal* (or group / target
marginal) coverage, not pointwise per-(x,a) guarantees — every docstring is
explicit about which.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ._betainc import beta_quantile, reg_incomplete_beta

_EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Finite-sample conformal quantile primitives
# ─────────────────────────────────────────────────────────────────────────────
def conformal_rank(n: int, alpha: float) -> int:
    """Return the 1-based order statistic ``ceil((n + 1)(1 - alpha))`` (doc L).

    This is the rank whose value is the conformal quantile. When it exceeds
    ``n`` (i.e. ``alpha < 1/(n+1)``) the interval is necessarily infinite and we
    signal that by returning ``n + 1``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    return math.ceil((n + 1) * (1.0 - alpha))


def split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile of nonconformity ``scores`` (E11).

    Returns ``+inf`` when ``n`` is too small to certify level ``alpha`` (the
    honest answer: no finite interval has the guarantee). Otherwise returns the
    ``ceil((n+1)(1-alpha))``-th smallest score.
    """
    s = np.sort(np.asarray(scores, dtype=float).ravel())
    n = s.size
    if n == 0:
        return math.inf
    k = conformal_rank(n, alpha)
    if k > n:
        return math.inf
    return float(s[k - 1])


def beta_corrected_level(n: int, alpha: float, delta: float) -> float:
    """Effective miscoverage giving "coverage >= 1-alpha w.p. >= 1-delta" (P6).

    Doc L's small-sample beta correction strengthens the deployment claim from a
    single-draw expectation to a probabilistic guarantee over the calibration
    draw itself. The coverage of split conformal with rank ``k`` is
    ``Beta(k, n + 1 - k)`` distributed; we pick the *largest* nominal level
    whose ``delta``-quantile of that Beta still clears ``1 - alpha``.

    Returns an adjusted ``alpha' <= alpha`` to feed back into
    :func:`split_conformal_quantile`. If even ``alpha' -> 0`` (use all of the
    calibration set) cannot meet the requirement, returns a small positive floor
    and the caller should treat the bound as best-effort.
    """
    target = 1.0 - alpha
    best = alpha
    # Search ranks downward from the un-corrected rank; larger rank == wider.
    for k in range(n, 0, -1):
        # coverage ~ Beta(k, n + 1 - k); its delta-quantile is the worst case.
        lower = beta_quantile(delta, k, n + 1 - k)
        if lower >= target:
            best = 1.0 - k / (n + 1)  # alpha' whose rank is k
            return float(max(best, _EPS))
    return float(max(1.0 / (n + 1), _EPS))


# ─────────────────────────────────────────────────────────────────────────────
# Split conformal regression (symmetric residuals)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class SplitConformalRegressor:
    """Split-conformal symmetric intervals around a point predictor (T1).

    Guarantee (doc L, T1): if the ``n`` calibration points and the test point
    are exchangeable, then ``P{Y in [mu - q, mu + q]} >= 1 - alpha`` where ``q``
    is the conformal quantile of absolute residuals ``S_i = |Y_i - mu(X_i)|``.
    Marginal coverage only. Use as the conservative fallback for near-symmetric
    residuals; prefer CQR for skewed operational deltas.
    """

    alpha: float = 0.10
    q: float = math.inf
    n_cal: int = 0
    delta: float | None = None  # if set, apply the beta correction (P6)

    def fit(self, y_cal: np.ndarray, mu_cal: np.ndarray) -> SplitConformalRegressor:
        scores = np.abs(np.asarray(y_cal, float) - np.asarray(mu_cal, float))
        self.n_cal = scores.size
        alpha = self.alpha
        if self.delta is not None:
            alpha = beta_corrected_level(self.n_cal, self.alpha, self.delta)
        self.q = split_conformal_quantile(scores, alpha)
        return self

    def interval(self, mu: float) -> tuple[float, float]:
        return (mu - self.q, mu + self.q)

    def intervals(self, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = np.asarray(mu, float)
        return mu - self.q, mu + self.q


# ─────────────────────────────────────────────────────────────────────────────
# Conformalized Quantile Regression (the default for KPI deltas)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ConformalizedQuantileRegressor:
    """CQR — conformalizes a base quantile model (T2; doc L default).

    Score (E12): ``S_i = max(q_L(X_i) - Y_i, Y_i - q_U(X_i))``; calibrated band
    ``[q_L(x) - q, q_U(x) + q]``. Guarantee: under exchangeability,
    ``P{Y in C(X)} >= 1 - alpha`` (marginal). CQR inherits the *adaptive width*
    of the base quantile model while restoring exact finite-sample validity,
    which is why doc L makes it the default for skewed, regime-sensitive KPI
    deltas.
    """

    alpha: float = 0.10
    q: float = math.inf
    n_cal: int = 0
    delta: float | None = None

    def fit(
        self,
        y_cal: np.ndarray,
        q_lo_cal: np.ndarray,
        q_hi_cal: np.ndarray,
    ) -> ConformalizedQuantileRegressor:
        y = np.asarray(y_cal, float)
        lo = np.asarray(q_lo_cal, float)
        hi = np.asarray(q_hi_cal, float)
        scores = np.maximum(lo - y, y - hi)
        self.n_cal = scores.size
        alpha = self.alpha
        if self.delta is not None:
            alpha = beta_corrected_level(self.n_cal, self.alpha, self.delta)
        self.q = split_conformal_quantile(scores, alpha)
        return self

    def interval(self, q_lo: float, q_hi: float) -> tuple[float, float]:
        return (q_lo - self.q, q_hi + self.q)

    def intervals(
        self, q_lo: np.ndarray, q_hi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(q_lo, float) - self.q, np.asarray(q_hi, float) + self.q


# ─────────────────────────────────────────────────────────────────────────────
# One-sided upper conformal (safety chance constraints, optimism bias)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class OneSidedUpperConformal:
    """One-sided upper bound ``U_alpha(x) = r_hat(x) + q^+`` (T4 / T6).

    Score (E13): ``S_i^+ = R_i - r_hat(X_i)``; upper conformal quantile
    ``q^+``. Guarantee: under exchangeability, ``P{R <= U_alpha(X)} >= 1 - alpha``
    (one-sided marginal). This is the correct conformal primitive for:

    * the **safety gate** (E17): admit ``a`` only if ``U_alpha_e(x, a) <= eps_e``
      for every residual safety risk ``e``; then
      ``P{certified-safe and R > eps} <= alpha``;
    * **optimism-bias control** (T6): with ``Z = excess risk vs fallback`` and
      gate ``U_alpha(x, a) <= 0`` we get ``P{deploy and Z > 0} <= alpha``.

    Distribution-free theory gives marginal control, not a per-(x,a) certificate.
    """

    alpha: float = 0.05
    q_plus: float = math.inf
    n_cal: int = 0
    delta: float | None = None

    def fit(self, r_true: np.ndarray, r_hat: np.ndarray) -> OneSidedUpperConformal:
        scores = np.asarray(r_true, float) - np.asarray(r_hat, float)
        self.n_cal = scores.size
        alpha = self.alpha
        if self.delta is not None:
            alpha = beta_corrected_level(self.n_cal, self.alpha, self.delta)
        self.q_plus = split_conformal_quantile(scores, alpha)
        return self

    def upper(self, r_hat: float) -> float:
        return r_hat + self.q_plus

    def uppers(self, r_hat: np.ndarray) -> np.ndarray:
        return np.asarray(r_hat, float) + self.q_plus

    def passes(self, r_hat: float, epsilon: float) -> bool:
        """Chance-constraint gate: ``True`` iff ``U_alpha(x) <= epsilon`` (E17)."""
        return self.upper(r_hat) <= epsilon


# ─────────────────────────────────────────────────────────────────────────────
# Mondrian (group-conditional) conformal
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class MondrianConformal:
    """Group-conditional conformal over prespecified slices (T3).

    For a slice label ``g = g(x)`` (doc L recommends
    ``(action family, horizon, workload family, topology/flavor family,
    source-health bucket)``) we compute a *separate* conformal quantile per
    group. Guarantee: under within-group exchangeability,
    ``P{Y in C^g(X) | G = g} >= 1 - alpha`` for every prespecified group.

    Full per-(x) conditional coverage is impossible distribution-free (doc L);
    group-conditional validity over a finite, prespecified partition is the
    achievable middle ground. When a group has too few calibration points
    (``n_g < min_group``) we **back off hierarchically** to a coarser parent
    label (or the pooled quantile), trading conditionality for a usable bound.
    """

    alpha: float = 0.10
    mode: str = "one_sided_upper"  # one_sided_upper | abs_residual | cqr
    min_group: int = 30
    quantiles_by_group: dict[object, float] = field(default_factory=dict)
    pooled_q: float = math.inf
    parent_of: dict[object, object] = field(default_factory=dict)
    delta: float | None = None

    def _score(self, y: np.ndarray, preds: dict[str, np.ndarray]) -> np.ndarray:
        if self.mode == "abs_residual":
            return np.abs(y - preds["mu"])
        if self.mode == "cqr":
            return np.maximum(preds["q_lo"] - y, y - preds["q_hi"])
        if self.mode == "one_sided_upper":
            return y - preds["r_hat"]
        raise ValueError(f"unknown Mondrian mode {self.mode!r}")

    def _q_from_scores(self, scores: np.ndarray) -> float:
        alpha = self.alpha
        if self.delta is not None and scores.size:
            alpha = beta_corrected_level(scores.size, self.alpha, self.delta)
        return split_conformal_quantile(scores, alpha)

    def fit(
        self,
        groups: np.ndarray,
        y_cal: np.ndarray,
        preds: dict[str, np.ndarray],
        parent_of: dict[object, object] | None = None,
    ) -> MondrianConformal:
        """Fit per-group quantiles; ``preds`` carries the mode's predictions."""
        groups = np.asarray(groups, dtype=object)
        y = np.asarray(y_cal, float)
        pred_arrays = {k: np.asarray(v, float) for k, v in preds.items()}
        all_scores = self._score(y, pred_arrays)
        self.pooled_q = self._q_from_scores(all_scores)
        self.parent_of = dict(parent_of or {})
        self.quantiles_by_group.clear()
        for g in np.unique(groups):
            mask = groups == g
            if mask.sum() < self.min_group:
                continue
            g_scores = all_scores[mask]
            self.quantiles_by_group[g] = self._q_from_scores(g_scores)
        return self

    def quantile_for(self, group: object) -> float:
        """Resolve a group's quantile, backing off to parent then pool (T3)."""
        seen: set[object] = set()
        g: object = group
        while g is not None and g not in seen:
            if g in self.quantiles_by_group:
                return self.quantiles_by_group[g]
            seen.add(g)
            g = self.parent_of.get(g)
        return self.pooled_q

    def upper(self, group: object, r_hat: float) -> float:
        """One-sided upper bound for ``group`` (mode ``one_sided_upper``)."""
        return r_hat + self.quantile_for(group)

    def interval(self, group: object, lo: float, hi: float) -> tuple[float, float]:
        """Two-sided interval for ``group`` (modes ``cqr`` / ``abs_residual``)."""
        q = self.quantile_for(group)
        return lo - q, hi + q


# ─────────────────────────────────────────────────────────────────────────────
# Weighted conformal (covariate shift within a fixed action bucket)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class WeightedConformal:
    """Weighted conformal for covariate shift in a fixed action bucket (T5).

    Assumptions (doc L E14): within a fixed intervention bucket ``a``,
    ``P_t^a(Y|X) = P_c^a(Y|X)`` (only the covariate marginal shifts) with
    support overlap and likelihood ratio ``w_a(x) = dP_t^a / dP_c^a``. Then the
    weighted conformal interval achieves *target-domain* marginal coverage
    ``P_{(X,Y)~P_t^a}{Y in C^w(X)} >= 1 - alpha``.

    The threshold is the smallest score whose **normalised weighted CDF**
    (including the test-point weight, which we approximate by the mean training
    weight per the standard construction) reaches ``1 - alpha``. Weights are
    clipped to ``[0, w_clip]`` to bound undercoverage when the density ratio is
    unstable (doc L clipped-weight method). What does **not** survive:
    conditional coverage, cross-action statements, or shifts in ``P(Y^a | X)``.
    """

    alpha: float = 0.10
    mode: str = "one_sided_upper"  # one_sided_upper | abs_residual | cqr
    w_clip: float = 20.0
    _scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _weights: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def _score(self, y: np.ndarray, preds: dict[str, np.ndarray]) -> np.ndarray:
        if self.mode == "abs_residual":
            return np.abs(y - preds["mu"])
        if self.mode == "cqr":
            return np.maximum(preds["q_lo"] - y, y - preds["q_hi"])
        if self.mode == "one_sided_upper":
            return y - preds["r_hat"]
        raise ValueError(f"unknown weighted-conformal mode {self.mode!r}")

    def fit(
        self,
        y_cal: np.ndarray,
        preds: dict[str, np.ndarray],
        weights: np.ndarray,
    ) -> WeightedConformal:
        y = np.asarray(y_cal, float)
        pred_arrays = {k: np.asarray(v, float) for k, v in preds.items()}
        self._scores = self._score(y, pred_arrays)
        w = np.clip(np.asarray(weights, float), 0.0, self.w_clip)
        self._weights = w
        return self

    def _weighted_quantile(self, test_weight: float) -> float:
        s = self._scores
        if s.size == 0:
            return math.inf
        w = np.append(self._weights, max(test_weight, 0.0))
        # Append +inf score for the test point (standard weighted-conformal).
        scores = np.append(s, math.inf)
        order = np.argsort(scores)
        scores_sorted = scores[order]
        w_sorted = w[order]
        total = w_sorted.sum()
        if total <= 0.0:
            return math.inf
        cum = np.cumsum(w_sorted) / total
        idx = int(np.searchsorted(cum, 1.0 - self.alpha, side="left"))
        idx = min(idx, scores_sorted.size - 1)
        return float(scores_sorted[idx])

    def upper(self, r_hat: float, test_weight: float = 1.0) -> float:
        return r_hat + self._weighted_quantile(test_weight)

    def interval(
        self, lo: float, hi: float, test_weight: float = 1.0
    ) -> tuple[float, float]:
        q = self._weighted_quantile(test_weight)
        return lo - q, hi + q


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Conformal Inference (online drift maintenance)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ACICalibrator:
    """Adaptive Conformal Inference online wrapper (E15).

    Maintains an effective miscoverage ``alpha_t`` updated after each revealed
    outcome: ``alpha_{t+1} = clip(alpha_t + gamma (alpha_target - err_t))`` where
    ``err_t = 1{Y_t not in C_t}``. Guarantee (doc L): *long-run / asymptotic*
    coverage close to ``1 - alpha_target`` under distribution shift — **not** the
    classical one-shot finite-sample guarantee for each future point.

    Used as a maintenance layer on top of a base conformal radius: the effective
    radius is the base score-quantile at the current ``alpha_t``.
    """

    alpha_target: float = 0.10
    gamma: float = 0.01
    alpha_t: float = field(init=False)
    alpha_min: float = 1e-3
    alpha_max: float = 0.5
    _err_hist: deque = field(default_factory=lambda: deque(maxlen=2000))

    def __post_init__(self) -> None:
        self.alpha_t = self.alpha_target

    def radius(self, scores: np.ndarray) -> float:
        """Current conformal radius at the adapted level ``alpha_t``."""
        return split_conformal_quantile(
            scores, float(np.clip(self.alpha_t, self.alpha_min, self.alpha_max))
        )

    def update(self, covered: bool) -> float:
        """Feed the realised coverage of the *last* interval; returns new alpha."""
        err = 0.0 if covered else 1.0
        self._err_hist.append(err)
        self.alpha_t = float(
            np.clip(
                self.alpha_t + self.gamma * (self.alpha_target - err),
                self.alpha_min,
                self.alpha_max,
            )
        )
        return self.alpha_t

    def empirical_coverage(self) -> float:
        if not self._err_hist:
            return float("nan")
        return 1.0 - float(np.mean(self._err_hist))


@dataclass(slots=True)
class BCACICalibrator:
    """Bias-Corrected ACI (BC-ACI) online wrapper (doc L, E15 + bias term).

    Standard ACI corrects the *level* but lags when the base forecaster develops
    a *persistent residual bias* (drift). BC-ACI additionally maintains an
    exponentially-weighted bias estimate ``b_t`` of the one-sided residual and
    conformalizes the **de-biased** scores ``S_i - b_t``. The level update is the
    same ACI projected gradient step. Guarantee: long-run / time-uniform target
    calibration with materially better finite-sample behaviour than ACI when the
    forecaster drifts (doc L cites the 2026 BC-ACI result).
    """

    alpha_target: float = 0.10
    gamma: float = 0.01
    bias_decay: float = 0.05  # EW step for the bias estimate b_t
    alpha_t: float = field(init=False)
    bias: float = 0.0
    alpha_min: float = 1e-3
    alpha_max: float = 0.5
    _err_hist: deque = field(default_factory=lambda: deque(maxlen=2000))

    def __post_init__(self) -> None:
        self.alpha_t = self.alpha_target

    def radius(self, scores: np.ndarray) -> float:
        """Radius on *de-biased* scores at the adapted level."""
        s = np.asarray(scores, float) - self.bias
        return split_conformal_quantile(
            s, float(np.clip(self.alpha_t, self.alpha_min, self.alpha_max))
        )

    def update(self, covered: bool, residual: float | None = None) -> float:
        """Update the level (ACI) and the EW bias estimate (BC).

        ``residual`` is the latest signed nonconformity residual driving the
        bias estimate; if omitted only the level is updated.
        """
        err = 0.0 if covered else 1.0
        self._err_hist.append(err)
        self.alpha_t = float(
            np.clip(
                self.alpha_t + self.gamma * (self.alpha_target - err),
                self.alpha_min,
                self.alpha_max,
            )
        )
        if residual is not None:
            self.bias = (1.0 - self.bias_decay) * self.bias + (
                self.bias_decay * float(residual)
            )
        return self.alpha_t

    def empirical_coverage(self) -> float:
        if not self._err_hist:
            return float("nan")
        return 1.0 - float(np.mean(self._err_hist))


# ─────────────────────────────────────────────────────────────────────────────
# Empirical coverage helper (acceptance check W1)
# ─────────────────────────────────────────────────────────────────────────────
def empirical_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of points inside ``[lo, hi]`` — the W1 acceptance check."""
    y = np.asarray(y, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    if y.size == 0:
        return float("nan")
    return float(np.mean((y >= lo) & (y <= hi)))


def one_sided_coverage(y: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of points with ``Y <= upper`` (one-sided acceptance check)."""
    y = np.asarray(y, float)
    upper = np.asarray(upper, float)
    if y.size == 0:
        return float("nan")
    return float(np.mean(y <= upper))


__all__ = [
    "ACICalibrator",
    "BCACICalibrator",
    "ConformalizedQuantileRegressor",
    "MondrianConformal",
    "OneSidedUpperConformal",
    "SplitConformalRegressor",
    "WeightedConformal",
    "beta_corrected_level",
    "conformal_rank",
    "empirical_coverage",
    "one_sided_coverage",
    "reg_incomplete_beta",
    "split_conformal_quantile",
]
