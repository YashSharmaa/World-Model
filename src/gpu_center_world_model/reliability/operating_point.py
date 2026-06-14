# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Budget-constrained, uncertainty-gated operating points (doc U §operating_point
and §rul_time_to_event).

Thresholding is **three-stage** (doc U §operating_point OI1), matching the repo
spine that safety is a *gate*, not a weighted metric:

1. **Exact-kernel safety gate.** Only kernel-certified-safe candidates reach the
   learned heads. This module assumes that filtering has already happened (it is
   owned by the simulator's exact kernel) and never expands the safe set.
2. **Budget-calibrated hazard / changepoint threshold ``u_B``.** Operators can
   inspect only a fixed alert load, so the threshold is chosen to respect an
   alert budget ``B`` over ``N`` decision windows (OEq2). A conservative
   *null-budget* threshold calibrated on incident-free slices is also provided.
3. **Uncertainty / observation-integrity gate ``kappa``** tied to abstention and
   conservative fallback (OD2). Emit time ``T_{u,kappa} = inf{t : s_t >= u and
   U_t <= kappa}``; if ``U_t > kappa`` the system abstains. By Theorem OT1,
   tightening ``kappa`` can only reduce alert volume and false alarms and can
   only weakly increase detection delay.

For rare faults, operating points are reported in **PR-space, not ROC-space**
(Theorem OT2): under a fixed alert budget precision directly encodes how much of
limited operator capacity is spent on true incidents, whereas AUROC can stay high
while deployed precision is unacceptable.

The RUL / time-to-event section turns the competing-risks survival object into
calibrated quantiles with **censoring-aware (IPCW) split-conformal intervals**
(doc U §rul REq1-REq4, RT1): a model-based interval is corrected on a calibration
set using inverse-probability-of-censoring weights so the certified interval
attains the advertised marginal coverage under conditionally-independent
censoring.

Pure NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .competing_risks import CauseSpecificForecast
from .quiet_failure import IntegrityPosterior


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — budget-constrained thresholds (doc U §operating_point OEq1-OEq3)
# ──────────────────────────────────────────────────────────────────────────────
def alert_rate(scores: np.ndarray, threshold: float) -> float:
    """Empirical alert rate ``A(u) = P(s_t >= u)`` (doc U OEq1)."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return 0.0
    return float(np.mean(scores >= threshold))


def budget_threshold(scores: np.ndarray, budget: int, n_windows: int) -> float:
    """Threshold ``u_B = inf{u : A(u) <= B/N}`` (doc U OEq2).

    Picks the smallest score threshold whose empirical alert rate does not exceed
    the per-window budget. Implemented as the upper ``(1 - B/N)`` empirical
    quantile of the score distribution — the standard budget-calibrated operating
    point. ``scores`` should be evaluation-slice alert scores (calibrate on the
    slice the budget applies to).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return float("inf")
    rate = max(min(budget / max(n_windows, 1), 1.0), 0.0)
    if rate <= 0.0:
        return float(np.max(scores)) + 1.0  # alert on nothing
    if rate >= 1.0:
        return float(np.min(scores)) - 1.0  # alert on everything
    # Smallest u with mean(scores >= u) <= rate  ==  (1 - rate) quantile.
    return float(np.quantile(scores, 1.0 - rate, method="higher"))


def null_budget_threshold(
    null_scores: np.ndarray, budget: int, n_windows: int
) -> float:
    """Conservative null-budget threshold ``u_B^null`` (doc U OEq2).

    Calibrated on *incident-free* slices: ``inf{u : P_inf(s_t >= u) <= B/N}``.
    Using the pre-change (null) score law makes the false-alarm budget hold even
    when the evaluation slice is event-poor.
    """
    return budget_threshold(null_scores, budget, n_windows)


def cost_weighted_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    c_fn: float,
    c_fp: float,
    grid: int = 256,
) -> float:
    """Cost-minimising threshold (doc U OEq3, detection-delay term dropped).

    Minimises ``c_FN * pi * (1 - TPR(u)) + c_FP * (1 - pi) * FPR(u)`` over a grid
    of candidate thresholds, where ``pi`` is the empirical prevalence. The
    detection-delay term ``c_D E[(T_u - nu)^+]`` of OEq3 is omitted here because
    it requires the temporal change point ``nu``; the changepoint detectors in
    :mod:`changepoint` own delay-aware tuning.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if scores.size == 0:
        return float("inf")
    pi = float(np.mean(labels))
    lo, hi = float(np.min(scores)), float(np.max(scores))
    candidates = np.linspace(lo, hi, grid)
    pos = labels
    neg = ~labels
    n_pos = max(int(pos.sum()), 1)
    n_neg = max(int(neg.sum()), 1)
    best_u, best_cost = candidates[0], np.inf
    for u in candidates:
        alarm = scores >= u
        tpr = float(np.sum(alarm & pos)) / n_pos
        fpr = float(np.sum(alarm & neg)) / n_neg
        cost = c_fn * pi * (1.0 - tpr) + c_fp * (1.0 - pi) * fpr
        if cost < best_cost:
            best_cost, best_u = cost, float(u)
    return best_u


# ──────────────────────────────────────────────────────────────────────────────
# PR-space operating points for rare faults (doc U Theorem OT2)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class OperatingPoint:
    """A single threshold's precision/recall/alert-rate snapshot (PR-space)."""

    threshold: float
    tpr: float  # recall
    fpr: float
    precision: float
    alert_rate: float


def precision_recall_curve(
    scores: np.ndarray, labels: np.ndarray, *, grid: int = 256
) -> list[OperatingPoint]:
    """PR-space operating points across a threshold grid (doc U OT2).

    Precision is prevalence-sensitive (doc U §operating_point O1), so for rare
    faults this curve is the operationally informative one — it shows directly how
    much operator capacity each threshold spends on true incidents.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pi = float(np.mean(labels)) if labels.size else 0.0
    pos, neg = labels, ~labels
    n_pos = max(int(pos.sum()), 1)
    n_neg = max(int(neg.sum()), 1)
    lo = float(np.min(scores)) if scores.size else 0.0
    hi = float(np.max(scores)) if scores.size else 1.0
    out: list[OperatingPoint] = []
    for u in np.linspace(lo, hi, grid):
        alarm = scores >= u
        tpr = float(np.sum(alarm & pos)) / n_pos
        fpr = float(np.sum(alarm & neg)) / n_neg
        denom = pi * tpr + (1.0 - pi) * fpr
        precision = (pi * tpr / denom) if denom > 0 else 1.0
        out.append(
            OperatingPoint(
                threshold=float(u),
                tpr=tpr,
                fpr=fpr,
                precision=precision,
                alert_rate=pi * tpr + (1.0 - pi) * fpr,
            )
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — uncertainty-gated three-stage decision (doc U OD2, OT1, OI1)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ThreeStageConfig:
    """Configuration of the three-stage gate (doc U OI1)."""

    score_threshold: float = 1.0  # stage-2 budget threshold u_B
    kappa_obs: float = 0.5  # stage-3 observation-integrity gate
    kappa_epistemic: float = 0.5  # stage-3 OOD/disagreement gate
    fallback: str = "conservative_rule"  # what to do on abstain


@dataclass(slots=True)
class GateDecision:
    """Outcome of the three-stage operating-point gate (auditable)."""

    emit: bool  # emit a model-based alert/ranking
    abstain: bool  # abstain → conservative fallback
    reason: str
    score_pass: bool
    integrity_pass: bool
    epistemic_pass: bool
    fallback: str | None = None


def three_stage_decision(
    *,
    score: float,
    integrity: IntegrityPosterior,
    epistemic_score: float,
    feasible: bool,
    config: ThreeStageConfig,
) -> GateDecision:
    """Three-stage operating point: exact gate → budget → abstention (doc U OI1).

    1. **Exact-kernel gate** — ``feasible`` reflects whether the exact kernel
       certified the candidate. An infeasible candidate is never emitted (the
       learned model can never expand the safe set).
    2. **Budget threshold** — the alert score must clear ``u_B``.
    3. **Uncertainty gate** — observation integrity ``u_t <= kappa_obs`` *and*
       OOD/disagreement ``<= kappa_epistemic`` (OD2). If telemetry integrity or
       epistemic uncertainty is too high, abstain and fall back conservatively.

    Per Theorem OT1, tightening ``kappa`` shrinks the acceptance set, so it can
    only reduce emitted alerts / false alarms (and only weakly increase delay).
    """
    if not feasible:
        return GateDecision(
            emit=False,
            abstain=False,
            reason="exact_kernel_infeasible",
            score_pass=False,
            integrity_pass=False,
            epistemic_pass=False,
            fallback=config.fallback,
        )

    score_pass = score >= config.score_threshold
    integrity_pass = integrity.u <= config.kappa_obs
    epistemic_pass = epistemic_score <= config.kappa_epistemic

    # The uncertainty gate governs whether we *trust* the model output at all.
    if not (integrity_pass and epistemic_pass):
        reason = (
            "abstain_observation_integrity"
            if not integrity_pass
            else "abstain_epistemic"
        )
        if integrity.is_quiet_failure:
            reason = "abstain_quiet_failure"
        return GateDecision(
            emit=False,
            abstain=True,
            reason=reason,
            score_pass=score_pass,
            integrity_pass=integrity_pass,
            epistemic_pass=epistemic_pass,
            fallback=config.fallback,
        )

    if score_pass:
        return GateDecision(
            emit=True,
            abstain=False,
            reason="emit",
            score_pass=True,
            integrity_pass=True,
            epistemic_pass=True,
        )
    return GateDecision(
        emit=False,
        abstain=False,
        reason="below_budget_threshold",
        score_pass=False,
        integrity_pass=True,
        epistemic_pass=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# RUL / time-to-event quantiles (doc U §rul REq1-REq3)
# ──────────────────────────────────────────────────────────────────────────────
def rul_survival(forecast: CauseSpecificForecast) -> np.ndarray:
    """All-cause RUL survival ``P(R_u > r) = S_u(r)`` (doc U REq1).

    ``forecast.survival`` already is ``S_u(r)`` on the bucket grid; returned as
    the RUL survival curve (index r-1 ↦ survival past bucket r).
    """
    return np.asarray(forecast.survival, dtype=np.float64)


def median_rul(forecast: CauseSpecificForecast) -> int:
    """Median RUL ``rhat_0.5 = inf{r : S_u(r) <= 0.5}`` (doc U REq2).

    Returns the 1-based bucket at which all-cause survival first drops to/below
    0.5, or the horizon length if it never does (right-censored RUL).
    """
    s = rul_survival(forecast)
    below = np.nonzero(s <= 0.5)[0]
    return int(below[0] + 1) if below.size else len(s)


def expected_rul(forecast: CauseSpecificForecast) -> float:
    """Mean RUL ``E[R_u] = sum_{r>=1} S_u(r-1)`` (doc U REq2).

    Uses ``S_u(r-1)`` (survival entering bucket r) so the first term is
    ``S_u(0) = 1``. Truncated at the modelled horizon (a lower bound when the
    survival curve has not yet reached 0).
    """
    s = rul_survival(forecast)
    s_prev = np.concatenate([[1.0], s[:-1]])
    return float(np.sum(s_prev))


def rul_quantile(forecast: CauseSpecificForecast, tau: float) -> int:
    """RUL quantile ``inf{r : S_u(r) <= 1 - tau}`` (doc U REq3).

    ``tau`` is the *event* probability mass (e.g. ``tau=0.9`` → the bucket by
    which 90% of the RUL mass has elapsed). Returns a 1-based bucket index.
    """
    s = rul_survival(forecast)
    target = 1.0 - float(np.clip(tau, 0.0, 1.0))
    below = np.nonzero(s <= target)[0]
    return int(below[0] + 1) if below.size else len(s)


# ──────────────────────────────────────────────────────────────────────────────
# Censoring-aware conformal RUL intervals (doc U §rul REq4, RT1)
# ──────────────────────────────────────────────────────────────────────────────
def km_censoring_survival(
    times: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier estimate of the *censoring* survival ``G(t) = P(C > t)``.

    Treats censoring as the "event" of interest (reverse-KM): ``observed`` is the
    *event* indicator, so a censored unit (``observed=False``) is an event for the
    censoring process. Returns the sorted unique times and the right-continuous
    step survival ``hatG`` evaluated just *before* each time (for IPCW at ``Y^-``).
    """
    times = np.asarray(times, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    order = np.argsort(times, kind="mergesort")
    t_sorted = times[order]
    cens_event = ~observed[order]  # censoring is the event for G
    uniq = np.unique(t_sorted)
    g = np.ones(len(uniq), dtype=np.float64)
    surv = 1.0
    n_at_risk = len(t_sorted)
    for i, ut in enumerate(uniq):
        at_t = t_sorted == ut
        d = int(np.sum(cens_event & at_t))  # censorings at ut
        if n_at_risk > 0:
            surv *= 1.0 - d / n_at_risk
        g[i] = surv
        n_at_risk -= int(np.sum(at_t))
    return uniq, g


def ipcw_weights(
    times: np.ndarray,
    observed: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Inverse-probability-of-censoring weights ``w_i = Delta_i / hatG(Y_i^-)``.

    (doc U REq4.) Only uncensored units carry weight (``Delta_i = observed``);
    each is up-weighted by the inverse censoring-survival just before its event
    time to correct for the units that were censored before they could be
    observed (conditionally-independent censoring, doc U A4 / §P1).
    """
    times = np.asarray(times, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    uniq, g = km_censoring_survival(times, observed)
    # hatG(Y_i^-): the censoring survival strictly before each event time. Use
    # the step value at the largest grid time < Y_i (1.0 before the first).
    idx = np.searchsorted(uniq, times, side="left") - 1
    g_minus = np.where(idx >= 0, g[np.clip(idx, 0, len(g) - 1)], 1.0)
    g_minus = np.maximum(g_minus, eps)
    return np.where(observed, 1.0 / g_minus, 0.0)


@dataclass(slots=True)
class ConformalRULInterval:
    """A censoring-aware conformal RUL prediction interval (doc U REq4, RT1)."""

    lower: float
    upper: float
    alpha: float  # miss rate target (coverage = 1 - alpha)
    q_hat: float  # the weighted conformal quantile of nonconformity scores


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted empirical ``q``-quantile (for IPCW conformal calibration)."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = weights > 0
    values, weights = values[mask], weights[mask]
    if values.size == 0:
        return float("inf")
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cum = np.cumsum(weights)
    total = cum[-1]
    if total <= 0:
        return float(values[-1])
    target = q * total
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(idx, len(values) - 1)
    return float(values[idx])


def conformal_rul_interval(
    point_pred: float,
    *,
    cal_point_preds: np.ndarray,
    cal_event_times: np.ndarray,
    cal_observed: np.ndarray,
    alpha: float = 0.1,
) -> ConformalRULInterval:
    """Split-conformal RUL interval with IPCW weights (doc U REq4, RT1).

    Nonconformity scores are absolute residuals ``a_i = |Yhat_i - Y_i|`` on the
    *uncensored* calibration units, IPCW-weighted to undo informative dropout of
    the censored ones. The weighted ``(1 - alpha)`` quantile ``q_hat`` gives the
    symmetric certified interval ``[Yhat - q_hat, Yhat + q_hat]`` (clipped at 0).
    Under conditionally-independent censoring and the weighted-exchangeability
    assumption (doc U A4 / RT1) this attains the advertised marginal coverage.

    Parameters
    ----------
    point_pred
        The point RUL prediction (e.g. :func:`median_rul`) for the test unit.
    cal_point_preds, cal_event_times, cal_observed
        Calibration-set point predictions, observed/censored times ``Y_i``, and
        event indicators ``Delta_i``.
    alpha
        Target miss rate; coverage is ``1 - alpha``.
    """
    cal_point_preds = np.asarray(cal_point_preds, dtype=np.float64)
    cal_event_times = np.asarray(cal_event_times, dtype=np.float64)
    cal_observed = np.asarray(cal_observed, dtype=bool)

    scores = np.abs(cal_point_preds - cal_event_times)
    w = ipcw_weights(cal_event_times, cal_observed)
    # Normalise weights to a simplex for a stable conformal level.
    if w.sum() > 0:
        w = w / w.sum()
    q_hat = weighted_quantile(scores, w, 1.0 - alpha)
    lo = max(0.0, point_pred - q_hat)
    hi = point_pred + q_hat
    return ConformalRULInterval(lower=lo, upper=hi, alpha=alpha, q_hat=q_hat)


__all__ = [
    "ConformalRULInterval",
    "GateDecision",
    "OperatingPoint",
    "ThreeStageConfig",
    "alert_rate",
    "budget_threshold",
    "conformal_rul_interval",
    "cost_weighted_threshold",
    "expected_rul",
    "ipcw_weights",
    "km_censoring_survival",
    "median_rul",
    "null_budget_threshold",
    "precision_recall_curve",
    "rul_quantile",
    "rul_survival",
    "three_stage_decision",
    "weighted_quantile",
]
