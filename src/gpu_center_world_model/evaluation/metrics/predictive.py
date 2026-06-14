# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Predictive-accuracy metrics for *factual replay* under the logged action.

Doc F places predictive accuracy fourth in the gated hierarchy — it is a
*diagnostic*, not the headline claim. It is measured on factual replay (the
action that really happened), horizon-sliced, and it must include proper
probabilistic scoring because the model is required to emit calibrated
uncertainty, never bare point estimates (doc F "metric families"; doc L
conformalized quantile regression; doc kpis.py).

Everything here is pure NumPy and operates on plain float arrays so the harness
never depends on the learned backend. The functions accept aligned
``pred``/``truth`` arrays already sliced by horizon; the higher-level runner is
responsible for the slicing.

Channels covered (doc F): queue depth, admitted GPU-hours, queue-wait deltas,
rack headroom, hotspot score, completion-time deltas (continuous, via
MAE/WAPE/pinball/coverage) and hazards PodsReadyTimeout, requeue, hotspot,
NodeFailure, XID fault (discrete, via Brier + alert-budget precision/recall).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ArrayLike = np.ndarray | list[float]


def _as_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


# ──────────────────────────────────────────────────────────────────────────────
# Continuous-outcome point metrics
# ──────────────────────────────────────────────────────────────────────────────
def mae(pred: ArrayLike, truth: ArrayLike) -> float:
    """Median *and* mean absolute error are both useful; this returns the mean.

    Doc F asks for "median absolute error plus a relative metric"; we expose the
    mean as ``mae`` and the median separately as :func:`medae` so both the
    average-case and tail-robust readouts are available.
    """
    p, y = _as_array(pred), _as_array(truth)
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.abs(p - y)))


def medae(pred: ArrayLike, truth: ArrayLike) -> float:
    """Median absolute error — robust to the heavy tails of queue dynamics."""
    p, y = _as_array(pred), _as_array(truth)
    if p.size == 0:
        return float("nan")
    return float(np.median(np.abs(p - y)))


def wape(pred: ArrayLike, truth: ArrayLike) -> float:
    """Weighted absolute percentage error = sum|p-y| / sum|y|.

    Doc F prefers WAPE over MAPE because operating scales differ across pods and
    scenarios, and WAPE is well-defined when individual truths are near zero
    (MAPE explodes). Returns NaN only if the total magnitude is exactly zero.
    """
    p, y = _as_array(pred), _as_array(truth)
    denom = float(np.sum(np.abs(y)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(np.abs(p - y)) / denom)


def rmse(pred: ArrayLike, truth: ArrayLike) -> float:
    """Root-mean-square error (tail-sensitive companion to MAE)."""
    p, y = _as_array(pred), _as_array(truth)
    if p.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - y) ** 2)))


# ──────────────────────────────────────────────────────────────────────────────
# Probabilistic scoring (quantile / interval)
# ──────────────────────────────────────────────────────────────────────────────
def pinball_loss(pred_q: ArrayLike, truth: ArrayLike, tau: float) -> float:
    """Quantile (pinball) loss at level ``tau`` — a proper scoring rule.

    ``pinball = mean( max(tau*(y-q), (tau-1)*(y-q)) )`` (doc F/L). Lower is
    better; the loss is asymmetric so it rewards correctly-placed quantiles, not
    merely centred ones.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0,1), got {tau}")
    q, y = _as_array(pred_q), _as_array(truth)
    if q.size == 0:
        return float("nan")
    diff = y - q
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def interval_coverage(lower: ArrayLike, upper: ArrayLike, truth: ArrayLike) -> float:
    """Empirical coverage of a prediction interval ``[lower, upper]``.

    Doc F acceptance: a nominal-90% interval should land in the 85%-95%
    empirical-coverage band.
    """
    lo, hi, y = _as_array(lower), _as_array(upper), _as_array(truth)
    if y.size == 0:
        return float("nan")
    inside = (y >= lo) & (y <= hi)
    return float(np.mean(inside))


def interval_sharpness(lower: ArrayLike, upper: ArrayLike) -> float:
    """Mean interval width — the sharpness companion to coverage.

    Coverage alone is gameable by emitting infinitely wide intervals; doc F
    requires reporting sharpness alongside coverage so the trade-off is visible.
    """
    lo, hi = _as_array(lower), _as_array(upper)
    if lo.size == 0:
        return float("nan")
    return float(np.mean(np.abs(hi - lo)))


def coverage_gap(
    lower: ArrayLike, upper: ArrayLike, truth: ArrayLike, nominal: float
) -> float:
    """Signed coverage error ``empirical - nominal`` (positive = over-cover)."""
    return interval_coverage(lower, upper, truth) - nominal


# ──────────────────────────────────────────────────────────────────────────────
# Hazard (binary event) metrics
# ──────────────────────────────────────────────────────────────────────────────
def brier_score(prob: ArrayLike, label: ArrayLike) -> float:
    """Brier score = mean( (p - y)^2 ) for binary event hazards (doc F)."""
    p, y = _as_array(prob), _as_array(label)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def precision_recall_at_budget(
    prob: ArrayLike, label: ArrayLike, alert_budget: float
) -> tuple[float, float, float]:
    """Precision/recall at a realistic *alert budget* (top-k fraction).

    Doc F: "Brier score plus precision-recall-oriented metrics at realistic
    alert budgets." We alert on the highest-probability ``alert_budget`` fraction
    of cases (an operations-grade framing — operators have finite attention) and
    return ``(precision, recall, threshold)``.
    """
    p, y = _as_array(prob), _as_array(label)
    n = p.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if not 0.0 < alert_budget <= 1.0:
        raise ValueError(f"alert_budget must be in (0,1], got {alert_budget}")
    k = max(1, int(np.ceil(alert_budget * n)))
    # Alert on the k highest-probability cases.
    order = np.argsort(-p, kind="stable")
    alerted = np.zeros(n, dtype=bool)
    alerted[order[:k]] = True
    threshold = float(p[order[k - 1]])
    tp = float(np.sum(alerted & (y > 0.5)))
    fp = float(np.sum(alerted & (y <= 0.5)))
    positives = float(np.sum(y > 0.5))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / positives if positives > 0 else float("nan")
    return precision, recall, threshold


# ──────────────────────────────────────────────────────────────────────────────
# Aggregated, horizon-sliced report object
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class PredictiveReport:
    """All factual-replay predictive metrics for one (channel, horizon) slice."""

    channel: str
    horizon_s: int
    n: int
    mae: float = float("nan")
    medae: float = float("nan")
    wape: float = float("nan")
    rmse: float = float("nan")
    pinball: dict[float, float] = field(default_factory=dict)
    coverage_90: float = float("nan")
    sharpness_90: float = float("nan")
    brier: float = float("nan")

    def as_dict(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "horizon_s": self.horizon_s,
            "n": self.n,
            "mae": self.mae,
            "medae": self.medae,
            "wape": self.wape,
            "rmse": self.rmse,
            "pinball": {str(k): v for k, v in self.pinball.items()},
            "coverage_90": self.coverage_90,
            "sharpness_90": self.sharpness_90,
            "brier": self.brier,
        }


def score_quantile_channel(
    channel: str,
    horizon_s: int,
    truth: ArrayLike,
    quantile_preds: dict[float, ArrayLike],
    *,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> PredictiveReport:
    """Score one continuous channel given a dict of ``{tau: pred_array}``.

    Uses the median (tau=0.5) for point metrics and the ``[lower_q, upper_q]``
    pair for the interval metrics. Matches the schema quantile grid
    ``QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)`` from kpis.py — a 0.05/0.95
    pair is a nominal-90% interval.
    """
    y = _as_array(truth)
    median = quantile_preds.get(0.5)
    rep = PredictiveReport(channel=channel, horizon_s=horizon_s, n=int(y.size))
    if median is not None:
        rep.mae = mae(median, y)
        rep.medae = medae(median, y)
        rep.wape = wape(median, y)
        rep.rmse = rmse(median, y)
    rep.pinball = {
        tau: pinball_loss(pred, y, tau) for tau, pred in quantile_preds.items()
    }
    if lower_q in quantile_preds and upper_q in quantile_preds:
        rep.coverage_90 = interval_coverage(
            quantile_preds[lower_q], quantile_preds[upper_q], y
        )
        rep.sharpness_90 = interval_sharpness(
            quantile_preds[lower_q], quantile_preds[upper_q]
        )
    return rep


__all__ = [
    "PredictiveReport",
    "brier_score",
    "coverage_gap",
    "interval_coverage",
    "interval_sharpness",
    "mae",
    "medae",
    "pinball_loss",
    "precision_recall_at_budget",
    "rmse",
    "score_quantile_channel",
    "wape",
]
