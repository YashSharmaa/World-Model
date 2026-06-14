# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Calibration metrics: ECE for hazards, interval coverage by slice.

Doc F warns that "models that look calibrated in-distribution can become
overconfident when the environment changes" — so calibration is reported *by
slice* (topology-shift, degraded-telemetry, unseen-workload) rather than as one
in-distribution number. This module provides:

* **Expected Calibration Error (ECE)** for binary hazard probabilities, with the
  standard equal-width reliability binning and the per-bin reliability curve;
* **quantile / interval coverage by slice** — the empirical coverage of nominal
  intervals, grouped by a slice key, with the doc-F 85%-95% acceptance band for
  the nominal-90% interval.

Pure NumPy; the runner feeds already-sliced arrays or a slice-label array.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ArrayLike = np.ndarray | list[float]


def _as_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


@dataclass(slots=True)
class ReliabilityBin:
    """One reliability-diagram bin (confidence band)."""

    lo: float
    hi: float
    count: int
    mean_pred: float
    mean_truth: float

    @property
    def gap(self) -> float:
        return abs(self.mean_pred - self.mean_truth)


@dataclass(slots=True)
class CalibrationReport:
    """ECE plus the reliability curve for a binary hazard channel / slice."""

    channel: str
    slice_name: str
    n: int
    ece: float
    mce: float  # maximum calibration error (worst bin)
    bins: list[ReliabilityBin] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "slice_name": self.slice_name,
            "n": self.n,
            "ece": self.ece,
            "mce": self.mce,
            "bins": [
                {
                    "lo": b.lo,
                    "hi": b.hi,
                    "count": b.count,
                    "mean_pred": b.mean_pred,
                    "mean_truth": b.mean_truth,
                    "gap": b.gap,
                }
                for b in self.bins
            ],
        }


def expected_calibration_error(
    prob: ArrayLike,
    label: ArrayLike,
    *,
    n_bins: int = 10,
    channel: str = "hazard",
    slice_name: str = "all",
) -> CalibrationReport:
    """Equal-width-binned ECE for binary hazard probabilities.

    ``ECE = sum_b (n_b/N) * |mean_pred_b - mean_truth_b|``. Also returns MCE (the
    worst-bin gap) and the full reliability curve so the report can render a
    diagram rather than collapsing to one number.
    """
    p, y = _as_array(prob), _as_array(label)
    n = p.size
    if n == 0:
        return CalibrationReport(
            channel=channel,
            slice_name=slice_name,
            n=0,
            ece=float("nan"),
            mce=float("nan"),
        )
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin is inclusive of 1.0
        mask = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        cnt = int(np.sum(mask))
        if cnt == 0:
            bins.append(ReliabilityBin(lo, hi, 0, float("nan"), float("nan")))
            continue
        mean_pred = float(np.mean(p[mask]))
        mean_truth = float(np.mean(y[mask]))
        gap = abs(mean_pred - mean_truth)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
        bins.append(ReliabilityBin(lo, hi, cnt, mean_pred, mean_truth))
    return CalibrationReport(
        channel=channel, slice_name=slice_name, n=n, ece=ece, mce=mce, bins=bins
    )


@dataclass(slots=True)
class CoverageBySliceReport:
    """Interval coverage grouped by a slice key (doc F shift-aware reporting)."""

    nominal: float
    band_lo: float
    band_hi: float
    per_slice: dict[str, dict[str, float]] = field(default_factory=dict)

    def in_band(self, slice_name: str) -> bool:
        s = self.per_slice.get(slice_name)
        if s is None:
            return False
        cov = s["coverage"]
        return self.band_lo <= cov <= self.band_hi

    def as_dict(self) -> dict[str, object]:
        return {
            "nominal": self.nominal,
            "band_lo": self.band_lo,
            "band_hi": self.band_hi,
            "per_slice": self.per_slice,
        }


def coverage_by_slice(
    lower: ArrayLike,
    upper: ArrayLike,
    truth: ArrayLike,
    slice_labels: list[str],
    *,
    nominal: float = 0.90,
    band: tuple[float, float] = (0.85, 0.95),
) -> CoverageBySliceReport:
    """Empirical interval coverage per slice, with the doc-F acceptance band.

    ``slice_labels[i]`` is the slice each (lower[i], upper[i], truth[i]) belongs
    to. The acceptance band defaults to doc F's 85%-95% for a nominal-90%
    interval.
    """
    lo, hi, y = _as_array(lower), _as_array(upper), _as_array(truth)
    labels = np.asarray(slice_labels)
    report = CoverageBySliceReport(nominal=nominal, band_lo=band[0], band_hi=band[1])
    for s in sorted(set(slice_labels)):
        m = labels == s
        ys = y[m]
        if ys.size == 0:
            continue
        inside = (ys >= lo[m]) & (ys <= hi[m])
        report.per_slice[s] = {
            "coverage": float(np.mean(inside)),
            "n": int(ys.size),
            "in_band": bool(band[0] <= float(np.mean(inside)) <= band[1]),
        }
    return report


__all__ = [
    "CalibrationReport",
    "CoverageBySliceReport",
    "ReliabilityBin",
    "coverage_by_slice",
    "expected_calibration_error",
]
