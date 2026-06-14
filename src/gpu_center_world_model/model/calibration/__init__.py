# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Post-hoc conformal calibration (doc L) — split / CQR / one-sided / Mondrian.

Calibrators are pure NumPy and implement the :class:`Calibrator` contract so the
serving and planning layers stay backend-agnostic.
"""

from __future__ import annotations

from .conformal import (
    ConformalBundleCalibrator,
    CQRCalibrator,
    MondrianConformal,
    OneSidedUpperConformal,
    SplitConformal,
    conformal_quantile,
)

__all__ = [
    "CQRCalibrator",
    "ConformalBundleCalibrator",
    "MondrianConformal",
    "OneSidedUpperConformal",
    "SplitConformal",
    "conformal_quantile",
]
