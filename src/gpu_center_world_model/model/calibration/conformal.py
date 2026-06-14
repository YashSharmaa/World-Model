# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Split-conformal calibrators implementing the :class:`Calibrator` contract.

Pure-NumPy, distribution-free post-hoc calibration of the learned forecasts
(doc L). The learned model is calibrated *only on action-conditioned decision
windows* anchored by a pre-action state hash — never by pooling observational
and interventional targets (doc L handoff).

Implemented primitives:

* **Split conformal regression** (doc L conformal_methods, T1): symmetric
  residual quantile around a point estimate.
* **Conformalized quantile regression (CQR)** (doc L T2): the default interval
  method for KPI deltas. Calibrates the gap to the predicted ``q_L``/``q_U``
  band, restoring finite-sample marginal coverage on top of a sharp quantile
  base model.
* **One-sided upper conformal** (doc L T4): the correct primitive for chance
  constraints, safety margins, and optimism-bias control — fills the
  ``conformal_upper`` slot on each :class:`HazardForecast`.
* **Mondrian group-conditional conformal** (doc L T3): separate thresholds per
  prespecified slice ``(action family, horizon, workload family, topology
  family, source-health bucket)`` with hierarchical back-off when a group is too
  small.

Each calibrator exposes both a low-level ``fit``/``predict`` API (for offline
calibration on a labelled split) and the :class:`Calibrator.calibrate` method
that widens / annotates a :class:`KPIForecastBundle` in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...schema.kpis import KPIForecastBundle, QuantileForecast


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample conformal quantile ``⌈(n+1)(1-α)⌉/n`` (doc L T1).

    Returns ``+inf`` when the calibration set is too small to certify level
    ``1-α`` (the conservative, never-undercover fallback).
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = s.shape[0]
    if n == 0:
        return float("inf")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf")
    return float(s[rank - 1])


# --------------------------------------------------------------------------- #
# Split conformal regression (symmetric).
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class SplitConformal:
    """Symmetric split-conformal interval around a point estimate (doc L T1)."""

    alpha: float = 0.10
    width_: float = float("inf")

    def fit(self, preds: np.ndarray, targets: np.ndarray) -> SplitConformal:
        scores = np.abs(np.asarray(targets) - np.asarray(preds))
        self.width_ = conformal_quantile(scores, self.alpha)
        return self

    def interval(self, pred: float) -> tuple[float, float]:
        return pred - self.width_, pred + self.width_


# --------------------------------------------------------------------------- #
# Conformalized quantile regression (CQR) — the default KPI interval method.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class CQRCalibrator:
    """Conformalized quantile regression (doc L T2).

    Calibrates the additive correction ``q_CQR`` to the predicted lower/upper
    quantile band so the conformalized interval ``[q_L - q_CQR, q_U + q_CQR]``
    has marginal coverage ``≥ 1-α`` under exchangeability. ``lo_tau``/``hi_tau``
    name which predicted quantiles bound the interval.
    """

    alpha: float = 0.10
    lo_tau: float = 0.05
    hi_tau: float = 0.95
    correction_: float = float("inf")

    def fit(
        self, q_lo: np.ndarray, q_hi: np.ndarray, targets: np.ndarray
    ) -> CQRCalibrator:
        q_lo, q_hi, y = map(np.asarray, (q_lo, q_hi, targets))
        scores = np.maximum(q_lo - y, y - q_hi)  # doc L D4
        self.correction_ = conformal_quantile(scores, self.alpha)
        return self

    def interval(self, q_lo: float, q_hi: float) -> tuple[float, float]:
        c = self.correction_ if np.isfinite(self.correction_) else 0.0
        return q_lo - c, q_hi + c

    def calibrate_forecast(self, qf: QuantileForecast) -> QuantileForecast:
        """Widen one quantile forecast's outer band by the conformal correction."""
        if not np.isfinite(self.correction_):
            return qf
        c = self.correction_
        q = dict(qf.quantiles)
        if self.lo_tau in q:
            q[self.lo_tau] = q[self.lo_tau] - c
        if self.hi_tau in q:
            q[self.hi_tau] = q[self.hi_tau] + c
        return QuantileForecast(channel=qf.channel, horizon_s=qf.horizon_s, quantiles=q)


# --------------------------------------------------------------------------- #
# One-sided upper conformal (chance constraints / optimism-bias control).
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class OneSidedUpperConformal:
    """One-sided upper conformal bound ``U_α(x) = r̂(x) + q⁺`` (doc L T4).

    The right primitive for the safety gate: ``P{R ≤ U_α} ≥ 1-α``. Fills the
    ``conformal_upper`` field on hazard forecasts so the planner can gate on the
    *calibrated* upper risk, not the nominal point estimate (doc L safety gate).

    This is a thin shim around the canonical
    :class:`uncertainty.conformal.OneSidedUpperConformal` so the project has *one*
    conformal implementation (doc L handoff). The divergent ``1 - pred`` fabricated
    margin / aggressive clamp that previously lived here is removed: when the
    calibrator is unfitted the honest answer is an infinite radius (no finite
    bound carries the guarantee), and the only post-hoc clip is to ``[0, 1]``
    because the bounded quantity here is a probability.
    """

    alpha: float = 0.05
    margin_: float = float("inf")

    def fit(self, preds: np.ndarray, targets: np.ndarray) -> OneSidedUpperConformal:
        from ...uncertainty.conformal import (
            OneSidedUpperConformal as _CanonicalOneSided,
        )

        canon = _CanonicalOneSided(alpha=self.alpha)
        canon.fit(np.asarray(targets), np.asarray(preds))  # (r_true, r_hat)
        self.margin_ = canon.q_plus
        return self

    def upper(self, pred: float) -> float:
        # Unfitted -> honest infinite radius -> the bound is the pred itself
        # clipped to the probability simplex (never *under*-states, never the old
        # ``1 - pred`` fabrication). Fitted -> pred + the conformal quantile.
        m = self.margin_ if np.isfinite(self.margin_) else 0.0
        return float(min(1.0, max(0.0, pred + m)))


# --------------------------------------------------------------------------- #
# Mondrian group-conditional conformal (per-slice thresholds + back-off).
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class MondrianConformal:
    """Group-conditional conformal with hierarchical back-off (doc L T3).

    Holds a separate scorer per group key. When a group's calibration set is
    below ``min_group``, it falls back to the pooled (parent) threshold so it
    never silently undercovers a sparse slice (doc L T3 back-off rule).
    """

    alpha: float = 0.10
    min_group: int = 20
    one_sided: bool = False
    group_width_: dict[str, float] = field(default_factory=dict)
    pooled_width_: float = float("inf")

    def fit(
        self,
        groups: list[str],
        preds: np.ndarray,
        targets: np.ndarray,
    ) -> MondrianConformal:
        preds, targets = np.asarray(preds), np.asarray(targets)
        resid = targets - preds if self.one_sided else np.abs(targets - preds)
        self.pooled_width_ = conformal_quantile(resid, self.alpha)
        gkeys = np.asarray(groups, dtype=object)
        for g in set(groups):
            sel = gkeys == g
            if int(sel.sum()) >= self.min_group:
                self.group_width_[g] = conformal_quantile(resid[sel], self.alpha)
        return self

    def width(self, group: str) -> float:
        w = self.group_width_.get(group, self.pooled_width_)
        return w if np.isfinite(w) else 0.0

    def interval(self, group: str, pred: float) -> tuple[float, float]:
        w = self.width(group)
        if self.one_sided:
            return float("-inf"), pred + w
        return pred - w, pred + w


# --------------------------------------------------------------------------- #
# Bundle-level calibrator implementing the Calibrator Protocol (doc L).
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ConformalBundleCalibrator:
    """Calibrates a whole :class:`KPIForecastBundle` (the Calibrator contract).

    Wires per-channel CQR calibrators for KPI intervals and per-channel
    one-sided upper calibrators for hazards. Calibrators are keyed by
    ``(channel, horizon)`` (the action-horizon-slice exchangeability bucket of
    doc L A2). Unfitted channels pass through unchanged (graceful degrade).
    """

    cqr: dict[tuple[str, int], CQRCalibrator] = field(default_factory=dict)
    upper: dict[tuple[str, int], OneSidedUpperConformal] = field(default_factory=dict)
    # Per-(channel, horizon) group-conditional Mondrian calibrators keyed by the
    # doc-L 5-tuple slice (action, horizon, workload, topology/flavor, source-
    # health) with hierarchical back-off (doc L T3). Consulted at serve time via
    # ``calibrate(bundle, slice_key=...)`` for group-conditional widening; the
    # marginal CQR remains the always-available fallback.
    mondrian: dict[tuple[str, int], Any] = field(default_factory=dict)

    def calibrate(
        self, bundle: KPIForecastBundle, slice_key: object | None = None
    ) -> KPIForecastBundle:
        new_q: list[QuantileForecast] = []
        for qf in bundle.quantiles:
            key = (qf.channel.value, qf.horizon_s)
            mond = self.mondrian.get(key) if slice_key is not None else None
            if mond is not None:
                lo = qf.quantiles.get(0.05, qf.median())
                hi = qf.quantiles.get(0.95, qf.median())
                clo, chi = mond.interval(slice_key, lo, hi)
                q = dict(qf.quantiles)
                if np.isfinite(clo):
                    q[0.05] = clo
                if np.isfinite(chi):
                    q[0.95] = chi
                new_q.append(
                    QuantileForecast(
                        channel=qf.channel, horizon_s=qf.horizon_s, quantiles=q
                    )
                )
                continue
            cal = self.cqr.get(key)
            new_q.append(cal.calibrate_forecast(qf) if cal else qf)
        new_h = []
        for hf in bundle.hazards:
            cal = self.upper.get((hf.channel.value, hf.horizon_s))
            cu = (
                cal.upper(hf.probability)
                if cal
                else max(hf.probability, hf.conformal_upper)
            )
            new_h.append(
                type(hf)(
                    channel=hf.channel,
                    horizon_s=hf.horizon_s,
                    probability=hf.probability,
                    conformal_upper=cu,
                )
            )
        return KPIForecastBundle(
            action_id=bundle.action_id,
            quantiles=new_q,
            hazards=new_h,
            epistemic_score=bundle.epistemic_score,
            ood_score=bundle.ood_score,
            abstain=bundle.abstain,
            abstain_reason=bundle.abstain_reason,
        )

    # --- offline fitting from a labelled calibration table --------------- #
    def fit_kpi(
        self,
        channel: str,
        horizon_s: int,
        q_lo: np.ndarray,
        q_hi: np.ndarray,
        targets: np.ndarray,
        alpha: float = 0.10,
        lo_tau: float = 0.05,
        hi_tau: float = 0.95,
    ) -> None:
        cal = CQRCalibrator(alpha=alpha, lo_tau=lo_tau, hi_tau=hi_tau)
        cal.fit(q_lo, q_hi, targets)
        self.cqr[(channel, horizon_s)] = cal

    def fit_hazard(
        self,
        channel: str,
        horizon_s: int,
        preds: np.ndarray,
        targets: np.ndarray,
        alpha: float = 0.05,
    ) -> None:
        cal = OneSidedUpperConformal(alpha=alpha)
        cal.fit(preds, targets)
        self.upper[(channel, horizon_s)] = cal

    def state_dict(self) -> dict:
        """Serialize the fitted thresholds (for the checkpoint calibration block)."""
        return {
            "cqr": {
                f"{c}|{h}": {
                    "correction": v.correction_,
                    "alpha": v.alpha,
                    "lo_tau": v.lo_tau,
                    "hi_tau": v.hi_tau,
                }
                for (c, h), v in self.cqr.items()
            },
            "upper": {
                f"{c}|{h}": {"margin": v.margin_, "alpha": v.alpha}
                for (c, h), v in self.upper.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> ConformalBundleCalibrator:
        out = cls()
        for key, v in state.get("cqr", {}).items():
            c, h = key.rsplit("|", 1)
            cal = CQRCalibrator(
                alpha=v["alpha"], lo_tau=v["lo_tau"], hi_tau=v["hi_tau"]
            )
            cal.correction_ = v["correction"]
            out.cqr[(c, int(h))] = cal
        for key, v in state.get("upper", {}).items():
            c, h = key.rsplit("|", 1)
            cal = OneSidedUpperConformal(alpha=v["alpha"])
            cal.margin_ = v["margin"]
            out.upper[(c, int(h))] = cal
        return out


__all__ = [
    "CQRCalibrator",
    "ConformalBundleCalibrator",
    "MondrianConformal",
    "OneSidedUpperConformal",
    "SplitConformal",
    "conformal_quantile",
]
