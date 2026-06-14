# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Tabular myopic learner baseline (doc F baseline #3).

A LightGBM / scikit-learn quantile-gradient-boosting model on **flattened
state-action features** that predicts 15/30/60-minute KPI deltas and ranks
candidate actions. This is the "plain ML" comparator the brief insists on: it
has *no temporal hierarchy and no typed object relations* — only a global
feature vector — so it is the foil that shows whether the relational world model
earns its complexity (doc F ablations).

The heavy learner is optional (HARD RULE 3): LightGBM is preferred, scikit-learn
``GradientBoostingRegressor`` (with ``loss='quantile'``) is the next choice, and
if neither is installed the baseline falls back to a closed-form **ridge
quantile-by-pinball** linear model fit with a couple of gradient steps in NumPy.
All three paths satisfy the same ``fit`` / ``score_actions`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...schema.actions import Action, action_workload_uid
from ...schema.kpis import (
    HORIZONS_S,
    QUANTILES,
    HazardChannel,
    HazardForecast,
    KPIChannel,
    KPIForecastBundle,
    QuantileForecast,
)
from ...schema.state import WorldState
from .features import action_onehot, flatten_state

try:  # preferred
    import lightgbm as _lgb  # type: ignore

    HAS_LGB = True
except ImportError:
    _lgb = None  # type: ignore
    HAS_LGB = False

try:  # next best
    from sklearn.ensemble import GradientBoostingRegressor  # type: ignore

    HAS_SKLEARN = True
except ImportError:
    GradientBoostingRegressor = None  # type: ignore
    HAS_SKLEARN = False


# Channels this baseline predicts (continuous deltas at the planner horizons).
_TARGET_CHANNELS: tuple[KPIChannel, ...] = (
    KPIChannel.QUEUE_WAIT_DELTA_S,
    KPIChannel.ADMITTED_GPU_HOURS_DELTA,
    KPIChannel.COMPLETION_TIME_DELTA_S,
)


def _pinball_linear_fit(
    X: np.ndarray, y: np.ndarray, tau: float, iters: int = 200, lr: float = 0.05
) -> np.ndarray:
    """Tiny gradient-descent quantile linear regression (NumPy fallback)."""
    n, d = X.shape
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
    w = np.zeros((d + 1,), dtype=np.float64)
    for _ in range(iters):
        pred = Xb @ w
        err = y - pred
        grad_per = np.where(err >= 0, -tau, (1.0 - tau))  # d ρ_τ / d pred
        grad = (Xb.T @ grad_per) / n
        w -= lr * grad
    return w


@dataclass(slots=True)
class TabularLearnerBaseline:
    """Flattened state-action quantile learner (LightGBM / sklearn / NumPy).

    ``models[(channel, horizon, tau)]`` holds one fitted quantile regressor.
    Fit with :meth:`fit` over a table of ``(state, action, target)`` rows; query
    with :meth:`score_actions`.
    """

    backend: str = "auto"  # auto | lightgbm | sklearn | numpy
    models: dict[tuple, object] = field(default_factory=dict)
    _resolved: str = "numpy"

    def __post_init__(self) -> None:
        if self.backend == "auto":
            self._resolved = (
                "lightgbm" if HAS_LGB else ("sklearn" if HAS_SKLEARN else "numpy")
            )
        else:
            self._resolved = self.backend

    @property
    def resolved_backend(self) -> str:
        return self._resolved

    def _featurize(self, state: WorldState, action: Action) -> np.ndarray:
        return np.concatenate([flatten_state(state), action_onehot(action)])

    def fit(
        self,
        rows: list[tuple[WorldState, Action]],
        targets: dict[tuple, np.ndarray],
    ) -> TabularLearnerBaseline:
        """Fit one quantile model per ``(channel, horizon, tau)`` (doc F #3).

        ``targets[(channel.value, horizon)]`` is an array aligned with ``rows``
        of realized KPI-delta values. A model is trained for every quantile in
        :data:`schema.QUANTILES`.
        """
        X = np.stack([self._featurize(s, a) for s, a in rows]).astype(np.float64)
        for ch in _TARGET_CHANNELS:
            for h in HORIZONS_S:
                y = targets.get((ch.value, h))
                if y is None:
                    continue
                y = np.asarray(y, dtype=np.float64)
                for tau in QUANTILES:
                    self.models[(ch.value, h, tau)] = self._fit_one(X, y, tau)
        return self

    def _fit_one(self, X: np.ndarray, y: np.ndarray, tau: float) -> object:
        if self._resolved == "lightgbm":
            ds = _lgb.Dataset(X, label=y)
            params = {
                "objective": "quantile",
                "alpha": tau,
                "num_leaves": 15,
                "learning_rate": 0.1,
                "min_data_in_leaf": 5,
                "verbosity": -1,
            }
            return _lgb.train(params, ds, num_boost_round=50)
        if self._resolved == "sklearn":
            gbr = GradientBoostingRegressor(
                loss="quantile", alpha=tau, n_estimators=80, max_depth=3
            )
            gbr.fit(X, y)
            return gbr
        return _pinball_linear_fit(X, y, tau)

    def _predict_one(self, model: object, x: np.ndarray) -> float:
        if self._resolved == "lightgbm":
            return float(model.predict(x[None, :])[0])  # type: ignore
        if self._resolved == "sklearn":
            return float(model.predict(x[None, :])[0])  # type: ignore
        xb = np.concatenate([x, [1.0]])
        return float(xb @ np.asarray(model))

    def score_actions(
        self, state: WorldState, actions: list[Action], horizon_s: int = 1800
    ) -> list[KPIForecastBundle]:
        out: list[KPIForecastBundle] = []
        for a in actions:
            x = self._featurize(state, a)
            quantiles: list[QuantileForecast] = []
            for ch in _TARGET_CHANNELS:
                for h in HORIZONS_S:
                    qd: dict[float, float] = {}
                    for tau in QUANTILES:
                        model = self.models.get((ch.value, h, tau))
                        qd[tau] = (
                            self._predict_one(model, x) if model is not None else 0.0
                        )
                    # enforce monotonicity (the boosters can cross)
                    qd = _sorted_quantiles(qd)
                    quantiles.append(
                        QuantileForecast(channel=ch, horizon_s=h, quantiles=qd)
                    )
            # myopic: a constant low base-rate hazard (no event modelling)
            hazards = [
                HazardForecast(
                    channel=HazardChannel.REQUEUE,
                    horizon_s=h,
                    probability=0.05,
                    conformal_upper=0.05,
                )
                for h in HORIZONS_S
            ]
            out.append(
                KPIForecastBundle(
                    action_id=f"{action_workload_uid(a) or '*'}::{a.type.value}",
                    quantiles=quantiles,
                    hazards=hazards,
                )
            )
        return out


def _sorted_quantiles(qd: dict[float, float]) -> dict[float, float]:
    taus = sorted(qd)
    vals = np.maximum.accumulate([qd[t] for t in taus])
    return {t: float(v) for t, v in zip(taus, vals)}


__all__ = ["HAS_LGB", "HAS_SKLEARN", "TabularLearnerBaseline"]
