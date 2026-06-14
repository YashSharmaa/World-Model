# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Learned world-model interface contract (backend-neutral).

The learned model is used only behind a small contract so the simulator and
planner never depend on MLX internals (doc E/H boundary rule). The model is
*residual, action-conditioned, and query-conditioned*: it forecasts planner KPI
deltas and event hazards with calibrated uncertainty over the **safe candidate
set** the exact kernel already certified. It may rank, forecast, or abstain — it
may never expand the safe set (docs C, F, M).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..schema.actions import Action
from ..schema.kpis import KPIForecastBundle
from ..schema.state import WorldState


@runtime_checkable
class WorldModel(Protocol):
    """The residual, action-conditioned, query-conditioned world model."""

    def encode(self, state: WorldState) -> object:
        """Encode the symbolic+observation state into the latent belief ``Z_t``."""
        ...

    def forecast(
        self, state: WorldState, action: Action, horizon_s: int
    ) -> KPIForecastBundle:
        """Roll an action forward and return calibrated KPI/hazard forecasts."""
        ...

    def score_actions(
        self, state: WorldState, actions: list[Action], horizon_s: int
    ) -> list[KPIForecastBundle]:
        """Forecast a batch of safe candidate actions (one bundle each).

        This is the single compiled hot path the serving layer drives.
        """
        ...


@runtime_checkable
class Calibrator(Protocol):
    """Post-hoc conformal calibration of forecasts (doc L)."""

    def calibrate(self, bundle: KPIForecastBundle) -> KPIForecastBundle: ...


__all__ = ["Calibrator", "WorldModel"]
