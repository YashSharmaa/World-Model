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


@runtime_checkable
class AmortizationHeads(Protocol):
    """The amortized planner-to-policy heads on top of the world model (doc M D1).

    Three heads distilled from offline full-enumeration replay (doc M
    ``amortization_plan`` D1-D3). They *accelerate* the planner — propose / warm-
    start / decide-to-trust — but never override the exact kernel: the emitted
    action is always selected from the shielded safe set, so amortization cannot
    break the hard-safety certificate (doc M G1). This Protocol is additive; a
    backend that does not implement it simply leaves the planner in exact mode.

    * ``proposal_scores`` — the proposal head ``π_φ(a|b)``: a recall score per
      safe candidate so the planner can keep only the top-k for scoring (D1).
    * ``value`` — the terminal value head ``V_ψ(b)`` / ``Q_ψ(b,a)`` aligned to the
      planner's scalarized lex score ``S_t``: warm-starts and early-terminates
      bounded search (D1/D3).
    * ``abstain_score`` — the abstention / OOD head ``u_ω(b,a)``: the disagreement
      / OOD signal the guarded inference path consults to decide whether to trust
      the amortized proposal or fall back to exhaustive scoring (D1/D3/G2).
    """

    def proposal_scores(
        self, state: WorldState, actions: list[Action], horizon_s: int
    ) -> list[float]:
        """Per-candidate recall score ``π_φ(a|b)`` (higher ⇒ more likely oracle)."""
        ...

    def value(
        self, state: WorldState, action: Action | None, horizon_s: int
    ) -> float:
        """Terminal value ``V_ψ(b)`` (``action is None``) or ``Q_ψ(b,a)``.

        Returned in the planner's scalarized lex-score units (lower is better),
        so it can warm-start / terminate bounded search directly.
        """
        ...

    def abstain_score(
        self, state: WorldState, action: Action, horizon_s: int
    ) -> float:
        """Disagreement / OOD score ``u_ω(b,a)`` in ``[0, 1]`` (higher ⇒ distrust)."""
        ...


__all__ = ["AmortizationHeads", "Calibrator", "WorldModel"]
