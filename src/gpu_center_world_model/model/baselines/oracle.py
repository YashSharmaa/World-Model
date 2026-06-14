# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Oracle upper-bound baseline (doc F baseline #6).

The oracle uses the **branch reference future** (hindsight) rather than a learned
prediction, so it separates *model limits from search limits* (doc F): if the
planner still picks badly given oracle forecasts, the bottleneck is search /
ranking, not the dynamics model.

It is not a deployable model — it requires the realized branch outcomes that are
only available offline in the counterfactual harness. It implements the same
``score_actions`` surface by reading per-action ground-truth KPI deltas and
hazard outcomes supplied as a lookup keyed by ``action_id``. Quantile bands are
degenerate (point mass at the realized value) because the oracle has no
uncertainty: it *is* the truth (relative to the branch reference environment).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...schema.actions import Action, action_workload_uid
from ...schema.kpis import (
    HazardChannel,
    HazardForecast,
    KPIChannel,
    KPIForecastBundle,
    QuantileForecast,
)
from ...schema.state import WorldState


@dataclass(slots=True)
class BranchOutcome:
    """A realized branch outcome for one action (the oracle's ground truth)."""

    kpi_deltas: dict[tuple[str, int], float] = field(
        default_factory=dict
    )  # (chan, h)->v
    hazard_events: dict[tuple[str, int], bool] = field(default_factory=dict)


@dataclass(slots=True)
class OracleBaseline:
    """Hindsight oracle that reports realized branch outcomes as forecasts.

    ``outcomes`` maps ``action_id`` -> :class:`BranchOutcome`. Unknown actions
    fall back to a neutral zero forecast (so the oracle never crashes on a
    candidate it has no branch for; that candidate simply looks neutral).
    """

    outcomes: dict[str, BranchOutcome] = field(default_factory=dict)

    def set_outcome(self, action_id: str, outcome: BranchOutcome) -> None:
        self.outcomes[action_id] = outcome

    def score_actions(
        self, state: WorldState, actions: list[Action], horizon_s: int = 1800
    ) -> list[KPIForecastBundle]:
        out: list[KPIForecastBundle] = []
        for a in actions:
            aid = f"{action_workload_uid(a) or '*'}::{a.type.value}"
            oc = self.outcomes.get(aid, BranchOutcome())
            quantiles = [
                QuantileForecast(
                    channel=KPIChannel(ch),
                    horizon_s=h,
                    quantiles={0.05: v, 0.5: v, 0.95: v},  # degenerate (truth)
                )
                for (ch, h), v in oc.kpi_deltas.items()
            ]
            hazards = [
                HazardForecast(
                    channel=HazardChannel(ch),
                    horizon_s=h,
                    probability=1.0 if occurred else 0.0,
                    conformal_upper=1.0 if occurred else 0.0,
                )
                for (ch, h), occurred in oc.hazard_events.items()
            ]
            out.append(
                KPIForecastBundle(action_id=aid, quantiles=quantiles, hazards=hazards)
            )
        return out


__all__ = ["BranchOutcome", "OracleBaseline"]
