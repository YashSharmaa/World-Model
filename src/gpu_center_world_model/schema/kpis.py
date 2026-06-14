# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Planner-facing KPI and hazard targets — the sigma-algebra worth predicting.

The learned world model is a *predictive-sufficient statistic for these
targets*, conditioned on the exact kernel and candidate action — not a
reconstruction code for raw telemetry (doc P, information bottleneck). KPIs are
expressed as **deltas relative to a declared baseline action** (typically
``hold``) and are predicted as distributions (quantiles) plus event hazards,
never as bare point estimates (docs F, J, L).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

#: Planning horizons in seconds (doc C/F: 15, 30, 60 minutes; +5 min fast probe).
HORIZONS_S: tuple[int, ...] = (300, 900, 1800, 3600)

#: Quantile grid for KPI-delta heads (doc L conformalized quantile regression).
QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.50, 0.90, 0.95)


class KPIChannel(str, Enum):
    """Continuous KPI-delta channels (quantile heads)."""

    QUEUE_WAIT_DELTA_S = "queue_wait_delta_s"
    ADMITTED_GPU_HOURS_DELTA = "admitted_gpu_hours_delta"
    TOPOLOGY_FRAGMENTATION_DELTA = "topology_fragmentation_delta"
    RACK_POWER_KW = "rack_power_kw"
    RACK_THERMAL_HEADROOM_C = "rack_thermal_headroom_c"
    COMPLETION_TIME_DELTA_S = "completion_time_delta_s"
    SLOWDOWN = "slowdown"
    CROSS_RACK_RATE = "cross_rack_rate"


class HazardChannel(str, Enum):
    """Event-hazard channels (discrete-time survival / competing risks).

    Mutually-exclusive terminal events are modelled as competing risks; co-
    occurring incidents use parallel binary hazards (docs L, U).
    """

    HOTSPOT = "hotspot"
    PODS_READY_TIMEOUT = "pods_ready_timeout"
    REQUEUE = "requeue"
    XID_FAULT = "xid_fault"
    HEADROOM_VIOLATION = "headroom_violation"
    NODE_FAILURE = "node_failure"


@dataclass(slots=True)
class QuantileForecast:
    """A horizon-indexed quantile forecast for one KPI channel."""

    channel: KPIChannel
    horizon_s: int
    quantiles: dict[float, float] = field(default_factory=dict)  # tau -> value

    def median(self) -> float:
        return self.quantiles.get(0.5, 0.0)


@dataclass(slots=True)
class HazardForecast:
    """A horizon-indexed hazard probability with a one-sided conformal bound."""

    channel: HazardChannel
    horizon_s: int
    probability: float = 0.0
    conformal_upper: float = 1.0  # one-sided (1-alpha) upper bound (doc L)


@dataclass(slots=True)
class KPIForecastBundle:
    """All planner-facing forecasts for one candidate action.

    This is the contract the planner consumes to rank safe candidates and the
    serving layer renders into the UI's branch-delta ribbon.
    """

    action_id: str
    quantiles: list[QuantileForecast] = field(default_factory=list)
    hazards: list[HazardForecast] = field(default_factory=list)
    # epistemic disagreement + OOD score feeding the abstention gate
    epistemic_score: float = 0.0
    ood_score: float = 0.0
    abstain: bool = False
    abstain_reason: str | None = None


__all__ = [
    "HORIZONS_S",
    "QUANTILES",
    "HazardChannel",
    "HazardForecast",
    "KPIChannel",
    "KPIForecastBundle",
    "QuantileForecast",
]
