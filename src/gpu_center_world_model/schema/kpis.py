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
    """Continuous KPI-delta channels (quantile heads).

    These reconcile one-to-one with the doc I generative-model ``kpi_map``
    (``KPI_t = g_K(X_t, Z_t)``): queue-wait delta, admitted GPU-hours delta,
    topology-fragmentation delta, rack power & thermal headroom quantiles, and
    completion-time/slowdown quantiles. Two channels are read-model refinements
    of that map rather than new physics:

    * :attr:`CROSS_RACK_RATE` is the **fragmentation proxy** — the fraction of a
      gang's replicas that land off its primary rack. It is the observable that
      *drives* :attr:`TOPOLOGY_FRAGMENTATION_DELTA` (a spread gang fragments the
      pod), so the two are folded together: ``cross_rack_rate`` is the raw rate
      and ``topology_fragmentation_delta`` its baseline-relative KPI delta.
    * :attr:`FAIRNESS_SHARE_DEBT_DELTA` surfaces the planner-utility fairness term
      ``lambda_unfair * share_debt`` (doc C planner utility, step 4) as its own
      KPI-delta channel, where ``share_debt`` is a ClusterQueue's
      Admission-Fair-Sharing ``weightedShare`` above its entitlement.
    """

    QUEUE_WAIT_DELTA_S = "queue_wait_delta_s"
    ADMITTED_GPU_HOURS_DELTA = "admitted_gpu_hours_delta"
    TOPOLOGY_FRAGMENTATION_DELTA = "topology_fragmentation_delta"
    RACK_POWER_KW = "rack_power_kw"
    RACK_THERMAL_HEADROOM_C = "rack_thermal_headroom_c"
    COMPLETION_TIME_DELTA_S = "completion_time_delta_s"
    SLOWDOWN = "slowdown"
    CROSS_RACK_RATE = "cross_rack_rate"
    # Fairness KPI delta surfacing the planner-utility term lambda_unfair*share_debt
    # (doc C planner utility; doc V Admission Fair Sharing weightedShare).
    FAIRNESS_SHARE_DEBT_DELTA = "fairness_share_debt_delta"


#: Reconciliation of the doc I ``kpi_map`` contents to concrete KPI channels.
#: Each doc-I KPI line maps to the channel(s) that realise it, so the generative
#: model's ``KPI_t = g_K(X_t, Z_t)`` is auditable against this enum.
KPI_MAP: dict[str, tuple[KPIChannel, ...]] = {
    "queue_wait_delta": (KPIChannel.QUEUE_WAIT_DELTA_S,),
    "admitted_gpu_hours_delta": (KPIChannel.ADMITTED_GPU_HOURS_DELTA,),
    "topology_fragmentation_delta": (
        KPIChannel.TOPOLOGY_FRAGMENTATION_DELTA,
        KPIChannel.CROSS_RACK_RATE,  # fragmentation proxy feeding the delta
    ),
    "rack_power_thermal_quantiles": (
        KPIChannel.RACK_POWER_KW,
        KPIChannel.RACK_THERMAL_HEADROOM_C,
    ),
    "completion_time_slowdown": (
        KPIChannel.COMPLETION_TIME_DELTA_S,
        KPIChannel.SLOWDOWN,
    ),
    "fairness_share_debt": (KPIChannel.FAIRNESS_SHARE_DEBT_DELTA,),
}


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
    "KPI_MAP",
    "QUANTILES",
    "HazardChannel",
    "HazardForecast",
    "KPIChannel",
    "KPIForecastBundle",
    "QuantileForecast",
]
