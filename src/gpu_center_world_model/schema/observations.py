# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Observation families, cadences, and informative-missingness state.

Two design commitments drive this module:

1. **Multi-rate observation.** GPU power/temp/util move on the scale of seconds;
   queueing/readiness on the scale of minutes; some thermal/network bottlenecks
   persist far longer. Each arriving observation *family* contributes a
   likelihood factor at its own cadence (doc T, Eq. E4) — we never force all
   channels onto one synchronous frame by zero-imputation.

2. **Missingness / staleness / source-health are first-class STATE, not noise.**
   Real GPU failures often surface first as missing or degraded telemetry. Each
   channel carries an availability indicator and an age-of-information, and
   missingness is modelled as MNAR conditional on latent state (docs C, D, I, T,
   U). Zero-filling missing telemetry is forbidden inside the filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ObservationFamily(str, Enum):
    """Canonical observation families with their nominal cadences."""

    SCHEDULER_STATE = "scheduler_state"  # event + 5 min
    PLACEMENT_STATE = "placement_state"  # event + 1 min
    GPU_TELEMETRY = "gpu_telemetry"  # 5-30 s
    PROFILING_TELEMETRY = "profiling_telemetry"  # 5-30 s
    NODE_RACK_TELEMETRY = "node_rack_telemetry"  # 30 s - 1 min
    EXOGENOUS_CONTEXT = "exogenous_context"  # 5-15 min


#: Nominal cadence (seconds) per family — used by the tensorizer's three lanes.
FAMILY_CADENCE_S: dict[ObservationFamily, float] = {
    ObservationFamily.SCHEDULER_STATE: 300.0,
    ObservationFamily.PLACEMENT_STATE: 60.0,
    ObservationFamily.GPU_TELEMETRY: 15.0,
    ObservationFamily.PROFILING_TELEMETRY: 15.0,
    ObservationFamily.NODE_RACK_TELEMETRY: 45.0,
    ObservationFamily.EXOGENOUS_CONTEXT: 600.0,
}


class SourceHealth(str, Enum):
    """Telemetry source-health regime (doc T, M_t observation-source state)."""

    OBSERVED = "observed"
    STALE = "stale"
    DELAYED = "delayed"
    DROPPED = "dropped"


# Canonical DCGM-style GPU telemetry field names used as default columns.
GPU_TELEMETRY_FIELDS: tuple[str, ...] = (
    "DCGM_FI_DEV_GPU_UTIL",
    "DCGM_FI_DEV_MEM_COPY_UTIL",
    "DCGM_FI_DEV_POWER_USAGE",
    "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION",
    "DCGM_FI_DEV_GPU_TEMP",
    "DCGM_FI_DEV_MEMORY_TEMP",
    "DCGM_FI_DEV_XID_ERRORS",
    "DCGM_FI_DEV_GET_GPU_RECOVERY_ACTION",
)
GPU_PROFILING_FIELDS: tuple[str, ...] = (
    "DCGM_FI_PROF_SM_ACTIVE",
    "DCGM_FI_PROF_DRAM_ACTIVE",
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
)


@dataclass(slots=True)
class Channel:
    """A single observed telemetry channel with explicit integrity metadata.

    ``value`` is meaningful only when ``available`` is True. A consumer must
    never read ``value`` as a real measurement when ``available`` is False; the
    missingness itself (and ``age_s``) is the signal.
    """

    name: str
    value: float = 0.0
    available: bool = True
    age_s: float = 0.0  # age-of-information / staleness
    source_health: SourceHealth = SourceHealth.OBSERVED
    censor: str = "none"  # none | left | right | interval
    censor_lo: float = float("-inf")
    censor_hi: float = float("inf")


@dataclass(slots=True)
class Observation:
    """A multi-family observation snapshot for one entity at one tick.

    ``channels`` is keyed by channel name; ``family`` records which observation
    family produced this snapshot so multi-rate likelihood assembly stays exact.
    """

    entity_kind: str
    entity_id: str
    family: ObservationFamily
    ts_event_ns: int
    channels: dict[str, Channel] = field(default_factory=dict)

    def get(self, name: str) -> Channel | None:
        return self.channels.get(name)

    def is_present(self, name: str) -> bool:
        ch = self.channels.get(name)
        return ch is not None and ch.available


__all__ = [
    "FAMILY_CADENCE_S",
    "GPU_PROFILING_FIELDS",
    "GPU_TELEMETRY_FIELDS",
    "Channel",
    "Observation",
    "ObservationFamily",
    "SourceHealth",
]
