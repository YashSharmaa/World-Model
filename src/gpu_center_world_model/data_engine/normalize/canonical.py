# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Canonical normalisation: Bronze → Silver → Gold (design doc D).

The data engine maps raw ingest snapshots/events into the canonical
:class:`~...schema.state.SymbolicState` + :class:`~...schema.observations.Observation`
streams through three layers (doc D, Data engine architecture):

* **Bronze** — an append-only event store of raw ``MetricSample`` / ``SchedulerEvent``
  rows tagged with origin, source, entity, cadence, and a quality mask. We never
  rewrite Bronze; it is the audit trail.
* **Silver** — raw rows aligned onto entity tables at *three synchronised lanes*
  (doc D, Silver layer): a **fast** lane (1-5 s GPU power/util/thermal), a **meso**
  lane (15-30 s node/rack/queue usage), and a **sparse event** lane (admissions,
  placements, evictions, drains, faults). Lanes are kept separate — we never force
  all channels onto one synchronous frame by zero-imputation.
* **Gold** — the decision-window object (built in :mod:`...windows`).

Two hard rules from doc D are enforced here:

* **Missingness / staleness / source-health is first-class STATE, never noise.**
  A missing channel keeps ``available=False`` and carries its ``age_s``; it is
  *never* zero-filled (doc D correlation_rules #5; doc state ``M_t``). Failures
  often surface first as missing telemetry, so the absence is the signal.
* **Origin is provenance, not a default model feature** (doc D Bronze schema): we
  carry it for auditing but it does not enter the canonical observation channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ...schema.observations import (
    FAMILY_CADENCE_S,
    Channel,
    Observation,
    ObservationFamily,
    SourceHealth,
)
from ...schema.state import SourceState


class Lane(str, Enum):
    """The three Silver alignment lanes (doc D Silver layer)."""

    FAST = "fast"  # 1-5 s : GPU power / util / thermal / host-net counters
    MESO = "meso"  # 15-30 s : node / rack / queue usage
    SPARSE = "sparse"  # event lane : admit / place / evict / drain / fault


#: Map each observation family to its Silver lane (doc D, three-lane design).
FAMILY_LANE: dict[ObservationFamily, Lane] = {
    ObservationFamily.GPU_TELEMETRY: Lane.FAST,
    ObservationFamily.PROFILING_TELEMETRY: Lane.FAST,
    ObservationFamily.NODE_RACK_TELEMETRY: Lane.MESO,
    ObservationFamily.PLACEMENT_STATE: Lane.MESO,
    ObservationFamily.SCHEDULER_STATE: Lane.SPARSE,
    ObservationFamily.EXOGENOUS_CONTEXT: Lane.MESO,
}


@dataclass(slots=True)
class MetricSample:
    """A Bronze metric row (doc D Bronze ``metric_sample`` schema)."""

    ts_event_ns: int
    origin: str  # real | sim — provenance only, not a model feature
    source: str  # dcgm | kueue | node_exporter | redfish | otlp
    entity_kind: str
    entity_id: str
    metric_name: str
    value: float
    unit: str = ""
    cadence_ms: int = 0
    quality_mask: int = 1  # 1 = good, 0 = degraded/missing (doc D)
    scenario_id: str | None = None


@dataclass(slots=True)
class SchedulerEventRow:
    """A Bronze scheduler/intervention row (doc D ``scheduler_event`` schema)."""

    ts_event_ns: int
    origin: str
    event_type: str  # submit | admit | reserve | place | evict | finish | ...
    object_kind: str
    object_uid: str
    actor: str = "simulator"
    reason: str = ""
    state_hash: str | None = None  # pre-action hash for interventions (doc D)


@dataclass(slots=True)
class SilverFrame:
    """A Silver-aligned bundle for one tick on one lane (doc D Silver layer)."""

    lane: Lane
    ts_ns: int
    observations: list[Observation] = field(default_factory=list)
    events: list[SchedulerEventRow] = field(default_factory=list)


#: Metric-name → DCGM canonical channel mapping (the auditable column names).
_METRIC_TO_CHANNEL: dict[str, str] = {
    "gpu_util": "DCGM_FI_DEV_GPU_UTIL",
    "mem_util": "DCGM_FI_DEV_MEM_COPY_UTIL",
    "power_w": "DCGM_FI_DEV_POWER_USAGE",
    "temp_c": "DCGM_FI_DEV_GPU_TEMP",
    "memory_temp_c": "DCGM_FI_DEV_MEMORY_TEMP",
    "xid": "DCGM_FI_DEV_XID_ERRORS",
    "energy_mj": "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION",
}


def _channel_name(metric_name: str) -> str:
    return _METRIC_TO_CHANNEL.get(metric_name, metric_name)


def _family_for_source(source: str, entity_kind: str) -> ObservationFamily:
    if source == "dcgm":
        return ObservationFamily.GPU_TELEMETRY
    if source in ("node_exporter", "redfish"):
        return ObservationFamily.NODE_RACK_TELEMETRY
    if source == "kueue":
        return (
            ObservationFamily.SCHEDULER_STATE
            if entity_kind in ("workload", "queue", "cluster_queue", "local_queue")
            else ObservationFamily.PLACEMENT_STATE
        )
    return ObservationFamily.EXOGENOUS_CONTEXT


def bronze_to_silver(
    samples: list[MetricSample],
    events: list[SchedulerEventRow] | None = None,
    *,
    now_ns: int | None = None,
) -> dict[Lane, list[SilverFrame]]:
    """Align Bronze rows onto Silver entity tables across the three lanes.

    Rows are grouped by (lane, tick, entity); each metric becomes a typed
    :class:`Channel`. A ``quality_mask == 0`` row is preserved as an *unavailable*
    channel carrying its age-of-information — never dropped or zero-filled
    (doc D correlation_rules #5). The resulting frames are sorted by time so the
    windows builder can slice strictly-before-t0 features.
    """
    events = events or []
    # group metric samples by (lane, tick_key, entity) keeping missingness
    by_lane: dict[Lane, dict[tuple[int, str, str], Observation]] = {
        Lane.FAST: {},
        Lane.MESO: {},
        Lane.SPARSE: {},
    }
    for s in samples:
        family = _family_for_source(s.source, s.entity_kind)
        lane = FAMILY_LANE[family]
        cadence_s = FAMILY_CADENCE_S[family]
        tick_key = int(s.ts_event_ns // int(max(cadence_s, 1) * 1e9))
        key = (tick_key, s.entity_kind, s.entity_id)
        obs = by_lane[lane].get(key)
        if obs is None:
            obs = Observation(
                entity_kind=s.entity_kind,
                entity_id=s.entity_id,
                family=family,
                ts_event_ns=s.ts_event_ns,
            )
            by_lane[lane][key] = obs
        available = s.quality_mask != 0
        ref_ns = now_ns if now_ns is not None else s.ts_event_ns
        age_s = max(0.0, (ref_ns - s.ts_event_ns) / 1e9)
        health = SourceHealth.OBSERVED if available else SourceHealth.DROPPED
        if available and age_s > 2 * FAMILY_CADENCE_S[family]:
            health = SourceHealth.STALE
        obs.channels[_channel_name(s.metric_name)] = Channel(
            name=_channel_name(s.metric_name),
            value=s.value,
            available=available,
            age_s=age_s,
            source_health=health,
        )

    out: dict[Lane, list[SilverFrame]] = {Lane.FAST: [], Lane.MESO: [], Lane.SPARSE: []}
    for lane, table in by_lane.items():
        # bucket observations into per-tick frames
        frames: dict[int, SilverFrame] = {}
        for (tick_key, _ek, _eid), obs in table.items():
            fr = frames.get(tick_key)
            if fr is None:
                fr = SilverFrame(lane=lane, ts_ns=obs.ts_event_ns)
                frames[tick_key] = fr
            fr.observations.append(obs)
            fr.ts_ns = min(fr.ts_ns, obs.ts_event_ns)
        out[lane] = [frames[k] for k in sorted(frames)]

    # sparse event lane carries the scheduler/intervention rows
    for ev in sorted(events, key=lambda e: e.ts_event_ns):
        out[Lane.SPARSE].append(
            SilverFrame(lane=Lane.SPARSE, ts_ns=ev.ts_event_ns, events=[ev])
        )
    out[Lane.SPARSE].sort(key=lambda f: f.ts_ns)
    return out


def source_state_from_silver(frames: list[SilverFrame]) -> SourceState:
    """Derive the observation-source state ``M_t`` from Silver missingness (doc T).

    Aggregates per-channel availability, staleness, and age into the canonical
    :class:`SourceState` block so the filter/planner can gate on telemetry
    integrity (doc policy AbstentionConfig observation-integrity gate). This keeps
    missingness as state rather than as silently dropped rows.
    """
    collector_health: dict[str, str] = {}
    age: dict[str, float] = {}
    missing: dict[str, float] = {}
    counts: dict[str, int] = {}
    miss_counts: dict[str, int] = {}
    for fr in frames:
        for obs in fr.observations:
            key = f"{obs.entity_kind}:{obs.entity_id}"
            for ch in obs.channels.values():
                counts[key] = counts.get(key, 0) + 1
                age[key] = max(age.get(key, 0.0), ch.age_s)
                if not ch.available:
                    miss_counts[key] = miss_counts.get(key, 0) + 1
                if ch.source_health != SourceHealth.OBSERVED:
                    collector_health[key] = ch.source_health.value
    for key, n in counts.items():
        missing[key] = miss_counts.get(key, 0) / max(n, 1)
    return SourceState(
        collector_health=collector_health,
        channel_missing_prob=missing,
        channel_age_s=age,
    )


def observations_to_metric_samples(
    observations: list[Observation], origin: str = "sim"
) -> list[MetricSample]:
    """Round-trip emitted :class:`Observation` streams back into Bronze rows.

    Lets the synthetic generator's emissions enter the same Bronze→Silver→Gold
    pipeline as real telemetry, so synthetic and real share one canonical schema
    (doc D prioritized recommendation). Missingness is preserved via the quality
    mask.
    """
    rows: list[MetricSample] = []
    source = {
        ObservationFamily.GPU_TELEMETRY: "dcgm",
        ObservationFamily.PROFILING_TELEMETRY: "dcgm",
        ObservationFamily.NODE_RACK_TELEMETRY: "node_exporter",
        ObservationFamily.PLACEMENT_STATE: "kueue",
        ObservationFamily.SCHEDULER_STATE: "kueue",
        ObservationFamily.EXOGENOUS_CONTEXT: "otlp",
    }
    for obs in observations:
        src = source.get(obs.family, "otlp")
        for ch in obs.channels.values():
            rows.append(
                MetricSample(
                    ts_event_ns=obs.ts_event_ns,
                    origin=origin,
                    source=src,
                    entity_kind=obs.entity_kind,
                    entity_id=obs.entity_id,
                    metric_name=ch.name,
                    value=ch.value,
                    quality_mask=1 if ch.available else 0,
                )
            )
    return rows


__all__ = [
    "FAMILY_LANE",
    "Lane",
    "MetricSample",
    "SchedulerEventRow",
    "SilverFrame",
    "bronze_to_silver",
    "observations_to_metric_samples",
    "source_state_from_silver",
]
