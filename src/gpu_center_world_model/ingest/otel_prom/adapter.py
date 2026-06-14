# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""OpenTelemetry / Prometheus adapter — the vendor-neutral collection backbone.

Prometheus exposition and the OpenTelemetry Collector are the open receivers that
keep the release vendor-neutral (doc D). Rather than re-implement each upstream
source, this adapter parses a raw Prometheus *exposition* scrape (or an
OTLP-style JSON metric list) into generic, label-keyed time series and routes
each series to a canonical :class:`Observation` by metric name.

It deliberately reuses the *same* canonical DCGM/host field names the dedicated
adapters use, so a scrape that already aggregates DCGM + node_exporter through a
Prometheus endpoint lands on identical channels. The routing table maps metric
prefixes to (entity_kind, family, id-label); unknown metrics are kept as generic
``misc`` channels on their labelled entity rather than dropped, so nothing is
silently lost. Parsing is pure-stdlib (no ``prometheus_client`` dependency).

Snapshot shape — either form is accepted::

    {"ts_ns": ..., "exposition": "<raw prometheus text>"}
    {"ts_ns": ..., "metrics": [{"name":..,"labels":{..},"value":..,"age_s":..}]}
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ...schema.observations import Observation, ObservationFamily
from ..base import (
    BaseAdapter,
    IngestPayload,
    make_channel,
    register_adapter,
)

# ── Prometheus exposition line grammar ───────────────────────────────────────
#   metric_name{label="v",...} value [timestamp]
_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[^\s]+)"
    r"(?:\s+(?P<ts>[0-9.eE+-]+))?\s*$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


@dataclass(slots=True)
class PromSample:
    """One parsed Prometheus/OTLP sample with labels."""

    name: str
    labels: dict[str, str]
    value: float
    age_s: float = 0.0


def parse_exposition(text: str) -> list[PromSample]:
    """Parse Prometheus exposition text into samples (tolerant of junk lines).

    Comments (``# HELP`` / ``# TYPE``) and unparseable / NaN lines are skipped
    rather than raising — a partial scrape still yields whatever it can.
    """
    out: list[PromSample] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        val_str = m.group("value")
        try:
            value = float(val_str)
        except ValueError:
            continue
        if value != value:  # NaN -> treat as missing, skip
            continue
        labels = dict(_LABEL_RE.findall(m.group("labels") or ""))
        out.append(PromSample(name=m.group("name"), labels=labels, value=value))
    return out


# ── Metric routing: prefix -> (entity_kind, family, id-label candidates) ──────
@dataclass(slots=True)
class Route:
    entity_kind: str
    family: ObservationFamily
    id_labels: tuple[str, ...]


_ROUTES: tuple[tuple[str, Route], ...] = (
    (
        "DCGM_FI_PROF_",
        Route(
            "gpu",
            ObservationFamily.PROFILING_TELEMETRY,
            ("UUID", "gpu", "gpu_uuid"),
        ),
    ),
    (
        "DCGM_FI_",
        Route("gpu", ObservationFamily.GPU_TELEMETRY, ("UUID", "gpu", "gpu_uuid")),
    ),
    (
        "node_",
        Route(
            "node",
            ObservationFamily.NODE_RACK_TELEMETRY,
            ("node", "instance", "node_uid"),
        ),
    ),
    (
        "kueue_",
        Route(
            "cluster_queue",
            ObservationFamily.SCHEDULER_STATE,
            ("cluster_queue", "queue", "name"),
        ),
    ),
)
_DEFAULT_ROUTE = Route(
    "misc", ObservationFamily.EXOGENOUS_CONTEXT, ("entity", "id", "instance")
)


def _route_for(name: str) -> Route:
    for prefix, route in _ROUTES:
        if name.startswith(prefix):
            return route
    return _DEFAULT_ROUTE


def _entity_id(sample: PromSample, route: Route) -> str:
    for lbl in route.id_labels:
        val = sample.labels.get(lbl)
        if val:
            return val
    return "unknown"


@register_adapter("otel_prom")
class OtelPromAdapter(BaseAdapter):
    """Parse a Prometheus/OTLP scrape and route series to canonical observations."""

    source_name = "otel_prom"
    default_family = ObservationFamily.EXOGENOUS_CONTEXT

    def parse(self, snapshot: dict[str, Any], payload: IngestPayload) -> None:
        ts_ns = int(snapshot.get("ts_ns") or 0)
        samples = self._collect_samples(snapshot)
        # group: (entity_kind, entity_id, family) -> {channel: (value, age)}
        grouped: dict[tuple[str, str, ObservationFamily], dict[str, PromSample]] = {}
        for s in samples:
            route = _route_for(s.name)
            eid = _entity_id(s, route)
            key = (route.entity_kind, eid, route.family)
            grouped.setdefault(key, {})[s.name] = s

        for (kind, eid, family), chans in grouped.items():
            channels = {
                name: make_channel(name, s.value, age_s=s.age_s)
                for name, s in chans.items()
            }
            payload.add_observation(
                Observation(
                    entity_kind=kind,
                    entity_id=eid,
                    family=family,
                    ts_event_ns=ts_ns,
                    channels=channels,
                )
            )

    @staticmethod
    def _collect_samples(snapshot: dict[str, Any]) -> list[PromSample]:
        samples: list[PromSample] = []
        exposition = snapshot.get("exposition")
        if isinstance(exposition, str):
            samples.extend(parse_exposition(exposition))
        # OTLP-style explicit metric list (already structured).
        for m in snapshot.get("metrics", []) or []:
            if not isinstance(m, dict):
                continue
            name = m.get("name")
            value = m.get("value")
            if name is None or value is None:
                continue
            try:
                fval = float(value)
            except (TypeError, ValueError):
                continue
            samples.append(
                PromSample(
                    name=str(name),
                    labels={str(k): str(v) for k, v in (m.get("labels") or {}).items()},
                    value=fval,
                    age_s=float(m.get("age_s", 0.0) or 0.0),
                )
            )
        return samples


__all__ = ["OtelPromAdapter", "PromSample", "parse_exposition"]
