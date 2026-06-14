# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Offline snapshot replay — the default ingest path (no live cluster).

A *snapshot directory* is one tick of the world captured from every source as a
set of JSON files::

    snapshot_dir/
      kueue.json          # scheduler / control-plane state
      kubernetes.json     # nodes, gang readiness, optional DRA
      dcgm.json           # per-GPU telemetry
      node_exporter.json  # host hwmon / powercap / thermal-zone
      redfish.json        # out-of-band power / cooling / thermal
      otel_prom.json      # vendor-neutral scrape backbone
      context.json        # optional exogenous context (regime, ambient, ...)

:func:`load_snapshot_dir` runs every available adapter, merges their partial
views into one :class:`WorldState`, and folds the multi-rate observations into
``M_t`` (source-health, missingness, staleness) without ever zero-imputing a
missing measurement. Missing source files mark that collector *dropped* — this
is the canonical way the offline pipeline preserves informative missingness for
the data engine and examples (docs C, D, T, U).

:func:`load_snapshot_series` walks a parent directory of timestamped snapshot
sub-directories and yields a :class:`WorldState` per tick — the replay stream the
data engine consumes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..schema.observations import Observation, ObservationFamily, SourceHealth
from ..schema.state import ExogenousContext, SourceState, WorldState

# Importing the adapter modules registers them in the shared REGISTRY.
from . import dcgm as _dcgm  # noqa: F401
from . import kubernetes as _kubernetes  # noqa: F401
from . import kueue as _kueue  # noqa: F401
from . import node_exporter as _node_exporter  # noqa: F401
from . import otel_prom as _otel_prom  # noqa: F401
from . import redfish as _redfish  # noqa: F401
from .base import (
    REGISTRY,
    BaseAdapter,
    IngestPayload,
    merge_source,
    merge_symbolic,
)

#: Default source-file -> adapter-name mapping for a snapshot directory.
DEFAULT_SOURCES: tuple[tuple[str, str], ...] = (
    ("kueue", "kueue.json"),
    ("kubernetes", "kubernetes.json"),
    ("dcgm", "dcgm.json"),
    ("node_exporter", "node_exporter.json"),
    ("redfish", "redfish.json"),
    ("otel_prom", "otel_prom.json"),
)


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────
def load_snapshot_dir(
    snapshot_dir: str | Path,
    *,
    sources: tuple[tuple[str, str], ...] = DEFAULT_SOURCES,
    dra_enabled: bool | None = None,
    tick: int = 0,
) -> WorldState:
    """Load one snapshot directory into a fully merged :class:`WorldState`.

    Every configured source is run through its adapter; a missing file produces
    an empty, ``dropped``-flagged payload (never a crash, never fabricated data).
    Observation families are folded into ``M_t`` so downstream consumers see
    per-channel availability, age, and source health.
    """
    root = Path(snapshot_dir)
    state = WorldState(tick=tick)
    payloads: list[IngestPayload] = []

    for name, filename in sources:
        if name not in REGISTRY:
            continue
        adapter = _make_adapter(name, dra_enabled=dra_enabled)
        payload = adapter.load_snapshot(root / filename)
        payloads.append(payload)
        merge_symbolic(state.symbolic, payload.symbolic)
        merge_source(state.source, payload.source)

    # Fold all observations into source-state integrity + pick a wall-clock ts.
    all_obs: list[Observation] = [o for p in payloads for o in p.observations]
    state.ts_ns = _infer_ts_ns(all_obs)
    _fold_observations_into_source(all_obs, state.source)
    _record_unavailable(payloads, state.source)

    # Optional exogenous context file.
    ctx_path = root / "context.json"
    if ctx_path.exists():
        state.context = _load_context(ctx_path)

    return state


def load_snapshot_observations(
    snapshot_dir: str | Path,
    *,
    sources: tuple[tuple[str, str], ...] = DEFAULT_SOURCES,
    dra_enabled: bool | None = None,
) -> list[Observation]:
    """Load just the raw multi-rate observations for one snapshot directory."""
    root = Path(snapshot_dir)
    obs: list[Observation] = []
    for name, filename in sources:
        if name not in REGISTRY:
            continue
        adapter = _make_adapter(name, dra_enabled=dra_enabled)
        payload = adapter.load_snapshot(root / filename)
        obs.extend(payload.observations)
    return obs


def load_snapshot_series(
    parent_dir: str | Path,
    *,
    sources: tuple[tuple[str, str], ...] = DEFAULT_SOURCES,
    dra_enabled: bool | None = None,
) -> Iterator[WorldState]:
    """Yield a :class:`WorldState` per timestamped snapshot sub-directory.

    Sub-directories are visited in lexical order (use zero-padded / ISO names for
    correct chronology). ``tick`` is assigned by enumeration order.
    """
    parent = Path(parent_dir)
    if not parent.is_dir():
        return
    subdirs = sorted(p for p in parent.iterdir() if p.is_dir())
    for tick, sub in enumerate(subdirs):
        yield load_snapshot_dir(
            sub, sources=sources, dra_enabled=dra_enabled, tick=tick
        )


# ──────────────────────────────────────────────────────────────────────────────
# Saving (fixtures / round-trip for tests & examples)
# ──────────────────────────────────────────────────────────────────────────────
def save_snapshot_dir(
    snapshot_dir: str | Path,
    raw_sources: dict[str, dict[str, Any]],
    *,
    context: ExogenousContext | None = None,
) -> Path:
    """Write a dict of ``{source_name: raw_snapshot}`` as a snapshot directory.

    Used to materialise demo / test fixtures that :func:`load_snapshot_dir` can
    then replay. Keys are adapter names; the file is ``<name>.json``.
    """
    root = Path(snapshot_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, raw in raw_sources.items():
        with (root / f"{name}.json").open("w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=2, sort_keys=True)
    if context is not None:
        with (root / "context.json").open("w", encoding="utf-8") as fh:
            json.dump(asdict(context), fh, indent=2, sort_keys=True)
    return root


# ──────────────────────────────────────────────────────────────────────────────
# internals
# ──────────────────────────────────────────────────────────────────────────────
def _make_adapter(name: str, *, dra_enabled: bool | None) -> BaseAdapter:
    """Create an adapter, passing ``dra_enabled`` to the k8s adapter only."""
    if name == "kubernetes":
        from .kubernetes.adapter import KubernetesAdapter

        return KubernetesAdapter(dra_enabled=dra_enabled)
    return REGISTRY.create(name)


def _infer_ts_ns(observations: list[Observation]) -> int:
    """Pick the max event timestamp across observations (best-effort wall clock)."""
    return max((o.ts_event_ns for o in observations if o.ts_event_ns), default=0)


def _fold_observations_into_source(
    observations: list[Observation], source: SourceState
) -> None:
    """Aggregate per-channel availability/age/health into ``M_t``.

    For each channel we record:
      * ``channel_age_s``      — max observed age-of-information,
      * ``channel_missing_prob`` — fraction of (entity) observations where the
        channel was unavailable (an empirical missingness rate, not imputation),
      * ``transport_delay_s``  — max staleness when a value was present.
    """
    seen: dict[str, int] = {}
    missing: dict[str, int] = {}
    for obs in observations:
        for cname, ch in obs.channels.items():
            seen[cname] = seen.get(cname, 0) + 1
            if not ch.available:
                missing[cname] = missing.get(cname, 0) + 1
            else:
                if ch.age_s > source.channel_age_s.get(cname, 0.0):
                    source.channel_age_s[cname] = ch.age_s
                if ch.source_health in (SourceHealth.STALE, SourceHealth.DELAYED):
                    prev = source.transport_delay_s.get(cname, 0.0)
                    source.transport_delay_s[cname] = max(prev, ch.age_s)
    for cname, total in seen.items():
        source.channel_missing_prob[cname] = missing.get(cname, 0) / float(total)


def _record_unavailable(payloads: list[IngestPayload], source: SourceState) -> None:
    """Mark collectors that declared expected-but-absent channels as degraded."""
    for p in payloads:
        if p.unavailable_channels and p.source_name:
            current = source.collector_health.get(
                p.source_name, SourceHealth.OBSERVED.value
            )
            # Only downgrade; never upgrade a dropped collector.
            if current == SourceHealth.OBSERVED.value:
                source.collector_health[p.source_name] = SourceHealth.DELAYED.value


def _load_context(path: Path) -> ExogenousContext:
    """Load optional exogenous context, tolerant of partial files."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return ExogenousContext()
    if not isinstance(raw, dict):
        return ExogenousContext()
    ctx = ExogenousContext()
    if "regime" in raw:
        ctx.regime = str(raw["regime"])
    if "ambient_temp_c" in raw:
        with suppress(TypeError, ValueError):
            ctx.ambient_temp_c = float(raw["ambient_temp_c"])
    if "carbon_intensity" in raw and raw["carbon_intensity"] is not None:
        with suppress(TypeError, ValueError):
            ctx.carbon_intensity = float(raw["carbon_intensity"])
    if "arrival_burst_regime" in raw:
        ctx.arrival_burst_regime = str(raw["arrival_burst_regime"])
    if "workload_mix_regime" in raw:
        ctx.workload_mix_regime = str(raw["workload_mix_regime"])
    if isinstance(raw.get("manual_drains"), list):
        ctx.manual_drains = [str(x) for x in raw["manual_drains"]]
    if isinstance(raw.get("maintenance_windows"), list):
        windows: list[tuple[float, float]] = []
        for w in raw["maintenance_windows"]:
            if isinstance(w, (list, tuple)) and len(w) == 2:
                try:
                    windows.append((float(w[0]), float(w[1])))
                except (TypeError, ValueError):
                    continue
        ctx.maintenance_windows = windows
    return ctx


# Re-export the canonical families so callers can size the three tensorizer lanes.
FAMILIES: tuple[ObservationFamily, ...] = tuple(ObservationFamily)


__all__ = [
    "DEFAULT_SOURCES",
    "FAMILIES",
    "load_snapshot_dir",
    "load_snapshot_observations",
    "load_snapshot_series",
    "save_snapshot_dir",
]
