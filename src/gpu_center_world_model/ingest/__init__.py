# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Ingest adapters: open scheduler/telemetry surfaces -> canonical schema.

This package maps the open data sources of doc D into the frozen world-state
schema, with **offline-snapshot replay as the default** so no live cluster is
required (doc H). Each adapter is tolerant of missing fields — a missing channel
is marked *unavailable* with its source-health and age-of-information, never
zero-imputed (docs C, D, T, U).

Public surface
--------------
* :class:`Adapter` / :class:`BaseAdapter` — the adapter protocol + base class.
* :class:`IngestPayload` — a partial, mergeable world view from one source.
* :data:`REGISTRY` / :func:`register_adapter` — the name->adapter registry.
* Per-source adapters: :class:`KueueAdapter`, :class:`KubernetesAdapter`,
  :class:`DCGMAdapter`, :class:`NodeExporterAdapter`, :class:`RedfishAdapter`,
  :class:`OtelPromAdapter`.
* Offline replay: :func:`load_snapshot_dir`, :func:`load_snapshot_series`,
  :func:`load_snapshot_observations`, :func:`save_snapshot_dir`.

Adapter modules are imported here so they self-register in :data:`REGISTRY`.
"""

from __future__ import annotations

from .base import (
    REGISTRY,
    Adapter,
    AdapterRegistry,
    BaseAdapter,
    IngestPayload,
    make_channel,
    merge_source,
    merge_symbolic,
    opt_bool,
    opt_float,
    opt_int,
    opt_str,
    read_json_snapshot,
    register_adapter,
)
from .dcgm import DCGMAdapter
from .kubernetes import KubernetesAdapter
from .kueue import EVICTION_REASONS, KueueAdapter
from .node_exporter import NodeExporterAdapter
from .offline_snapshots import (
    DEFAULT_SOURCES,
    FAMILIES,
    load_snapshot_dir,
    load_snapshot_observations,
    load_snapshot_series,
    save_snapshot_dir,
)
from .otel_prom import OtelPromAdapter, PromSample, parse_exposition
from .redfish import RedfishAdapter

__all__ = [
    "DEFAULT_SOURCES",
    "EVICTION_REASONS",
    "FAMILIES",
    "REGISTRY",
    "Adapter",
    "AdapterRegistry",
    "BaseAdapter",
    "DCGMAdapter",
    "IngestPayload",
    "KubernetesAdapter",
    "KueueAdapter",
    "NodeExporterAdapter",
    "OtelPromAdapter",
    "PromSample",
    "RedfishAdapter",
    "load_snapshot_dir",
    "load_snapshot_observations",
    "load_snapshot_series",
    "make_channel",
    "merge_source",
    "merge_symbolic",
    "opt_bool",
    "opt_float",
    "opt_int",
    "opt_str",
    "parse_exposition",
    "read_json_snapshot",
    "register_adapter",
    "save_snapshot_dir",
]
