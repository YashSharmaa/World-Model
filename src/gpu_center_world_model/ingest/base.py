# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Adapter protocol, payload container, and registry for ingest sources.

Ingest maps *open* scheduler / telemetry surfaces (Kueue, Kubernetes DRA, NVIDIA
DCGM, node_exporter, Redfish, OpenTelemetry/Prometheus) into the canonical
world-state schema. The whole layer is **offline-snapshot first**: every adapter
loads a JSON snapshot of one source's exported state and returns a partial view
of the world; live-cluster IO is intentionally kept behind the same interface so
nothing downstream needs a running cluster (docs D, H).

Three design commitments are enforced here so they cannot be violated per-source:

1. **Never fabricate.** A missing field maps to an *unavailable* channel or a
   left-unset symbolic value, never to a zero-imputed measurement. Real GPU
   failures surface first as missing/degraded telemetry, so missingness is
   first-class state, not noise (docs C, D, I, T, U). The :func:`opt_float` /
   :func:`make_channel` helpers encode this rule once for every adapter.

2. **Partial, mergeable views.** Each source observes only part of the world.
   An adapter returns an :class:`IngestPayload` carrying whatever entities and
   observations it can build plus per-channel source-health, and the offline
   loader merges payloads from many sources into one :class:`WorldState`.

3. **Symbolic stays exact, learned stays out.** Adapters populate ``X_t`` (the
   symbolic tables) and ``M_t`` (source state) and emit raw observations; they
   never touch the learned residual belief ``Z_t``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..schema.entities import GPU, DeviceHealth, Node, NodeHealth, Workload
from ..schema.observations import (
    Channel,
    Observation,
    ObservationFamily,
    SourceHealth,
)
from ..schema.state import SourceState, SymbolicState


# ──────────────────────────────────────────────────────────────────────────────
# Tolerant field extraction helpers (the "never fabricate" rule, once)
# ──────────────────────────────────────────────────────────────────────────────
def opt_float(d: dict[str, Any], *keys: str) -> float | None:
    """Return the first present, numeric value among ``keys`` or ``None``.

    Returns ``None`` (not ``0.0``) when the field is absent or non-numeric so the
    caller can decide whether to leave a symbolic default in place or to mark a
    channel unavailable. We deliberately do not coerce booleans to floats.
    """
    for k in keys:
        if k in d and d[k] is not None and not isinstance(d[k], bool):
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return None


def opt_int(d: dict[str, Any], *keys: str) -> int | None:
    """Return the first present value among ``keys`` coerced to ``int`` or ``None``."""
    for k in keys:
        if k in d and d[k] is not None and not isinstance(d[k], bool):
            try:
                return int(d[k])
            except (TypeError, ValueError):
                continue
    return None


def opt_str(d: dict[str, Any], *keys: str) -> str | None:
    """Return the first present value among ``keys`` as ``str`` or ``None``."""
    for k in keys:
        if k in d and d[k] is not None:
            return str(d[k])
    return None


def opt_bool(d: dict[str, Any], *keys: str, default: bool | None = None) -> bool | None:
    """Return the first present boolean-ish value among ``keys`` or ``default``."""
    for k in keys:
        if k in d and d[k] is not None:
            v = d[k]
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                return v.strip().lower() in ("true", "1", "yes", "ready", "ok")
    return default


def make_channel(
    name: str,
    value: float | None,
    *,
    age_s: float = 0.0,
    source_health: SourceHealth | None = None,
    stale_after_s: float | None = None,
    censor: str = "none",
    censor_lo: float = float("-inf"),
    censor_hi: float = float("inf"),
) -> Channel:
    """Build a :class:`Channel`, marking it *unavailable* when ``value is None``.

    Source health defaults to ``OBSERVED`` for present values; if ``value`` is
    ``None`` it becomes ``DROPPED`` unless an explicit health is given. When
    ``stale_after_s`` is provided and ``age_s`` exceeds it, a present value is
    downgraded to ``STALE`` — staleness is the integrity signal, not the value
    (doc T M_t, doc U telemetry-degradation rare event).
    """
    available = value is not None
    if source_health is None:
        if not available:
            source_health = SourceHealth.DROPPED
        elif stale_after_s is not None and age_s > stale_after_s:
            source_health = SourceHealth.STALE
        else:
            source_health = SourceHealth.OBSERVED
    return Channel(
        name=name,
        value=float(value) if available else 0.0,
        available=available,
        age_s=float(age_s),
        source_health=source_health,
        censor=censor,
        censor_lo=censor_lo,
        censor_hi=censor_hi,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Payload container — a partial, mergeable view of the world from one source
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class IngestPayload:
    """A partial world view produced by one adapter from one snapshot.

    ``symbolic`` holds whatever entity tables this source can reconstruct (other
    tables stay empty). ``observations`` is the list of raw multi-rate snapshots
    this source emits. ``source`` carries the ``M_t`` integrity state for the
    channels this source owns. ``unavailable_channels`` records channels the
    source *should* provide but could not this tick (so downstream knows the
    difference between "not this source's job" and "this source dropped it").
    """

    source: SourceState = field(default_factory=SourceState)
    symbolic: SymbolicState = field(default_factory=SymbolicState)
    observations: list[Observation] = field(default_factory=list)
    unavailable_channels: list[str] = field(default_factory=list)
    source_name: str = ""

    def add_observation(self, obs: Observation) -> None:
        self.observations.append(obs)

    def mark_collector(self, status: str) -> None:
        """Record this source's collector health (e.g. ``observed``/``dropped``)."""
        if self.source_name:
            self.source.collector_health[self.source_name] = status


def _merge_gpus(dst: dict[str, GPU], src: dict[str, GPU]) -> None:
    """Field-aware GPU merge — different sources own different GPU fields.

    Kubernetes/DRA owns inventory + the latent ``health_state``; DCGM owns
    telemetry (power/temp/util/energy/xid/recovery). A naive whole-object replace
    would let whichever source runs last clobber fields it never observed (e.g.
    DCGM resetting the DRA-derived ``health_state`` back to HEALTHY). We merge
    per-field, keeping a non-default value from ``src`` over a default in ``dst``.
    """
    defaults = GPU(gpu_uuid="", node_uid="")
    for uuid, s in src.items():
        d = dst.get(uuid)
        if d is None:
            dst[uuid] = s
            continue
        d.node_uid = s.node_uid or d.node_uid
        if s.mig_profile != defaults.mig_profile:
            d.mig_profile = s.mig_profile
        if s.memory_total_mb != defaults.memory_total_mb:
            d.memory_total_mb = s.memory_total_mb
        if s.interconnect_class != defaults.interconnect_class:
            d.interconnect_class = s.interconnect_class
        if s.allocated_to is not None:
            d.allocated_to = s.allocated_to
        # telemetry (DCGM-owned): take any non-default reading
        for attr in (
            "util",
            "mem_util",
            "temp_c",
            "memory_temp_c",
            "power_w",
            "energy_mj",
        ):
            sv = getattr(s, attr)
            if sv != getattr(defaults, attr):
                setattr(d, attr, sv)
        if s.xid_code != defaults.xid_code:
            d.xid_code = s.xid_code
        if s.recovery_action != defaults.recovery_action:
            d.recovery_action = s.recovery_action
        # latent device health (DRA/reliability-owned): keep any non-HEALTHY belief
        if s.health_state != DeviceHealth.HEALTHY:
            d.health_state = s.health_state


def _merge_nodes(dst: dict[str, Node], src: dict[str, Node]) -> None:
    """Field-aware Node merge — k8s owns inventory/health, host metrics own load.

    The kubernetes adapter creates the node with static capacity + health; the
    node_exporter adapter enriches dynamic host load (cpu_alloc / mem / pods) and
    must not be lost when both are present in the same tick.
    """
    defaults = Node(node_uid="", rack_id="", block_id="", switch_id="", flavor_id="")
    for uid, s in src.items():
        d = dst.get(uid)
        if d is None:
            dst[uid] = s
            continue
        for attr in ("rack_id", "block_id", "switch_id", "flavor_id"):
            sv = getattr(s, attr)
            if sv:
                setattr(d, attr, sv)
        if s.labels:
            d.labels = {**d.labels, **s.labels}
        for attr in (
            "cpu_cap",
            "mem_cap_gb",
            "gpu_cap",
            "cpu_alloc",
            "mem_alloc_gb",
            "running_pods",
        ):
            sv = getattr(s, attr)
            if sv != getattr(defaults, attr):
                setattr(d, attr, sv)
        if s.node_health != NodeHealth.READY:
            d.node_health = s.node_health
        # ready/schedulable: only propagate an explicit "not ready/unschedulable"
        if not s.ready:
            d.ready = False
        if not s.schedulable:
            d.schedulable = False


def _merge_workloads(dst: dict[str, Workload], src: dict[str, Workload]) -> None:
    """Field-aware Workload merge — kueue owns identity, k8s owns gang readiness.

    The kueue adapter is authoritative for workload identity, status, podsets,
    queues, and quota fields; the kubernetes PodGroup adapter contributes the
    gang ``ready_fraction``. A whole-object replace would let whichever runs last
    wipe the other's contribution, so we merge per-field: non-empty identity /
    non-default dynamic values from ``src`` win over defaults in ``dst``.
    """
    defaults = Workload(
        workload_uid="", tenant_id="", local_queue_id="", cluster_queue_id=""
    )
    for uid, s in src.items():
        d = dst.get(uid)
        if d is None:
            dst[uid] = s
            continue
        # identity / categorical fields: take a non-empty value from src
        for attr in ("tenant_id", "local_queue_id", "cluster_queue_id"):
            sv = getattr(s, attr)
            if sv:
                setattr(d, attr, sv)
        if s.admitted_flavor_id is not None:
            d.admitted_flavor_id = s.admitted_flavor_id
        if s.admitted_domain_id is not None:
            d.admitted_domain_id = s.admitted_domain_id
        if s.podsets:
            d.podsets = s.podsets
        if s.labels:
            d.labels = {**d.labels, **s.labels}
        if s.status != defaults.status:
            d.status = s.status
        # dynamic progression / readiness: take any non-default reading
        for attr in (
            "priority",
            "submit_ts",
            "max_exec_s",
            "age_s",
            "elapsed_run_s",
            "est_remaining_s",
            "ready_fraction",
            "requeue_count",
        ):
            sv = getattr(s, attr)
            if sv != getattr(defaults, attr):
                setattr(d, attr, sv)
        if s.topology_policy != defaults.topology_policy:
            d.topology_policy = s.topology_policy
        if s.comm_class != defaults.comm_class:
            d.comm_class = s.comm_class


def merge_symbolic(dst: SymbolicState, src: SymbolicState) -> None:
    """In-place merge of ``src`` entity tables into ``dst``.

    Most per-kind dicts are updated by key (last writer wins). :class:`Workload`,
    :class:`GPU`, and :class:`Node` use field-aware merges because their fields
    are owned by *different* sources within the same tick (kueue identity vs. k8s
    gang readiness; DCGM telemetry vs. DRA latent health vs. host-metric load). The
    ``operator_policy`` is replaced only if ``src`` carries a non-default
    ``policy_id`` (so an empty payload does not clobber a set policy);
    ``relations`` merge by key.
    """
    _merge_workloads(dst.workloads, src.workloads)
    dst.podsets.update(src.podsets)
    dst.local_queues.update(src.local_queues)
    dst.cluster_queues.update(src.cluster_queues)
    dst.flavors.update(src.flavors)
    dst.domains.update(src.domains)
    _merge_nodes(dst.nodes, src.nodes)
    _merge_gpus(dst.gpus, src.gpus)
    dst.switches.update(src.switches)
    dst.racks.update(src.racks)
    dst.power_domains.update(src.power_domains)
    dst.cooling_zones.update(src.cooling_zones)
    dst.relations.update(src.relations)
    if src.operator_policy.policy_id not in ("", "default"):
        dst.operator_policy = src.operator_policy


def merge_source(dst: SourceState, src: SourceState) -> None:
    """In-place merge of telemetry-source integrity state across adapters."""
    dst.collector_health.update(src.collector_health)
    dst.channel_missing_prob.update(src.channel_missing_prob)
    dst.channel_age_s.update(src.channel_age_s)
    dst.transport_delay_s.update(src.transport_delay_s)


# ──────────────────────────────────────────────────────────────────────────────
# The adapter protocol + base class
# ──────────────────────────────────────────────────────────────────────────────
@runtime_checkable
class Adapter(Protocol):
    """Structural contract every ingest source adapter satisfies.

    Adapters are stateless mappers: given a path to an offline snapshot (a JSON
    file or a directory), produce an :class:`IngestPayload`. Live collection, if
    ever added, is expected to populate the same payload behind this interface.
    """

    #: Stable source name, e.g. ``"kueue"`` / ``"dcgm"`` (matches doc D ``source``).
    source_name: str
    #: Default observation family this source contributes to.
    default_family: ObservationFamily

    def load_snapshot(self, path: str | Path) -> IngestPayload:
        """Map an offline snapshot of this source into a partial world view."""
        ...


class BaseAdapter:
    """Convenience base implementing JSON loading + payload bootstrapping.

    Subclasses implement :meth:`parse` over an already-decoded snapshot dict and
    inherit tolerant file loading (missing file -> empty payload flagged
    ``dropped``, never a crash) and a default observation family.
    """

    source_name: str = "base"
    default_family: ObservationFamily = ObservationFamily.SCHEDULER_STATE

    def new_payload(self) -> IngestPayload:
        p = IngestPayload(source_name=self.source_name)
        return p

    def load_snapshot(self, path: str | Path) -> IngestPayload:
        """Read a JSON snapshot file (or directory of files) and parse it."""
        snap = read_json_snapshot(path)
        payload = self.new_payload()
        if snap is None:
            # Source unreachable / file absent: mark collector dropped, no fabricate.
            payload.mark_collector(SourceHealth.DROPPED.value)
            return payload
        payload.mark_collector(SourceHealth.OBSERVED.value)
        self.parse(snap, payload)
        return payload

    def parse(self, snapshot: dict[str, Any], payload: IngestPayload) -> None:
        """Populate ``payload`` from a decoded snapshot. Override in subclasses."""
        raise NotImplementedError


def read_json_snapshot(path: str | Path) -> dict[str, Any] | None:
    """Read a JSON snapshot tolerantly.

    Accepts either a single ``*.json`` file, or a directory containing a file
    named ``<source>.json`` is handled by the offline loader, not here — this
    helper only resolves a concrete path. Returns ``None`` if the path is missing
    or unparseable so callers degrade to an empty, source-dropped payload.
    """
    p = Path(path)
    if not p.exists() or p.is_dir():
        return None
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else {"items": data}


# ──────────────────────────────────────────────────────────────────────────────
# Registry — name -> adapter factory, so the offline loader can discover sources
# ──────────────────────────────────────────────────────────────────────────────
class AdapterRegistry:
    """A small name->factory registry of ingest adapters.

    Kept deliberately simple (no entry points / plugin scanning) so the package
    imports cleanly on plain CI with no optional dependencies. Adapters register
    themselves at import time via :meth:`register`.
    """

    def __init__(self) -> None:
        self._factories: dict[str, type[BaseAdapter]] = {}

    def register(self, name: str, factory: type[BaseAdapter]) -> type[BaseAdapter]:
        self._factories[name] = factory
        return factory

    def create(self, name: str) -> BaseAdapter:
        if name not in self._factories:
            raise KeyError(f"No ingest adapter registered under '{name}'")
        return self._factories[name]()

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def __contains__(self, name: str) -> bool:
        return name in self._factories


#: Process-wide registry populated by each adapter module on import.
REGISTRY = AdapterRegistry()


def register_adapter(name: str):
    """Class decorator that registers an adapter under ``name`` in ``REGISTRY``."""

    def _wrap(cls: type[BaseAdapter]) -> type[BaseAdapter]:
        return REGISTRY.register(name, cls)

    return _wrap


__all__ = [
    "REGISTRY",
    "Adapter",
    "AdapterRegistry",
    "BaseAdapter",
    "IngestPayload",
    "make_channel",
    "merge_source",
    "merge_symbolic",
    "opt_bool",
    "opt_float",
    "opt_int",
    "opt_str",
    "read_json_snapshot",
    "register_adapter",
]
