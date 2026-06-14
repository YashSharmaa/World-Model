# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""NVIDIA DCGM adapter — the canonical per-GPU telemetry surface.

The DCGM exporter is the standard open source of per-GPU power, utilization,
energy, temperatures, MIG visibility, Xid errors, and recovery actions, and it
can map GPU metrics to Kubernetes pods via the Kubelet PodResources API (doc D).
This adapter consumes a snapshot of DCGM fields keyed by the schema's canonical
field names (:data:`GPU_TELEMETRY_FIELDS` / :data:`GPU_PROFILING_FIELDS`) and:

  * updates the symbolic :class:`GPU` telemetry block (power/temp/util/energy/
    xid/recovery) — only for fields actually present, leaving the rest at their
    prior/default value rather than zero-imputing;
  * emits :class:`Observation` snapshots in the GPU_TELEMETRY and
    PROFILING_TELEMETRY families with explicit availability + staleness, because
    failures often surface first as *missing* DCGM samples (docs C, D, T, U);
  * carries the pod->GPU mapping into ``GPU.allocated_to`` when PodResources data
    is included.

Snapshot shape::

    {
      "ts_ns": ...,
      "gpus": [
        {
          "gpu_uuid": "GPU-abc", "node_uid": "node-1",
          "pod": "workload-7",            # PodResources mapping (optional)
          "age_s": 4.0,
          "fields": {                      # canonical DCGM_FI_* -> value
            "DCGM_FI_DEV_POWER_USAGE": 412.0,
            "DCGM_FI_DEV_GPU_TEMP": 71.0, ...
          }
        }, ...
      ]
    }
"""

from __future__ import annotations

from typing import Any

from ...schema.entities import GPU
from ...schema.observations import (
    GPU_PROFILING_FIELDS,
    GPU_TELEMETRY_FIELDS,
    Observation,
    ObservationFamily,
)
from ..base import (
    BaseAdapter,
    IngestPayload,
    make_channel,
    opt_float,
    opt_int,
    opt_str,
    register_adapter,
)

#: GPU_TELEMETRY samples ~5-30 s; flag stale beyond ~3 nominal cadences.
TELEMETRY_STALE_AFTER_S = 90.0
PROFILING_STALE_AFTER_S = 90.0

#: Map canonical DCGM fields onto the symbolic ``GPU`` telemetry attributes.
_FIELD_TO_ATTR: dict[str, str] = {
    "DCGM_FI_DEV_GPU_UTIL": "util",
    "DCGM_FI_DEV_MEM_COPY_UTIL": "mem_util",
    "DCGM_FI_DEV_POWER_USAGE": "power_w",
    "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION": "energy_mj",
    "DCGM_FI_DEV_GPU_TEMP": "temp_c",
    "DCGM_FI_DEV_MEMORY_TEMP": "memory_temp_c",
}


@register_adapter("dcgm")
class DCGMAdapter(BaseAdapter):
    """Map a DCGM exporter snapshot onto GPU telemetry + observations."""

    source_name = "dcgm"
    default_family = ObservationFamily.GPU_TELEMETRY

    def parse(self, snapshot: dict[str, Any], payload: IngestPayload) -> None:
        ts_ns = int(snapshot.get("ts_ns") or 0)
        for g in snapshot.get("gpus", []) or []:
            self._parse_gpu(g, payload, ts_ns)

    def _parse_gpu(self, g: dict[str, Any], payload: IngestPayload, ts_ns: int) -> None:
        uuid = opt_str(g, "gpu_uuid", "uuid", "UUID")
        if uuid is None:
            return
        node_uid = opt_str(g, "node_uid", "node", "Hostname") or ""
        age_s = opt_float(g, "age_s") or 0.0
        fields = g.get("fields", {}) or {}

        gpu = payload.symbolic.gpus.get(uuid) or GPU(gpu_uuid=uuid, node_uid=node_uid)
        gpu.node_uid = node_uid or gpu.node_uid

        # PodResources pod->GPU mapping (optional). Never overwrite with None.
        pod = opt_str(g, "pod", "allocated_to", "exported_pod")
        if pod is not None:
            gpu.allocated_to = pod

        # Continuous numeric telemetry — only assign present fields (the
        # _FIELD_TO_ATTR map is the authority on which GPU attribute each DCGM
        # field updates; missing fields are left at their prior value).
        for field_name, attr in _FIELD_TO_ATTR.items():
            val = opt_float(fields, field_name)
            if val is not None:
                setattr(gpu, attr, val)

        # Discrete fields: Xid error code + recovery action.
        xid = opt_int(fields, "DCGM_FI_DEV_XID_ERRORS")
        if xid is not None:
            gpu.xid_code = xid
        recovery = opt_str(fields, "DCGM_FI_DEV_GET_GPU_RECOVERY_ACTION")
        if recovery is not None:
            gpu.recovery_action = recovery
        payload.symbolic.gpus[uuid] = gpu

        # Build the GPU_TELEMETRY observation over ALL canonical fields so that
        # missing ones are explicitly unavailable (the missingness is the signal).
        self._emit_family(
            uuid,
            node_uid,
            fields,
            GPU_TELEMETRY_FIELDS,
            ObservationFamily.GPU_TELEMETRY,
            TELEMETRY_STALE_AFTER_S,
            age_s,
            ts_ns,
            payload,
        )
        self._emit_family(
            uuid,
            node_uid,
            fields,
            GPU_PROFILING_FIELDS,
            ObservationFamily.PROFILING_TELEMETRY,
            PROFILING_STALE_AFTER_S,
            age_s,
            ts_ns,
            payload,
        )

        # Track per-channel age + collector reachability in M_t.
        payload.source.channel_age_s[f"dcgm:{uuid}"] = age_s

    def _emit_family(
        self,
        uuid: str,
        node_uid: str,
        fields: dict[str, Any],
        field_names: tuple[str, ...],
        family: ObservationFamily,
        stale_after_s: float,
        age_s: float,
        ts_ns: int,
        payload: IngestPayload,
    ) -> None:
        channels: dict[str, Any] = {}
        any_present = False
        for fn in field_names:
            val = opt_float(fields, fn)
            if val is not None:
                any_present = True
            channels[fn] = make_channel(
                fn, val, age_s=age_s, stale_after_s=stale_after_s
            )
        if not any_present:
            payload.unavailable_channels.append(f"dcgm:{uuid}:{family.value}")
        payload.add_observation(
            Observation(
                entity_kind="gpu",
                entity_id=uuid,
                family=family,
                ts_event_ns=ts_ns,
                channels=channels,
            )
        )


__all__ = [
    "PROFILING_STALE_AFTER_S",
    "TELEMETRY_STALE_AFTER_S",
    "DCGMAdapter",
]
