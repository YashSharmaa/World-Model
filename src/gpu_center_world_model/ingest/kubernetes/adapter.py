# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Kubernetes adapter — PodGroup gang state, nodes, and (optional) DRA.

This adapter reconstructs the scheduler-adjacent inventory that Kueue does not
own directly: the schedulable ``Node`` set with health, the gang-readiness of
PodGroups (workload-aware scheduling cycles), and — behind a feature flag —
Dynamic Resource Allocation (DRA) device state. DRA separates admission-time
quota from later device allocation, so it is the right surface for device
health, ``allocatedResourcesStatus``, and per-pod device assignment, but it is
stable-but-newer (k8s v1.35+) and must stay optional so v1 runs on the classic
device-plugin path too (docs C, D, H §55).

DRA mapping (when ``enabled``):
  * ``DeviceClass``        -> GPU ``mig_profile`` / ``interconnect_class`` hints
  * ``ResourceSlice``      -> per-node advertised devices (GPU inventory)
  * ``ResourceClaim``      -> pod->device allocation (``GPU.allocated_to``)
  * device health / ``allocatedResourcesStatus`` -> :class:`DeviceHealth`

Snapshot shape::

    {
      "ts_ns": ...,
      "nodes":      [ {Node status + labels + health} , ... ],
      "podgroups":  [ {gang readiness} , ... ],
      "dra": {                       # optional; ignored unless enabled
        "enabled": true,
        "deviceclasses":  [ ... ],
        "resourceslices": [ ... ],
        "resourceclaims": [ ... ]
      }
    }
"""

from __future__ import annotations

from typing import Any

from ...schema.entities import GPU, DeviceHealth, Node, NodeHealth, Workload
from ...schema.observations import Observation, ObservationFamily
from ..base import (
    BaseAdapter,
    IngestPayload,
    make_channel,
    opt_bool,
    opt_float,
    opt_int,
    opt_str,
    register_adapter,
)

_NODE_HEALTH_ALIASES: dict[str, NodeHealth] = {
    "ready": NodeHealth.READY,
    "degraded": NodeHealth.DEGRADED,
    "unschedulable": NodeHealth.UNSCHEDULABLE,
    "draining": NodeHealth.DRAINING,
    "down": NodeHealth.DOWN,
    "notready": NodeHealth.DOWN,
}

#: DRA ``allocatedResourcesStatus`` health strings -> latent device health.
_DEVICE_HEALTH_ALIASES: dict[str, DeviceHealth] = {
    "healthy": DeviceHealth.HEALTHY,
    "degraded": DeviceHealth.DEGRADED,
    "intermittentlyfaulty": DeviceHealth.INTERMITTENTLY_FAULTY,
    "intermittently_faulty": DeviceHealth.INTERMITTENTLY_FAULTY,
    "unhealthy": DeviceHealth.DEGRADED,
    "resetrecovery": DeviceHealth.RESET_RECOVERY,
    "reset_recovery": DeviceHealth.RESET_RECOVERY,
    "masked": DeviceHealth.MASKED,
    "unknown": DeviceHealth.MASKED,
}


@register_adapter("kubernetes")
class KubernetesAdapter(BaseAdapter):
    """Map nodes, gang readiness, and optional DRA device state to the schema."""

    source_name = "kubernetes"
    default_family = ObservationFamily.PLACEMENT_STATE

    def __init__(self, *, dra_enabled: bool | None = None) -> None:
        # ``None`` => honor the snapshot's own dra.enabled flag; an explicit bool
        # overrides it (operator feature-flag). DRA stays OFF unless asked.
        self._dra_override = dra_enabled

    def parse(self, snapshot: dict[str, Any], payload: IngestPayload) -> None:
        ts_ns = int(snapshot.get("ts_ns") or 0)
        for nd in snapshot.get("nodes", []) or []:
            self._parse_node(nd, payload)
        for pg in snapshot.get("podgroups", snapshot.get("podGroups", [])) or []:
            self._parse_podgroup(pg, payload, ts_ns)

        dra = snapshot.get("dra") or {}
        enabled = (
            self._dra_override
            if self._dra_override is not None
            else bool(dra.get("enabled", False))
        )
        if enabled and isinstance(dra, dict):
            self._parse_dra(dra, payload, ts_ns)
        else:
            # Record that DRA is intentionally not collected (channel unavailable).
            payload.unavailable_channels.append("dra.devices")

    # ── Node ──────────────────────────────────────────────────────────────────
    def _parse_node(self, nd: dict[str, Any], payload: IngestPayload) -> None:
        uid = opt_str(nd, "node_uid", "uid", "name")
        if uid is None:
            return
        health_raw = (opt_str(nd, "node_health", "health", "status") or "ready").lower()
        node = Node(
            node_uid=uid,
            rack_id=opt_str(nd, "rack_id", "rack") or "",
            block_id=opt_str(nd, "block_id", "block") or "",
            switch_id=opt_str(nd, "switch_id", "switch") or "",
            flavor_id=opt_str(nd, "flavor_id", "flavor") or "",
            labels={str(k): str(v) for k, v in (nd.get("labels") or {}).items()},
            cpu_cap=opt_float(nd, "cpu_cap", "cpuCapacity") or 0.0,
            mem_cap_gb=opt_float(nd, "mem_cap_gb", "memCapacityGb") or 0.0,
            gpu_cap=opt_int(nd, "gpu_cap", "gpuCapacity") or 0,
            ready=opt_bool(nd, "ready", default=True),
            schedulable=opt_bool(nd, "schedulable", default=True),
            cpu_alloc=opt_float(nd, "cpu_alloc") or 0.0,
            mem_alloc_gb=opt_float(nd, "mem_alloc_gb") or 0.0,
            running_pods=opt_int(nd, "running_pods", "pods") or 0,
            node_health=_NODE_HEALTH_ALIASES.get(health_raw, NodeHealth.READY),
        )
        payload.symbolic.nodes[uid] = node

    # ── PodGroup gang readiness ───────────────────────────────────────────────
    def _parse_podgroup(
        self, pg: dict[str, Any], payload: IngestPayload, ts_ns: int
    ) -> None:
        wl = opt_str(pg, "workload_uid", "owner", "name")
        if wl is None:
            return
        min_member = opt_int(pg, "min_member", "minMember")
        scheduled = opt_int(pg, "scheduled", "scheduledCount")
        ready = opt_int(pg, "ready", "readyCount")
        running = opt_int(pg, "running", "runningCount")
        # gang fraction is only meaningful when min_member is known.
        gang_frac = None
        if min_member and min_member > 0 and ready is not None:
            gang_frac = max(0.0, min(1.0, ready / min_member))
        ch = {
            "gang_min_member": make_channel(
                "gang_min_member", float(min_member) if min_member is not None else None
            ),
            "gang_scheduled": make_channel(
                "gang_scheduled", float(scheduled) if scheduled is not None else None
            ),
            "gang_ready": make_channel(
                "gang_ready", float(ready) if ready is not None else None
            ),
            "gang_running": make_channel(
                "gang_running", float(running) if running is not None else None
            ),
            "gang_ready_fraction": make_channel("gang_ready_fraction", gang_frac),
        }
        payload.add_observation(
            Observation(
                entity_kind="workload",
                entity_id=wl,
                family=ObservationFamily.PLACEMENT_STATE,
                ts_event_ns=ts_ns,
                channels=ch,
            )
        )
        # Upsert a thin Workload carrying ONLY the gang-readiness fraction this
        # source observes; the field-aware merge (base._merge_workloads) folds it
        # into the identity Workload the kueue adapter created without either
        # source clobbering the other's fields within a tick.
        if gang_frac is not None:
            existing = payload.symbolic.workloads.get(wl) or Workload(
                workload_uid=wl, tenant_id="", local_queue_id="", cluster_queue_id=""
            )
            existing.ready_fraction = gang_frac
            payload.symbolic.workloads[wl] = existing

    # ── DRA (optional / flagged) ──────────────────────────────────────────────
    def _parse_dra(
        self, dra: dict[str, Any], payload: IngestPayload, ts_ns: int
    ) -> None:
        # DeviceClass -> per-class profile/interconnect hints.
        class_hints: dict[str, dict[str, str]] = {}
        for dc in dra.get("deviceclasses", dra.get("deviceClasses", [])) or []:
            cname = opt_str(dc, "name")
            if cname is None:
                continue
            class_hints[cname] = {
                "mig_profile": opt_str(dc, "mig_profile", "migProfile") or "full",
                "interconnect_class": opt_str(dc, "interconnect_class", "interconnect")
                or "nvlink",
            }

        # ResourceSlice -> advertised devices (GPU inventory per node).
        for slc in dra.get("resourceslices", dra.get("resourceSlices", [])) or []:
            node_uid = opt_str(slc, "node_uid", "nodeName", "node") or ""
            for dev in slc.get("devices", []) or []:
                self._parse_dra_device(dev, node_uid, class_hints, payload, ts_ns)

        # ResourceClaim -> pod->device allocation (sets GPU.allocated_to).
        for claim in dra.get("resourceclaims", dra.get("resourceClaims", [])) or []:
            wl = opt_str(claim, "workload_uid", "owner")
            for alloc in claim.get("allocations", claim.get("devices", [])) or []:
                dev_uuid = opt_str(alloc, "gpu_uuid", "device", "deviceName")
                if dev_uuid is None:
                    continue
                gpu = payload.symbolic.gpus.get(dev_uuid)
                if gpu is not None and wl is not None:
                    gpu.allocated_to = wl

    def _parse_dra_device(
        self,
        dev: dict[str, Any],
        node_uid: str,
        class_hints: dict[str, dict[str, str]],
        payload: IngestPayload,
        ts_ns: int,
    ) -> None:
        uuid = opt_str(dev, "gpu_uuid", "uuid", "name")
        if uuid is None:
            return
        cls = opt_str(dev, "device_class", "deviceClassName") or ""
        hint = class_hints.get(cls, {})
        # allocatedResourcesStatus / health -> latent device health belief.
        health_raw = (
            opt_str(dev, "health", "allocatedResourcesStatus", "deviceHealth")
            or "healthy"
        ).lower()
        gpu = payload.symbolic.gpus.get(uuid) or GPU(gpu_uuid=uuid, node_uid=node_uid)
        gpu.node_uid = node_uid or gpu.node_uid
        gpu.mig_profile = (
            opt_str(dev, "mig_profile", "migProfile")
            or hint.get("mig_profile")
            or gpu.mig_profile
        )
        gpu.interconnect_class = (
            opt_str(dev, "interconnect_class")
            or hint.get("interconnect_class")
            or gpu.interconnect_class
        )
        mem = opt_int(dev, "memory_total_mb", "memoryTotalMb")
        if mem is not None:
            gpu.memory_total_mb = mem
        gpu.health_state = _DEVICE_HEALTH_ALIASES.get(health_raw, DeviceHealth.HEALTHY)
        payload.symbolic.gpus[uuid] = gpu

        # Emit DRA device-health observation (the integrity signal).
        payload.add_observation(
            Observation(
                entity_kind="gpu",
                entity_id=uuid,
                family=ObservationFamily.PLACEMENT_STATE,
                ts_event_ns=ts_ns,
                channels={
                    "dra_device_present": make_channel("dra_device_present", 1.0),
                    "dra_device_allocated": make_channel(
                        "dra_device_allocated",
                        1.0 if gpu.allocated_to else 0.0,
                    ),
                },
            )
        )


__all__ = ["KubernetesAdapter"]
