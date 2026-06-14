# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""node_exporter adapter — host CPU/mem/network + hwmon/powercap/thermal-zone.

node_exporter exposes the host-side signals that connect GPU behavior to node
and rack context: CPU/memory/network counters plus the ``hwmon`` (temperatures),
``powercap`` (RAPL energy/power caps), and ``thermal_zone`` (host thermal zones)
collectors (doc D). These feed the NODE_RACK_TELEMETRY family that the thermal /
power-headroom residual heads consume.

This adapter:
  * fills the dynamic :class:`Node` host fields (cpu_alloc, mem_alloc, pods) that
    are present, leaving the rest untouched;
  * emits NODE_RACK_TELEMETRY observations carrying hwmon/powercap/thermal-zone
    channels with explicit availability + staleness;
  * optionally aggregates per-node power into a rack-power observation when a
    rack mapping is provided (a coarse headroom signal, never a learned value).

Snapshot shape::

    {
      "ts_ns": ...,
      "nodes": [
        {
          "node_uid": "node-1", "rack_id": "rack-a", "age_s": 12.0,
          "cpu_util": 0.62, "mem_used_gb": 380.0, "running_pods": 14,
          "hwmon_temp_c": 48.0,
          "powercap_w": 1850.0, "powercap_limit_w": 2200.0,
          "thermal_zone_c": 52.0,
          "net_rx_bps": 1.2e9, "net_tx_bps": 9.8e8
        }, ...
      ]
    }
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ...schema.entities import Node
from ...schema.observations import Observation, ObservationFamily
from ..base import (
    BaseAdapter,
    IngestPayload,
    make_channel,
    opt_float,
    opt_int,
    opt_str,
    register_adapter,
)

#: NODE_RACK_TELEMETRY samples ~30-60 s; stale beyond ~3 cadences.
NODE_STALE_AFTER_S = 180.0

#: Host channel name -> snapshot key aliases, all optional.
_NODE_CHANNELS: dict[str, tuple[str, ...]] = {
    "cpu_util": ("cpu_util", "cpu_load", "node_cpu_util"),
    "mem_used_gb": ("mem_used_gb", "memory_used_gb"),
    "hwmon_temp_c": ("hwmon_temp_c", "hwmon_temp", "host_temp_c"),
    "powercap_w": ("powercap_w", "rapl_power_w", "powercap_power_w"),
    "powercap_limit_w": ("powercap_limit_w", "rapl_limit_w"),
    "thermal_zone_c": ("thermal_zone_c", "thermal_zone_temp_c"),
    "net_rx_bps": ("net_rx_bps", "network_receive_bps"),
    "net_tx_bps": ("net_tx_bps", "network_transmit_bps"),
}


@register_adapter("node_exporter")
class NodeExporterAdapter(BaseAdapter):
    """Map a node_exporter host-metrics snapshot to node/rack telemetry."""

    source_name = "node_exporter"
    default_family = ObservationFamily.NODE_RACK_TELEMETRY

    def parse(self, snapshot: dict[str, Any], payload: IngestPayload) -> None:
        ts_ns = int(snapshot.get("ts_ns") or 0)
        rack_power: dict[str, float] = defaultdict(float)
        rack_seen: dict[str, bool] = defaultdict(bool)
        for nd in snapshot.get("nodes", []) or []:
            self._parse_node(nd, payload, ts_ns, rack_power, rack_seen)
        # Coarse rack-power aggregation (a headroom signal, not a model output).
        for rack_id, present in rack_seen.items():
            if not present:
                continue
            payload.add_observation(
                Observation(
                    entity_kind="rack",
                    entity_id=rack_id,
                    family=ObservationFamily.NODE_RACK_TELEMETRY,
                    ts_event_ns=ts_ns,
                    channels={
                        "rack_host_power_w": make_channel(
                            "rack_host_power_w", rack_power[rack_id]
                        )
                    },
                )
            )

    def _parse_node(
        self,
        nd: dict[str, Any],
        payload: IngestPayload,
        ts_ns: int,
        rack_power: dict[str, float],
        rack_seen: dict[str, bool],
    ) -> None:
        uid = opt_str(nd, "node_uid", "uid", "name", "instance")
        if uid is None:
            return
        age_s = opt_float(nd, "age_s") or 0.0
        channels: dict[str, Any] = {}
        for ch_name, aliases in _NODE_CHANNELS.items():
            val = opt_float(nd, *aliases)
            channels[ch_name] = make_channel(
                ch_name, val, age_s=age_s, stale_after_s=NODE_STALE_AFTER_S
            )
        payload.add_observation(
            Observation(
                entity_kind="node",
                entity_id=uid,
                family=ObservationFamily.NODE_RACK_TELEMETRY,
                ts_event_ns=ts_ns,
                channels=channels,
            )
        )

        # Upsert a thin Node carrying ONLY the dynamic host-load fields this
        # source observes; the field-aware merge (base._merge_nodes) folds these
        # into the inventory Node the kubernetes adapter created, without the two
        # sources clobbering each other within a tick.
        node = payload.symbolic.nodes.get(uid) or Node(
            node_uid=uid, rack_id="", block_id="", switch_id="", flavor_id=""
        )
        rack_id = opt_str(nd, "rack_id", "rack")
        if rack_id is not None:
            node.rack_id = rack_id
        cpu = opt_float(nd, "cpu_alloc")
        if cpu is not None:
            node.cpu_alloc = cpu
        mem = opt_float(nd, "mem_alloc_gb", "mem_used_gb")
        if mem is not None:
            node.mem_alloc_gb = mem
        pods = opt_int(nd, "running_pods", "pods")
        if pods is not None:
            node.running_pods = pods
        payload.symbolic.nodes[uid] = node

        # Accumulate host power into the rack bucket when both are present.
        host_w = opt_float(nd, *_NODE_CHANNELS["powercap_w"])
        if rack_id is not None and host_w is not None:
            rack_power[rack_id] += host_w
            rack_seen[rack_id] = True

        payload.source.channel_age_s[f"node_exporter:{uid}"] = age_s


__all__ = ["NODE_STALE_AFTER_S", "NodeExporterAdapter"]
