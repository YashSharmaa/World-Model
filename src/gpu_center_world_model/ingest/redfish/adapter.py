# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Redfish adapter — out-of-band power / cooling / thermal metadata.

Redfish is the standards-based path for out-of-band inventory plus power-
distribution and liquid-cooling telemetry in heterogeneous environments (doc D).
It is the authoritative source for facility-level constraint objects that no
in-band collector sees: :class:`PowerDomain` budgets and headroom, and
:class:`CoolingZone` inlet/return temperatures and margins. These objects are
the aggregation points for the headroom / hotspot hazards (docs C, U).

Redfish support is marked experimental in v1 (doc H §422); the adapter degrades
gracefully when a chassis/PDU/cooling resource is absent.

Snapshot shape (flattened Redfish resources)::

    {
      "ts_ns": ...,
      "power_domains": [
        {"power_domain_id":"pdu-1","pdu_class":"rack","cap_kw":48.0,
         "used_kw":31.5,"reserve_kw":4.0,"age_s":30.0}, ...
      ],
      "cooling_zones": [
        {"cooling_zone_id":"cz-1","loop_type":"liquid","zone_cap_kw":120.0,
         "inlet_temp_c":24.0,"return_temp_c":33.0,"cooling_margin":0.7}, ...
      ],
      "racks": [
        {"rack_id":"rack-a","power_domain_id":"pdu-1","cooling_zone_id":"cz-1",
         "rack_power_kw":28.0,"rack_headroom_kw":12.0,
         "inferred_thermal_margin_c":6.0}, ...
      ]
    }
"""

from __future__ import annotations

from typing import Any

from ...schema.entities import CoolingZone, PowerDomain, Rack
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

#: Out-of-band cadence is slow (~30-60 s); flag stale beyond a few minutes.
REDFISH_STALE_AFTER_S = 300.0


@register_adapter("redfish")
class RedfishAdapter(BaseAdapter):
    """Map Redfish out-of-band power/cooling/thermal metadata to the schema."""

    source_name = "redfish"
    default_family = ObservationFamily.NODE_RACK_TELEMETRY

    def parse(self, snapshot: dict[str, Any], payload: IngestPayload) -> None:
        ts_ns = int(snapshot.get("ts_ns") or 0)
        for pd in snapshot.get("power_domains", snapshot.get("powerDomains", [])) or []:
            self._parse_power_domain(pd, payload, ts_ns)
        for cz in snapshot.get("cooling_zones", snapshot.get("coolingZones", [])) or []:
            self._parse_cooling_zone(cz, payload, ts_ns)
        for rk in snapshot.get("racks", []) or []:
            self._parse_rack(rk, payload, ts_ns)

    # ── PowerDomain ───────────────────────────────────────────────────────────
    def _parse_power_domain(
        self, pd: dict[str, Any], payload: IngestPayload, ts_ns: int
    ) -> None:
        pid = opt_str(pd, "power_domain_id", "id", "name")
        if pid is None:
            return
        age_s = opt_float(pd, "age_s") or 0.0
        used = opt_float(pd, "used_kw", "PowerConsumedWatts_kw")
        cap = opt_float(pd, "cap_kw", "PowerCapacityWatts_kw")
        reserve = opt_float(pd, "reserve_kw")
        domain = payload.symbolic.power_domains.get(pid) or PowerDomain(
            power_domain_id=pid
        )
        domain.pdu_class = opt_str(pd, "pdu_class", "class") or domain.pdu_class
        if cap is not None:
            domain.cap_kw = cap
        if used is not None:
            domain.used_kw = used
        if reserve is not None:
            domain.reserve_kw = reserve
        payload.symbolic.power_domains[pid] = domain

        payload.add_observation(
            Observation(
                entity_kind="power_domain",
                entity_id=pid,
                family=ObservationFamily.NODE_RACK_TELEMETRY,
                ts_event_ns=ts_ns,
                channels={
                    "power_used_kw": make_channel(
                        "power_used_kw",
                        used,
                        age_s=age_s,
                        stale_after_s=REDFISH_STALE_AFTER_S,
                    ),
                    "power_cap_kw": make_channel(
                        "power_cap_kw",
                        cap,
                        age_s=age_s,
                        stale_after_s=REDFISH_STALE_AFTER_S,
                    ),
                },
            )
        )
        payload.source.channel_age_s[f"redfish:power:{pid}"] = age_s

    # ── CoolingZone ───────────────────────────────────────────────────────────
    def _parse_cooling_zone(
        self, cz: dict[str, Any], payload: IngestPayload, ts_ns: int
    ) -> None:
        cid = opt_str(cz, "cooling_zone_id", "id", "name")
        if cid is None:
            return
        age_s = opt_float(cz, "age_s") or 0.0
        inlet = opt_float(cz, "inlet_temp_c", "SupplyTemperatureCelsius")
        ret = opt_float(cz, "return_temp_c", "ReturnTemperatureCelsius")
        margin = opt_float(cz, "cooling_margin")
        cap = opt_float(cz, "zone_cap_kw")
        zone = payload.symbolic.cooling_zones.get(cid) or CoolingZone(
            cooling_zone_id=cid
        )
        zone.loop_type = opt_str(cz, "loop_type", "CoolingType") or zone.loop_type
        if cap is not None:
            zone.zone_cap_kw = cap
        if inlet is not None:
            zone.inlet_temp_c = inlet
        if ret is not None:
            zone.return_temp_c = ret
        if margin is not None:
            zone.cooling_margin = margin
        payload.symbolic.cooling_zones[cid] = zone

        payload.add_observation(
            Observation(
                entity_kind="cooling_zone",
                entity_id=cid,
                family=ObservationFamily.NODE_RACK_TELEMETRY,
                ts_event_ns=ts_ns,
                channels={
                    "inlet_temp_c": make_channel(
                        "inlet_temp_c",
                        inlet,
                        age_s=age_s,
                        stale_after_s=REDFISH_STALE_AFTER_S,
                    ),
                    "return_temp_c": make_channel(
                        "return_temp_c",
                        ret,
                        age_s=age_s,
                        stale_after_s=REDFISH_STALE_AFTER_S,
                    ),
                    "cooling_margin": make_channel(
                        "cooling_margin",
                        margin,
                        age_s=age_s,
                        stale_after_s=REDFISH_STALE_AFTER_S,
                    ),
                },
            )
        )
        payload.source.channel_age_s[f"redfish:cooling:{cid}"] = age_s

    # ── Rack (out-of-band power/thermal facing) ───────────────────────────────
    def _parse_rack(
        self, rk: dict[str, Any], payload: IngestPayload, ts_ns: int
    ) -> None:
        rid = opt_str(rk, "rack_id", "id", "name")
        if rid is None:
            return
        rack = payload.symbolic.racks.get(rid) or Rack(
            rack_id=rid,
            block_id=opt_str(rk, "block_id") or "",
            power_domain_id=opt_str(rk, "power_domain_id") or "",
            cooling_zone_id=opt_str(rk, "cooling_zone_id") or "",
        )
        # back-fill containment ids if newly learned
        rack.power_domain_id = opt_str(rk, "power_domain_id") or rack.power_domain_id
        rack.cooling_zone_id = opt_str(rk, "cooling_zone_id") or rack.cooling_zone_id
        rack.block_id = opt_str(rk, "block_id") or rack.block_id
        cap = opt_int(rk, "designed_gpu_cap")
        if cap is not None:
            rack.designed_gpu_cap = cap
        for keys, attr in (
            (("rack_power_kw",), "rack_power_kw"),
            (("rack_headroom_kw",), "rack_headroom_kw"),
            (("inferred_thermal_margin_c",), "inferred_thermal_margin_c"),
        ):
            val = opt_float(rk, *keys)
            if val is not None:
                setattr(rack, attr, val)
        payload.symbolic.racks[rid] = rack


__all__ = ["REDFISH_STALE_AFTER_S", "RedfishAdapter"]
