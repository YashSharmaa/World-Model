# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Redfish ingest adapter package (out-of-band power/cooling/thermal metadata)."""

from __future__ import annotations

from .adapter import REDFISH_STALE_AFTER_S, RedfishAdapter

__all__ = ["REDFISH_STALE_AFTER_S", "RedfishAdapter"]
