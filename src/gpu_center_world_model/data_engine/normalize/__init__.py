# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Canonical normalisation (Bronze → Silver → Gold) — public surface.

Maps raw ingest snapshots/events into the canonical schema across three
synchronised lanes (fast / meso / sparse events), keeping missingness and
staleness as first-class state and never zero-filling (design doc D).
"""

from __future__ import annotations

from .canonical import (
    FAMILY_LANE,
    Lane,
    MetricSample,
    SchedulerEventRow,
    SilverFrame,
    bronze_to_silver,
    observations_to_metric_samples,
    source_state_from_silver,
)

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
