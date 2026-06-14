# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Deterministic seed derivation for common-random-number counterfactuals.

Paired counterfactual branches MUST share one exogenous seed bundle so branch
deltas estimate intervention effects rather than intervention-plus-noise
(docs D, R, O). This module derives sub-stream seeds deterministically from a
master seed + a named stream, so every generator (arrivals, phases, faults,
thermal, network, telemetry) draws reproducibly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

#: Named exogenous noise streams (doc I/R exogenous bundle U).
STREAMS: tuple[str, ...] = (
    "regime",
    "arrivals",
    "phase",
    "health",
    "cooling",
    "operator",
    "source_health",
    "network",
    "telemetry",
)


def derive_seed(master: int, stream: str) -> int:
    """Derive a stable 64-bit sub-seed for a named noise stream."""
    blob = f"{master}:{stream}".encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class SeedBundle:
    """A reproducible bundle of per-stream seeds shared across branches."""

    master: int

    def stream(self, name: str) -> int:
        return derive_seed(self.master, name)

    def all_streams(self) -> dict[str, int]:
        return {s: self.stream(s) for s in STREAMS}


__all__ = ["STREAMS", "SeedBundle", "derive_seed"]
