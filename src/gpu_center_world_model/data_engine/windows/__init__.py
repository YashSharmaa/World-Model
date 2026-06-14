# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Decision-window builder — public surface.

Builds :class:`...data_engine.contracts.DecisionWindow` records anchored by a
pre-action state hash, with a strict no-post-action-leakage guarantee (design
docs D, I, R).
"""

from __future__ import annotations

from ..contracts import DecisionWindow, LabelPack, Origin, Split
from .builder import (
    WindowFeatures,
    assemble_features,
    assert_no_leakage,
    build_window,
    derive_seed_bundle,
    labels_from_pack,
    temporal_split,
)

__all__ = [
    "WindowFeatures",
    "assemble_features",
    "assert_no_leakage",
    "labels_from_pack",
    "build_window",
    "temporal_split",
    "derive_seed_bundle",
    # re-export contracts for one clean surface
    "DecisionWindow",
    "LabelPack",
    "Origin",
    "Split",
]
