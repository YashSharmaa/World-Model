# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Backend-neutral tensorizer — public surface.

Turns a WorldState + window features into a padded
:class:`~...schema.tensors.TensorBatch` with per-entity masks and one-hot
incidence matrices ``S_r``/``T_r`` for typed message passing (design docs C, Q, E).
"""

from __future__ import annotations

from ...schema.tensors import ShapeCaps, TensorBatch, pick_bucket
from .tensorizer import (
    GPU_FEATURES,
    NODE_FEATURES,
    QUEUE_FEATURES,
    RACK_FEATURES,
    WORKLOAD_FEATURES,
    EntityIndex,
    tensorize,
)

__all__ = [
    "tensorize",
    "EntityIndex",
    "WORKLOAD_FEATURES",
    "NODE_FEATURES",
    "GPU_FEATURES",
    "RACK_FEATURES",
    "QUEUE_FEATURES",
    # re-export schema tensor contract for one clean surface
    "ShapeCaps",
    "TensorBatch",
    "pick_bucket",
]
