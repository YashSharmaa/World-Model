# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Flat feature extraction shared by the *non-relational* baselines (doc F).

The tabular and flat-sequence baselines deliberately throw away typed object
structure: they consume a single global feature vector. This module produces
that vector so the baselines stay honest foils — any advantage of the relational
world model must come from structure these features cannot express.
"""

from __future__ import annotations

import numpy as np

from ...schema.actions import Action
from ...schema.entities import DeviceHealth, WorkloadStatus
from ...schema.state import WorldState

#: Fixed-width flat state summary (global aggregates only — no object identity).
FLAT_STATE_DIM = 16
_ACTION_TYPES = ("hold", "admit", "place", "admit_and_place", "mask_domain")


def flatten_state(state: WorldState) -> np.ndarray:
    """Aggregate the symbolic state into a fixed-width global feature vector."""
    X = state.symbolic
    gpus = list(X.gpus.values())
    nodes = list(X.nodes.values())
    wls = list(X.workloads.values())
    n_gpu = max(len(gpus), 1)
    busy = sum(1 for g in gpus if g.allocated_to) / n_gpu
    util = float(np.mean([g.util for g in gpus])) if gpus else 0.0
    temp = float(np.mean([g.temp_c for g in gpus])) if gpus else 0.0
    power = float(np.mean([g.power_w for g in gpus])) if gpus else 0.0
    unhealthy = sum(1 for g in gpus if g.health_state != DeviceHealth.HEALTHY) / n_gpu
    pending = sum(1 for w in wls if w.status == WorkloadStatus.PENDING)
    running = sum(1 for w in wls if w.status == WorkloadStatus.RUNNING)
    free_gpu = sum(d.free_gpu for d in X.domains.values())
    headroom = (
        float(np.mean([d.headroom_score for d in X.domains.values()]))
        if X.domains
        else 1.0
    )
    hotspot = (
        float(np.mean([d.hotspot_score for d in X.domains.values()]))
        if X.domains
        else 0.0
    )
    ready_avail = sum(1 for n in nodes if n.ready and n.schedulable)
    feat = np.array(
        [
            busy,
            util,
            temp / 100.0,
            power / 1000.0,
            unhealthy,
            float(pending),
            float(running),
            float(free_gpu),
            headroom,
            hotspot,
            float(ready_avail),
            float(len(nodes)),
            float(len(gpus)) / 256.0,
            float(len(X.racks)),
            state.context.ambient_temp_c / 30.0,
            1.0 if state.context.regime == "normal" else 0.0,
        ],
        dtype=np.float64,
    )
    return feat


def action_onehot(action: Action) -> np.ndarray:
    """One-hot encode the action type (the only action signal these baselines see)."""
    v = np.zeros((len(_ACTION_TYPES),), dtype=np.float64)
    idx = _ACTION_TYPES.index(action.type.value)
    v[idx] = 1.0
    return v


__all__ = ["FLAT_STATE_DIM", "action_onehot", "flatten_state"]
