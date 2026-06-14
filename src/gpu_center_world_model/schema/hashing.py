# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Deterministic pre-action state hashing for counterfactual branch alignment.

A **state hash** is taken immediately before every operator/scheduler action.
Every real or simulated rollout points back to that same pre-action hash so
counterfactual branches differ *only* by the clamped action — never by drifting
exogenous conditions. Without this, action effects get contaminated by changed
context, the single biggest source of false optimism in learned world models
(docs D, I, R).

The hash is a stable, canonical digest of the *symbolic* state (``X_t``) plus
the exogenous context (``C_t``). The learned residual belief ``Z_t`` is excluded
because it is an estimate, not ground truth, and would make the hash
non-reproducible across model versions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .state import ExogenousContext, SymbolicState, WorldState


def _canonical(obj: Any) -> Any:
    """Recursively convert dataclasses/enums/sets to canonically-ordered JSON."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _canonical(getattr(obj, f.name))
            for f in sorted(dataclasses.fields(obj), key=lambda x: x.name)
        }
    if isinstance(obj, dict):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(x) for x in obj]
    if isinstance(obj, set):
        return sorted(_canonical(x) for x in obj)
    if hasattr(obj, "value") and hasattr(obj, "name"):  # Enum
        return obj.value
    if isinstance(obj, float):
        # Quantize floats so micro-jitter does not change the hash.
        return round(obj, 6)
    return obj


def state_hash(
    symbolic: SymbolicState,
    context: ExogenousContext | None = None,
    *,
    algo: str = "sha256",
) -> str:
    """Return a stable hex digest of the pre-action symbolic state (+ context).

    Determinism guarantees: same canonical state -> same hash, independent of
    dict insertion order, float jitter below 1e-6, and Python process.
    """
    payload = {"X": _canonical(symbolic)}
    if context is not None:
        payload["C"] = _canonical(context)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.new(algo, blob).hexdigest()


def world_state_hash(state: WorldState, *, algo: str = "sha256") -> str:
    """Convenience: hash a full :class:`WorldState`'s symbolic + context blocks."""
    return state_hash(state.symbolic, state.context, algo=algo)


__all__ = ["state_hash", "world_state_hash"]
