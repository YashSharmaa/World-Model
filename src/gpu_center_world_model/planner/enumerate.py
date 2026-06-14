# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Safe-candidate enumeration — the constructive zero-violation front door.

The planner *never* invents actions. It asks the exact kernel (the safety shield,
injected via the :class:`~..simulator.contracts.ExactKernel` Protocol) for the
complete, hard-feasible candidate set ``A_K(b) = {a : K(b, a) = 1}`` for the
queued workload under consideration (doc M, D3; doc S, D1).

Design rationale — *enum-now, search-later* (doc M ``enumeration_vs_search``):
while the v1 action ontology is ``{hold, admit, place, admit_and_place}`` over a
2-4 rack demo pod, the safe set is finite and small, so exhaustive enumeration is
**decision-theoretically optimal**: search error is exactly zero whenever every
safe candidate is scored. We therefore enumerate fully here and defer
MPC/CEM/MCTS to a later action family. The learned model may rank / forecast /
abstain over this set but may *never expand it* (doc C / F / M).

This module also guarantees that ``hold`` is always present as the conservative
fallback (doc F step 6: "if the safe set is empty except hold, return hold"),
and it canonicalises a stable, content-addressed ``action_id`` per candidate so
the ledger, forecasts, and UI all key on the same identifier.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from ..schema.actions import Action, ActionType, Hold, action_workload_uid
from ..schema.policy import PolicyConfig
from ..schema.state import WorldState
from ..simulator.contracts import ExactKernel

__all__ = [
    "action_id_for",
    "enumerate_safe_candidates",
    "filter_by_policy_whitelist",
]


# ──────────────────────────────────────────────────────────────────────────────
# Stable action identity
# ──────────────────────────────────────────────────────────────────────────────
def _canonical(obj: Any) -> Any:
    """Recursively canonicalise a dataclass/enum/collection for hashing.

    Mirrors :mod:`schema.hashing` semantics so the same action always maps to the
    same id regardless of dict ordering or float micro-jitter.
    """
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
        return round(obj, 6)
    return obj


def action_id_for(action: Action) -> str:
    """Return a stable, content-addressed identifier for a candidate action.

    The id is ``"<type>:<12-hex digest>"`` — human-skimmable (the action type is
    visible) yet collision-resistant and reproducible across processes, so the
    forecast bundle, ledger entries, and UI ribbon can all key on it (doc F).
    """
    payload = _canonical(action)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()[:12]
    return f"{action.type.value}:{digest}"


# ──────────────────────────────────────────────────────────────────────────────
# Policy whitelist (operator guardrail, applied *before* the kernel where cheap)
# ──────────────────────────────────────────────────────────────────────────────
def filter_by_policy_whitelist(
    actions: list[Action], policy: PolicyConfig
) -> list[Action]:
    """Drop any action whose type is not in the operator's ``allowed_actions``.

    The exact kernel already enforces the action whitelist as a hard blocker
    (``BlockerGroup.ACTION_WHITELIST``); applying it here too is a cheap,
    defence-in-depth pre-filter so we never even forecast a disallowed action.
    ``mask_domain`` is treated as an exogenous guardrail, not an actor action,
    and is excluded from the actor candidate set regardless of the whitelist.
    """
    allowed = set(policy.allowed_actions)
    out: list[Action] = []
    for a in actions:
        if a.type is ActionType.MASK_DOMAIN:
            continue
        if a.type.value in allowed:
            out.append(a)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# The enumeration entry point
# ──────────────────────────────────────────────────────────────────────────────
def enumerate_safe_candidates(
    kernel: ExactKernel,
    state: WorldState,
    workload_uid: str,
    policy: PolicyConfig,
) -> list[tuple[str, Action]]:
    """Return the kernel-certified safe candidate set for one queued workload.

    Steps:

    1. Ask the injected exact kernel for ``safe_candidates(state, workload_uid)``.
       By contract these are already hard-feasible (``K(b, a) = 1``).
    2. Apply the operator action whitelist as a defence-in-depth pre-filter and
       drop any exogenous ``mask_domain`` guardrail actions.
    3. Guarantee ``hold`` is present as the conservative fallback (doc F step 6;
       ``hold`` for a pending workload is always hard-feasible).
    4. Deduplicate by stable ``action_id`` while preserving the kernel's order.

    Returns a list of ``(action_id, action)`` pairs. The constructive
    zero-violation property of the whole planner rests on this set: every action
    the planner can ever emit is drawn from here, so it is hard-safe by
    construction (doc M, T1 / doc S, T1).
    """
    raw = list(kernel.safe_candidates(state, workload_uid))
    allowed = filter_by_policy_whitelist(raw, policy)

    # Ensure hold is always available as the conservative fallback. The kernel
    # should already include it, but the planner must never be left without an
    # emittable safe action when hold is operator-permitted.
    if ActionType.HOLD.value in set(policy.allowed_actions):
        has_hold = any(
            a.type is ActionType.HOLD and action_workload_uid(a) == workload_uid
            for a in allowed
        )
        if not has_hold:
            allowed.append(Hold(workload_uid=workload_uid))

    seen: set[str] = set()
    out: list[tuple[str, Action]] = []
    for a in allowed:
        aid = action_id_for(a)
        if aid in seen:
            continue
        seen.add(aid)
        out.append((aid, a))
    return out
