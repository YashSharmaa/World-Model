# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The v1 action alphabet (the primitive treatment / intervention space).

Interventions are *command-level*: ``do(a)`` acts on the command presented to
the exact kernel, not on a physically impossible post-action state (doc I,
Definition 6). If a command is infeasible, the interventional semantics are
*reject / no-op with a blocker ledger*, never teleportation into an infeasible
allocation.

The v1 controllable alphabet is intentionally small and clean — ``hold``,
``admit``, ``place`` — because in GPU-center operations actions are already
explicit, logged, and semantically unambiguous (doc C: do *not* learn latent
actions for v1). ``admit_and_place`` is the most common composed control;
``mask_domain`` is an operator guardrail / scenario intervention, not an
autonomous actor action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ActionType(str, Enum):
    HOLD = "hold"
    ADMIT = "admit"
    PLACE = "place"
    ADMIT_AND_PLACE = "admit_and_place"
    MASK_DOMAIN = "mask_domain"  # operator guardrail / scenario control


#: The learned-counterfactual actor alphabet for v1 (doc C / J / P).
ACTOR_ACTIONS: tuple[ActionType, ...] = (
    ActionType.HOLD,
    ActionType.ADMIT,
    ActionType.PLACE,
    ActionType.ADMIT_AND_PLACE,
)


@dataclass(slots=True)
class Hold:
    """Defer a workload by one or more scheduling windows."""

    workload_uid: str
    hold_windows: int = 1
    type: ActionType = field(default=ActionType.HOLD, init=False)


@dataclass(slots=True)
class Admit:
    """Reserve quota for a workload in a (cluster_queue, flavor)."""

    workload_uid: str
    cluster_queue_id: str
    resource_flavor_id: str
    type: ActionType = field(default=ActionType.ADMIT, init=False)


@dataclass(slots=True)
class Place:
    """Choose the topology domain (or domain set) that hosts the gang."""

    workload_uid: str
    domain_policy: str = "preferred"  # required | preferred | unconstrained
    domain_id_or_set: tuple[str, ...] = ()
    type: ActionType = field(default=ActionType.PLACE, init=False)


@dataclass(slots=True)
class AdmitAndPlace:
    """The most common v1 control: admit + place in one command."""

    workload_uid: str
    cluster_queue_id: str
    resource_flavor_id: str
    domain_policy: str = "preferred"
    domain_id_or_set: tuple[str, ...] = ()
    type: ActionType = field(default=ActionType.ADMIT_AND_PLACE, init=False)


@dataclass(slots=True)
class MaskDomain:
    """Operator override: exclude a domain/rack from eligibility.

    Treated as an exogenous guardrail / scenario intervention (a change to the
    policy/context), *not* a normal actor action in v1.
    """

    domain_id_or_set: tuple[str, ...]
    reason: str = "operator_mask"
    type: ActionType = field(default=ActionType.MASK_DOMAIN, init=False)


Action = Hold | Admit | Place | AdmitAndPlace | MaskDomain


def action_workload_uid(action: Action) -> str | None:
    """Return the workload a command targets (``None`` for ``mask_domain``)."""
    return getattr(action, "workload_uid", None)


def is_actor_action(action: Action) -> bool:
    """True if the action belongs to the learned-counterfactual actor alphabet."""
    return action.type in ACTOR_ACTIONS


__all__ = [
    "ACTOR_ACTIONS",
    "Action",
    "ActionType",
    "Admit",
    "AdmitAndPlace",
    "Hold",
    "MaskDomain",
    "Place",
    "action_workload_uid",
    "is_actor_action",
]
