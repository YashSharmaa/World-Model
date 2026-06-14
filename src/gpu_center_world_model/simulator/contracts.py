# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Exact-kernel and branch-runner interface contracts.

The exact kernel is the project's **safety shield** and the source of truth for
hard feasibility. It is deterministic and auditable, never learned (docs C, I,
M, S). These Protocols are the fixed boundary between the simulator and the
planner / data engine / evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from ..schema.actions import Action
from ..schema.state import WorldState


class BlockerGroup(str, Enum):
    """Semantic blocker families for why-not certificates (doc S)."""

    ACTION_WHITELIST = "action_whitelist"
    STATUS_PRECONDITION = "status_precondition"
    ADMISSION_CHECKS = "admission_checks"
    QUEUE_FLAVOR_ELIGIBILITY = "queue_flavor_eligibility"
    PROTECTED_DOMAIN_MASK = "protected_domain_mask"
    TOPOLOGY_RULE = "topology_rule"
    HEALTH_SCHEDULABILITY = "health_schedulability"
    NODE_CAPACITIES = "node_capacities"
    QUOTA = "quota"
    BORROWING_LENDING = "borrowing_lending"
    GANG_ATOMICITY = "gang_atomicity"


@dataclass(slots=True)
class Blocker:
    """A single exact reason an action is infeasible, with a deficit if known."""

    group: BlockerGroup
    detail: str
    deficit: dict[str, float] = field(default_factory=dict)  # e.g. {"gpu": 4}


@dataclass(slots=True)
class FeasibilityResult:
    """Exact yes/no plus blocker ledger and the feasible domain set."""

    feasible: bool
    blockers: list[Blocker] = field(default_factory=list)
    feasible_domains: list[str] = field(default_factory=list)
    # the concrete integer assignment (podset -> {node: replicas}) if feasible
    assignment: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass(slots=True)
class WhyNot:
    """A minimal (group-level) infeasibility certificate (doc S, IIS analogue)."""

    action_id: str
    minimal_groups: list[BlockerGroup] = field(default_factory=list)
    blockers: list[Blocker] = field(default_factory=list)
    deficits: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class ExactKernel(Protocol):
    """Deterministic symbolic feasibility gate + transition (the shield)."""

    def feasible(self, state: WorldState, action: Action) -> FeasibilityResult:
        """Return exact feasibility + blockers + feasible domains for one action."""
        ...

    def safe_candidates(self, state: WorldState, workload_uid: str) -> list[Action]:
        """Enumerate the kernel-certified safe candidate set for a workload."""
        ...

    def why_not(self, state: WorldState, action: Action) -> WhyNot:
        """Return a minimal semantic infeasibility certificate for a rejected action."""
        ...

    def step(self, state: WorldState, action: Action) -> WorldState:
        """Apply ``X_{t+1} = K(X_t, a_t, ...)`` deterministically (symbolic block)."""
        ...


@dataclass(slots=True)
class BranchRoot:
    """A counterfactual branch root: a frozen pre-action state hash + seed bundle."""

    state_hash: str
    seed_bundle: int
    horizon_s: int


@runtime_checkable
class BranchRunner(Protocol):
    """Deterministic branch replay from a fixed pre-action hash and seed bundle."""

    def root(self, state: WorldState, horizon_s: int, seed: int) -> BranchRoot: ...

    def rollout(
        self, root: BranchRoot, state: WorldState, action: Action
    ) -> list[WorldState]:
        """Roll a single action branch forward under common random numbers."""
        ...


__all__ = [
    "Blocker",
    "BlockerGroup",
    "BranchRoot",
    "BranchRunner",
    "ExactKernel",
    "FeasibilityResult",
    "WhyNot",
]
