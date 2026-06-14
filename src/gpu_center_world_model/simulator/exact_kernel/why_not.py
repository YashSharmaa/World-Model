# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Group-minimal infeasibility certificates (semantic IIS) — doc S, Algorithm A1.

A "why-not" answer is a *group-minimal* infeasibility certificate: a subset of
semantic :class:`BlockerGroup`\\ s that is jointly infeasible and becomes
feasible if any single group is removed (the grouped analogue of an
Irreducible Inconsistent Subsystem / MUS, doc S Definitions D15/D16,
Theorem T14).

Extraction follows the four phases of doc S Algorithm A1:

1. **Singleton guards.** Whitelist, status, masks, flavor eligibility, and quota
   are exact singleton screens; if exactly one such guard fails it is already a
   minimal certificate and is returned immediately.
2. **Collect the full blocker ledger** from :func:`feasibility.evaluate`.
3. **Deletion filtering** to a group-minimal core: tentatively *relax* each group
   (treat its blockers as satisfied) and re-test; a group that is unnecessary for
   infeasibility is dropped. Because we already hold the exact blocker ledger,
   relaxation is done by filtering the ledger rather than re-solving — exact for
   the grouped model.
4. **Deficit quantification.** Per-resource shortages are aggregated into the
   certificate (e.g. ``{"gpu": 4.0}`` -> "needs 4 more healthy GPUs").

The certificate is exact relative to the semantic grouping but need not be unique
(doc S, Theorem T14); we canonicalise with a deterministic group ordering.
"""

from __future__ import annotations

from ...schema.actions import Action, action_workload_uid
from ...schema.state import SymbolicState
from ..contracts import Blocker, BlockerGroup, FeasibilityResult, WhyNot
from . import feasibility as feas
from .health_masks import HealthMasks
from .topology_masks import TopologyIndex

#: Deterministic canonical order over semantic groups (doc S, Theorem T14).
GROUP_ORDER: tuple[BlockerGroup, ...] = (
    BlockerGroup.ACTION_WHITELIST,
    BlockerGroup.STATUS_PRECONDITION,
    BlockerGroup.ADMISSION_CHECKS,
    BlockerGroup.QUEUE_FLAVOR_ELIGIBILITY,
    BlockerGroup.PROTECTED_DOMAIN_MASK,
    BlockerGroup.HEALTH_SCHEDULABILITY,
    BlockerGroup.TOPOLOGY_RULE,
    BlockerGroup.NODE_CAPACITIES,
    BlockerGroup.QUOTA,
    BlockerGroup.BORROWING_LENDING,
    BlockerGroup.GANG_ATOMICITY,
)

#: Groups that are deterministic singleton screens (phase 1 of Algorithm A1).
SINGLETON_GROUPS: frozenset[BlockerGroup] = frozenset(
    {
        BlockerGroup.ACTION_WHITELIST,
        BlockerGroup.STATUS_PRECONDITION,
        BlockerGroup.ADMISSION_CHECKS,
        BlockerGroup.QUEUE_FLAVOR_ELIGIBILITY,
        BlockerGroup.PROTECTED_DOMAIN_MASK,
    }
)


def _groups_present(blockers: list[Blocker]) -> list[BlockerGroup]:
    present = {b.group for b in blockers}
    return [g for g in GROUP_ORDER if g in present]


def _ledger_infeasible(blockers: list[Blocker], drop: BlockerGroup) -> bool:
    """Would the action still be infeasible if ``drop``'s blockers were satisfied?

    Exact for the grouped model: relaxing a group means deleting its rows from
    the ledger; the action is still infeasible iff any other blocker remains.
    """
    return any(b.group is not drop for b in blockers)


def deletion_filter(blockers: list[Blocker]) -> list[BlockerGroup]:
    """Shrink the blocker ledger to a group-minimal core (doc S, phase 3).

    A group is *necessary* iff removing it (alone) makes the residual ledger
    feasible. We iterate in canonical order so the returned core is deterministic.
    """
    present = _groups_present(blockers)
    core: list[BlockerGroup] = []
    for g in present:
        # g is necessary iff dropping g would restore feasibility, i.e. nothing
        # else blocks. Equivalently: if the model is infeasible without g, g is
        # redundant for the minimal core.
        if not _ledger_infeasible(blockers, g):
            core.append(g)
    # If every group is jointly required (no single removal restores feasibility)
    # the whole present set is the minimal core.
    return core or present


def _aggregate_deficits(blockers: list[Blocker]) -> dict[str, float]:
    """Sum per-resource deficits across the certificate's blockers."""
    deficits: dict[str, float] = {}
    for b in blockers:
        for resource, amount in b.deficit.items():
            deficits[resource] = deficits.get(resource, 0.0) + amount
    return deficits


def certificate(
    state: SymbolicState,
    action: Action,
    *,
    result: FeasibilityResult | None = None,
    topology_index: TopologyIndex | None = None,
    health: HealthMasks | None = None,
) -> WhyNot:
    """Return a minimal semantic infeasibility certificate for ``action``.

    If ``result`` is supplied (already computed by the kernel) we reuse its
    blocker ledger; otherwise we run the exact feasibility predicate.
    """
    action_id = action_workload_uid(action) or action.type.value
    wl_uid = action_workload_uid(action)

    if result is None:
        wl = state.workloads.get(wl_uid) if wl_uid else None
        if wl is None:
            return WhyNot(
                action_id=action_id,
                minimal_groups=[BlockerGroup.STATUS_PRECONDITION],
                blockers=[
                    Blocker(
                        BlockerGroup.STATUS_PRECONDITION,
                        f"workload '{wl_uid}' not found",
                    )
                ],
            )
        result = feas.evaluate(
            state, wl, action, topology_index=topology_index, health=health
        )

    if result.feasible:
        # Feasible actions have an empty certificate by definition.
        return WhyNot(action_id=action_id, minimal_groups=[], blockers=[])

    # Phase 1: singleton guards — a lone failing singleton is already minimal.
    present = _groups_present(result.blockers)
    singletons = [g for g in present if g in SINGLETON_GROUPS]
    if len(present) == 1 and singletons:
        g = present[0]
        rows = [b for b in result.blockers if b.group is g]
        return WhyNot(
            action_id=action_id,
            minimal_groups=[g],
            blockers=rows,
            deficits=_aggregate_deficits(rows),
        )

    # Phases 3-4: deletion-filter to a group-minimal core and quantify deficits.
    core = deletion_filter(result.blockers)
    core_blockers = [b for b in result.blockers if b.group in core]
    return WhyNot(
        action_id=action_id,
        minimal_groups=core,
        blockers=core_blockers,
        deficits=_aggregate_deficits(core_blockers),
    )


__all__ = [
    "GROUP_ORDER",
    "SINGLETON_GROUPS",
    "certificate",
    "deletion_filter",
]
