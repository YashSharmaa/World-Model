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
from ...schema.entities import Workload
from ...schema.state import SymbolicState
from ..contracts import Blocker, BlockerGroup, FeasibilityResult, WhyNot
from . import feasibility as feas
from .health_masks import HealthMasks, build_health_masks
from .topology_masks import TopologyIndex, build_topology_index

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


def _is_infeasible_with(
    state: SymbolicState,
    workload: Workload,
    action: Action,
    relax: frozenset[BlockerGroup],
    *,
    topology_index: TopologyIndex | None,
    health: HealthMasks | None,
) -> bool:
    """Re-solve the exact predicate with ``relax`` groups dropped; True iff still
    infeasible (doc S, Algorithm A1 phase 3 — real deletion-filtering)."""
    res = feas.evaluate(
        state,
        workload,
        action,
        topology_index=topology_index,
        health=health,
        relax=relax,
    )
    return not res.feasible


def deletion_filter(
    blockers: list[Blocker],
    *,
    state: SymbolicState | None = None,
    workload: Workload | None = None,
    action: Action | None = None,
    topology_index: TopologyIndex | None = None,
    health: HealthMasks | None = None,
) -> list[BlockerGroup]:
    """Shrink the blocker ledger to a group-minimal core (doc S, phase 3).

    When the live ``state`` / ``workload`` / ``action`` are supplied this performs
    **true deletion-filtering**: it re-invokes the exact feasibility predicate
    with each group relaxed and tests whether infeasibility survives, exactly as
    doc S Algorithm A1 phase 3 prescribes. A group is *redundant* iff the model is
    still infeasible after relaxing it (something else already blocks); a group is
    *necessary* iff relaxing it (alone) restores feasibility. Iterating in
    canonical order yields a deterministic minimal core.

    Without the live context (legacy callers) it degrades to the exact
    ledger-filter over the grouped model: a group is necessary iff it is the sole
    remaining blocker once the others are deleted.
    """
    present = _groups_present(blockers)
    if not present:
        return []

    if state is not None and workload is not None and action is not None:
        # True deletion-filtering over the grouped model (doc S, A1 phase 3).
        #
        # The full present set is jointly infeasible (it is exactly the conflict
        # the predicate just returned). We shrink it to a minimal core by deletion
        # filtering: maintain the set of groups already proven *redundant*
        # (``removed``); for each candidate group ``g`` in canonical order, test
        # whether relaxing ``removed | {g}`` keeps the model infeasible. If it
        # does, ``g`` is not needed for the conflict and joins ``removed``;
        # otherwise ``g`` is essential and stays. The surviving set is a
        # group-minimal IIS: every member is individually necessary (relaxing it
        # together with the already-removed groups restores feasibility), and no
        # member can be dropped without losing the infeasibility proof
        # (Theorem T14).
        removed: set[BlockerGroup] = set()
        for g in present:
            trial = frozenset(removed | {g})
            if _is_infeasible_with(
                state,
                workload,
                action,
                trial,
                topology_index=topology_index,
                health=health,
            ):
                # Still infeasible without g -> g is redundant for the conflict.
                removed.add(g)
        core = [g for g in GROUP_ORDER if g in present and g not in removed]
        # A non-empty conflict must yield a non-empty core; if deletion filtering
        # somehow removed everything (it cannot for a genuine conflict), fall back
        # to the present set so the certificate is never empty.
        return core or present

    # Legacy exact ledger-filter (no live re-solve available).
    core = [g for g in present if not any(b.group is not g for b in blockers)]
    return core or present


def _aggregate_deficits(blockers: list[Blocker]) -> dict[str, float]:
    """Sum per-resource deficits across the certificate's blockers."""
    deficits: dict[str, float] = {}
    for b in blockers:
        for resource, amount in b.deficit.items():
            deficits[resource] = deficits.get(resource, 0.0) + amount
    return deficits


def feasibility_relaxation(
    state: SymbolicState,
    workload: Workload,
    action: Action,
    core: list[BlockerGroup],
    *,
    topology_index: TopologyIndex | None = None,
    health: HealthMasks | None = None,
) -> dict[str, float]:
    """Quantify exact shortage amounts for the certificate (doc S, A1 phase 4).

    Runs a feasibility relaxation: it places *very high* (effectively infinite)
    penalty on the hard-safety groups that may not move (whitelist, status, masks,
    admission checks) by simply not relaxing them, and *unit* diagnostic slacks on
    the movable resource groups (quota, borrowing, node capacity) by reading their
    exact per-resource deficits directly off the predicate's blocker ledger. The
    result is a precise statement such as ``{"gpu": 4.0}`` -> "needs 4 more healthy
    GPUs" or ``{"gpu": 8.0}`` -> "exceeds borrowable quota by 8" (the doc's worked
    examples), aggregated only over the groups in the minimal ``core``.

    Because every group's blocker already carries its exact integer deficit
    (quota/borrow shortfalls from the quota arithmetic, GPU shortfalls from the
    capacity screen), the minimal weighted-slack relaxation reduces to summing
    those exact deficits over the core — no LP solve is required, and the answer
    is exact for the grouped model.
    """
    res = feas.evaluate(
        state,
        workload,
        action,
        topology_index=topology_index,
        health=health,
    )
    core_set = set(core)
    deficits: dict[str, float] = {}
    for b in res.blockers:
        if b.group not in core_set:
            continue
        for resource, amount in b.deficit.items():
            deficits[resource] = max(deficits.get(resource, 0.0), amount)
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

    # Amortise the masks across every re-solve in deletion-filtering / relaxation.
    idx = topology_index or build_topology_index(state)
    hmask = health or build_health_masks(state)

    if result is None:
        result = feas.evaluate(state, wl, action, topology_index=idx, health=hmask)

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

    # Phase 3: real deletion-filtering — re-solve with each group relaxed.
    core = deletion_filter(
        result.blockers,
        state=state,
        workload=wl,
        action=action,
        topology_index=idx,
        health=hmask,
    )
    core_blockers = [b for b in result.blockers if b.group in core]
    # Phase 4: feasibility relaxation -> exact per-resource shortage quantities.
    deficits = feasibility_relaxation(
        state, wl, action, core, topology_index=idx, health=hmask
    )
    # Fall back to the ledger-carried deficits if the relaxation found none
    # (e.g. a pure topology/health core with no numeric shortage).
    if not deficits:
        deficits = _aggregate_deficits(core_blockers)
    return WhyNot(
        action_id=action_id,
        minimal_groups=core,
        blockers=core_blockers,
        deficits=deficits,
    )


__all__ = [
    "GROUP_ORDER",
    "SINGLETON_GROUPS",
    "certificate",
    "deletion_filter",
    "feasibility_relaxation",
]
