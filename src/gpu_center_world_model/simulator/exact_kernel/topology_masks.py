# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Topology domain hierarchy, node->domain membership, and candidate enumeration.

The kernel must reason about *topology selections* ``sigma in Sigma_w`` — domains
or domain-sets admissible under a workload's locality intent (doc S, Definition
D3). This module compiles the Kueue Topology-Aware-Scheduling (TAS) hierarchy
(block / rack / node / compute-domain) into:

* ``node_to_domains`` — for each node, the set of domain ids it belongs to at
  every level (a node belongs to its rack, that rack's block, etc.);
* ``domain_nodes`` — the inverse map, domain id -> member node ids;
* a candidate-selection enumerator that, given the workload's
  :class:`TopologyPolicy`, returns the admissible ``sigma`` set with operator and
  protected-domain masks already applied.

Locality semantics implemented (doc S / doc C):

* ``REQUIRED``   — the whole gang must fit inside *one* domain. Candidate
  selections are the individual leaf+ancestor domains (same-rack, same-block).
* ``PREFERRED``  — prefer one domain; fall back to a wider domain or, if none
  fits, to an unconstrained fill. We enumerate single domains *plus* the
  unconstrained whole-pod selection so the feasibility predicate can pick the
  tightest that fits.
* ``UNCONSTRAINED`` — fill gaps anywhere; the single selection is the set of all
  unmasked schedulable nodes (fragmentation-filling mode).

Everything here is exact and read-only; it never learns a topology preference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...schema.entities import TopologyLevel, TopologyPolicy
from ...schema.state import SymbolicState

#: Synthetic domain id used for the "unconstrained / whole-pod" selection. Kept
#: out of the real domain namespace by an unambiguous prefix.
UNCONSTRAINED_DOMAIN_ID = "__unconstrained__"

#: Synthetic rack-level domains derived from a node's ``rack_id`` are tagged with
#: this level so per-level enumeration (rack-locality vs block-locality) can
#: distinguish them from explicit block/compute domains (doc S, Definition D3).
_SYNTHETIC_RACK_LEVEL = TopologyLevel.RACK


@dataclass(slots=True)
class TopologyIndex:
    """Compiled node<->domain membership and the domain hierarchy.

    ``ancestors[domain_id]`` lists the domain plus all of its ancestor domains
    (rack -> block -> ...). ``node_domain[node_uid]`` is the *leaf* domain a node
    sits in if one is declared via :class:`TopologyDomain`; otherwise we derive a
    synthetic per-rack domain from the node's ``rack_id`` so that same-rack
    locality still works without an explicit domain table.
    """

    domain_nodes: dict[str, list[str]] = field(default_factory=dict)
    node_to_domains: dict[str, set[str]] = field(default_factory=dict)
    ancestors: dict[str, list[str]] = field(default_factory=dict)
    all_node_uids: list[str] = field(default_factory=list)
    #: ``domain_id -> TopologyLevel`` for every real or synthetic domain. Drives
    #: per-level candidate enumeration (rack-locality vs block-locality, doc S
    #: Definition D3 / multi-layer slice handling).
    domain_level: dict[str, TopologyLevel] = field(default_factory=dict)

    def level_of(self, domain_id: str) -> TopologyLevel | None:
        """Return the topology level of a domain id (``None`` if unknown)."""
        return self.domain_level.get(domain_id)

    def nodes_in_selection(self, selection: tuple[str, ...]) -> list[str]:
        """Union of member nodes across every domain id in ``selection``.

        The synthetic :data:`UNCONSTRAINED_DOMAIN_ID` expands to every node.
        """
        if UNCONSTRAINED_DOMAIN_ID in selection:
            return list(self.all_node_uids)
        seen: dict[str, None] = {}
        for dom in selection:
            for n in self.domain_nodes.get(dom, ()):  # type: ignore[arg-type]
                seen.setdefault(n, None)
        return list(seen)


def _synthetic_rack_domain(rack_id: str) -> str:
    """Stable synthetic domain id for a rack with no explicit domain entry."""
    return f"rack::{rack_id}"


def build_topology_index(state: SymbolicState) -> TopologyIndex:
    """Compile node<->domain membership from the symbolic state.

    The hierarchy is built from two complementary sources so the kernel works
    whether or not an explicit :class:`TopologyDomain` table is populated:

    1. **Explicit domains.** If ``state.domains`` declares parent links we walk
       ``parent_domain_id`` chains to compute ancestor closures and use a node's
       ``rack_id``/``block_id`` to attach it to matching rack/block domains.
    2. **Implicit rack domains.** Every node is always attached to a synthetic
       ``rack::<rack_id>`` domain so same-rack required locality is exact even on
       minimal fixtures.
    """
    idx = TopologyIndex()
    idx.all_node_uids = list(state.nodes.keys())

    # Ancestor closure + declared level for every explicit domain.
    for dom_id, dom in state.domains.items():
        chain: list[str] = []
        cur: str | None = dom_id
        guard = 0
        while cur is not None and guard < 64:
            chain.append(cur)
            parent = state.domains.get(cur)
            cur = parent.parent_domain_id if parent is not None else None
            guard += 1
        idx.ancestors[dom_id] = chain
        idx.domain_level[dom_id] = dom.level

    # Attach nodes to domains.
    for node_uid, node in state.nodes.items():
        doms: set[str] = set()
        # Synthetic per-rack domain (always present).
        rack_dom = _synthetic_rack_domain(node.rack_id)
        doms.add(rack_dom)
        idx.domain_nodes.setdefault(rack_dom, []).append(node_uid)
        idx.ancestors.setdefault(rack_dom, [rack_dom])
        idx.domain_level.setdefault(rack_dom, _SYNTHETIC_RACK_LEVEL)
        # Explicit domains whose id matches the node's rack/block, plus ancestors.
        for cand in (node.rack_id, node.block_id):
            if cand in state.domains:
                for anc in idx.ancestors.get(cand, [cand]):
                    doms.add(anc)
                    idx.domain_nodes.setdefault(anc, [])
                    if node_uid not in idx.domain_nodes[anc]:
                        idx.domain_nodes[anc].append(node_uid)
        idx.node_to_domains[node_uid] = doms

    # Ensure every declared domain has a (possibly empty) member list.
    for dom_id in state.domains:
        idx.domain_nodes.setdefault(dom_id, [])
    return idx


def _is_masked(domain_id: str, masked: set[str], protected: set[str]) -> bool:
    """A selection domain is excluded if it is operator-masked or protected."""
    return domain_id in masked or domain_id in protected


def candidate_selections(
    state: SymbolicState,
    index: TopologyIndex,
    policy: TopologyPolicy,
    *,
    masked_domains: set[str] | None = None,
    protected_domains: set[str] | None = None,
    requested: tuple[str, ...] = (),
    level: TopologyLevel | None = None,
) -> list[tuple[str, ...]]:
    """Enumerate admissible topology selections ``Sigma_w`` for one workload.

    Operator/protected masks (``pi_sigma(s)`` in doc S, Eq. E10) are applied here
    so masked domains never enter the candidate set in the first place. If the
    caller pinned an explicit ``requested`` domain set (the ``Place`` action's
    ``domain_id_or_set``), only that selection is returned (still mask-checked).

    ``level`` restricts the enumerated single domains to one topology level
    (doc S, Definition D3 / multi-layer slice handling): pass
    :attr:`TopologyLevel.RACK` for same-rack locality, :attr:`TopologyLevel.BLOCK`
    for same-block locality, etc. With ``level=None`` every level's domains are
    enumerated (the prior behaviour). The unconstrained fill is independent of
    level and is added under PREFERRED/UNCONSTRAINED as before.

    Returns a deterministic, de-duplicated list of selection tuples; each tuple
    is a topology selection ``sigma`` whose member nodes are looked up via
    :meth:`TopologyIndex.nodes_in_selection`.
    """
    masked = masked_domains or set()
    protected = protected_domains or set()

    def allowed(sel: tuple[str, ...]) -> bool:
        return not any(_is_masked(d, masked, protected) for d in sel)

    # Explicitly requested selection short-circuits enumeration.
    if requested:
        sel = tuple(requested)
        return [sel] if allowed(sel) else []

    # All single explicit/synthetic domains, sorted for determinism, optionally
    # filtered to the requested topology level.
    single_domains = sorted(
        d
        for d in index.domain_nodes
        if index.domain_nodes[d]
        and (level is None or index.domain_level.get(d) is level)
    )
    selections: list[tuple[str, ...]] = []

    if policy is TopologyPolicy.UNCONSTRAINED:
        # One selection: fill anywhere (fragmentation mode).
        if allowed((UNCONSTRAINED_DOMAIN_ID,)):
            selections.append((UNCONSTRAINED_DOMAIN_ID,))
        return selections

    # REQUIRED and PREFERRED both consider single domains (tightest first by
    # size). PREFERRED additionally allows the unconstrained fallback.
    for dom in single_domains:
        sel = (dom,)
        if allowed(sel):
            selections.append(sel)

    if policy is TopologyPolicy.PREFERRED and allowed((UNCONSTRAINED_DOMAIN_ID,)):
        selections.append((UNCONSTRAINED_DOMAIN_ID,))

    return selections


def prune_dominated_selections(
    index: TopologyIndex,
    selections: list[tuple[str, ...]],
    cap_of: dict[tuple[str, ...], tuple[int, float, float, int]],
) -> list[tuple[str, ...]]:
    """Dominance + symmetry reduction over candidate selections (doc S pruning).

    The doc's pruning order (``latency_budget.pruning``) calls for: (a) **symmetry
    reduction** — collapse selections that are permutations of one another (here,
    selections whose member-node *set* is identical, e.g. two synthetic rack
    domains over the same nodes) to a single representative; and (b) **domain
    dominance** — when two selections sit at the *same topology level* and one's
    residual resource vector dominates the other's componentwise, the dominated
    one carries strictly less placement capability and can be discarded.

    Only a *dominated* selection is removed, and only at the same level, so this
    is an exact reduction: it never deletes a selection that could host a gang the
    survivor cannot. ``cap_of`` maps each selection to its aggregate usable
    ``(gpu, cpu, mem, pods)`` residual vector. Returns the survivors in the input
    order (determinism).
    """
    # (a) Symmetry: dedupe by frozenset of member nodes, keeping the first.
    seen_nodes: dict[frozenset[str], tuple[str, ...]] = {}
    deduped: list[tuple[str, ...]] = []
    for sel in selections:
        key = frozenset(index.nodes_in_selection(sel))
        if key in seen_nodes:
            continue
        seen_nodes[key] = sel
        deduped.append(sel)

    # (b) Dominance at equal level: drop sel if another *distinct* survivor at the
    # same level dominates it componentwise (>= on every resource, > on at least
    # one) — that survivor can host every gang sel could, plus strictly more.
    def level_key(sel: tuple[str, ...]) -> object:
        levels = {index.domain_level.get(d) for d in sel}
        return next(iter(levels)) if len(levels) == 1 else None

    survivors: list[tuple[str, ...]] = []
    for sel in deduped:
        cap = cap_of.get(sel)
        if cap is None:
            survivors.append(sel)
            continue
        lvl = level_key(sel)
        dominated = False
        for other in deduped:
            if other is sel or level_key(other) != lvl:
                continue
            ocap = cap_of.get(other)
            if ocap is None:
                continue
            ge_all = all(o >= c for o, c in zip(ocap, cap, strict=False))
            gt_any = any(o > c for o, c in zip(ocap, cap, strict=False))
            if ge_all and gt_any:
                dominated = True
                break
        if not dominated:
            survivors.append(sel)
    return survivors


def selection_label(selection: tuple[str, ...]) -> str:
    """Human/feasible-domain label for a selection (used in FeasibilityResult)."""
    if selection == (UNCONSTRAINED_DOMAIN_ID,):
        return "unconstrained"
    return "+".join(selection)


__all__ = [
    "UNCONSTRAINED_DOMAIN_ID",
    "TopologyIndex",
    "build_topology_index",
    "candidate_selections",
    "prune_dominated_selections",
    "selection_label",
]
