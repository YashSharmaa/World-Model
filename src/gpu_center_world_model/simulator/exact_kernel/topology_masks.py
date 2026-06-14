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

from ...schema.entities import TopologyPolicy
from ...schema.state import SymbolicState

#: Synthetic domain id used for the "unconstrained / whole-pod" selection. Kept
#: out of the real domain namespace by an unambiguous prefix.
UNCONSTRAINED_DOMAIN_ID = "__unconstrained__"


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

    # Ancestor closure for every explicit domain.
    for dom_id in state.domains:
        chain: list[str] = []
        cur: str | None = dom_id
        guard = 0
        while cur is not None and guard < 64:
            chain.append(cur)
            parent = state.domains.get(cur)
            cur = parent.parent_domain_id if parent is not None else None
            guard += 1
        idx.ancestors[dom_id] = chain

    # Attach nodes to domains.
    for node_uid, node in state.nodes.items():
        doms: set[str] = set()
        # Synthetic per-rack domain (always present).
        rack_dom = _synthetic_rack_domain(node.rack_id)
        doms.add(rack_dom)
        idx.domain_nodes.setdefault(rack_dom, []).append(node_uid)
        idx.ancestors.setdefault(rack_dom, [rack_dom])
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
) -> list[tuple[str, ...]]:
    """Enumerate admissible topology selections ``Sigma_w`` for one workload.

    Operator/protected masks (``pi_sigma(s)`` in doc S, Eq. E10) are applied here
    so masked domains never enter the candidate set in the first place. If the
    caller pinned an explicit ``requested`` domain set (the ``Place`` action's
    ``domain_id_or_set``), only that selection is returned (still mask-checked).

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

    # All single explicit/synthetic domains, sorted for determinism.
    single_domains = sorted(d for d in index.domain_nodes if index.domain_nodes[d])
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
    "selection_label",
]
