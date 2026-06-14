# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The exact feasibility predicate ``Phi_s(w, action)`` (doc S, Definition D6).

This is the heart of the safety shield: a deterministic, exact yes/no over the
integer feasibility model of doc S (equations E1-E14). It is *never* learned.

Implementation strategy (doc S, "structure & complexity" + "latency budget"):

* **Action / status guards** (E1, E4): whitelist + status preconditions are exact
  singleton screens checked before any placement work.
* **Queue / flavor eligibility** (E5, E7): ResourceFlavor label/taint
  compatibility is an exact boolean mask.
* **Quota / borrowing / lending** (E11-E13): delegated to :mod:`quota`.
* **Topology & masks** (E8, E10): the candidate selection's member nodes are
  taken from :mod:`topology_masks`; protected/operator masks are pre-applied.
* **Placement feasibility** (E6-E9): the gang must be packed all-or-nothing into
  the selection's nodes under per-node multi-resource capacity. Two solvers:

  - **Flow / b-matching fast path** (doc S, Theorem T1). When the workload has a
    single dominant scalar resource (gpu) per replica and homogeneous replicas,
    feasibility reduces to ``sum_n floor(B_n / a) >= demand`` — an exact,
    LP-integral max-flow check done in closed form.
  - **Exact greedy + backtracking** (doc S, general NP-hard case, T2). For
    multi-resource / heterogeneous-podset gangs we run a depth-first exact
    placement search with best-fit ordering and capacity pruning. At demo-pod
    scale (<=256 GPUs, <=32 nodes) this is correct and fast.

Both solvers return the *concrete integer assignment* ``x_pn`` (podset -> {node:
replicas}) so the planner/serving can show the placement and so the shield can do
an exact integer recheck before emission (doc S, Condition C4).
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from dataclasses import field as _field

from ...schema.actions import (
    Action,
    Admit,
    AdmitAndPlace,
    Place,
)
from ...schema.entities import (
    Node,
    PodSet,
    ResourceFlavor,
    TopologyLevel,
    TopologyPolicy,
    Workload,
    WorkloadStatus,
    admission_checks_positive,
)
from ...schema.state import SymbolicState
from ..contracts import Blocker, BlockerGroup, FeasibilityResult
from . import quota as quota_mod
from .health_masks import HealthMasks, build_health_masks
from .topology_masks import (
    UNCONSTRAINED_DOMAIN_ID,
    TopologyIndex,
    build_topology_index,
    candidate_selections,
    prune_dominated_selections,
    selection_label,
)

#: Hard scalar resources in v1 (doc S, Definition D1). Order matters only for
#: deterministic deficit reporting.
RESOURCES: tuple[str, ...] = ("gpu", "cpu", "mem", "pods")
_EPS = 1e-9


@dataclass(slots=True)
class _ResVec:
    """A small multi-resource vector with exact integer GPU/pods semantics."""

    gpu: int = 0
    cpu: float = 0.0
    mem: float = 0.0
    pods: int = 0

    def fits_in(self, cap: _ResVec) -> bool:
        return (
            self.gpu <= cap.gpu
            and self.cpu <= cap.cpu + _EPS
            and self.mem <= cap.mem + _EPS
            and self.pods <= cap.pods
        )

    def subtract(self, other: _ResVec) -> _ResVec:
        return _ResVec(
            gpu=self.gpu - other.gpu,
            cpu=self.cpu - other.cpu,
            mem=self.mem - other.mem,
            pods=self.pods - other.pods,
        )


def _podset_req(p: PodSet) -> _ResVec:
    """Per-replica demand vector ``a_pr`` for a podset (doc S, Definition D2)."""
    return _ResVec(gpu=int(p.gpu_req), cpu=p.cpu_req, mem=p.mem_req_gb, pods=1)


def _node_capacity(node: Node, health: HealthMasks) -> _ResVec:
    """Effective residual capacity ``B_nr`` for a node (doc S, Definition D4).

    GPU capacity is the *healthy* GPU count from the health masks. CPU/mem are
    declared capacity minus current allocation; pods capacity is taken as the
    healthy-GPU count's natural pod ceiling when GPUs are the binding resource,
    otherwise a generous per-node pod budget. The snapshot already subtracts
    admitted usage (Assumption A1/A3).
    """
    gpu = health.gpu_cap(node.node_uid, node.gpu_cap)
    cpu = max(0.0, node.cpu_cap - node.cpu_alloc)
    mem = max(0.0, node.mem_cap_gb - node.mem_alloc_gb)
    # Pods headroom: model as "at least as many pods as healthy GPUs, but never
    # negative". For CPU-only flavors with no GPUs we allow a flat pod budget.
    pods = gpu if gpu > 0 else max(0, int(cpu))
    return _ResVec(gpu=gpu, cpu=cpu, mem=mem, pods=pods)


def flavor_eligible(state: SymbolicState, flavor: ResourceFlavor, node: Node) -> bool:
    """Exact ResourceFlavor compatibility ``eta_pnf`` (doc S, Eq. E7).

    A node is eligible for a flavor iff its ``flavor_id`` matches *or* it
    satisfies every ``node_label_constraints`` label of the flavor. Taints are
    honoured by requiring the flavor's tolerations to cover any node taint label
    of the form ``taint:<name>`` present in ``node.labels`` (a light-weight but
    exact label-driven model faithful to Kueue flavor gating).
    """
    if node.flavor_id == flavor.name:
        label_ok = True
    else:
        label_ok = all(
            node.labels.get(k) == v for k, v in flavor.node_label_constraints.items()
        )
        if not flavor.node_label_constraints and node.flavor_id != flavor.name:
            # No label constraints and a different declared flavor -> not eligible.
            label_ok = False
    if not label_ok:
        return False
    # Taint / toleration check.
    node_taints = [v for k, v in node.labels.items() if k.startswith("taint")]
    return all(t in flavor.tolerations for t in node_taints)


# ──────────────────────────────────────────────────────────────────────────────
# Bounded per-action time / search budget (doc S latency_budget.live_demo_rule)
# ──────────────────────────────────────────────────────────────────────────────
#: Default wall-clock budget for one feasibility evaluation's packing work, in
#: seconds. The doc's live-demo rule caps fixed-candidate checks at "tens of
#: milliseconds" and mandates a conservative hold/abstain on exhaustion; 50 ms is
#: comfortably inside that frontier for the demo caps (<=32 nodes, <=256 GPUs).
DEFAULT_BUDGET_S: float = 0.05

#: Hard cap on DFS node expansions per packing attempt. This is the *boundedness*
#: guarantee independent of the wall clock: the backtracking search can expand at
#: most this many nodes before it conservatively aborts, so the packer always
#: terminates even under adversarial fragmentation (doc S T2 NP-hardness — we
#: never run an unbounded DFS). Sized generously for demo-pod instances.
DEFAULT_MAX_EXPANSIONS: int = 200_000


class BudgetExceededError(Exception):
    """Raised when a packing attempt exhausts its time or expansion budget.

    The feasibility predicate treats this as *unknown*, and — because safety is a
    gate, never a weighted metric — conservatively as **not feasible** for that
    selection (doc S: "a timeout must never turn an unknown candidate into a safe
    candidate; it must instead cause abstention or conservative fallback").
    """


@dataclass(slots=True)
class TimeBudget:
    """A per-action time + search budget for the bounded placement solver.

    Tracks a monotonic wall-clock deadline *and* a hard DFS expansion counter.
    :meth:`tick` is called at every search-node expansion; it raises
    :class:`BudgetExceededError` the moment either bound is crossed. ``exhausted``
    records that an abort happened so the predicate can surface an abstention.
    """

    budget_s: float = DEFAULT_BUDGET_S
    max_expansions: int = DEFAULT_MAX_EXPANSIONS
    _deadline: float = _field(default=0.0, init=False)
    _expansions: int = _field(default=0, init=False)
    exhausted: bool = _field(default=False, init=False)

    def start(self) -> None:
        self._deadline = _time.monotonic() + self.budget_s
        self._expansions = 0

    def tick(self) -> None:
        self._expansions += 1
        # Check the cheap counter first; only sample the clock periodically to
        # keep the hot DFS loop fast (the clock check is the expensive one).
        if self._expansions > self.max_expansions:
            self.exhausted = True
            raise BudgetExceededError("expansion bound exceeded")
        if (self._expansions & 0x3FF) == 0 and _time.monotonic() > self._deadline:
            self.exhausted = True
            raise BudgetExceededError("time budget exceeded")


# ──────────────────────────────────────────────────────────────────────────────
# Placement solvers
# ──────────────────────────────────────────────────────────────────────────────
def _is_single_dominant_gpu(podsets: list[PodSet]) -> bool:
    """True when GPU is the only binding resource and replicas are homogeneous.

    This recognises the flow / b-matching regime of doc S, Theorem T1: a single
    podset whose replicas each demand a positive GPU count and (effectively) no
    CPU/mem pressure relative to nodes. We keep it conservative — if any podset
    requests CPU or memory we fall through to the exact multi-resource solver.
    """
    if len(podsets) != 1:
        return False
    p = podsets[0]
    return p.gpu_req > 0 and p.cpu_req <= _EPS and p.mem_req_gb <= _EPS


def _flow_feasible(
    podsets: list[PodSet],
    nodes: list[tuple[str, _ResVec]],
) -> dict[str, dict[str, int]] | None:
    """Exact closed-form b-matching feasibility (doc S, Theorem T1).

    Each node ``n`` admits ``b_n = floor(B_n.gpu / a.gpu)`` replicas. The gang of
    ``c`` replicas fits iff ``sum_n b_n >= c``. Because this is a capacitated
    bipartite network with a single supply type, max-flow saturates the demand
    iff that inequality holds, and the LP relaxation is integral. We then realise
    a concrete integer assignment greedily (any saturating assignment is valid).
    """
    p = podsets[0]
    a = int(p.gpu_req)
    demand = p.count
    # Sort by capacity descending for a deterministic, compact assignment.
    caps = sorted(
        ((nid, cap.gpu // a) for nid, cap in nodes if cap.gpu >= a),
        key=lambda t: (-t[1], t[0]),
    )
    total = sum(b for _, b in caps)
    if total < demand:
        return None
    assignment: dict[str, int] = {}
    remaining = demand
    for nid, b in caps:
        if remaining <= 0:
            break
        take = min(b, remaining)
        if take > 0:
            assignment[nid] = take
            remaining -= take
    return {p.name: assignment}


def _exact_pack(
    podsets: list[PodSet],
    nodes: list[tuple[str, _ResVec]],
    budget: TimeBudget | None = None,
) -> dict[str, dict[str, int]] | None:
    """Exact bounded backtracking multi-resource gang placement (doc S, T2).

    We flatten the gang into individual replica demands (one item per replica)
    and place them via depth-first search over residual node capacities with:

    * **best-fit ordering** — try the node with the tightest residual that still
      fits (reduces fragmentation, finds a packing fast when one exists);
    * **capacity lower-bound pruning** — abort a branch as soon as remaining
      aggregate demand exceeds remaining aggregate capacity on any resource;
    * **symmetry reduction across equal-residual nodes** — within one item's
      candidate list, nodes with an identical residual vector are interchangeable,
      so only the first representative is tried (this collapses the ``N!``-style
      permutation blow-up that pure DFS suffers on homogeneous pods);
    * **a hard budget** — every expansion calls :meth:`TimeBudget.tick`, which
      raises :class:`BudgetExceededError` once the wall-clock or expansion bound is
      crossed. Boundedness is therefore guaranteed independent of input: the DFS
      expands at most ``budget.max_expansions`` nodes and then aborts.

    **Exactness.** Within the budget the search is complete: it enumerates, up to
    the equal-residual symmetry quotient, every distinct way to seat the items, so
    a returned ``None`` (without a raised :class:`BudgetExceededError`) means *no*
    integer packing exists — an exact infeasibility witness. The symmetry quotient
    is sound because two nodes with identical residual vectors are mutually
    substitutable: any packing using one has a mirror using the other.

    Raises :class:`BudgetExceededError` if the budget is exhausted before a definitive
    answer; the caller converts that into a conservative (not-feasible) result.
    """
    # Build the item list: (podset_name, demand vector). Larger items first so
    # the hardest-to-place replicas are seated early (first-fit-decreasing seed).
    items: list[tuple[str, _ResVec]] = []
    for p in podsets:
        req = _podset_req(p)
        items.extend((p.name, req) for _ in range(p.count))
    items.sort(key=lambda it: (-it[1].gpu, -it[1].cpu, -it[1].mem), reverse=False)
    n_items = len(items)

    residual: dict[str, _ResVec] = {nid: cap for nid, cap in nodes}
    node_order = [nid for nid, _ in nodes]
    assignment: dict[str, dict[str, int]] = {}

    # Suffix sums of remaining demand: ``suffix[i]`` is the aggregate demand of
    # items[i:], precomputed so the lower-bound prune is O(1) per node.
    suffix: list[_ResVec] = [_ResVec()] * (n_items + 1)
    acc = _ResVec()
    for i in range(n_items - 1, -1, -1):
        req = items[i][1]
        acc = _ResVec(
            gpu=acc.gpu + req.gpu,
            cpu=acc.cpu + req.cpu,
            mem=acc.mem + req.mem,
            pods=acc.pods + req.pods,
        )
        suffix[i] = acc

    def remaining_capacity() -> _ResVec:
        agg = _ResVec()
        for cap in residual.values():
            agg = _ResVec(
                gpu=agg.gpu + cap.gpu,
                cpu=agg.cpu + cap.cpu,
                mem=agg.mem + cap.mem,
                pods=agg.pods + cap.pods,
            )
        return agg

    def candidate_nodes(req: _ResVec) -> list[str]:
        """Best-fit, symmetry-reduced candidate nodes for one item."""
        cands = [nid for nid in node_order if req.fits_in(residual[nid])]
        cands.sort(
            key=lambda nid: (
                residual[nid].gpu,
                residual[nid].cpu,
                residual[nid].mem,
                nid,
            )
        )
        out: list[str] = []
        seen: set[tuple[int, float, float, int]] = set()
        for nid in cands:
            r = residual[nid]
            sig = (r.gpu, r.cpu, r.mem, r.pods)
            # Symmetry reduction: nodes with identical residual are substitutable.
            if sig in seen:
                continue
            seen.add(sig)
            out.append(nid)
        return out

    def seat(nid: str, name: str, req: _ResVec) -> None:
        residual[nid] = residual[nid].subtract(req)
        bucket = assignment.setdefault(name, {})
        bucket[nid] = bucket.get(nid, 0) + 1

    def unseat(nid: str, name: str, req: _ResVec) -> None:
        bucket = assignment[name]
        bucket[nid] -= 1
        if bucket[nid] == 0:
            del bucket[nid]
        residual[nid] = _ResVec(
            gpu=residual[nid].gpu + req.gpu,
            cpu=residual[nid].cpu + req.cpu,
            mem=residual[nid].mem + req.mem,
            pods=residual[nid].pods + req.pods,
        )

    # Explicit-stack iterative DFS (no Python recursion limit, so the packer
    # scales to large homogeneous gangs). Each stack frame is the mutable
    # ``[item_index, candidate_list, cursor]``; advancing the cursor tries the
    # next node, exhausting it pops (backtrack). This is the exact same search
    # tree as the recursive form, just trampolined.
    found = False
    if n_items == 0:
        found = True
    else:
        # Initial prune before descending.
        if suffix[0].fits_in(remaining_capacity()):
            _, req0 = items[0]
            stack: list[list] = [[0, candidate_nodes(req0), 0]]
            while stack:
                if budget is not None:
                    budget.tick()
                frame = stack[-1]
                i, cands, cursor = frame
                name, req = items[i]
                if cursor >= len(cands):
                    # Exhausted this item's choices -> backtrack one level.
                    stack.pop()
                    if stack:
                        pf = stack[-1]
                        pname, preq = items[pf[0]]
                        unseat(pf[1][pf[2]], pname, preq)
                        pf[2] += 1
                    continue
                nid = cands[cursor]
                seat(nid, name, req)
                nxt = i + 1
                if nxt >= n_items:
                    found = True
                    break
                # Lower-bound prune at the child before pushing its frame.
                if suffix[nxt].fits_in(remaining_capacity()):
                    _, nreq = items[nxt]
                    stack.append([nxt, candidate_nodes(nreq), 0])
                else:
                    # Child infeasible: undo this seat and advance the cursor.
                    unseat(nid, name, req)
                    frame[2] += 1

    if found:
        # Drop any empty buckets for a clean assignment.
        return {k: v for k, v in assignment.items() if v}
    return None


def place_gang(
    podsets: list[PodSet],
    nodes: list[tuple[str, _ResVec]],
    budget: TimeBudget | None = None,
) -> dict[str, dict[str, int]] | None:
    """Dispatch to the flow fast path or the exact bounded backtracking solver.

    Returns the concrete integer assignment ``x_pn`` if the gang fits all-or-
    nothing in ``nodes``, else ``None`` (an exact infeasibility witness). May
    raise :class:`BudgetExceededError` when ``budget`` is supplied and exhausted.
    """
    if _is_single_dominant_gpu(podsets):
        # The flow regime is closed-form (T1): no search, so no budget needed.
        return _flow_feasible(podsets, nodes)
    return _exact_pack(podsets, nodes, budget)


def _place_in_selection(
    podsets: list[PodSet],
    nodes: list[tuple[str, _ResVec]],
    budget: TimeBudget | None,
) -> dict[str, dict[str, int]] | None:
    """Budget-guarded :func:`place_gang` used by the feasibility loop.

    Swallows :class:`BudgetExceededError` into a conservative ``None`` (this selection
    is treated as *not* known-feasible). The ``budget.exhausted`` flag stays set
    so the predicate can attach an abstention note to the result.
    """
    try:
        return place_gang(podsets, nodes, budget)
    except BudgetExceededError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Placement-constraint helpers (E14 podset-groups, levels, blocker attribution)
# ──────────────────────────────────────────────────────────────────────────────
def _has(blockers: list[Blocker], group: BlockerGroup) -> bool:
    """True iff the ledger already carries a blocker in ``group``."""
    return any(b.group is group for b in blockers)


def _has_podset_group(podsets: list[PodSet]) -> bool:
    """True iff any podset declares a Kueue ``podset_group`` (shared-domain unit)."""
    return any(p.podset_group for p in podsets)


def _requested_level(
    idx: TopologyIndex, requested: tuple[str, ...]
) -> TopologyLevel | None:
    """Infer the topology level pinned by an explicit ``requested`` domain set.

    When the operator pins specific domains, the locality level is the common
    level of those domains (e.g. all rack-domains -> rack locality). Mixed or
    unknown levels return ``None`` (no level restriction). With no request the
    predicate enumerates every level (doc S, Definition D3 multi-layer handling).
    """
    if not requested:
        return None
    levels = {idx.domain_level.get(d) for d in requested}
    levels.discard(None)
    if len(levels) == 1:
        return next(iter(levels))
    return None


def _groups_within_single_selection(
    assignment: dict[str, dict[str, int]],
    podsets: list[PodSet],
    idx: TopologyIndex,
) -> bool:
    """E14: every podset-group must be packed inside one shared domain (z_sigma).

    Kueue's ``podset-group-name`` forces all PodSets in a group into a single
    common topology-fitting unit (doc S, Eq. E14: a shared ``z_sigma`` across the
    group, or group-specific ``z_{sigma g}`` with equality across PodSets). We
    enforce it post-pack: for each declared group, gather the nodes its podsets
    landed on and require that some single domain contains *all* of them. A
    grouped assignment that straddles two domains is rejected (returns ``False``).

    Ungrouped podsets and the degenerate single-domain case pass trivially.
    """
    group_nodes: dict[str, set[str]] = {}
    for p in podsets:
        if not p.podset_group:
            continue
        nodes = assignment.get(p.name, {})
        group_nodes.setdefault(p.podset_group, set()).update(nodes.keys())

    for nodes in group_nodes.values():
        if len(nodes) <= 1:
            continue  # a single node is trivially within one domain
        # The group's nodes share a domain iff the intersection of their
        # per-node domain memberships is non-empty (some common z_sigma).
        common: set[str] | None = None
        for nid in nodes:
            doms = idx.node_to_domains.get(nid, set())
            common = set(doms) if common is None else (common & doms)
            if not common:
                return False
    return True


def _attribute_placement_failure(
    *,
    gang_demand_gpu: int,
    saw_nonempty_selection: bool,
    saw_any_usable_node: bool,
    aggregate_gpu_any_single: int,
    total_usable_gpu: int,
    grouped: bool,
) -> Blocker:
    """Attribute a placement failure to its precise blocker family (doc S, D15).

    Splits the former single ``NODE_CAPACITIES`` catch-all into the three
    families the doc's blocker taxonomy distinguishes:

    * **HEALTH_SCHEDULABILITY** — selections existed with member nodes, but every
      such node was filtered out by health/schedulability/flavor masks, so no
      bookable capacity remains at all (the obstruction is health, not size).
    * **TOPOLOGY_RULE** — bookable nodes exist and the *cluster-wide aggregate*
      GPU capacity (``total_usable_gpu``) meets demand, but no *single* domain
      selection (max ``aggregate_gpu_any_single``) can hold the gang (or the
      grouped podsets). Under REQUIRED locality this is the same-domain rule
      biting; for a grouped pod it is the shared-domain (E14) rule.
    * **NODE_CAPACITIES** — even the whole cluster's usable GPUs fall short of the
      gang demand: a genuine capacity shortfall, reported with the exact deficit.
    """
    if saw_nonempty_selection and not saw_any_usable_node:
        return Blocker(
            BlockerGroup.HEALTH_SCHEDULABILITY,
            "all member nodes of every candidate domain are unschedulable, "
            "unhealthy, or flavor-ineligible",
        )
    if (
        total_usable_gpu >= gang_demand_gpu
        and aggregate_gpu_any_single < gang_demand_gpu
    ):
        # Aggregate capacity exists across domains, but no single domain (or the
        # grouped podsets' shared domain) can hold the gang all-or-nothing.
        reason = (
            "grouped podsets cannot share a single domain (E14 shared z_sigma)"
            if grouped
            else "no single topology domain hosts the required gang "
            "(aggregate capacity exists but spans domains)"
        )
        return Blocker(BlockerGroup.TOPOLOGY_RULE, reason)
    deficit = float(gang_demand_gpu - total_usable_gpu)
    return Blocker(
        BlockerGroup.NODE_CAPACITIES,
        f"insufficient healthy GPU capacity for the gang: need {deficit:g} more",
        deficit={"gpu": deficit},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Feasibility predicate
# ──────────────────────────────────────────────────────────────────────────────
def _action_params(
    action: Action,
) -> tuple[bool, bool, str | None, str | None, tuple[str, ...], str]:
    """Decode (does_admit, does_place, queue, flavor, requested_domains, policy)."""
    if isinstance(action, Admit):
        return (
            True,
            False,
            action.cluster_queue_id,
            action.resource_flavor_id,
            (),
            "preferred",
        )
    if isinstance(action, Place):
        return False, True, None, None, action.domain_id_or_set, action.domain_policy
    if isinstance(action, AdmitAndPlace):
        return (
            True,
            True,
            action.cluster_queue_id,
            action.resource_flavor_id,
            action.domain_id_or_set,
            action.domain_policy,
        )
    # Hold / MaskDomain reach feasibility only via guards; treat as no-op here.
    return False, False, None, None, (), "preferred"


def _policy_enum(domain_policy: str) -> TopologyPolicy:
    try:
        return TopologyPolicy(domain_policy)
    except ValueError:
        return TopologyPolicy.PREFERRED


def evaluate(
    state: SymbolicState,
    workload: Workload,
    action: Action,
    *,
    topology_index: TopologyIndex | None = None,
    health: HealthMasks | None = None,
    budget: TimeBudget | None = None,
    relax: frozenset[BlockerGroup] | None = None,
) -> FeasibilityResult:
    """Exact ``Phi_s(w, action)`` returning blockers + feasible domains + assignment.

    The caller may pass precomputed ``topology_index`` / ``health`` masks to amortise
    them across a :meth:`safe_candidates` sweep (doc S latency budget).

    ``budget`` bounds the per-action placement search (doc S
    ``latency_budget.live_demo_rule``): on exhaustion the predicate
    conservatively treats the affected selection as *not* feasible and sets
    :attr:`FeasibilityResult.budget_exhausted`, so a timeout can never turn an
    unknown candidate into a safe one. A fresh default budget is created when none
    is supplied.

    ``relax`` is the set of semantic :class:`BlockerGroup`\\ s whose constraints
    are *dropped* for this evaluation (doc S Algorithm A1 phase 3 — real
    deletion-filtering re-solves the predicate with one group relaxed at a time).
    Relaxing a group both suppresses its blockers *and* applies the matching
    constraint relaxation (e.g. masked domains become eligible when
    ``PROTECTED_DOMAIN_MASK`` is relaxed, node capacity becomes unbounded when
    ``NODE_CAPACITIES`` is relaxed), so the returned ``feasible`` flag is the true
    feasibility of the relaxed submodel — not merely a filtered ledger.
    """
    relax = relax or frozenset()
    idx = topology_index or build_topology_index(state)
    hmask = health or build_health_masks(state)
    if budget is None:
        budget = TimeBudget()
    budget.start()
    blockers: list[Blocker] = []

    does_admit, does_place, queue_id, flavor_id, requested, policy_str = _action_params(
        action
    )

    # ── E4: action whitelist + status preconditions ────────────────────────
    allowed = state.operator_policy.allowed_actions
    if action.type.value not in allowed and BlockerGroup.ACTION_WHITELIST not in relax:
        blockers.append(
            Blocker(
                BlockerGroup.ACTION_WHITELIST,
                f"action '{action.type.value}' not in operator whitelist",
            )
        )
    # Placement is only meaningful for non-running, non-finished workloads; a
    # pure place query on an already-running gang is a status violation (an
    # admit_and_place handles its own status precondition via the admit branch).
    if BlockerGroup.STATUS_PRECONDITION not in relax:
        if (
            does_place
            and not does_admit
            and workload.status
            not in (
                WorkloadStatus.ADMITTED,
                WorkloadStatus.PENDING,
                WorkloadStatus.RESERVING,
            )
        ):
            blockers.append(
                Blocker(
                    BlockerGroup.STATUS_PRECONDITION,
                    f"cannot place workload in status '{workload.status.value}'",
                )
            )
        if does_admit and workload.status in (
            WorkloadStatus.ADMITTED,
            WorkloadStatus.RUNNING,
            WorkloadStatus.FINISHED,
        ):
            blockers.append(
                Blocker(
                    BlockerGroup.STATUS_PRECONDITION,
                    f"workload already in status '{workload.status.value}'",
                )
            )

    # ── Queue / flavor eligibility (E5/E7) ─────────────────────────────────
    flavor: ResourceFlavor | None = None
    relax_qf = BlockerGroup.QUEUE_FLAVOR_ELIGIBILITY in relax
    if does_admit:
        if not relax_qf and (queue_id is None or queue_id not in state.cluster_queues):
            blockers.append(
                Blocker(
                    BlockerGroup.QUEUE_FLAVOR_ELIGIBILITY,
                    f"cluster queue '{queue_id}' not found",
                )
            )
        if flavor_id is None or flavor_id not in state.flavors:
            if not relax_qf:
                blockers.append(
                    Blocker(
                        BlockerGroup.QUEUE_FLAVOR_ELIGIBILITY,
                        f"resource flavor '{flavor_id}' not found",
                    )
                )
        else:
            flavor = state.flavors[flavor_id]

        # ── E5: AdmissionCheck gate (y_qf <= chi_qf(s)) ────────────────────
        # The conjunction of all required AdmissionChecks for the (queue,
        # flavor) must be positive before admission. This is the only why_not
        # GROUP_ORDER group with no placement producer: an exact singleton screen
        # (doc S, Eq. E5 / Algorithm A1 phase 1; Kueue admits only once every
        # referenced check is READY).
        if (
            BlockerGroup.ADMISSION_CHECKS not in relax
            and queue_id in state.cluster_queues
            and flavor is not None
            and not admission_checks_positive(
                state.cluster_queues[queue_id], flavor.name
            )
        ):
            blockers.append(
                Blocker(
                    BlockerGroup.ADMISSION_CHECKS,
                    f"admission checks not positive for queue '{queue_id}' "
                    f"flavor '{flavor.name}' (chi_qf = 0)",
                )
            )

    podsets = [p for p in workload.podsets]
    demand = quota_mod.workload_demand(
        [
            (
                p.count,
                {
                    "gpu": p.gpu_req,
                    "cpu": p.cpu_req,
                    "mem": p.mem_req_gb,
                    "pods": 1.0,
                },
            )
            for p in podsets
        ]
    )

    # ── Quota / borrowing / lending (E11-E13) ──────────────────────────────
    if does_admit and queue_id in state.cluster_queues and flavor is not None:
        cq = state.cluster_queues[queue_id]
        cohort = quota_mod.resolve_cohort(state.cluster_queues, cq)
        qd = quota_mod.quota_check(cq, cohort, flavor.name, demand)
        if not qd.feasible:
            # A deficit means demand exceeded nominal + borrowable quota. If the
            # queue can borrow this resource at all, the binding cap is the
            # borrowing/lending limit, so attribute the blocker there; otherwise
            # it is a pure nominal-quota shortfall (doc S, Eq. E11 vs E12/E13).
            for resource, short in qd.deficit.items():
                can_borrow = (
                    cq.borrowing_limit.get(flavor.name, {}).get(resource, 0.0) > 0.0
                )
                grp = (
                    BlockerGroup.BORROWING_LENDING if can_borrow else BlockerGroup.QUOTA
                )
                if grp in relax:
                    continue
                blockers.append(
                    Blocker(
                        grp,
                        f"quota exceeded for {resource}: need {short:g} more",
                        deficit={resource: short},
                    )
                )

    # ── Placement: topology, masks, health, capacities, gang atomicity ─────
    feasible_domains: list[str] = []
    assignment: dict[str, dict[str, int]] = {}

    if does_place:
        # E7 on *any* placement: a pure ``place`` query has no (q, f) of its own,
        # so the flavor is the one the workload was already admitted under
        # (``admitted_flavor_id``). Resolving it here applies flavor eligibility
        # to candidate nodes for placement too, not only for admit_and_place.
        if flavor is None and workload.admitted_flavor_id is not None:
            flavor = state.flavors.get(workload.admitted_flavor_id)

        # Placement-group relaxations for deletion-filtering (doc S A1 phase 3):
        # each drops a real constraint, not merely the blocker row.
        relax_mask = BlockerGroup.PROTECTED_DOMAIN_MASK in relax
        relax_health = BlockerGroup.HEALTH_SCHEDULABILITY in relax
        relax_cap = BlockerGroup.NODE_CAPACITIES in relax
        relax_topo = BlockerGroup.TOPOLOGY_RULE in relax
        relax_gang = BlockerGroup.GANG_ATOMICITY in relax

        masked = (
            set()
            if relax_mask
            else set(getattr(state.operator_policy, "protected_domains", []))
        )
        protected = (
            set()
            if relax_mask
            else set(state.operator_policy.protected_domains)
        )
        req_level = _requested_level(idx, requested)
        # Context-level masked domains (operator MaskDomain guardrail) live on
        # the operator policy's protected list in v1; both exclude a domain.
        # Under REQUIRED locality we enumerate per topology level (rack-domains for
        # rack locality, block-domains for block locality, ...) so a same-rack and
        # a same-block check are both offered; the predicate keeps the tightest
        # that fits (doc S, Definition D3 multi-layer slice handling).
        eff_policy = _policy_enum(policy_str)
        selections = candidate_selections(
            state,
            idx,
            eff_policy,
            masked_domains=masked,
            protected_domains=protected,
            requested=() if relax_mask else requested,
            level=req_level,
        )
        # Relaxing the topology rule means dropping same-domain locality: add the
        # whole-pod (unconstrained) selection so the gang may span domains.
        if relax_topo:
            uncon = (UNCONSTRAINED_DOMAIN_ID,)
            if uncon not in selections:
                selections = [*selections, uncon]
        if requested and not selections and not relax_mask:
            blockers.append(
                Blocker(
                    BlockerGroup.PROTECTED_DOMAIN_MASK,
                    f"requested domain(s) {requested} are masked/protected",
                )
            )
        # Nodes that belong to *any* masked/protected domain are forbidden in
        # every selection — including the unconstrained fill, which would
        # otherwise re-admit a protected rack's nodes (doc S, Eq. E10).
        forbidden_nodes: set[str] = set()
        for dom in masked | protected:
            forbidden_nodes.update(idx.domain_nodes.get(dom, ()))
        gang_demand_gpu = sum(p.count * int(p.gpu_req) for p in podsets)

        # Per-selection usable-node analysis. We separately track *why* a
        # selection fails so the blocker is attributed to the right family
        # (doc S, Definition D15) instead of a single NODE_CAPACITIES catch-all.
        usable_by_sel: dict[tuple[str, ...], list[tuple[str, _ResVec]]] = {}
        cap_by_sel: dict[tuple[str, ...], tuple[int, float, float, int]] = {}
        # Diagnostics across the whole candidate set:
        saw_nonempty_selection = False  # at least one selection had member nodes
        saw_any_usable_node = False  # at least one node survived health+flavor
        aggregate_gpu_any_single = 0  # max single-selection healthy GPU total
        # Distinct usable node -> healthy GPU cap, so the cluster-wide aggregate
        # (the doc's "aggregate capacity exists") is the sum over distinct nodes,
        # never double-counted across overlapping selections.
        distinct_usable_gpu: dict[str, int] = {}
        for sel in selections:
            node_uids = idx.nodes_in_selection(sel)
            if node_uids:
                saw_nonempty_selection = True
            usable: list[tuple[str, _ResVec]] = []
            agg = _ResVec()
            for nid in node_uids:
                if nid in forbidden_nodes:
                    continue
                node = state.nodes.get(nid)
                if node is None:
                    continue
                # Relaxing health/schedulability treats every present node as
                # schedulable (drops the TAS ready+schedulable+device-health mask).
                if not relax_health and not hmask.node_schedulable.get(nid, False):
                    continue
                if flavor is not None and not flavor_eligible(state, flavor, node):
                    continue
                cap = _node_capacity(node, hmask)
                # Relaxing node capacities makes each node an unbounded bin (the
                # capacity rows E9 are dropped), so the gang trivially packs.
                if relax_cap:
                    cap = _ResVec(
                        gpu=gang_demand_gpu + cap.gpu,
                        cpu=cap.cpu + 1e18,
                        mem=cap.mem + 1e18,
                        pods=gang_demand_gpu + cap.pods + 1,
                    )
                agg = _ResVec(
                    gpu=agg.gpu + cap.gpu,
                    cpu=agg.cpu + cap.cpu,
                    mem=agg.mem + cap.mem,
                    pods=agg.pods + cap.pods,
                )
                distinct_usable_gpu[nid] = cap.gpu
                usable.append((nid, cap))
            if usable:
                saw_any_usable_node = True
            aggregate_gpu_any_single = max(aggregate_gpu_any_single, agg.gpu)
            usable_by_sel[sel] = usable
            cap_by_sel[sel] = (agg.gpu, agg.cpu, agg.mem, agg.pods)
        total_usable_gpu = sum(distinct_usable_gpu.values())

        # Dominance + symmetry reduction over the surviving selections before any
        # packing work (doc S latency_budget.pruning): identical-node selections
        # collapse and same-level dominated selections are dropped.
        pruned = prune_dominated_selections(idx, list(usable_by_sel), cap_by_sel)

        any_fit = False
        best_assignment: dict[str, dict[str, int]] | None = None
        for sel in pruned:
            usable = usable_by_sel[sel]
            agg_gpu = cap_by_sel[sel][0]
            # Resource lower-bound screen (doc S pruning): skip pack work when the
            # selection cannot possibly hold the gang's GPUs.
            if agg_gpu < gang_demand_gpu:
                continue
            packed = _place_in_selection(podsets, usable, budget)
            # Relaxing GANG_ATOMICITY drops the all-or-nothing E14/E6 rule: a
            # group spanning domains is then admissible too.
            group_ok = relax_gang or (
                packed is not None
                and _groups_within_single_selection(packed, podsets, idx)
            )
            if packed is not None and group_ok:
                any_fit = True
                feasible_domains.append(selection_label(sel))
                if best_assignment is None:
                    best_assignment = packed
        if best_assignment is not None:
            assignment = best_assignment
        if (
            not any_fit
            and not feasible_domains
            and not _has(blockers, BlockerGroup.PROTECTED_DOMAIN_MASK)
        ):
            if budget.exhausted:
                # The search frontier blew past the per-action budget before a
                # definitive yes/no. Safety is a gate: we conservatively report
                # infeasible-by-abstention (GANG_ATOMICITY family) rather than
                # guess feasible (doc S latency_budget.live_demo_rule).
                if BlockerGroup.GANG_ATOMICITY not in relax:
                    blockers.append(
                        Blocker(
                            BlockerGroup.GANG_ATOMICITY,
                            "placement search exhausted its per-action time "
                            "budget; conservatively holding (abstain) rather than "
                            "guessing a feasible gang assignment",
                        )
                    )
            else:
                fail = _attribute_placement_failure(
                    gang_demand_gpu=gang_demand_gpu,
                    saw_nonempty_selection=saw_nonempty_selection,
                    saw_any_usable_node=saw_any_usable_node,
                    aggregate_gpu_any_single=aggregate_gpu_any_single,
                    total_usable_gpu=total_usable_gpu,
                    grouped=_has_podset_group(podsets),
                )
                # Deletion-filtering: if the attributed failure group is the one
                # being relaxed, the obstruction is gone — record a synthetic
                # feasible domain so the relaxed submodel reads as placeable.
                if fail.group in relax:
                    feasible_domains.append("__relaxed__")
                else:
                    blockers.append(fail)

    feasible = len(blockers) == 0 and (not does_place or len(feasible_domains) > 0)
    # Hold and admit-only actions are feasible once their guards pass.
    if not does_place and not does_admit:
        feasible = len(blockers) == 0  # hold

    return FeasibilityResult(
        feasible=feasible,
        blockers=blockers,
        feasible_domains=feasible_domains,
        assignment=assignment,
        budget_exhausted=budget.exhausted,
    )


__all__ = [
    "DEFAULT_BUDGET_S",
    "DEFAULT_MAX_EXPANSIONS",
    "RESOURCES",
    "BudgetExceededError",
    "TimeBudget",
    "evaluate",
    "flavor_eligible",
    "place_gang",
]
