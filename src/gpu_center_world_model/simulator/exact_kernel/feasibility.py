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

from dataclasses import dataclass

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
    TopologyPolicy,
    Workload,
    WorkloadStatus,
)
from ...schema.state import SymbolicState
from ..contracts import Blocker, BlockerGroup, FeasibilityResult
from . import quota as quota_mod
from .health_masks import HealthMasks, build_health_masks
from .topology_masks import (
    TopologyIndex,
    build_topology_index,
    candidate_selections,
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
) -> dict[str, dict[str, int]] | None:
    """Exact greedy + backtracking multi-resource gang placement (doc S, T2).

    We flatten the gang into individual replica demands (one item per replica)
    and place them via depth-first search over residual node capacities with:

    * **best-fit ordering** — try the node with the tightest residual that still
      fits (reduces fragmentation, finds a packing fast when one exists);
    * **capacity lower-bound pruning** — abort a branch as soon as remaining
      aggregate demand exceeds remaining aggregate capacity on any resource.

    Correct and complete at demo-pod scale: it returns ``None`` only when *no*
    integer packing exists, so a ``None`` is an exact infeasibility witness.
    """
    # Build the item list: (podset_name, demand vector). Larger items first so
    # the hardest-to-place replicas are seated early (first-fit-decreasing seed).
    items: list[tuple[str, _ResVec]] = []
    for p in podsets:
        req = _podset_req(p)
        items.extend((p.name, req) for _ in range(p.count))
    items.sort(key=lambda it: (-it[1].gpu, -it[1].cpu, -it[1].mem), reverse=False)

    residual: dict[str, _ResVec] = {nid: cap for nid, cap in nodes}
    node_order = [nid for nid, _ in nodes]
    assignment: dict[str, dict[str, int]] = {}

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

    def remaining_demand(i: int) -> _ResVec:
        agg = _ResVec()
        for _, req in items[i:]:
            agg = _ResVec(
                gpu=agg.gpu + req.gpu,
                cpu=agg.cpu + req.cpu,
                mem=agg.mem + req.mem,
                pods=agg.pods + req.pods,
            )
        return agg

    def place(i: int) -> bool:
        if i >= len(items):
            return True
        # Lower-bound prune: total remaining demand must fit in total capacity.
        dem = remaining_demand(i)
        cap = remaining_capacity()
        if not dem.fits_in(cap):
            return False
        name, req = items[i]
        # Best-fit candidate ordering: smallest residual that still fits.
        candidates = [nid for nid in node_order if req.fits_in(residual[nid])]
        candidates.sort(
            key=lambda nid: (
                residual[nid].gpu,
                residual[nid].cpu,
                residual[nid].mem,
                nid,
            )
        )
        for nid in candidates:
            residual[nid] = residual[nid].subtract(req)
            bucket = assignment.setdefault(name, {})
            bucket[nid] = bucket.get(nid, 0) + 1
            if place(i + 1):
                return True
            # backtrack
            bucket[nid] -= 1
            if bucket[nid] == 0:
                del bucket[nid]
            residual[nid] = _ResVec(
                gpu=residual[nid].gpu + req.gpu,
                cpu=residual[nid].cpu + req.cpu,
                mem=residual[nid].mem + req.mem,
                pods=residual[nid].pods + req.pods,
            )
        return False

    if place(0):
        # Drop any empty buckets for a clean assignment.
        return {k: v for k, v in assignment.items() if v}
    return None


def place_gang(
    podsets: list[PodSet],
    nodes: list[tuple[str, _ResVec]],
) -> dict[str, dict[str, int]] | None:
    """Dispatch to the flow fast path or the exact backtracking solver.

    Returns the concrete integer assignment ``x_pn`` if the gang fits all-or-
    nothing in ``nodes``, else ``None`` (an exact infeasibility witness).
    """
    if _is_single_dominant_gpu(podsets):
        return _flow_feasible(podsets, nodes)
    return _exact_pack(podsets, nodes)


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
) -> FeasibilityResult:
    """Exact ``Phi_s(w, action)`` returning blockers + feasible domains + assignment.

    The caller may pass precomputed ``topology_index`` / ``health`` masks to amortise
    them across a :meth:`safe_candidates` sweep (doc S latency budget).
    """
    idx = topology_index or build_topology_index(state)
    hmask = health or build_health_masks(state)
    blockers: list[Blocker] = []

    does_admit, does_place, queue_id, flavor_id, requested, policy_str = _action_params(
        action
    )

    # ── E4: action whitelist + status preconditions ────────────────────────
    allowed = state.operator_policy.allowed_actions
    if action.type.value not in allowed:
        blockers.append(
            Blocker(
                BlockerGroup.ACTION_WHITELIST,
                f"action '{action.type.value}' not in operator whitelist",
            )
        )
    # Placement is only meaningful for non-running, non-finished workloads; a
    # pure place query on an already-running gang is a status violation (an
    # admit_and_place handles its own status precondition via the admit branch).
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
    if does_admit:
        if queue_id is None or queue_id not in state.cluster_queues:
            blockers.append(
                Blocker(
                    BlockerGroup.QUEUE_FLAVOR_ELIGIBILITY,
                    f"cluster queue '{queue_id}' not found",
                )
            )
        if flavor_id is None or flavor_id not in state.flavors:
            blockers.append(
                Blocker(
                    BlockerGroup.QUEUE_FLAVOR_ELIGIBILITY,
                    f"resource flavor '{flavor_id}' not found",
                )
            )
        else:
            flavor = state.flavors[flavor_id]

    podsets = [p for p in workload.podsets]
    demand = quota_mod.workload_demand(
        [
            (p.count, {"gpu": p.gpu_req, "cpu": p.cpu_req, "mem": p.mem_req_gb})
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
        masked = set(getattr(state.operator_policy, "protected_domains", []))
        protected = set(state.operator_policy.protected_domains)
        # Context-level masked domains (operator MaskDomain guardrail) live on
        # the operator policy's protected list in v1; both exclude a domain.
        selections = candidate_selections(
            state,
            idx,
            _policy_enum(policy_str),
            masked_domains=masked,
            protected_domains=protected,
            requested=requested,
        )
        if requested and not selections:
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
        any_fit = False
        best_assignment: dict[str, dict[str, int]] | None = None
        for sel in selections:
            node_uids = idx.nodes_in_selection(sel)
            usable: list[tuple[str, _ResVec]] = []
            healthy_gpu_total = 0
            for nid in node_uids:
                if nid in forbidden_nodes:
                    continue
                node = state.nodes.get(nid)
                if node is None or not hmask.node_schedulable.get(nid, False):
                    continue
                if flavor is not None and not flavor_eligible(state, flavor, node):
                    continue
                cap = _node_capacity(node, hmask)
                healthy_gpu_total += cap.gpu
                usable.append((nid, cap))
            # Resource lower-bound screen (doc S pruning): skip MILP/pack work
            # when the selection cannot possibly hold the gang's GPUs.
            if healthy_gpu_total < gang_demand_gpu:
                continue
            packed = place_gang(podsets, usable)
            if packed is not None:
                any_fit = True
                feasible_domains.append(selection_label(sel))
                if best_assignment is None:
                    best_assignment = packed
        if best_assignment is not None:
            assignment = best_assignment
        if not any_fit and not feasible_domains:
            # Distinguish health/schedulability, topology, and capacity blockers.
            blockers.append(
                Blocker(
                    BlockerGroup.NODE_CAPACITIES,
                    "no topology selection can hold the gang all-or-nothing",
                    deficit={"gpu": float(gang_demand_gpu)},
                )
            )

    feasible = len(blockers) == 0 and (not does_place or len(feasible_domains) > 0)
    # Hold and admit-only actions are feasible once their guards pass.
    if not does_place and not does_admit:
        feasible = len(blockers) == 0  # hold

    return FeasibilityResult(
        feasible=feasible,
        blockers=blockers,
        feasible_domains=feasible_domains,
        assignment=assignment,
    )


__all__ = [
    "RESOURCES",
    "evaluate",
    "flavor_eligible",
    "place_gang",
]
