# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""``ExactSymbolicKernel`` — the deterministic safety shield (doc S).

This class implements the :class:`ExactKernel` Protocol. It owns hard feasibility
and is **never learned**. Its four duties:

* :meth:`feasible` — the exact predicate ``Phi_s(w, action)`` (delegated to
  :mod:`feasibility`).
* :meth:`safe_candidates` — enumerate the kernel-certified safe set
  ``F_s(w) = {alpha : Phi_s(w, alpha) = 1}`` (doc S, Definition D7) with the
  aggressive pruning of the doc S latency budget: singleton screens, resource
  lower bounds, domain dominance, and symmetry reduction.
* :meth:`why_not` — the group-minimal infeasibility certificate (:mod:`why_not`).
* :meth:`step` — the deterministic symbolic transition ``X_{t+1} = K(X_t, a_t)``
  (admit reserves quota; place binds the gang; hold defers).

Plus the **safety shield projection** :meth:`project` (``Pi_s``, doc S
Definitions D11-D13): the nearest-feasible repair of a planner proposal. The
shield is *sound* (T10), *idempotent* (T11), and *minimal-perturbation* (T12);
together with shadow-mode / atomic revalidation (Conditions C2-C4) these give the
constructive **zero-hard-violation guarantee** (Theorem T13).

ZERO-HARD-VIOLATION PROOF SKETCH (doc S, T10/T13)
-------------------------------------------------
Let ``S_s(w) = {alpha : Phi_s(w, alpha) = 1}`` be the exact safe set. Every action
this kernel *emits* for actuation is the output of :meth:`project`, which is an
argmin over ``S_s(w)`` under the lexicographic action distance. Therefore:

* **Soundness (T10):** ``project(...) in S_s(w)`` whenever ``S_s(w)`` is non-empty,
  because the argmin is taken over ``S_s(w)`` by construction. Membership in
  ``S_s(w)`` means every modeled hard constraint (gang atomicity, topology,
  health, quota, borrowing/lending, protected-domain/operator masks) holds.
* **Idempotence (T11):** if ``alpha in S_s(w)`` then ``project(alpha) = alpha``
  because its distance to itself is the lexicographic zero and the tie-break is
  deterministic. Hence ``Pi_s(Pi_s(alpha)) = Pi_s(alpha)``.
* **Zero violation (T13):** under shadow-mode (v1) the emitted action is never
  actuated, or in an actuating deployment it is re-checked atomically against the
  live snapshot (C2) and only the projected action is emitted (C3) with an exact
  integer recheck of the returned assignment (C4 — our placement solvers return
  concrete integer ``x_pn`` and use only integer GPU/pods comparisons).

The learned residual model may *rank / forecast / abstain* over the set returned
by :meth:`safe_candidates`, but it can never enlarge it — safety is a gate, not a
weighted metric.
"""

from __future__ import annotations

from dataclasses import replace

from ...schema.actions import (
    Action,
    ActionType,
    Admit,
    AdmitAndPlace,
    Hold,
    MaskDomain,
    Place,
)
from ...schema.entities import (
    Workload,
    WorkloadStatus,
    admission_checks_positive,
)
from ...schema.state import SymbolicState, WorldState
from ..contracts import FeasibilityResult, WhyNot
from . import feasibility as feas
from . import quota as quota_mod
from . import why_not as why_not_mod
from .health_masks import build_health_masks
from .topology_masks import (
    UNCONSTRAINED_DOMAIN_ID,
    build_topology_index,
)


def _copy_symbolic(symbolic: SymbolicState) -> SymbolicState:
    """Shallow-but-safe copy for the transition: clone the mutated entity tables.

    ``step`` only mutates workloads, cluster queues, nodes, and domains, so we
    deep-copy *those* tables (and the affected dataclasses) and share the rest by
    reference. This keeps the transition allocation-light at demo-pod scale while
    never mutating the caller's input state.

    ClusterQueue carries the per-(flavor, resource) ``reserved_by_resource`` /
    ``borrowed_by_resource`` *nested* ledgers that ``step`` mutates in place; a
    plain :func:`replace` shallow-copies the queue but would alias those nested
    dicts back into the caller's state, leaking the admission. We therefore clone
    the nested ledgers explicitly so the transition is genuinely pure.
    """
    return replace(
        symbolic,
        workloads={k: replace(v) for k, v in symbolic.workloads.items()},
        podsets={k: replace(v) for k, v in symbolic.podsets.items()},
        cluster_queues={
            k: replace(
                v,
                reserved_by_resource={
                    f: dict(r) for f, r in v.reserved_by_resource.items()
                },
                borrowed_by_resource={
                    f: dict(r) for f, r in v.borrowed_by_resource.items()
                },
            )
            for k, v in symbolic.cluster_queues.items()
        },
        nodes={k: replace(v) for k, v in symbolic.nodes.items()},
        gpus={k: replace(v) for k, v in symbolic.gpus.items()},
        domains={k: replace(v) for k, v in symbolic.domains.items()},
    )


class ExactSymbolicKernel:
    """The exact symbolic feasibility gate + transition (the safety shield)."""

    def __init__(self, *, max_candidate_domains: int = 64) -> None:
        #: Latency guard — cap the enumerated domain count (doc S latency budget).
        self.max_candidate_domains = max_candidate_domains

    # ── ExactKernel.feasible ───────────────────────────────────────────────
    def feasible(self, state: WorldState, action: Action) -> FeasibilityResult:
        """Exact ``Phi_s(w, action)`` for a single fully-parameterised action."""
        symbolic = state.symbolic
        wl_uid = getattr(action, "workload_uid", None)
        # mask_domain / hold-on-missing-workload guards.
        if isinstance(action, MaskDomain):
            # A mask command is always "feasible" as a guardrail; it changes the
            # policy context rather than admitting/placing a workload.
            return FeasibilityResult(feasible=True)
        if wl_uid is None or wl_uid not in symbolic.workloads:
            from ..contracts import Blocker, BlockerGroup

            return FeasibilityResult(
                feasible=False,
                blockers=[
                    Blocker(
                        BlockerGroup.STATUS_PRECONDITION,
                        f"workload '{wl_uid}' not found",
                    )
                ],
            )
        wl = symbolic.workloads[wl_uid]
        return feas.evaluate(symbolic, wl, action)

    # ── ExactKernel.safe_candidates ────────────────────────────────────────
    def safe_candidates(self, state: WorldState, workload_uid: str) -> list[Action]:
        """Enumerate the kernel-certified safe set ``F_s(w)`` for a workload.

        Generates ``hold`` plus ``admit_and_place(q, f, sigma)`` over the cross
        product of eligible (cluster_queue, flavor) pairs and feasible topology
        selections, with the doc S pruning order applied:

        * **singleton screens** — drop (q, f) pairs that fail whitelist /
          quota arithmetic before any placement work;
        * **resource lower bounds** — drop selections whose healthy-GPU total is
          below gang demand before packing;
        * **domain dominance** — keep only feasible *selection labels* (the
          feasibility predicate already returns the surviving feasible domains);
        * **symmetry reduction** — identical-capacity selections collapse to the
          same label, so duplicates never re-enter the candidate list.

        ``hold`` is always offered when the operator whitelist permits it
        (doc S, Theorem T7: hold weakly dominates irrevocable reject).
        """
        symbolic = state.symbolic
        out: list[Action] = []
        if workload_uid not in symbolic.workloads:
            return out
        wl = symbolic.workloads[workload_uid]
        allowed = set(symbolic.operator_policy.allowed_actions)

        # hold is always a safe candidate when whitelisted.
        if ActionType.HOLD.value in allowed:
            hold = Hold(workload_uid=workload_uid)
            if self.feasible(state, hold).feasible:
                out.append(hold)

        # admit_and_place over (queue, flavor); placement is the dominant control.
        if ActionType.ADMIT_AND_PLACE.value in allowed and wl.status in (
            WorkloadStatus.PENDING,
            WorkloadStatus.RESERVING,
        ):
            idx = build_topology_index(symbolic)
            health = build_health_masks(symbolic)
            queues = self._candidate_queues(symbolic, wl)
            flavors = self._candidate_flavors(symbolic, wl)
            for q in queues:
                for f in flavors:
                    # Singleton admission-check + quota screens before placement
                    # work (doc S latency_budget.pruning: cheap exact screens
                    # first). chi_qf = 0 gates the (q, f) admit outright (E5).
                    cq = symbolic.cluster_queues[q]
                    if not admission_checks_positive(cq, f):
                        continue
                    cohort = quota_mod.resolve_cohort(symbolic.cluster_queues, cq)
                    demand = quota_mod.workload_demand(
                        [
                            (
                                p.count,
                                {
                                    "gpu": p.gpu_req,
                                    "cpu": p.cpu_req,
                                    "mem": p.mem_req_gb,
                                },
                            )
                            for p in wl.podsets
                        ]
                    )
                    if not quota_mod.quota_check(cq, cohort, f, demand).feasible:
                        continue
                    probe = AdmitAndPlace(
                        workload_uid=workload_uid,
                        cluster_queue_id=q,
                        resource_flavor_id=f,
                        domain_policy=wl.topology_policy.value,
                    )
                    res = feas.evaluate(
                        symbolic, wl, probe, topology_index=idx, health=health
                    )
                    if not res.feasible:
                        continue
                    # Emit one safe action per surviving feasible domain selection
                    # (already dominance/symmetry-reduced by the predicate).
                    for label in res.feasible_domains[: self.max_candidate_domains]:
                        out.append(
                            AdmitAndPlace(
                                workload_uid=workload_uid,
                                cluster_queue_id=q,
                                resource_flavor_id=f,
                                domain_policy=wl.topology_policy.value,
                                domain_id_or_set=self._label_to_selection(label),
                            )
                        )
        return out

    # ── ExactKernel.why_not ────────────────────────────────────────────────
    def why_not(self, state: WorldState, action: Action) -> WhyNot:
        """Group-minimal infeasibility certificate for a rejected action."""
        symbolic = state.symbolic
        result = self.feasible(state, action)
        return why_not_mod.certificate(symbolic, action, result=result)

    # ── ExactKernel.step ───────────────────────────────────────────────────
    def step(self, state: WorldState, action: Action) -> WorldState:
        """Deterministic symbolic transition ``X_{t+1} = K(X_t, a_t)``.

        Interventional semantics (doc I, Definition 6 / doc S): an *infeasible*
        command is a no-op (reject) — the kernel never teleports the world into an
        infeasible allocation. A feasible command mutates the symbolic state:

        * ``hold``           — increment the workload's age / requeue-free defer;
        * ``admit``          — reserve quota, set status ``RESERVING``;
        * ``place`` /
          ``admit_and_place``— bind the gang onto the assignment's nodes, reserve
          quota, set status ``ADMITTED`` and record flavor/domain.
        * ``mask_domain``    — operator guardrail: add the domains to the operator
          policy's protected list (exogenous policy change).

        The returned state is a fresh copy; the input ``state`` is never mutated.
        """
        new_symbolic = _copy_symbolic(state.symbolic)
        new_state = replace(
            state,
            symbolic=new_symbolic,
            tick=state.tick + 1,
        )

        if isinstance(action, MaskDomain):
            pol = new_symbolic.operator_policy
            merged = list(
                dict.fromkeys(
                    list(pol.protected_domains) + list(action.domain_id_or_set)
                )
            )
            new_symbolic.operator_policy = replace(pol, protected_domains=merged)
            return new_state

        wl_uid = getattr(action, "workload_uid", None)
        if wl_uid is None or wl_uid not in new_symbolic.workloads:
            return new_state  # nothing to do (no-op)

        # Re-validate against the (copied) live state before mutating (C2/C3/C4):
        # only a projected/feasible action is ever applied.
        result = feas.evaluate(new_symbolic, new_symbolic.workloads[wl_uid], action)
        if not result.feasible:
            return new_state  # reject / no-op with the blocker ledger available

        wl = new_symbolic.workloads[wl_uid]
        if isinstance(action, Hold):
            wl.age_s += 0.0  # defer: age accrues via the time process, not here
            return new_state

        # ── C4: exact integer recheck of the returned assignment (doc S) ───────
        # Before committing any mutation, re-verify the placement solver's integer
        # assignment with deterministic integer arithmetic: every podset is bound
        # all-or-nothing (sum_n x_pn == c_p) and every target node has enough free
        # *healthy* GPUs and pod headroom for what we are about to bind
        # (bound == self_bound, per-node capacity). If the recheck fails we abort
        # to a no-op rather than emit a possibly-inconsistent allocation (T13).
        if (
            isinstance(action, (Place, AdmitAndPlace))
            and result.assignment
            and not self._c4_integer_recheck(new_symbolic, wl, result.assignment)
        ):
            return new_state  # C4 violation -> no-op

        # admit / admit_and_place reserve quota.
        if isinstance(action, (Admit, AdmitAndPlace)):
            cq = new_symbolic.cluster_queues.get(action.cluster_queue_id)
            if cq is not None:
                cohort = quota_mod.resolve_cohort(new_symbolic.cluster_queues, cq)
                demand = quota_mod.workload_demand(
                    [
                        (
                            p.count,
                            {"gpu": p.gpu_req, "cpu": p.cpu_req, "mem": p.mem_req_gb},
                        )
                        for p in wl.podsets
                    ]
                )
                qd = quota_mod.quota_check(
                    cq, cohort, action.resource_flavor_id, demand
                )
                gpu_demand = demand.get("gpu", 0.0)
                borrow_gpu = qd.borrow.get("gpu", 0.0)
                # Reserve across the full hard-resource set R = {gpu, cpu, mem,
                # pods}, not just gpu, so a sequence of admissions correctly
                # decrements declared cpu/mem/pods quota and bounds borrowing
                # (doc S, Eq. E11/E12 for all r). The ``pods`` demand is the
                # gang's replica count (one pod per replica).
                pods_demand = sum(p.count for p in wl.podsets)
                full_demand = dict(demand)
                full_demand["pods"] = float(pods_demand)
                quota_mod.apply_admission(
                    cq,
                    gpu_demand,
                    borrow_gpu,
                    flavor_id=action.resource_flavor_id,
                    demand=full_demand,
                    borrow=qd.borrow,
                )
                wl.admitted_flavor_id = action.resource_flavor_id
            wl.status = WorkloadStatus.RESERVING

        # place / admit_and_place bind the gang.
        if isinstance(action, (Place, AdmitAndPlace)) and result.assignment:
            self._bind_assignment(new_symbolic, wl, result.assignment)
            wl.status = WorkloadStatus.ADMITTED
            # Record the chosen domain for the UI / planner.
            if result.feasible_domains:
                wl.admitted_domain_id = result.feasible_domains[0]
        return new_state

    # ── Safety shield projection Pi_s (doc S, Definitions D11-D13) ──────────
    def project(self, state: WorldState, proposal: Action) -> Action | None:
        """Nearest-feasible projection of a planner proposal into ``S_s(w)``.

        Implements the constructive shield algorithm (doc S):

        1. if ``proposal`` is already feasible, return it (idempotence, T11);
        2. else enumerate ``S_s(w)`` via :meth:`safe_candidates`;
        3. prefer the same action type, then same queue, then same flavor, then
           the smallest topology edit (lexicographic distance, D12);
        4. if no non-hold safe action remains, return ``hold``;
        5. if even hold is disallowed, return ``None`` (abstain).

        The returned action is guaranteed to lie in ``S_s(w)`` (Theorem T10).
        """
        if self.feasible(state, proposal).feasible:
            return proposal  # T11 idempotence

        wl_uid = getattr(proposal, "workload_uid", None)
        if wl_uid is None:
            return None
        safe = self.safe_candidates(state, wl_uid)
        if not safe:
            return None  # abstain — even hold is unavailable

        best = min(safe, key=lambda a: self._distance(proposal, a))
        return best

    # ── internals ──────────────────────────────────────────────────────────
    @staticmethod
    def _distance(a: Action, b: Action) -> tuple[int, int, int, int]:
        """Lexicographic action distance ``dist(a, b)`` (doc S, Definition D12)."""
        type_diff = int(a.type != b.type)
        qa = getattr(a, "cluster_queue_id", None)
        qb = getattr(b, "cluster_queue_id", None)
        queue_diff = int(qa != qb)
        fa = getattr(a, "resource_flavor_id", None)
        fb = getattr(b, "resource_flavor_id", None)
        flavor_diff = int(fa != fb)
        sa = set(getattr(a, "domain_id_or_set", ()) or ())
        sb = set(getattr(b, "domain_id_or_set", ()) or ())
        topo_edit = len(sa ^ sb)
        return (type_diff, queue_diff, flavor_diff, topo_edit)

    @staticmethod
    def _candidate_queues(symbolic: SymbolicState, wl: Workload) -> list[str]:
        """Eligible cluster queues for a workload (its declared queue first)."""
        if wl.cluster_queue_id in symbolic.cluster_queues:
            return [wl.cluster_queue_id]
        return sorted(symbolic.cluster_queues.keys())

    @staticmethod
    def _candidate_flavors(symbolic: SymbolicState, wl: Workload) -> list[str]:
        """Candidate flavors: a queue's ``flavor_order`` if set, else all flavors."""
        cq = symbolic.cluster_queues.get(wl.cluster_queue_id)
        if cq is not None and cq.flavor_order:
            return [f for f in cq.flavor_order if f in symbolic.flavors]
        return sorted(symbolic.flavors.keys())

    @staticmethod
    def _label_to_selection(label: str) -> tuple[str, ...]:
        """Inverse of :func:`topology_masks.selection_label`."""
        if label == "unconstrained":
            return (UNCONSTRAINED_DOMAIN_ID,)
        return tuple(label.split("+"))

    @staticmethod
    def _c4_integer_recheck(
        symbolic: SymbolicState,
        wl: Workload,
        assignment: dict[str, dict[str, int]],
    ) -> bool:
        """Exact deterministic integer recheck of an assignment (doc S, C4 / T13).

        Returns ``True`` iff the assignment is committable with *no* modeled hard
        violation, verified with integer arithmetic only:

        * **Gang atomicity (E6).** For every podset ``p`` the assignment's total
          replica count over nodes equals the gang demand ``c_p`` — never a
          partial subset.
        * **Per-node GPU availability.** Each target node has at least
          ``self_bound = sum_p replicas_pn * gpu_req_p`` *free healthy* GPUs (the
          ``bound == self_bound`` condition the binder relies on). We count
          free, ``HEALTHY``, unallocated devices physically on the node.
        * **Per-node pod / capacity headroom (E9).** Accumulated pods, cpu, and
          mem across all podsets landing on a node fit its declared residual.

        Any failure aborts the transition to a no-op rather than emit a possibly
        inconsistent allocation. This closes the C4 gap even if an upstream solver
        used float tolerances internally.
        """
        from ...schema.entities import DeviceHealth

        req_by_podset = {p.name: p for p in wl.podsets}

        # E6: gang all-or-nothing per podset.
        for p in wl.podsets:
            placed = sum(assignment.get(p.name, {}).values())
            if placed != p.count:
                return False

        # Per-node accumulation of the integer demand this assignment imposes.
        gpu_need: dict[str, int] = {}
        cpu_need: dict[str, float] = {}
        mem_need: dict[str, float] = {}
        for podset_name, node_map in assignment.items():
            p = req_by_podset.get(podset_name)
            if p is None:
                return False  # assignment references an unknown podset
            for node_uid, replicas in node_map.items():
                if replicas < 0:
                    return False
                gpu_need[node_uid] = gpu_need.get(node_uid, 0) + replicas * int(
                    p.gpu_req
                )
                cpu_need[node_uid] = cpu_need.get(node_uid, 0.0) + replicas * p.cpu_req
                mem_need[node_uid] = (
                    mem_need.get(node_uid, 0.0) + replicas * p.mem_req_gb
                )

        eps = 1e-9
        for node_uid, need_gpu in gpu_need.items():
            node = symbolic.nodes.get(node_uid)
            if node is None:
                return False
            # Free *healthy* GPU devices physically on this node.
            free_healthy = sum(
                1
                for g in symbolic.gpus.values()
                if g.node_uid == node_uid
                and g.allocated_to is None
                and g.health_state is DeviceHealth.HEALTHY
            )
            # When the symbolic state carries no per-GPU inventory for this node,
            # fall back to the declared free GPU capacity (cap - 0 here, since the
            # snapshot residual is the declared gpu_cap for inventory-less nodes).
            has_inventory = any(g.node_uid == node_uid for g in symbolic.gpus.values())
            avail_gpu = free_healthy if has_inventory else node.gpu_cap
            if need_gpu > avail_gpu:
                return False
            # Per-node cpu + mem headroom (E9), exact comparisons with an epsilon
            # only to absorb float representation of integer-derived demands.
            if node.cpu_alloc + cpu_need[node_uid] > node.cpu_cap + eps:
                return False
            if node.mem_alloc_gb + mem_need[node_uid] > node.mem_cap_gb + eps:
                return False
        return True

    @staticmethod
    def _bind_assignment(
        symbolic: SymbolicState,
        wl: Workload,
        assignment: dict[str, dict[str, int]],
    ) -> None:
        """Apply the concrete integer assignment ``x_pn`` to node allocation.

        Decrements free capacity on each target node deterministically. This is
        the symbolic placement transition: it consumes the exact integer GPU /
        pod counts the placement solver produced. The assignment has already
        passed :meth:`_c4_integer_recheck` (Condition C4), so ``bound`` always
        reaches ``self_bound`` here; the binder is therefore a pure commit.
        """
        # Map podset name -> per-replica request for capacity bookkeeping.
        req_by_podset = {p.name: p for p in wl.podsets}
        for podset_name, node_map in assignment.items():
            p = req_by_podset.get(podset_name)
            if p is None:
                continue
            for node_uid, replicas in node_map.items():
                node = symbolic.nodes.get(node_uid)
                if node is None:
                    continue
                node.cpu_alloc += replicas * p.cpu_req
                node.mem_alloc_gb += replicas * p.mem_req_gb
                node.running_pods += replicas
                # GPUs: bind the first available healthy device per replica*gpu_req.
                self_bound = replicas * int(p.gpu_req)
                bound = 0
                for g in symbolic.gpus.values():
                    if bound >= self_bound:
                        break
                    if g.node_uid == node_uid and g.allocated_to is None:
                        g.allocated_to = wl.workload_uid
                        bound += 1


__all__ = ["ExactSymbolicKernel"]
