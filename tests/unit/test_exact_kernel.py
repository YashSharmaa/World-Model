# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Unit tests for the exact symbolic kernel — the project's safety shield (doc S).

The kernel owns *hard feasibility*: quota, gang atomicity, topology rules, health
masks, operator policy, and exact blockers. It is deterministic and never
learned. The contract these tests pin:

* ``feasible`` is exact: ``hold`` is always feasible; an over-large or
  protected/masked placement is infeasible with a *blocker ledger*.
* ``safe_candidates`` returns *only* actions that are individually feasible — the
  set the learned model may rank/forecast/abstain over but never expand (doc S
  Definition D7 + the constructive zero-violation property).
* ``why_not`` returns a non-empty minimal certificate for a rejected action.
* ``step`` applies the deterministic transition and never mutates the input.
"""

from __future__ import annotations

import copy

from gpu_center_world_model.schema import (
    AdmitAndPlace,
    Hold,
    MaskDomain,
    WorkloadStatus,
    world_state_hash,
)
from gpu_center_world_model.simulator import BlockerGroup


def _flavor(state) -> str:
    return next(iter(state.symbolic.flavors))


def _cq(state, uid) -> str:
    return state.symbolic.workloads[uid].cluster_queue_id


# ──────────────────────────────────────────────────────────────────────────────
# feasible(): exact predicate
# ──────────────────────────────────────────────────────────────────────────────
def test_hold_is_always_feasible(kernel, demo_state, queued_workload_uid):
    res = kernel.feasible(demo_state, Hold(workload_uid=queued_workload_uid))
    assert res.feasible
    assert res.blockers == []


def test_unknown_workload_is_infeasible_with_blocker(kernel, demo_state):
    res = kernel.feasible(demo_state, Hold(workload_uid="does-not-exist"))
    assert not res.feasible
    assert res.blockers
    assert res.blockers[0].group is BlockerGroup.STATUS_PRECONDITION


def test_feasible_admit_and_place_returns_feasible_domains(
    kernel, demo_state, queued_workload_uid
):
    act = AdmitAndPlace(
        workload_uid=queued_workload_uid,
        cluster_queue_id=_cq(demo_state, queued_workload_uid),
        resource_flavor_id=_flavor(demo_state),
        domain_policy="preferred",
    )
    res = kernel.feasible(demo_state, act)
    assert res.feasible
    assert res.feasible_domains  # at least one domain hosts the 4-GPU gang


def test_oversized_required_gang_is_infeasible(kernel, demo_state, queued_workload_uid):
    # Force the gang to demand more GPUs than any single domain can host, with
    # `required` locality, so single-domain packing is impossible.
    state = copy.deepcopy(demo_state)
    wl = state.symbolic.workloads[queued_workload_uid]
    for ps in wl.podsets:
        ps.count = 999  # absurd gang
    act = AdmitAndPlace(
        workload_uid=queued_workload_uid,
        cluster_queue_id=wl.cluster_queue_id,
        resource_flavor_id=_flavor(state),
        domain_policy="required",
        domain_id_or_set=(next(iter(state.symbolic.domains)),),
    )
    res = kernel.feasible(state, act)
    assert not res.feasible
    assert res.blockers
    # The blocker carries a resource deficit or a hard topology/capacity reason.
    groups = {b.group for b in res.blockers}
    assert groups & {
        BlockerGroup.NODE_CAPACITIES,
        BlockerGroup.TOPOLOGY_RULE,
        BlockerGroup.GANG_ATOMICITY,
        BlockerGroup.QUOTA,
    }


def test_mask_domain_is_treated_as_guardrail_feasible(kernel, demo_state):
    # A mask command changes policy context; the kernel treats it as feasible.
    res = kernel.feasible(demo_state, MaskDomain(domain_id_or_set=("rack-0",)))
    assert res.feasible


# ──────────────────────────────────────────────────────────────────────────────
# safe_candidates(): only feasible actions, every one re-verifiable
# ──────────────────────────────────────────────────────────────────────────────
def test_safe_candidates_nonempty_and_contains_hold(
    kernel, demo_state, queued_workload_uid
):
    cands = kernel.safe_candidates(demo_state, queued_workload_uid)
    assert cands
    assert any(a.type.value == "hold" for a in cands)


def test_safe_candidates_are_all_individually_feasible(
    kernel, demo_state, queued_workload_uid
):
    # The defining property of the safe set: every member passes feasible() (doc S).
    cands = kernel.safe_candidates(demo_state, queued_workload_uid)
    for a in cands:
        assert kernel.feasible(demo_state, a).feasible, a


def test_safe_candidates_offers_placement_options(
    kernel, demo_state, queued_workload_uid
):
    cands = kernel.safe_candidates(demo_state, queued_workload_uid)
    placements = [a for a in cands if a.type.value == "admit_and_place"]
    assert placements, "a 4-GPU gang in a 2-rack pod should have placement options"


def test_safe_candidates_unknown_workload_is_empty(kernel, demo_state):
    assert kernel.safe_candidates(demo_state, "nope") == []


def test_protected_domain_shrinks_safe_set(kernel, demo_state, queued_workload_uid):
    # Masking every domain via the operator policy removes all placements but
    # leaves hold available — the shield never enlarges, only restricts.
    state = copy.deepcopy(demo_state)
    pol = state.symbolic.operator_policy
    state.symbolic.operator_policy = type(pol)(
        policy_id=pol.policy_id,
        allowed_actions=list(pol.allowed_actions),
        protected_domains=list(state.symbolic.domains),
        risk_thresholds=dict(pol.risk_thresholds),
        active_since_ts=pol.active_since_ts,
    )
    cands = kernel.safe_candidates(state, queued_workload_uid)
    assert all(a.type.value == "hold" for a in cands)
    # And every survivor is still feasible.
    for a in cands:
        assert kernel.feasible(state, a).feasible


def test_whitelist_removes_placement_from_safe_set(
    kernel, demo_state, queued_workload_uid
):
    state = copy.deepcopy(demo_state)
    pol = state.symbolic.operator_policy
    state.symbolic.operator_policy = type(pol)(
        policy_id=pol.policy_id,
        allowed_actions=["hold"],  # only hold permitted
        protected_domains=list(pol.protected_domains),
        risk_thresholds=dict(pol.risk_thresholds),
        active_since_ts=pol.active_since_ts,
    )
    cands = kernel.safe_candidates(state, queued_workload_uid)
    assert cands and all(a.type.value == "hold" for a in cands)


# ──────────────────────────────────────────────────────────────────────────────
# why_not(): minimal infeasibility certificate
# ──────────────────────────────────────────────────────────────────────────────
def test_why_not_certificate_for_rejected_action(
    kernel, demo_state, queued_workload_uid
):
    state = copy.deepcopy(demo_state)
    wl = state.symbolic.workloads[queued_workload_uid]
    for ps in wl.podsets:
        ps.count = 999
    act = AdmitAndPlace(
        workload_uid=queued_workload_uid,
        cluster_queue_id=wl.cluster_queue_id,
        resource_flavor_id=_flavor(state),
        domain_policy="required",
        domain_id_or_set=(next(iter(state.symbolic.domains)),),
    )
    assert not kernel.feasible(state, act).feasible
    wn = kernel.why_not(state, act)
    assert wn.minimal_groups, "an infeasible action must carry a minimal cert"
    assert wn.blockers


# ──────────────────────────────────────────────────────────────────────────────
# step(): deterministic transition, input never mutated
# ──────────────────────────────────────────────────────────────────────────────
def test_step_does_not_mutate_input_state(kernel, demo_state, queued_workload_uid):
    before = world_state_hash(demo_state)
    act = AdmitAndPlace(
        workload_uid=queued_workload_uid,
        cluster_queue_id=_cq(demo_state, queued_workload_uid),
        resource_flavor_id=_flavor(demo_state),
        domain_policy="preferred",
    )
    kernel.step(demo_state, act)
    assert world_state_hash(demo_state) == before  # untouched


def test_step_admit_and_place_admits_workload(kernel, demo_state, queued_workload_uid):
    act = AdmitAndPlace(
        workload_uid=queued_workload_uid,
        cluster_queue_id=_cq(demo_state, queued_workload_uid),
        resource_flavor_id=_flavor(demo_state),
        domain_policy="preferred",
    )
    nxt = kernel.step(demo_state, act)
    assert nxt.tick == demo_state.tick + 1
    assert nxt.symbolic.workloads[queued_workload_uid].status is (
        WorkloadStatus.ADMITTED
    )


def test_step_is_deterministic(kernel, demo_state, queued_workload_uid):
    act = AdmitAndPlace(
        workload_uid=queued_workload_uid,
        cluster_queue_id=_cq(demo_state, queued_workload_uid),
        resource_flavor_id=_flavor(demo_state),
        domain_policy="preferred",
    )
    a = kernel.step(demo_state, act)
    b = kernel.step(demo_state, act)
    assert world_state_hash(a) == world_state_hash(b)


def test_step_rejects_infeasible_as_noop(kernel, demo_state, queued_workload_uid):
    # An infeasible command is a reject/no-op (doc I, Definition 6) — the kernel
    # never teleports the world into an infeasible allocation.
    state = copy.deepcopy(demo_state)
    wl = state.symbolic.workloads[queued_workload_uid]
    for ps in wl.podsets:
        ps.count = 999
    act = AdmitAndPlace(
        workload_uid=queued_workload_uid,
        cluster_queue_id=wl.cluster_queue_id,
        resource_flavor_id=_flavor(state),
        domain_policy="required",
        domain_id_or_set=(next(iter(state.symbolic.domains)),),
    )
    nxt = kernel.step(state, act)
    # Status unchanged (still pending); the no-op did not admit an oversized gang.
    assert nxt.symbolic.workloads[queued_workload_uid].status is (
        WorkloadStatus.PENDING
    )


# ──────────────────────────────────────────────────────────────────────────────
# E5 AdmissionCheck gate (KERNEL-ADMISSION-CHECK, doc S Eq. E5)
# ──────────────────────────────────────────────────────────────────────────────
def _aap(state, uid, **kw) -> AdmitAndPlace:
    return AdmitAndPlace(
        workload_uid=uid,
        cluster_queue_id=_cq(state, uid),
        resource_flavor_id=_flavor(state),
        domain_policy=kw.pop("domain_policy", "preferred"),
        **kw,
    )


def test_admission_check_not_ready_blocks_admit(
    kernel, demo_state, queued_workload_uid
):
    # chi_qf = 0 (a required AdmissionCheck is not READY) gates the (q,f) admit.
    state = copy.deepcopy(demo_state)
    cq = state.symbolic.cluster_queues[_cq(state, queued_workload_uid)]
    cq.admission_check_state = {"*": False}
    res = kernel.feasible(state, _aap(state, queued_workload_uid))
    assert not res.feasible
    assert BlockerGroup.ADMISSION_CHECKS in {b.group for b in res.blockers}


def test_admission_check_ready_does_not_block(
    kernel, demo_state, queued_workload_uid
):
    # chi_qf = 1 (every required check READY) leaves admission feasible.
    state = copy.deepcopy(demo_state)
    cq = state.symbolic.cluster_queues[_cq(state, queued_workload_uid)]
    cq.admission_check_state = {"*": True}
    assert kernel.feasible(state, _aap(state, queued_workload_uid)).feasible


def test_admission_check_is_a_singleton_certificate(
    kernel, demo_state, queued_workload_uid
):
    state = copy.deepcopy(demo_state)
    cq = state.symbolic.cluster_queues[_cq(state, queued_workload_uid)]
    cq.admission_check_state = {"*": False}
    wn = kernel.why_not(state, _aap(state, queued_workload_uid))
    assert wn.minimal_groups == [BlockerGroup.ADMISSION_CHECKS]


def test_admission_check_empty_map_is_vacuously_satisfied(
    kernel, demo_state, queued_workload_uid
):
    # No declared checks -> admission is not gated here (Kueue: admit on quota).
    state = copy.deepcopy(demo_state)
    cq = state.symbolic.cluster_queues[_cq(state, queued_workload_uid)]
    cq.admission_check_state = {}
    assert kernel.feasible(state, _aap(state, queued_workload_uid)).feasible


# ──────────────────────────────────────────────────────────────────────────────
# Non-GPU quota counters (KERNEL-NONGPU-QUOTA, doc S Eq. E11/E12 for all r)
# ──────────────────────────────────────────────────────────────────────────────
def test_cpu_quota_is_decremented_across_a_sequence(
    kernel, demo_state, queued_workload_uid
):
    from gpu_center_world_model.simulator.exact_kernel import quota as q

    state = copy.deepcopy(demo_state)
    uid = queued_workload_uid
    cq_id = _cq(state, uid)
    fl = _flavor(state)
    cq = state.symbolic.cluster_queues[cq_id]
    # The demo gang is 4 replicas * 8 cpu = 32 cpu. Declare a 40-cpu quota so a
    # single admit fits but a second one must not.
    cq.nominal_quota[fl]["cpu"] = 40.0
    act = _aap(state, uid)
    assert kernel.feasible(state, act).feasible
    nxt = kernel.step(state, act)
    cq2 = nxt.symbolic.cluster_queues[cq_id]
    assert q.reserved_for(cq2, fl, "cpu") == 32.0  # exactly the admitted cpu
    assert q._nominal_headroom(cq2, fl, "cpu") == 8.0  # 40 - 32
    # Re-queue the same workload and re-admit: 32 cpu > 8 headroom -> cpu quota.
    nxt.symbolic.workloads[uid].status = WorkloadStatus.PENDING
    res2 = kernel.feasible(nxt, _aap(nxt, uid))
    assert not res2.feasible
    grp_def = {(b.group, tuple(sorted(b.deficit))) for b in res2.blockers}
    assert any(g is BlockerGroup.QUOTA and "cpu" in d for g, d in grp_def)


def test_quota_release_returns_counters_and_prunes_ledger(demo_state):
    from gpu_center_world_model.simulator.exact_kernel import quota as q

    state = copy.deepcopy(demo_state)
    cq = next(iter(state.symbolic.cluster_queues.values()))
    fl = _flavor(state)
    q.apply_admission(
        cq, 4.0, 0.0, flavor_id=fl, demand={"gpu": 4.0, "cpu": 16.0}, borrow={}
    )
    assert q.reserved_for(cq, fl, "cpu") == 16.0
    q.apply_release(
        cq, 4.0, 0.0, flavor_id=fl, demand={"gpu": 4.0, "cpu": 16.0}, borrow={}
    )
    assert q.reserved_for(cq, fl, "cpu") == 0.0
    # Empty inner maps are pruned so a released queue hashes like a fresh one.
    assert cq.reserved_by_resource.get(fl, {}) == {}


# ──────────────────────────────────────────────────────────────────────────────
# Placement-constraint blocker attribution (KERNEL-PLACEMENT-CONSTRAINTS)
# ──────────────────────────────────────────────────────────────────────────────
def test_health_filtered_nodes_emit_health_schedulability(
    kernel, demo_state, queued_workload_uid
):
    from gpu_center_world_model.schema.entities import NodeHealth

    state = copy.deepcopy(demo_state)
    # Drain every node: member nodes exist but none is schedulable -> the blocker
    # is HEALTH_SCHEDULABILITY, not a capacity catch-all.
    for node in state.symbolic.nodes.values():
        node.node_health = NodeHealth.DRAINING
        node.schedulable = False
    res = kernel.feasible(state, _aap(state, queued_workload_uid))
    assert not res.feasible
    assert BlockerGroup.HEALTH_SCHEDULABILITY in {b.group for b in res.blockers}


def _rack_only_topology(state):
    """Strip the all-encompassing block domain so only per-rack domains host nodes.

    The demo pod nests every node under a single ``block-0`` domain that spans the
    whole pod; that block would always host any pod-sized gang. Removing it (and
    pointing each node's ``block_id`` at its own rack) leaves only the per-rack
    domains, so same-domain (rack) locality is exercised exactly.
    """
    state.symbolic.domains.clear()
    for node in state.symbolic.nodes.values():
        node.block_id = node.rack_id


def test_aggregate_capacity_across_domains_emits_topology_rule(
    kernel, demo_state, queued_workload_uid
):
    # A required gang larger than any single rack but no larger than the cluster's
    # aggregate has capacity yet no single domain hosts it -> TOPOLOGY_RULE.
    state = copy.deepcopy(demo_state)
    _rack_only_topology(state)
    wl = state.symbolic.workloads[queued_workload_uid]
    # demo pod: 2 racks * 2 nodes * 4 GPUs = 16 GPUs, 8 per rack. Demand 12 GPUs
    # (no cpu/mem so it is GPU-binding) exceeds one rack's 8 but fits the cluster.
    wl.podsets = wl.podsets[:1]
    wl.podsets[0].count = 12
    wl.podsets[0].gpu_req = 1
    wl.podsets[0].cpu_req = 0.0
    wl.podsets[0].mem_req_gb = 0.0
    act = AdmitAndPlace(
        workload_uid=queued_workload_uid,
        cluster_queue_id=wl.cluster_queue_id,
        resource_flavor_id=_flavor(state),
        domain_policy="required",  # one domain only -> no whole-pod fallback
    )
    res = kernel.feasible(state, act)
    assert not res.feasible
    assert BlockerGroup.TOPOLOGY_RULE in {b.group for b in res.blockers}


def test_podset_group_must_share_one_domain(kernel, demo_state, queued_workload_uid):
    # Two grouped podsets that together exceed a single rack cannot co-locate in
    # one domain under required locality -> E14 shared-z rejection.
    from gpu_center_world_model.schema.entities import PodSet

    state = copy.deepcopy(demo_state)
    _rack_only_topology(state)
    wl = state.symbolic.workloads[queued_workload_uid]
    uid = queued_workload_uid
    # 8 GPUs per rack; a grouped 12-replica demand cannot fit one rack.
    wl.podsets = [
        PodSet(
            workload_uid=uid,
            name="a",
            count=7,
            gpu_req=1,
            cpu_req=0.0,
            mem_req_gb=0.0,
            podset_group="g",
        ),
        PodSet(
            workload_uid=uid,
            name="b",
            count=5,
            gpu_req=1,
            cpu_req=0.0,
            mem_req_gb=0.0,
            podset_group="g",
        ),
    ]
    act = AdmitAndPlace(
        workload_uid=uid,
        cluster_queue_id=wl.cluster_queue_id,
        resource_flavor_id=_flavor(state),
        domain_policy="required",
    )
    res = kernel.feasible(state, act)
    # No single rack can hold the grouped 12 replicas, so it is rejected.
    assert not res.feasible


def test_place_resolves_flavor_from_admitted_flavor_id(
    kernel, demo_state, queued_workload_uid
):
    # A pure place query (E7 on any placement) resolves the flavor from the
    # workload's admitted_flavor_id and applies flavor eligibility.
    from gpu_center_world_model.schema import Place

    state = copy.deepcopy(demo_state)
    wl = state.symbolic.workloads[queued_workload_uid]
    wl.status = WorkloadStatus.ADMITTED
    wl.admitted_flavor_id = _flavor(state)
    res = kernel.feasible(state, Place(workload_uid=queued_workload_uid))
    assert res.feasible
    assert res.feasible_domains


# ──────────────────────────────────────────────────────────────────────────────
# Why-not: real deletion-filter + feasibility-relaxation (KERNEL-WHYNOT-IIS-MILP)
# ──────────────────────────────────────────────────────────────────────────────
def test_why_not_quantifies_exact_gpu_deficit(
    kernel, demo_state, queued_workload_uid
):
    state = copy.deepcopy(demo_state)
    wl = state.symbolic.workloads[queued_workload_uid]
    wl.podsets = wl.podsets[:1]
    wl.podsets[0].count = 20  # 20 GPUs vs a 16-GPU pod and 16-GPU quota
    wl.podsets[0].gpu_req = 1
    wl.podsets[0].cpu_req = 0.0
    wl.podsets[0].mem_req_gb = 0.0
    act = AdmitAndPlace(
        workload_uid=queued_workload_uid,
        cluster_queue_id=wl.cluster_queue_id,
        resource_flavor_id=_flavor(state),
        domain_policy="preferred",
    )
    wn = kernel.why_not(state, act)
    assert wn.minimal_groups
    # The feasibility-relaxation deficit is exact: 20 demanded - 16 available = 4.
    assert wn.deficits.get("gpu") == 4.0


def test_why_not_deletion_filter_is_group_minimal(
    kernel, demo_state, queued_workload_uid
):
    # Whitelist + quota + capacity all fail jointly. The re-solving deletion
    # filter must return a non-empty, group-minimal core: relaxing the whole core
    # restores feasibility, and every member is individually necessary (relaxing
    # the core minus any one member is still infeasible).
    from gpu_center_world_model.simulator.exact_kernel import feasibility as feas

    state = copy.deepcopy(demo_state)
    pol = state.symbolic.operator_policy
    state.symbolic.operator_policy = type(pol)(
        policy_id=pol.policy_id,
        allowed_actions=["hold"],  # admit_and_place not whitelisted
        protected_domains=list(pol.protected_domains),
        risk_thresholds=dict(pol.risk_thresholds),
        active_since_ts=pol.active_since_ts,
    )
    wl = state.symbolic.workloads[queued_workload_uid]
    wl.podsets[0].count = 999  # also quota/capacity infeasible
    act = AdmitAndPlace(
        workload_uid=queued_workload_uid,
        cluster_queue_id=wl.cluster_queue_id,
        resource_flavor_id=_flavor(state),
        domain_policy="preferred",
    )
    wn = kernel.why_not(state, act)
    core = wn.minimal_groups
    assert core, "an infeasible action must carry a non-empty minimal cert"
    # The core is drawn from the actually-failing groups.
    full = feas.evaluate(state.symbolic, wl, act)
    present = {b.group for b in full.blockers}
    assert set(core) <= present
    # Relaxing *all* present blocker groups makes the model feasible (the
    # relaxation machinery genuinely drops constraints, not just blocker rows).
    relaxed_all = feas.evaluate(state.symbolic, wl, act, relax=frozenset(present))
    assert relaxed_all.feasible
    # Minimality of the returned core (a single IIS): the submodel induced by the
    # core alone is infeasible (relax everything outside the core), yet relaxing
    # any one core member on top of that restores feasibility.
    outside = frozenset(present - set(core))
    assert not feas.evaluate(state.symbolic, wl, act, relax=outside).feasible
    for g in core:
        without_g = frozenset(outside | {g})
        assert feas.evaluate(state.symbolic, wl, act, relax=without_g).feasible, (
            f"core member {g} is not necessary"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Bounded packer + per-action time budget (doc S latency_budget.live_demo_rule)
# ──────────────────────────────────────────────────────────────────────────────
def test_budget_exhaustion_abstains_conservatively(demo_state, queued_workload_uid):
    from gpu_center_world_model.simulator.exact_kernel import feasibility as feas

    state = copy.deepcopy(demo_state)
    wl = state.symbolic.workloads[queued_workload_uid]
    act = AdmitAndPlace(
        workload_uid=queued_workload_uid,
        cluster_queue_id=wl.cluster_queue_id,
        resource_flavor_id=_flavor(state),
        domain_policy="preferred",
    )
    # A budget of a single expansion forces the multi-resource DFS packer to abort.
    budget = feas.TimeBudget(budget_s=10.0, max_expansions=1)
    res = feas.evaluate(state.symbolic, wl, act, budget=budget)
    # Safety is a gate: a timeout must never produce a feasible verdict.
    assert not res.feasible
    assert res.budget_exhausted
    assert BlockerGroup.GANG_ATOMICITY in {b.group for b in res.blockers}


def test_bounded_packer_terminates_on_large_gang(demo_state, queued_workload_uid):
    # Boundedness: even a huge gang into unbounded-looking capacity terminates
    # (the iterative DFS + expansion bound guarantees it; no RecursionError).
    from gpu_center_world_model.simulator.exact_kernel import feasibility as feas

    state = copy.deepcopy(demo_state)
    wl = state.symbolic.workloads[queued_workload_uid]
    wl.podsets[0].count = 5000
    act = AdmitAndPlace(
        workload_uid=queued_workload_uid,
        cluster_queue_id=wl.cluster_queue_id,
        resource_flavor_id=_flavor(state),
        domain_policy="preferred",
    )
    # Must return a definitive (infeasible) result without raising.
    res = feas.evaluate(state.symbolic, wl, act)
    assert not res.feasible


# ──────────────────────────────────────────────────────────────────────────────
# C4 exact integer recheck (doc S, Condition C4 / Theorem T13)
# ──────────────────────────────────────────────────────────────────────────────
def test_c4_recheck_rejects_partial_and_overbind(
    kernel, demo_state, queued_workload_uid
):
    state = copy.deepcopy(demo_state)
    wl = state.symbolic.workloads[queued_workload_uid]
    name = wl.podsets[0].name
    node_uid = next(iter(state.symbolic.nodes))
    # Partial gang (fewer replicas than the gang count) is rejected.
    partial = {name: {node_uid: 1}}
    assert not kernel._c4_integer_recheck(state.symbolic, wl, partial)
    # Over-binding a node beyond its healthy GPU capacity is rejected.
    over = {name: {node_uid: 999}}
    assert not kernel._c4_integer_recheck(state.symbolic, wl, over)


def test_c4_accepts_the_solvers_own_assignment(
    kernel, demo_state, queued_workload_uid
):
    state = copy.deepcopy(demo_state)
    wl = state.symbolic.workloads[queued_workload_uid]
    res = kernel.feasible(state, _aap(state, queued_workload_uid))
    assert res.feasible
    # The placement solver's returned integer assignment passes the C4 recheck.
    assert kernel._c4_integer_recheck(state.symbolic, wl, res.assignment)
