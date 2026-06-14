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


# ──────────────────────────────────────────────────────────────────────────────
# Safety shield projection Pi_s — soundness / idempotence / minimal-perturbation
# (TEST-KERNEL-INGEST-SERVE-FE, doc S Theorems T10/T11/T12, Definitions D11-D13)
# ──────────────────────────────────────────────────────────────────────────────
def test_project_is_sound_returns_member_of_safe_set_T10(
    kernel, demo_state, queued_workload_uid
):
    # T10: when S_s(w) is non-empty, Pi_s(proposal) must lie in S_s(w). We feed a
    # proposal the kernel rejects (a masked/garbage domain under `required`
    # locality) and assert the *repaired* action is individually feasible — the
    # constructive zero-violation basis (doc S, Theorem T10).
    state = copy.deepcopy(demo_state)
    bad = AdmitAndPlace(
        workload_uid=queued_workload_uid,
        cluster_queue_id=_cq(state, queued_workload_uid),
        resource_flavor_id=_flavor(state),
        domain_policy="required",
        domain_id_or_set=("no-such-domain",),  # cannot host the gang -> infeasible
    )
    assert not kernel.feasible(state, bad).feasible  # the proposal really is bad
    projected = kernel.project(state, bad)
    assert projected is not None
    # Soundness: the projection is a member of the exact safe set.
    assert kernel.feasible(state, projected).feasible
    # And it is drawn from the kernel-certified candidate set (never enlarged).
    safe = kernel.safe_candidates(state, queued_workload_uid)
    assert any(
        kernel._distance(projected, c) == (0, 0, 0, 0) for c in safe
    ), "projection must be one of the enumerated safe candidates"


def test_project_is_idempotent_T11(kernel, demo_state, queued_workload_uid):
    # T11: Pi_s(Pi_s(alpha)) == Pi_s(alpha). A feasible action projects to itself
    # (distance zero), and re-projecting a projection is a fixed point.
    state = copy.deepcopy(demo_state)
    feasible_action = AdmitAndPlace(
        workload_uid=queued_workload_uid,
        cluster_queue_id=_cq(state, queued_workload_uid),
        resource_flavor_id=_flavor(state),
        domain_policy="preferred",
    )
    assert kernel.feasible(state, feasible_action).feasible
    once = kernel.project(state, feasible_action)
    # A feasible proposal projects to *itself* (zero lexicographic distance, T11).
    assert once is feasible_action
    # Now repair an infeasible proposal, then re-project the repair: fixed point.
    bad = AdmitAndPlace(
        workload_uid=queued_workload_uid,
        cluster_queue_id=_cq(state, queued_workload_uid),
        resource_flavor_id=_flavor(state),
        domain_policy="required",
        domain_id_or_set=("no-such-domain",),
    )
    repaired = kernel.project(state, bad)
    assert repaired is not None
    twice = kernel.project(state, repaired)
    # Idempotence: a projected action is a fixed point of the projection.
    assert kernel._distance(twice, repaired) == (0, 0, 0, 0)


def test_project_minimal_perturbation_prefers_same_type_queue_flavor_T12(
    kernel, demo_state, queued_workload_uid
):
    # T12: the projection is the *nearest* feasible action under the lexicographic
    # action distance (type, queue, flavor, topology-edit). The repair of a bad
    # admit_and_place must therefore stay an admit_and_place on the same (q, f)
    # whenever such a safe action exists — only the topology selection is edited.
    state = copy.deepcopy(demo_state)
    q = _cq(state, queued_workload_uid)
    f = _flavor(state)
    bad = AdmitAndPlace(
        workload_uid=queued_workload_uid,
        cluster_queue_id=q,
        resource_flavor_id=f,
        domain_policy="required",
        domain_id_or_set=("no-such-domain",),
    )
    projected = kernel.project(state, bad)
    assert projected is not None
    # Minimal perturbation: same action type, same queue, same flavor are all
    # preserved (the three highest-priority components of dist are zero).
    type_diff, queue_diff, flavor_diff, _topo = kernel._distance(bad, projected)
    assert type_diff == 0, "nearest repair keeps the same action type"
    assert queue_diff == 0, "nearest repair keeps the same cluster queue"
    assert flavor_diff == 0, "nearest repair keeps the same resource flavor"
    # It is the global argmin over the safe set under the lexicographic metric.
    safe = kernel.safe_candidates(state, queued_workload_uid)
    best = min(kernel._distance(bad, c) for c in safe)
    assert kernel._distance(bad, projected) == best


def test_project_abstains_when_even_hold_is_unavailable(
    kernel, demo_state, queued_workload_uid
):
    # When S_s(w) is empty (no whitelisted action, not even hold), Pi_s abstains
    # by returning None rather than emitting an unsafe action (doc S step 6).
    state = copy.deepcopy(demo_state)
    pol = state.symbolic.operator_policy
    state.symbolic.operator_policy = type(pol)(
        policy_id=pol.policy_id,
        allowed_actions=[],  # nothing whitelisted, not even hold
        protected_domains=list(pol.protected_domains),
        risk_thresholds=dict(pol.risk_thresholds),
        active_since_ts=pol.active_since_ts,
    )
    bad = AdmitAndPlace(
        workload_uid=queued_workload_uid,
        cluster_queue_id=_cq(state, queued_workload_uid),
        resource_flavor_id=_flavor(state),
        domain_policy="required",
        domain_id_or_set=("no-such-domain",),
    )
    assert kernel.project(state, bad) is None


def test_project_falls_back_to_hold_when_no_placement_is_safe(
    kernel, demo_state, queued_workload_uid
):
    # When placement is impossible but hold is whitelisted, the nearest safe
    # action is hold (doc S step 6: hold weakly dominates irrevocable reject).
    from gpu_center_world_model.schema import ActionType

    state = copy.deepcopy(demo_state)
    pol = state.symbolic.operator_policy
    # Mask every domain so admit_and_place has no feasible target, leave hold on.
    state.symbolic.operator_policy = type(pol)(
        policy_id=pol.policy_id,
        allowed_actions=list(pol.allowed_actions),
        protected_domains=list(state.symbolic.domains),
        risk_thresholds=dict(pol.risk_thresholds),
        active_since_ts=pol.active_since_ts,
    )
    bad = AdmitAndPlace(
        workload_uid=queued_workload_uid,
        cluster_queue_id=_cq(state, queued_workload_uid),
        resource_flavor_id=_flavor(state),
        domain_policy="required",
        domain_id_or_set=(next(iter(state.symbolic.domains)),),
    )
    projected = kernel.project(state, bad)
    assert projected is not None
    assert projected.type is ActionType.HOLD
    assert kernel.feasible(state, projected).feasible


# ──────────────────────────────────────────────────────────────────────────────
# Quota / borrowing / lending / cohort arithmetic
# (TEST-KERNEL-INGEST-SERVE-FE, doc S Eq. E11 / E12 / E13)
# ──────────────────────────────────────────────────────────────────────────────
def _two_queue_cohort():
    """Build a 2-queue cohort sharing flavor `f0`, resource `gpu`.

    `q-borrow` has a small nominal quota; `q-lend` declares a lending limit so the
    borrower's cohort headroom is bounded by both unused cohort nominal and the
    lender's lending cap (doc S, Eq. E13).
    """
    from gpu_center_world_model.schema.entities import ClusterQueue

    borrower = ClusterQueue(
        name="q-borrow",
        cohort_id="coh",
        nominal_quota={"f0": {"gpu": 4.0}},
        borrowing_limit={"f0": {"gpu": 8.0}},
    )
    lender = ClusterQueue(
        name="q-lend",
        cohort_id="coh",
        nominal_quota={"f0": {"gpu": 10.0}},
        lending_limit={"f0": {"gpu": 6.0}},
    )
    return borrower, lender


def test_E11_demand_within_nominal_needs_no_borrow():
    # E11: d_wr <= U_nom_qfr -> admissible with beta = 0 (no borrowing recorded).
    from gpu_center_world_model.simulator.exact_kernel import quota as q

    borrower, lender = _two_queue_cohort()
    cohort = [borrower, lender]
    dec = q.quota_check(borrower, cohort, "f0", {"gpu": 4.0})  # exactly nominal
    assert dec.feasible
    assert dec.borrow.get("gpu", 0.0) == 0.0
    assert dec.borrowed_resources == []


def test_E11_E12_borrowing_is_exact_and_bounded_by_cohort_headroom():
    # E11/E12: a demand above nominal borrows exactly need = d - U_nom, and that
    # borrow is admissible only because it is within min(borrowing_limit,
    # cohort_headroom). Borrower nominal=4, demand=9 -> need=5; cohort headroom is
    # min(lender unused nominal 10, lender lending limit 6) = 6, borrowing_limit
    # 8, so cap = min(8, 6) = 6 >= 5 -> feasible with beta_gpu = 5.
    from gpu_center_world_model.simulator.exact_kernel import quota as q

    borrower, lender = _two_queue_cohort()
    cohort = [borrower, lender]
    assert q.cohort_headroom(borrower, cohort, "f0", "gpu") == 6.0
    dec = q.quota_check(borrower, cohort, "f0", {"gpu": 9.0})
    assert dec.feasible
    assert dec.borrow["gpu"] == 5.0  # exact need = 9 - 4
    assert dec.borrowed_resources == ["gpu"]


def test_E12_E13_lending_limit_caps_cohort_borrow_and_yields_deficit():
    # E13: the cohort cap is the conservative min of unused cohort nominal and the
    # lenders' aggregate lending limit. The lender lends at most 6, so a demand
    # needing 7 of borrow (nominal 4, demand 11) cannot be satisfied even though
    # borrowing_limit (8) alone would allow it -> infeasible with exact deficit.
    from gpu_center_world_model.simulator.exact_kernel import quota as q

    borrower, lender = _two_queue_cohort()
    cohort = [borrower, lender]
    dec = q.quota_check(borrower, cohort, "f0", {"gpu": 11.0})  # need = 7 > cap 6
    assert not dec.feasible
    assert "gpu" not in dec.borrow
    # deficit = need - cap = 7 - 6 = 1 (doc S feasibility-relaxation shortage).
    assert dec.deficit["gpu"] == 1.0


def test_E13_lending_drops_when_lender_consumes_its_own_quota():
    # Lending headroom is *conservative*: as the lender reserves its own nominal
    # quota, the unused-cohort-nominal term of E13 shrinks, so the borrower's
    # cohort headroom can fall below the static lending_limit. Reserving 8 of the
    # lender's 10 nominal leaves unused cohort nominal = 2, so cohort headroom =
    # min(2, lending 6) = 2 (down from 6).
    from gpu_center_world_model.simulator.exact_kernel import quota as q

    borrower, lender = _two_queue_cohort()
    cohort = [borrower, lender]
    q.apply_admission(lender, 8.0, 0.0, flavor_id="f0", demand={"gpu": 8.0}, borrow={})
    assert q.reserved_for(lender, "f0", "gpu") == 8.0
    assert q.cohort_headroom(borrower, cohort, "f0", "gpu") == 2.0
    # A borrow of need=5 (demand 9, nominal 4) now exceeds the 2-GPU cohort cap.
    dec = q.quota_check(borrower, cohort, "f0", {"gpu": 9.0})
    assert not dec.feasible
    assert dec.deficit["gpu"] == 3.0  # need 5 - cohort cap 2


def test_singleton_cohort_cannot_borrow():
    # A queue with no cohort_id is its own singleton cohort: it can neither borrow
    # nor lend, so a demand above its nominal quota is infeasible (Kueue cohort
    # semantics; doc S resolve_cohort).
    from gpu_center_world_model.schema.entities import ClusterQueue
    from gpu_center_world_model.simulator.exact_kernel import quota as q

    solo = ClusterQueue(
        name="solo",
        cohort_id=None,
        nominal_quota={"f0": {"gpu": 4.0}},
        borrowing_limit={"f0": {"gpu": 8.0}},
    )
    cohort = q.resolve_cohort({"solo": solo}, solo)
    assert cohort == [solo]
    assert q.cohort_headroom(solo, cohort, "f0", "gpu") == 0.0
    dec = q.quota_check(solo, cohort, "f0", {"gpu": 6.0})  # 2 over nominal, no lend
    assert not dec.feasible
    assert dec.deficit["gpu"] == 2.0


def test_resolve_cohort_groups_only_same_cohort_id():
    # resolve_cohort returns exactly the queues sharing the cohort id (incl self),
    # so cohort arithmetic never pulls in an unrelated cohort's lendable quota.
    from gpu_center_world_model.schema.entities import ClusterQueue
    from gpu_center_world_model.simulator.exact_kernel import quota as q

    a = ClusterQueue(name="a", cohort_id="coh-1")
    b = ClusterQueue(name="b", cohort_id="coh-1")
    c = ClusterQueue(name="c", cohort_id="coh-2")
    queues = {"a": a, "b": b, "c": c}
    cohort = q.resolve_cohort(queues, a)
    names = {x.name for x in cohort}
    assert names == {"a", "b"}  # c is in a different cohort, excluded


def test_workload_demand_aggregates_podset_counts():
    # d_wr = sum_p c_p * a_pr (doc S, Eq. E11): the aggregate per-resource demand
    # is the count-weighted sum across podsets, not a re-learned estimate.
    from gpu_center_world_model.simulator.exact_kernel import quota as q

    demand = q.workload_demand(
        [
            (4, {"gpu": 1.0, "cpu": 8.0, "mem": 64.0}),
            (2, {"gpu": 0.0, "cpu": 4.0, "mem": 16.0}),
        ]
    )
    assert demand["gpu"] == 4.0  # 4*1 + 2*0
    assert demand["cpu"] == 40.0  # 4*8 + 2*4
    assert demand["mem"] == 288.0  # 4*64 + 2*16
