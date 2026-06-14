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
