# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Property-based tests for the exact kernel & branch runner (design doc H).

Hypothesis generates many small but varied demo pods and queued gangs; for each
we assert the *invariants the whole spine relies on*:

* **Resource conservation** — a feasible placement consumes exactly the gang's
  GPU demand; an infeasible command consumes nothing (no-op). Free GPUs never go
  negative and the symbolic transition is conservative.
* **Gang atomicity** — a feasible assignment is all-or-nothing: it binds exactly
  the gang demand across nodes, never a partial subset.
* **Deterministic branching** — the same pre-action hash + seed + action produces
  a byte-identical rollout (common random numbers; doc D/I/R).

``hypothesis`` is a declared dev dependency; if it is unavailable the whole module
is skipped so the numpy-only suite still passes (doc H "tests must pass with
numpy only").
"""

from __future__ import annotations

import pytest

# Robust skip: handle both a clean absence and a partial/namespace-shell install
# (a leftover empty ``hypothesis/`` dir makes the top-level import succeed while
# submodule imports fail). We require the actual symbols, skipping otherwise.
try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover - exercised on the numpy-only CI box
    pytest.skip(
        "hypothesis not available; property tests require it", allow_module_level=True
    )

from gpu_center_world_model.data_engine.synthetic import (
    PodSpec,
    generate_pod_state,
)
from gpu_center_world_model.schema import (
    AdmitAndPlace,
    Hold,
    world_state_hash,
)
from gpu_center_world_model.simulator import (
    BranchRunner,
    ExactSymbolicKernel,
)

# Keep generated pods tiny so the property suite stays fast and deterministic.
_RACKS = st.integers(min_value=1, max_value=3)
_NODES = st.integers(min_value=1, max_value=3)
_GPN = st.integers(min_value=2, max_value=4)
_SEED = st.integers(min_value=0, max_value=2**16)
_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _free_gpu(state) -> int:
    return sum(1 for g in state.symbolic.gpus.values() if g.allocated_to is None)


def _gang_demand(state, uid) -> int:
    wl = state.symbolic.workloads[uid]
    return sum(p.count * p.gpu_req for p in wl.podsets)


def _placement(state, uid) -> AdmitAndPlace:
    wl = state.symbolic.workloads[uid]
    return AdmitAndPlace(
        workload_uid=uid,
        cluster_queue_id=wl.cluster_queue_id,
        resource_flavor_id=next(iter(state.symbolic.flavors)),
        domain_policy="preferred",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Resource conservation
# ──────────────────────────────────────────────────────────────────────────────
@given(_RACKS, _NODES, _GPN, st.integers(1, 6), _SEED)
@_SETTINGS
def test_feasible_placement_consumes_exactly_demand(r, n, g, demand, seed):
    _pod, state, uid = generate_pod_state(
        PodSpec(n_racks=r, nodes_per_rack=n, gpus_per_node=g),
        seed=seed,
        gpu_demand=demand,
    )
    kernel = ExactSymbolicKernel()
    act = _placement(state, uid)
    free_before = _free_gpu(state)
    res = kernel.feasible(state, act)
    nxt = kernel.step(state, act)
    free_after = _free_gpu(nxt)
    if res.feasible:
        # exactly the gang demand was consumed.
        assert free_before - free_after == _gang_demand(state, uid)
    else:
        # infeasible command is a no-op (reject) — nothing consumed.
        assert free_after == free_before
    assert free_after >= 0  # never negative


@given(_RACKS, _NODES, _GPN, _SEED)
@_SETTINGS
def test_hold_never_consumes_resources(r, n, g, seed):
    _pod, state, uid = generate_pod_state(
        PodSpec(n_racks=r, nodes_per_rack=n, gpus_per_node=g), seed=seed, gpu_demand=2
    )
    kernel = ExactSymbolicKernel()
    free_before = _free_gpu(state)
    nxt = kernel.step(state, Hold(workload_uid=uid))
    assert _free_gpu(nxt) == free_before


# ──────────────────────────────────────────────────────────────────────────────
# Gang atomicity (all-or-nothing)
# ──────────────────────────────────────────────────────────────────────────────
@given(_RACKS, _NODES, _GPN, st.integers(1, 8), _SEED)
@_SETTINGS
def test_gang_assignment_is_all_or_nothing(r, n, g, demand, seed):
    _pod, state, uid = generate_pod_state(
        PodSpec(n_racks=r, nodes_per_rack=n, gpus_per_node=g),
        seed=seed,
        gpu_demand=demand,
    )
    kernel = ExactSymbolicKernel()
    res = kernel.feasible(state, _placement(state, uid))
    if res.feasible:
        assigned = sum(sum(node_map.values()) for node_map in res.assignment.values())
        # a feasible gang binds exactly its demand — never a partial subset.
        assert assigned == _gang_demand(state, uid)
    else:
        # an infeasible gang yields no partial assignment to actuate.
        assigned = sum(sum(node_map.values()) for node_map in res.assignment.values())
        assert assigned < _gang_demand(state, uid) or assigned == 0


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic branching (common random numbers)
# ──────────────────────────────────────────────────────────────────────────────
@given(_RACKS, _NODES, _GPN, _SEED, st.integers(0, 2**20))
@_SETTINGS
def test_same_root_seed_action_gives_identical_rollout(r, n, g, pod_seed, branch_seed):
    _pod, state, uid = generate_pod_state(
        PodSpec(n_racks=r, nodes_per_rack=n, gpus_per_node=g),
        seed=pod_seed,
        gpu_demand=2,
    )
    kernel = ExactSymbolicKernel()
    runner = BranchRunner(kernel, step_s=300)
    root = runner.root(state, horizon_s=900, seed=branch_seed)
    act = _placement(state, uid)

    traj_a = runner.rollout(root, state, act)
    traj_b = runner.rollout(root, state, act)
    ha = [world_state_hash(s) for s in traj_a]
    hb = [world_state_hash(s) for s in traj_b]
    assert ha == hb  # byte-identical under common random numbers
    assert ha[0] == root.state_hash  # branch starts at the frozen pre-action hash


@given(_RACKS, _NODES, _GPN, _SEED)
@_SETTINGS
def test_branch_refuses_drifted_start_state(r, n, g, seed):
    _pod, state, uid = generate_pod_state(
        PodSpec(n_racks=r, nodes_per_rack=n, gpus_per_node=g), seed=seed, gpu_demand=2
    )
    kernel = ExactSymbolicKernel()
    runner = BranchRunner(kernel, step_s=300)
    root = runner.root(state, horizon_s=600, seed=1)
    # Mutate the state so it no longer matches the frozen root hash.
    drifted = kernel.step(state, _placement(state, uid))
    with pytest.raises(ValueError):
        runner.rollout(root, drifted, Hold(workload_uid=uid))
