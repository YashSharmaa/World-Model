# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""DE-UNIFY-KERNEL-AND-RARE-EVENTS — real kernel + one rare-event path.

Verifies the doc-R [D3]/[Eq1]/[T2] kernel-unification and the single unified
rare-event injection path (doc D rare_event_library; doc R rare_event_simulation):

* the generator's :func:`feasible` / :func:`step` are **byte-identical** to the
  deployed :class:`ExactSymbolicKernel` (so generator and shield certify the same
  safe set, doc R T2);
* a scenario's ``cfg.rare_event`` is translated into one scheduled onset applied
  during the rollout via the simulator state-mutating library, and that onset
  propagates causally (fault burst → XID hazard; cooling derate → lower headroom;
  queue burst → pods-ready pressure); and
* the rollout stays deterministic and the rare event is applied identically to
  every paired branch (common random numbers).
"""

from __future__ import annotations

import gpu_center_world_model.data_engine.synthetic as S  # noqa: N812
from gpu_center_world_model.data_engine.synthetic.counterfactual import (
    RareEventSpec,
    _apply_rare_onset,
    _onset_step,
    _zone_derated,
    _zone_fault_present,
)
from gpu_center_world_model.schema.kpis import HazardChannel, KPIChannel
from gpu_center_world_model.simulator.exact_kernel import ExactSymbolicKernel
from gpu_center_world_model.utils.seeding import SeedBundle


def _state():
    pod, state, uid = S.generate_pod_state(seed=0, gpu_demand=16)
    return pod, state, uid


# ── kernel unification: generator feasible/step == real shield ──────────────
def test_generator_feasible_is_byte_identical_to_kernel():
    _, state, uid = _state()
    actions = S.default_action_catalog(state, uid)
    kernel = ExactSymbolicKernel()
    assert len(actions) >= 3
    for a in actions:
        rk = kernel.feasible(state, a)
        rg = S.feasible(state, a)
        assert rk.feasible == rg.feasible
        assert [b.group for b in rk.blockers] == [b.group for b in rg.blockers]
        assert rk.feasible_domains == rg.feasible_domains


def test_generator_step_bindings_identical_to_kernel():
    _, state, uid = _state()
    actions = S.default_action_catalog(state, uid)
    kernel = ExactSymbolicKernel()
    for a in actions:
        sg = S.step(state, a)
        sk = kernel.step(state, a)
        bound_g = sorted(g for g, gp in sg.symbolic.gpus.items() if gp.allocated_to)
        bound_k = sorted(g for g, gp in sk.symbolic.gpus.items() if gp.allocated_to)
        assert bound_g == bound_k


# ── unified rare-event onset path ───────────────────────────────────────────
def test_onset_step_schedule():
    spec0 = RareEventSpec("cooling_derate", onset_s=0.0)
    spec_mid = RareEventSpec("cooling_derate", onset_s=300.0)
    assert _onset_step(spec0, 30.0, 60) == 0
    assert _onset_step(spec_mid, 30.0, 60) == 10
    # a too-late onset clamps to the last step (never silently dropped).
    assert _onset_step(RareEventSpec("x", onset_s=1e9), 30.0, 60) == 59


def test_rare_onset_uses_simulator_library_and_mutates_state():
    _, state, _ = _state()
    spec = S.rare_event_spec_for("gpu_fault_burst")
    post = _apply_rare_onset(spec, state, SeedBundle(7), step_i=0)
    # the simulator library flips GPUs into a fault state (the shared parent).
    from gpu_center_world_model.data_engine.synthetic.scm import PodSpec, build_pod

    pod = build_pod(PodSpec(n_racks=2))
    assert _zone_fault_present(post, pod)
    # cooling derate perturbs the shared zone first (doc R causal-consistency).
    cool = _apply_rare_onset(
        S.rare_event_spec_for("cooling_derate"), state, SeedBundle(7), step_i=0
    )
    assert _zone_derated(cool)


def test_fault_burst_raises_xid_hazard():
    # The gpu_fault_burst onset masks GPUs exogenously, so it raises the XID
    # hazard on every branch that runs the rollout (incl. hold) regardless of the
    # branch's placement feasibility — the burst is not gated on active workloads.
    normal = S.generate_pack(seed=7, scenario="normal")
    fault = S.generate_pack(seed=7, scenario="fault_stress")

    def xid(pack):
        return max(
            b.hazards.get(HazardChannel.XID_FAULT.value, 0.0)
            for b in pack.branches
            if b.feasible
        )

    assert xid(fault) > xid(normal)


def test_cooling_derate_lowers_headroom():
    normal = S.generate_pack(seed=7, scenario="normal")
    thermal = S.generate_pack(seed=7, scenario="thermal_stress")

    def headroom(pack):
        return min(
            b.kpis.get(KPIChannel.RACK_THERMAL_HEADROOM_C.value, 99.0)
            for b in pack.branches
            if b.feasible
        )

    assert headroom(thermal) < headroom(normal)


def test_rare_event_rollout_is_deterministic():
    a = S.generate_pack(seed=11, scenario="fault_stress")
    b = S.generate_pack(seed=11, scenario="fault_stress")
    assert [br.hazards for br in a.branches] == [br.hazards for br in b.branches]
    assert [br.kpis for br in a.branches] == [br.kpis for br in b.branches]


def test_rare_event_applied_identically_across_paired_branches():
    # Every feasible branch that runs the rollout must see the SAME exogenous
    # event (CRN). The gpu_fault_burst onset masks GPUs identically across paired
    # branches, so the XID flag is raised on every feasible branch.
    fault = S.generate_pack(seed=3, scenario="fault_stress")
    feas = [b for b in fault.branches if b.feasible]
    assert feas, "expected at least one feasible branch"
    xids = {b.hazards.get(HazardChannel.XID_FAULT.value, 0.0) for b in feas}
    assert xids == {1.0}


def test_normal_scenario_has_no_rare_event_onset():
    # a normal scenario translates to no rare-event spec (single-path: None).
    assert S.rare_event_spec_for(None) is None
    assert S.rare_event_spec_for("not_a_family") is None
    spec = S.rare_event_spec_for("queue_burst")
    assert spec is not None and spec.name == "queue_burst"
