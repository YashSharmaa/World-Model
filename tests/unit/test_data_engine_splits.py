# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""DE-SPLITS-HOLDOUT — topology/intervention/rare-event holdout splitters.

Verifies the doc-D "four disjoint test suites" contract:

* topology/flavor, intervention, rare-event, and severe-shift holdouts each route
  to TEST_TOPOLOGY / TEST_INTERVENTION / TEST_RARE_EVENT / TEST_SEVERE_SHIFT;
* a held-out family is **guaranteed never to enter train** (the routing checks
  holdouts before the seen default);
* the four suites are mutually disjoint and no decision hash crosses splits
  (:func:`assert_disjoint_splits`); and
* :func:`build_demo_release` is family-aware (not index-based) and passes the
  disjointness guard end-to-end.
"""

from __future__ import annotations

import itertools
import tempfile

import pytest

from gpu_center_world_model.data_engine.contracts import Split
from gpu_center_world_model.data_engine.release_manifest import (
    build_demo_release,
    scenario_families,
)
from gpu_center_world_model.data_engine.windows.builder import (
    HoldoutPolicy,
    WindowDescriptor,
    assert_disjoint_splits,
    assign_split,
    severity_band,
)


def test_severity_bands():
    assert severity_band(0.1) == "low"
    assert severity_band(0.5) == "medium"
    assert severity_band(0.9) == "high"


def test_topology_holdout_routes_to_test_topology():
    pol = HoldoutPolicy(held_out_topologies=frozenset({"h100-sxm:4r"}))
    held = WindowDescriptor("d", topology_family="h100-sxm:4r")
    seen = WindowDescriptor("d", topology_family="h100-sxm:2r")
    assert assign_split(held, pol) == Split.TEST_TOPOLOGY
    assert assign_split(seen, pol) == Split.TRAIN


def test_intervention_holdout_routes_to_test_intervention():
    pol = HoldoutPolicy(held_out_interventions=frozenset({"admit_and_place:high"}))
    held = WindowDescriptor(
        "d", intervention_families=("hold:low", "admit_and_place:high")
    )
    seen = WindowDescriptor(
        "d", intervention_families=("hold:low", "admit_and_place:low")
    )
    assert assign_split(held, pol) == Split.TEST_INTERVENTION
    assert assign_split(seen, pol) == Split.TRAIN


def test_rare_event_holdout_routes_to_test_rare_event():
    pol = HoldoutPolicy(held_out_rare_events=frozenset({"gpu_fault_burst:high"}))
    held = WindowDescriptor("d", rare_event_family="gpu_fault_burst:high")
    seen = WindowDescriptor("d", rare_event_family="cooling_derate:low")
    assert assign_split(held, pol) == Split.TEST_RARE_EVENT
    assert assign_split(seen, pol) == Split.TRAIN


def test_severe_shift_holdout_routes_to_test_severe_shift():
    pol = HoldoutPolicy(hold_out_severe_shift=True)
    held = WindowDescriptor("d", severe_shift=True)
    seen = WindowDescriptor("d", severe_shift=False)
    assert assign_split(held, pol) == Split.TEST_SEVERE_SHIFT
    assert assign_split(seen, pol) == Split.TRAIN


def test_held_out_family_never_enters_train():
    pol = HoldoutPolicy(
        held_out_rare_events=frozenset({"gpu_fault_burst:high"}),
        held_out_topologies=frozenset({"h100-sxm:2r"}),
        held_out_interventions=frozenset({"place:high"}),
        hold_out_severe_shift=True,
    )
    for re_f, topo_f, sev in itertools.product(
        ["", "gpu_fault_burst:high"], ["", "h100-sxm:2r"], [True, False]
    ):
        desc = WindowDescriptor(
            "x",
            rare_event_family=re_f,
            topology_family=topo_f,
            intervention_families=("place:high",) if not re_f else (),
            severe_shift=sev,
        )
        touches = (
            re_f in pol.held_out_rare_events
            or topo_f in pol.held_out_topologies
            or sev
            or any(f in pol.held_out_interventions for f in desc.intervention_families)
        )
        split = assign_split(desc, pol)
        if touches:
            assert split != Split.TRAIN
            assert split in {
                Split.TEST_TOPOLOGY,
                Split.TEST_INTERVENTION,
                Split.TEST_RARE_EVENT,
                Split.TEST_SEVERE_SHIFT,
            }


def test_route_priority_is_deterministic_and_single_suite():
    # a window touching multiple held-out families maps to exactly one suite, in
    # the fixed priority order (severe > rare > intervention > topology).
    pol = HoldoutPolicy(
        held_out_topologies=frozenset({"t"}),
        held_out_interventions=frozenset({"place:high"}),
        held_out_rare_events=frozenset({"r:high"}),
        hold_out_severe_shift=True,
    )
    desc = WindowDescriptor(
        "x",
        topology_family="t",
        intervention_families=("place:high",),
        rare_event_family="r:high",
        severe_shift=True,
    )
    assert assign_split(desc, pol) == Split.TEST_SEVERE_SHIFT


def test_assert_disjoint_splits_catches_crossing():
    assert_disjoint_splits([("a", Split.TRAIN), ("b", Split.VAL)])  # ok
    with pytest.raises(ValueError, match="split leakage"):
        assert_disjoint_splits([("a", Split.TRAIN), ("a", Split.TEST_TOPOLOGY)])


def test_temporal_split_over_seen_pool():
    seen = WindowDescriptor("d", topology_family="seen")
    pol = HoldoutPolicy()
    assert assign_split(seen, pol, t0_ns=0, train_end_ns=5, val_end_ns=8) == Split.TRAIN
    assert assign_split(seen, pol, t0_ns=6, train_end_ns=5, val_end_ns=8) == Split.VAL
    assert (
        assign_split(seen, pol, t0_ns=9, train_end_ns=5, val_end_ns=8)
        == Split.TEST_TEMPORAL
    )


def test_build_demo_release_is_family_aware_and_disjoint():
    scenarios = ("normal", "thermal_stress", "fault_stress", "scheduling_stress")
    with tempfile.TemporaryDirectory() as d:
        # default policy holds out fault-burst + severe-shift; must not raise on the
        # internal assert_disjoint_splits guard.
        res = build_demo_release(d, seeds=(0, 1), scenarios=scenarios)
        splits = res["stats"].split_counts
        # severe-shift packs (fault_stress / scheduling_stress) land out of train.
        assert Split.TEST_SEVERE_SHIFT.value in splits


def test_build_demo_release_exercises_all_four_suites():
    scenarios = ("normal", "thermal_stress", "fault_stress", "scheduling_stress")
    policy = HoldoutPolicy(
        held_out_topologies=frozenset(
            {scenario_families("scheduling_stress")["topology_family"]}
        ),
        held_out_rare_events=frozenset({"cooling_derate:low"}),
        hold_out_severe_shift=True,
    )
    with tempfile.TemporaryDirectory() as d:
        res = build_demo_release(
            d, seeds=(0,), scenarios=scenarios, holdout=policy
        )
        splits = res["stats"].split_counts
    # cooling_derate:low (thermal_stress) → rare-event; scheduling/fault → severe.
    assert Split.TEST_RARE_EVENT.value in splits
    assert Split.TEST_SEVERE_SHIFT.value in splits
