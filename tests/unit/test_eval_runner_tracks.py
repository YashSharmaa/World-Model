# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Scoped tests for EVAL-RUNNER-TRACKS (doc F three-track wiring + ingestion).

These live inside the owned ``evaluation/reports`` package (the repo pins
``testpaths=["tests"]`` so a bare ``pytest`` does not collect them, keeping other
lanes undisturbed). Run them explicitly (path elided for line length)::

    python3 -m pytest .../evaluation/reports/test_eval_runner_tracks.py -q

They cover, per the EVAL-RUNNER-TRACKS spec:

* ``track_inputs_from_records`` round-trips a saved pack (payload mapping AND a
  flat tagged-record list) into a populated :class:`TrackInputs`;
* the runner wires predictive (quantile + hazard PR-at-budget), calibration ECE,
  coverage-by-slice, and operator-usefulness tracks;
* the operator-usefulness metrics and the doc-O L10 ledger binding;
* per-slice power certification against ``MIN_EFFECTIVE_WINDOWS_PER_SLICE``;
* the counterfactual ``coverage>=0.60`` gate and the 85%-95% interval band check;
* the hardened red-team predicates FAIL when a required trigger flag is absent,
  and the scenario-case generator constructs properly-flagged passing cases.
"""

from __future__ import annotations

import math

import pytest

from gpu_center_world_model.evaluation.metrics.operator_usefulness import (
    OperatorCase,
    evaluate_operator_usefulness,
)
from gpu_center_world_model.evaluation.metrics.predictive import (
    HAZARD_CHANNELS,
    score_hazard_channel,
)
from gpu_center_world_model.evaluation.reports.provable_vs_empirical import (
    _CLAIM_BINDINGS,
    generate_ledger,
)
from gpu_center_world_model.evaluation.runner import (
    THRESHOLDS,
    TrackInputs,
    apply_acceptance,
    compute_slices,
    counterfactual_coverage,
    run_benchmark,
    track_inputs_from_records,
)
from gpu_center_world_model.evaluation.slice_defs import (
    MIN_EFFECTIVE_WINDOWS_PER_SLICE,
)
from gpu_center_world_model.evaluation.suites.redteam import (
    RED_TEAM_SCENARIOS,
    evaluate_red_team,
    generate_red_team_suite,
    make_red_team_case,
)
from gpu_center_world_model.simulator.contracts import BlockerGroup


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: a small, fully-populated saved pack payload
# ──────────────────────────────────────────────────────────────────────────────
def _full_payload() -> dict:
    return {
        "n_branches": 4,
        "rho_pair": 0.4,
        "tau_mix": 2.0,
        "n_severe_shift_zero_events": 300,
        "safety_judgements": [
            {
                "action_id": "a1",
                "recommended": True,
                "abstained": False,
                "kernel_feasible": True,
            },
            {
                "action_id": "a2",
                "recommended": False,
                "abstained": True,
                "kernel_feasible": False,
                "kernel_blockers": ["topology_rule"],
                "emitted_blockers": ["topology_rule"],
            },
        ],
        "branch_windows": [
            {
                "window_id": "w1",
                "action_ids": ["hold", "admit", "place"],
                "pred_energy": {"hold": 1.0, "admit": 0.5, "place": 0.8},
                "true_energy": {"hold": 1.0, "admit": 0.4, "place": 0.9},
                "recommended": {"admit": True},
                "ref_unsafe": {"admit": False},
                "pred_risk": {"admit": 0.1},
            },
            {
                "window_id": "w2",
                "action_ids": ["hold", "admit"],
                "pred_energy": {"hold": 2.0, "admit": 1.0},
                "true_energy": {"hold": 2.0, "admit": 1.2},
                "recommended": {"admit": True},
                "ref_unsafe": {"admit": False},
                "pred_risk": {"admit": 0.2},
            },
        ],
        "window_features": [
            {
                "action_family": "admit",
                "horizon_s": 1800,
                "workload_family": "pipeline",
                "topology_family": "intra_rack",
                "source_health": "observed",
            },
            {
                "action_family": "admit",
                "horizon_s": 1800,
                "workload_family": "single_pod",
                "topology_family": "intra_node",
                "source_health": "observed",
            },
        ],
        "planning_windows": [
            {
                "window_id": "w1",
                "chosen_action_id": "admit",
                "true_lex_keys": {"hold": [0.0, 1.0], "admit": [0.0, 0.4]},
                "safe_action_ids": ["hold", "admit"],
            }
        ],
        "baseline_planning_windows": [
            {
                "window_id": "w1",
                "chosen_action_id": "hold",
                "true_lex_keys": {"hold": [0.0, 1.0], "admit": [0.0, 0.4]},
                "safe_action_ids": ["hold", "admit"],
            }
        ],
        "quantile_channel_inputs": [
            {
                "channel": "queue_depth",
                "horizon_s": 1800,
                "truth": [1.0, 2.0, 3.0, 4.0],
                "quantile_preds": {
                    "0.05": [0.5, 1.5, 2.5, 3.5],
                    "0.5": [1.1, 2.1, 2.9, 4.1],
                    "0.95": [1.5, 2.5, 3.5, 4.5],
                },
            }
        ],
        "hazard_channel_inputs": [
            {
                "channel": "pods_ready_timeout",
                "horizon_s": 1800,
                "prob": [0.9, 0.1, 0.8, 0.2, 0.7],
                "label": [1, 0, 1, 0, 0],
            }
        ],
        "calibration_inputs": [
            {
                "channel": "hotspot",
                "slice_name": "topology_shift",
                "prob": [0.1, 0.2, 0.8, 0.9, 0.5],
                "label": [0, 0, 1, 1, 1],
            }
        ],
        "coverage_by_slice_inputs": {
            "lower": [0.0, 0.0, 0.0, 0.0],
            "upper": [2.0, 3.0, 4.0, 5.0],
            "truth": [1.0, 2.0, 3.0, 4.0],
            "slice_labels": ["topology=A", "topology=A", "topology=B", "topology=B"],
        },
        "operator_cases": [
            {
                "case_id": "c1",
                "assistant_spoke": True,
                "assistant_correct": True,
                "operator_followed": True,
                "operator_decision_correct": True,
                "unaided_decision_correct": False,
                "decision_time_s": 10.0,
                "unaided_decision_time_s": 20.0,
                "is_blocked_case": True,
                "emitted_blockers": ["topology_rule"],
                "operator_blockers": ["topology_rule"],
            },
            {
                "case_id": "c2",
                "assistant_spoke": True,
                "assistant_correct": False,
                "operator_followed": False,
                "operator_decision_correct": True,
                "unaided_decision_correct": True,
                "decision_time_s": 12.0,
                "unaided_decision_time_s": 18.0,
            },
        ],
        "shift_abstained_or_fellback": [1.0] * 5,
        "shift_false_safe": [0.0] * 5,
    }


# ──────────────────────────────────────────────────────────────────────────────
# track_inputs_from_records — payload + flat-record deserialization
# ──────────────────────────────────────────────────────────────────────────────
def test_deserialize_payload_populates_all_tracks() -> None:
    inputs = track_inputs_from_records(_full_payload())
    assert len(inputs.safety_judgements) == 2
    assert len(inputs.branch_windows) == 2
    assert len(inputs.planning_windows) == 1
    assert len(inputs.baseline_planning_windows) == 1
    assert len(inputs.quantile_channel_inputs) == 1
    assert len(inputs.hazard_channel_inputs) == 1
    assert len(inputs.calibration_inputs) == 1
    assert len(inputs.operator_cases) == 2
    assert inputs.coverage_by_slice_inputs["slice_labels"]
    # enum coercion of blocker strings
    j2 = inputs.safety_judgements[1]
    assert j2.kernel_blockers == frozenset({BlockerGroup.TOPOLOGY_RULE})


def test_deserialize_flat_tagged_records() -> None:
    records = [
        {"kind": "safety_judgement", "action_id": "a", "recommended": True,
         "abstained": False, "kernel_feasible": True},
        {"kind": "branch_window", "window_id": "w", "action_ids": ["hold", "admit"],
         "pred_energy": {"hold": 1.0, "admit": 0.5},
         "true_energy": {"hold": 1.0, "admit": 0.4}},
        {"kind": "red_team_case", "scenario": "unseen_workload_or_topology_regime",
         "case_id": "rt", "is_ood": True, "fell_back": True,
         "uncertainty_increased": True, "triggers_present": []},
    ]
    inputs = track_inputs_from_records(records)
    assert len(inputs.safety_judgements) == 1
    assert len(inputs.branch_windows) == 1
    assert "unseen_workload_or_topology_regime" in inputs.red_team_cases


def test_deserialize_unknown_kind_raises() -> None:
    with pytest.raises(ValueError):
        track_inputs_from_records([{"kind": "nonsense_record"}])


def test_track_inputs_from_records_no_longer_raises_notimplemented() -> None:
    # the v1 stub raised NotImplementedError; the wired version must not.
    out = track_inputs_from_records({})
    assert isinstance(out, TrackInputs)


# ──────────────────────────────────────────────────────────────────────────────
# Runner wiring: predictive / calibration / coverage / operator tracks
# ──────────────────────────────────────────────────────────────────────────────
def test_runner_scores_all_tracks_from_pack() -> None:
    inputs = track_inputs_from_records(_full_payload())
    inputs.red_team_cases = generate_red_team_suite(n_per_scenario=1)
    report = run_benchmark(inputs, run_id="t").as_dict()

    assert report["safety_gate"] is not None
    assert report["counterfactual"] is not None
    assert report["planning"] is not None
    # predictive: one continuous + one hazard report
    channels = {p["channel"] for p in report["predictive"]}
    assert {"queue_depth", "pods_ready_timeout"} <= channels
    # calibration scored from raw inputs
    assert any(c["channel"] == "hotspot" for c in report["calibration"])
    # coverage-by-slice + operator usefulness present
    assert report["coverage_by_slice"] is not None
    assert report["operator_usefulness"] is not None


def test_hazard_pr_at_budget_fields() -> None:
    rep = score_hazard_channel(
        "pods_ready_timeout", 1800, [0.9, 0.1, 0.8, 0.2, 0.7], [1, 0, 1, 0, 0]
    )
    assert "pods_ready_timeout" in HAZARD_CHANNELS
    assert not math.isnan(rep.brier)
    assert set(rep.pr_at_budget) == {0.01, 0.05, 0.10}
    for entry in rep.pr_at_budget.values():
        assert set(entry) == {"precision", "recall", "threshold"}
    d = rep.as_dict()
    assert "pr_at_budget" in d


# ──────────────────────────────────────────────────────────────────────────────
# Operator usefulness + L10 ledger binding
# ──────────────────────────────────────────────────────────────────────────────
def test_operator_usefulness_metrics() -> None:
    cases = [
        OperatorCase(
            case_id="g",
            assistant_spoke=True,
            assistant_correct=True,
            operator_followed=True,
            operator_decision_correct=True,
            unaided_decision_correct=False,
            decision_time_s=8.0,
            unaided_decision_time_s=20.0,
            is_blocked_case=True,
            emitted_blockers=frozenset({"topology_rule"}),
            operator_blockers=frozenset({"topology_rule"}),
        ),
        OperatorCase(
            case_id="h",
            assistant_spoke=True,
            assistant_correct=False,
            operator_followed=False,
            operator_decision_correct=True,
            unaided_decision_correct=True,
            decision_time_s=9.0,
            unaided_decision_time_s=18.0,
        ),
    ]
    rep = evaluate_operator_usefulness(cases)
    # 20s->8s and 18s->9s => median reduction well over 20%
    assert rep.decision_time.meets_time_target
    assert rep.overrides.no_safety_loss
    assert rep.passed
    assert rep.provisional  # never release-blocking
    # appropriate reliance: followed-correct + overrode-incorrect = 2/2
    assert rep.reliance.appropriate_reliance_rate == 1.0
    # explanation agreement: exact match on the one blocked case
    assert rep.explanation.exact_match_rate == 1.0


def test_l10_is_bound_in_ledger() -> None:
    assert "L10" in _CLAIM_BINDINGS
    assert _CLAIM_BINDINGS["L10"] == ("operator_usefulness",)
    results = {"operator_usefulness": {"passed": True, "provisional": True}}
    claims = {c.claim_id: c for c in generate_ledger(results)}
    assert claims["L10"].empirical_binding is not None
    assert "operator_usefulness" in claims["L10"].empirical_binding


# ──────────────────────────────────────────────────────────────────────────────
# Per-slice power certification against MIN_EFFECTIVE_WINDOWS_PER_SLICE
# ──────────────────────────────────────────────────────────────────────────────
def test_compute_slices_flags_under_powered() -> None:
    inputs = track_inputs_from_records(_full_payload())
    slices = compute_slices(inputs)
    summary = slices["_power_summary"]
    assert summary["min_effective_windows_per_slice"] == MIN_EFFECTIVE_WINDOWS_PER_SLICE
    # only 2 windows -> every cell is under-powered vs the ~750 requirement
    assert summary["n_under_powered_cells"] == summary["n_cells"]
    assert not summary["all_cells_sufficiently_powered"]
    # each cell carries a power block flagging under-power
    fam = slices["action_family"]
    for cell in fam.values():
        assert cell["power"]["under_powered"] is True


def test_compute_slices_power_certifies_when_large() -> None:
    # synthesize 2000 windows in one cell -> effective N clears 750 at tau_mix=2
    payload = _full_payload()
    bw = payload["branch_windows"][0]
    feat = payload["window_features"][0]
    payload["branch_windows"] = [
        {**bw, "window_id": f"w{i}"} for i in range(2000)
    ]
    payload["window_features"] = [dict(feat) for _ in range(2000)]
    inputs = track_inputs_from_records(payload)
    slices = compute_slices(inputs)
    cell = slices["action_family"]["action_family=admit"]
    assert cell["power"]["effective_n_windows"] >= MIN_EFFECTIVE_WINDOWS_PER_SLICE
    assert cell["power"]["sufficiently_powered"] is True


# ──────────────────────────────────────────────────────────────────────────────
# Counterfactual coverage gate + interval-coverage-band acceptance
# ──────────────────────────────────────────────────────────────────────────────
def test_counterfactual_coverage_gate_fails_below_floor() -> None:
    from gpu_center_world_model.evaluation.metrics.counterfactual import (
        CounterfactualReport,
    )

    cf = CounterfactualReport(
        n_windows=10,
        sign_accuracy=0.9,
        sign_accuracy_pairs=100,
        sign_accuracy_non_abstained=0.95,
        sign_accuracy_non_abstained_pairs=20,
        delta_mae=0.1,
        delta_wape=0.1,
        top1_near_oracle=0.9,
        optimism_bias_rate=0.0,
        optimism_bias_n=10,
    )
    # high accuracy but coverage below 0.60 -> fails the gate
    acc_low = apply_acceptance(
        safety_gate=None,
        red_team=None,
        counterfactual=cf,
        planning=None,
        abstention_shift=None,
        risk_coverage=None,
        counterfactual_coverage_value=0.40,
    )
    assert acc_low["counterfactual_ranking"]["passed"] is False
    # same accuracy with adequate coverage -> passes
    acc_ok = apply_acceptance(
        safety_gate=None,
        red_team=None,
        counterfactual=cf,
        planning=None,
        abstention_shift=None,
        risk_coverage=None,
        counterfactual_coverage_value=0.80,
    )
    assert acc_ok["counterfactual_ranking"]["passed"] is True


def test_counterfactual_coverage_helper() -> None:
    inputs = track_inputs_from_records(_full_payload())
    cov = counterfactual_coverage(inputs.branch_windows)
    # both windows recommend admit without abstaining -> full coverage
    assert cov == 1.0


def test_interval_coverage_band_acceptance() -> None:
    from gpu_center_world_model.evaluation.metrics.calibration import coverage_by_slice

    # all four truths inside their intervals -> coverage 1.0 -> OUT of [0.85,0.95]
    cbs = coverage_by_slice(
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 3.0, 4.0, 5.0],
        [1.0, 2.0, 3.0, 4.0],
        ["A", "A", "B", "B"],
        band=(THRESHOLDS["coverage_band_lo"], THRESHOLDS["coverage_band_hi"]),
    )
    acc = apply_acceptance(
        safety_gate=None,
        red_team=None,
        counterfactual=None,
        planning=None,
        abstention_shift=None,
        risk_coverage=None,
        coverage_by_slice=cbs,
    )
    assert "interval_coverage_band" in acc
    # over-covered (1.0) slices are out of the 85-95 band -> fails
    assert acc["interval_coverage_band"]["passed"] is False


# ──────────────────────────────────────────────────────────────────────────────
# Red-team: must FAIL on absent trigger; generator builds passing cases
# ──────────────────────────────────────────────────────────────────────────────
def test_generated_suite_passes() -> None:
    suite = generate_red_team_suite(n_per_scenario=2)
    rep = evaluate_red_team(suite)
    assert rep.passed
    assert rep.overall_pass_rate == 1.0


@pytest.mark.parametrize("scen", [s.name for s in RED_TEAM_SCENARIOS])
def test_missing_trigger_fails(scen: str) -> None:
    # a case that omits the scenario-defining trigger must FAIL the predicate
    case = make_red_team_case(scen, "mt", should_pass=True, omit_trigger=True)
    suite = generate_red_team_suite(n_per_scenario=1)
    suite[scen] = [case]
    rep = evaluate_red_team(suite)
    assert not rep.passed, f"{scen}: missing trigger must FAIL the suite"
    result = next(r for r in rep.results if r.scenario == scen)
    assert result.n_passed == 0
    assert any("trigger" in f["reason"] for f in result.failures)


@pytest.mark.parametrize("scen", [s.name for s in RED_TEAM_SCENARIOS])
def test_generated_failing_case_fails(scen: str) -> None:
    # a triggered-but-mishandled case must also FAIL (for a non-trigger reason)
    bad = make_red_team_case(scen, "bad", should_pass=False, omit_trigger=False)
    from gpu_center_world_model.evaluation.suites.redteam import RED_TEAM_BY_NAME

    ok, reason = RED_TEAM_BY_NAME[scen].passes(bad)
    assert ok is False
    assert "trigger" not in reason  # failed for the behaviour, not the trigger
