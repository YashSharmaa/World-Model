# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The SAFETY-GATED EVALUATION HARNESS — the source of truth for claims (doc F).

Design spine reflected here:

* **Safety is a gate, not a weighted metric.** The :mod:`metrics.safety_gate`
  family and the red-team :mod:`suites` are *release-blocking*; a hard-safety
  failure blocks the release regardless of any other number.
* **No single blended score.** The :mod:`reports` package emits a profile of
  separate metric objects (JSON + Markdown), led by the safety gate.
* **Counterfactual ranking is the primary technical target** (doc F / doc O
  GB4), scored on branched replay from one pre-action state hash and seed bundle.
* **Dependent-data sample complexity** (:mod:`sample_complexity`, doc O) certifies
  whether a release has enough *effective* samples to make each claim.
* **Provable vs empirical ledger** (doc O L1-L12) ships with every report.

Three benchmark tracks (doc F): semantics, counterfactual rollout, safe action
selection — orchestrated by :class:`runner.BenchmarkRunner`.

This package is pure NumPy and depends only on the frozen schema and contracts,
so it imports and runs on plain Linux CI without MLX or the learned backend.

Done by Yash Sharma — https://www.linkedin.com/in/yashsharmaa/
"""

from __future__ import annotations

# Metric families (re-exported as one clean surface) ---------------------------
from .metrics import (
    BranchWindow,
    CalibrationReport,
    CounterfactualReport,
    CoverageBySliceReport,
    PlanningReport,
    PlanningWindow,
    PredictiveReport,
    ReliabilityBin,
    RiskCoveragePoint,
    RiskCoverageReport,
    SafetyGateReport,
    SafetyJudgement,
    ShiftAbstentionReport,
    blocker_set_exact_match,
    brier_score,
    chose_oracle,
    chose_unsafe,
    coverage_by_slice,
    coverage_gap,
    delta_error,
    evaluate_counterfactual,
    evaluate_planning,
    evaluate_safety_gate,
    expected_calibration_error,
    false_safe_rate,
    false_safe_under_shift,
    interval_coverage,
    interval_sharpness,
    lex_argmin,
    lex_less,
    mae,
    medae,
    optimism_bias_rate,
    pairwise_sign_accuracy,
    pinball_loss,
    precision_recall_at_budget,
    risk_coverage_curve,
    rmse,
    safe_regret,
    score_hazard_channel,
    score_quantile_channel,
    spurious_block_count,
    top1_near_oracle_rate,
    violation_count,
    wape,
)
from .metrics.operator_usefulness import (
    OperatorCase,
    OperatorUsefulnessReport,
    evaluate_operator_usefulness,
)

# Reports + provable-vs-empirical ledger ---------------------------------------
from .reports import (
    ClaimStatus,
    EvaluationReport,
    LedgerClaim,
    generate_ledger,
    ledger_as_dict,
    ledger_claims,
    render_markdown,
    write_reports,
)

# Runner (the three-track orchestrator) ---------------------------------------
from .runner import (
    THRESHOLDS,
    TRACKS,
    BenchmarkRunner,
    TrackInputs,
    apply_acceptance,
    compute_slices,
    compute_strongest_baseline,
    counterfactual_coverage,
    run_abstention,
    run_benchmark,
    run_coverage_by_slice,
    run_operator_usefulness,
    run_track_calibration,
    run_track_counterfactual,
    run_track_planning,
    run_track_predictive,
    run_track_semantics,
    track_inputs_from_records,
)

# Sample-complexity calculators (doc O) ----------------------------------------
from .sample_complexity import (
    CertificationCheck,
    SampleComplexityPlan,
    build_certification_plan,
    design_effect,
    effective_pairwise_samples,
    mixing_effective_n,
    pairwise_per_window,
    required_n_one_sided,
    required_n_paired_bernstein,
    required_n_two_sided,
    rule_of_three_upper,
    weighted_ess,
    windows_for_pairwise_target,
)

# Prespecified Mondrian slices -------------------------------------------------
from .slice_defs import (
    MIN_EFFECTIVE_WINDOWS_PER_SLICE,
    SLICE_FAMILIES,
    SLICE_FAMILIES_BY_NAME,
    SliceFamily,
    all_slice_labels,
    assign_slices,
)

# Red-team suite ---------------------------------------------------------------
from .suites import (
    RED_TEAM_SCENARIOS,
    RedTeamCase,
    RedTeamReport,
    RedTeamScenario,
    ScenarioResult,
    evaluate_red_team,
    evaluate_scenario,
    generate_red_team_suite,
    make_red_team_case,
)

__all__ = [
    "MIN_EFFECTIVE_WINDOWS_PER_SLICE",
    "RED_TEAM_SCENARIOS",
    "SLICE_FAMILIES",
    "SLICE_FAMILIES_BY_NAME",
    "THRESHOLDS",
    "TRACKS",
    # runner / orchestration
    "BenchmarkRunner",
    # counterfactual
    "BranchWindow",
    "CalibrationReport",
    # sample complexity
    "CertificationCheck",
    "ClaimStatus",
    "CounterfactualReport",
    "CoverageBySliceReport",
    # reports + ledger
    "EvaluationReport",
    "LedgerClaim",
    # operator usefulness (doc O L10)
    "OperatorCase",
    "OperatorUsefulnessReport",
    "PlanningReport",
    # planning
    "PlanningWindow",
    # predictive
    "PredictiveReport",
    # red-team
    "RedTeamCase",
    "RedTeamReport",
    "RedTeamScenario",
    # calibration
    "ReliabilityBin",
    # abstention
    "RiskCoveragePoint",
    "RiskCoverageReport",
    "SafetyGateReport",
    # safety gate
    "SafetyJudgement",
    "SampleComplexityPlan",
    "ScenarioResult",
    "ShiftAbstentionReport",
    # slices
    "SliceFamily",
    "TrackInputs",
    "all_slice_labels",
    "apply_acceptance",
    "assign_slices",
    "blocker_set_exact_match",
    "brier_score",
    "build_certification_plan",
    "chose_oracle",
    "chose_unsafe",
    "compute_slices",
    "compute_strongest_baseline",
    "counterfactual_coverage",
    "coverage_by_slice",
    "coverage_gap",
    "delta_error",
    "design_effect",
    "effective_pairwise_samples",
    "evaluate_counterfactual",
    "evaluate_operator_usefulness",
    "evaluate_planning",
    "evaluate_red_team",
    "evaluate_safety_gate",
    "evaluate_scenario",
    "expected_calibration_error",
    "false_safe_rate",
    "false_safe_under_shift",
    "generate_ledger",
    "generate_red_team_suite",
    "interval_coverage",
    "interval_sharpness",
    "ledger_as_dict",
    "ledger_claims",
    "lex_argmin",
    "lex_less",
    "mae",
    "make_red_team_case",
    "medae",
    "mixing_effective_n",
    "optimism_bias_rate",
    "pairwise_per_window",
    "pairwise_sign_accuracy",
    "pinball_loss",
    "precision_recall_at_budget",
    "render_markdown",
    "required_n_one_sided",
    "required_n_paired_bernstein",
    "required_n_two_sided",
    "risk_coverage_curve",
    "rmse",
    "rule_of_three_upper",
    "run_abstention",
    "run_benchmark",
    "run_coverage_by_slice",
    "run_operator_usefulness",
    # CLI entrypoint (gcwm eval)
    "run_suite",
    "run_track_calibration",
    "run_track_counterfactual",
    "run_track_planning",
    "run_track_predictive",
    "run_track_semantics",
    "safe_regret",
    "score_hazard_channel",
    "score_quantile_channel",
    "spurious_block_count",
    "top1_near_oracle_rate",
    "track_inputs_from_records",
    "violation_count",
    "wape",
    "weighted_ess",
    "windows_for_pairwise_target",
    "write_reports",
]

from .suite_entry import run_suite
