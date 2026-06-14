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
    score_quantile_channel,
    spurious_block_count,
    top1_near_oracle_rate,
    violation_count,
    wape,
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
    run_abstention,
    run_benchmark,
    run_track_counterfactual,
    run_track_planning,
    run_track_semantics,
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
)

__all__ = [
    # runner / orchestration
    "BenchmarkRunner",
    "TrackInputs",
    "run_benchmark",
    "THRESHOLDS",
    "TRACKS",
    "apply_acceptance",
    "compute_slices",
    "run_abstention",
    "run_track_semantics",
    "run_track_counterfactual",
    "run_track_planning",
    # reports + ledger
    "EvaluationReport",
    "render_markdown",
    "write_reports",
    "ClaimStatus",
    "LedgerClaim",
    "ledger_claims",
    "generate_ledger",
    "ledger_as_dict",
    # safety gate
    "SafetyJudgement",
    "SafetyGateReport",
    "evaluate_safety_gate",
    "violation_count",
    "false_safe_rate",
    "blocker_set_exact_match",
    "spurious_block_count",
    # counterfactual
    "BranchWindow",
    "CounterfactualReport",
    "evaluate_counterfactual",
    "pairwise_sign_accuracy",
    "delta_error",
    "top1_near_oracle_rate",
    "optimism_bias_rate",
    # planning
    "PlanningWindow",
    "PlanningReport",
    "evaluate_planning",
    "safe_regret",
    "lex_less",
    "lex_argmin",
    "chose_oracle",
    "chose_unsafe",
    # predictive
    "PredictiveReport",
    "score_quantile_channel",
    "mae",
    "medae",
    "wape",
    "rmse",
    "pinball_loss",
    "interval_coverage",
    "interval_sharpness",
    "coverage_gap",
    "brier_score",
    "precision_recall_at_budget",
    # calibration
    "ReliabilityBin",
    "CalibrationReport",
    "CoverageBySliceReport",
    "expected_calibration_error",
    "coverage_by_slice",
    # abstention
    "RiskCoveragePoint",
    "RiskCoverageReport",
    "ShiftAbstentionReport",
    "risk_coverage_curve",
    "false_safe_under_shift",
    # sample complexity
    "CertificationCheck",
    "SampleComplexityPlan",
    "build_certification_plan",
    "pairwise_per_window",
    "design_effect",
    "effective_pairwise_samples",
    "weighted_ess",
    "mixing_effective_n",
    "required_n_one_sided",
    "required_n_two_sided",
    "rule_of_three_upper",
    "required_n_paired_bernstein",
    "windows_for_pairwise_target",
    # slices
    "SliceFamily",
    "SLICE_FAMILIES",
    "SLICE_FAMILIES_BY_NAME",
    "assign_slices",
    "all_slice_labels",
    # red-team
    "RedTeamCase",
    "RedTeamScenario",
    "ScenarioResult",
    "RedTeamReport",
    "RED_TEAM_SCENARIOS",
    "evaluate_red_team",
    "evaluate_scenario",
    # CLI entrypoint (gcwm eval)
    "run_suite",
]

from .suite_entry import run_suite  # noqa: E402  (placed last to avoid cycles)
