# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Metric families for the safety-gated evaluation harness (doc F).

The families are deliberately kept as *separate reported objects*, never folded
into one blended score (doc F: "always publish slice-wise metrics ... rather than
a single winner score"; doc O publishes the provable-vs-empirical ledger
alongside). The gated hierarchy is:

    1. safety_gate      (release-blocking; exact-kernel agreement)
    2. counterfactual   (the primary technical target; branch fidelity)
    3. planning         (safe regret vs oracle; lexicographic)
    4. predictive       (factual-replay diagnostic)
    5. calibration      (ECE + coverage by slice, shift-aware)
    6. abstention       (risk-coverage; false-safe under shift)
"""

from __future__ import annotations

from .abstention import (
    RiskCoveragePoint,
    RiskCoverageReport,
    ShiftAbstentionReport,
    false_safe_under_shift,
    risk_coverage_curve,
)
from .calibration import (
    CalibrationReport,
    CoverageBySliceReport,
    ReliabilityBin,
    coverage_by_slice,
    expected_calibration_error,
)
from .counterfactual import (
    BranchWindow,
    CounterfactualReport,
    delta_error,
    evaluate_counterfactual,
    optimism_bias_rate,
    pairwise_sign_accuracy,
    top1_near_oracle_rate,
)
from .operator_usefulness import (
    CoverageWinRatePoint,
    DecisionTimeReport,
    ExplanationAgreement,
    OperatorCase,
    OperatorUsefulnessReport,
    OverrideReport,
    RelianceBreakdown,
    appropriate_reliance,
    coverage_vs_win_rate,
    decision_time_reduction,
    evaluate_operator_usefulness,
    explanation_blocker_agreement,
    override_harness,
)
from .planning import (
    LexKey,
    PlanningReport,
    PlanningWindow,
    chose_oracle,
    chose_unsafe,
    evaluate_planning,
    lex_argmin,
    lex_less,
    safe_regret,
)
from .predictive import (
    ALERT_BUDGETS,
    HAZARD_CHANNELS,
    PredictiveReport,
    brier_score,
    coverage_gap,
    interval_coverage,
    interval_sharpness,
    mae,
    medae,
    pinball_loss,
    precision_recall_at_budget,
    rmse,
    score_hazard_channel,
    score_quantile_channel,
    wape,
)
from .safety_gate import (
    SafetyGateReport,
    SafetyJudgement,
    blocker_set_exact_match,
    evaluate_safety_gate,
    false_safe_rate,
    spurious_block_count,
    violation_count,
)

__all__ = [
    "ALERT_BUDGETS",
    "HAZARD_CHANNELS",
    # counterfactual
    "BranchWindow",
    "CalibrationReport",
    "CounterfactualReport",
    "CoverageBySliceReport",
    "CoverageWinRatePoint",
    "DecisionTimeReport",
    "ExplanationAgreement",
    # planning
    "LexKey",
    # operator usefulness (deployment-readiness layer; doc O L10)
    "OperatorCase",
    "OperatorUsefulnessReport",
    "OverrideReport",
    "PlanningReport",
    "PlanningWindow",
    # predictive
    "PredictiveReport",
    # calibration
    "ReliabilityBin",
    "RelianceBreakdown",
    # abstention
    "RiskCoveragePoint",
    "RiskCoverageReport",
    "SafetyGateReport",
    # safety_gate (release-blocking)
    "SafetyJudgement",
    "ShiftAbstentionReport",
    "appropriate_reliance",
    "blocker_set_exact_match",
    "brier_score",
    "chose_oracle",
    "chose_unsafe",
    "coverage_by_slice",
    "coverage_gap",
    "coverage_vs_win_rate",
    "decision_time_reduction",
    "delta_error",
    "evaluate_counterfactual",
    "evaluate_operator_usefulness",
    "evaluate_planning",
    "evaluate_safety_gate",
    "expected_calibration_error",
    "explanation_blocker_agreement",
    "false_safe_rate",
    "false_safe_under_shift",
    "interval_coverage",
    "interval_sharpness",
    "lex_argmin",
    "lex_less",
    "mae",
    "medae",
    "optimism_bias_rate",
    "override_harness",
    "pairwise_sign_accuracy",
    "pinball_loss",
    "precision_recall_at_budget",
    "risk_coverage_curve",
    "rmse",
    "safe_regret",
    "score_hazard_channel",
    "score_quantile_channel",
    "spurious_block_count",
    "top1_near_oracle_rate",
    "violation_count",
    "wape",
]
