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
    # safety_gate (release-blocking)
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
    "LexKey",
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
]
