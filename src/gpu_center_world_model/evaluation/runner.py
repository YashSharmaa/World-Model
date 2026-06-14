# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The benchmark runner over the three tracks (doc F three-track release plan).

The runner is the orchestration layer that turns collected per-window evaluation
records into a structured :class:`EvaluationReport`, applies the doc-F acceptance
thresholds, and certifies sample sizes via doc O. It runs the three benchmark
tracks:

* **Track A — semantics**: feasibility / blocker-set / safe-candidate agreement
  with the exact kernel (safety gate + red-team). Release-blocking.
* **Track B — counterfactual rollout**: branched-replay action ranking, deltas,
  optimism bias, calibration. The primary technical target.
* **Track C — safe action selection**: lexicographic safe-regret vs oracle and
  abstention. Planning utility.

The runner is *pure aggregation*: it does not run a model. The caller (training
/ serving / a notebook) collects the per-window records by running the model and
the exact kernel, then hands them in. This keeps the harness backend-neutral and
import-clean on CI (no MLX, no model dependency) — the harness is the *source of
truth for claims*, so it must be runnable and auditable on its own.

Acceptance thresholds (doc F ``acceptance_thresholds``) are encoded as the
release-gating logic in :func:`apply_acceptance`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..version import SCHEMA_VERSION
from .metrics.abstention import (
    RiskCoverageReport,
    ShiftAbstentionReport,
    false_safe_under_shift,
    risk_coverage_curve,
)
from .metrics.calibration import CalibrationReport
from .metrics.counterfactual import (
    BranchWindow,
    CounterfactualReport,
    evaluate_counterfactual,
)
from .metrics.planning import (
    PlanningReport,
    PlanningWindow,
    evaluate_planning,
)
from .metrics.predictive import PredictiveReport
from .metrics.safety_gate import (
    SafetyGateReport,
    SafetyJudgement,
    evaluate_safety_gate,
)
from .reports.report import EvaluationReport
from .sample_complexity import SampleComplexityPlan, build_certification_plan
from .slice_defs import SLICE_FAMILIES, assign_slices
from .suites.redteam import RedTeamCase, RedTeamReport, evaluate_red_team

# ──────────────────────────────────────────────────────────────────────────────
# Doc F acceptance thresholds (the single source of truth for the gating)
# ──────────────────────────────────────────────────────────────────────────────
THRESHOLDS: dict[str, float] = {
    "max_false_safe_rate": 0.0,  # spine: 0 in-distribution
    "max_false_safe_rate_shift": 0.01,  # doc F shift pack
    "min_sign_accuracy": 0.70,
    "min_sign_accuracy_non_abstained": 0.80,
    "min_coverage": 0.60,
    "min_regret_reduction": 0.15,
    "coverage_band_lo": 0.85,
    "coverage_band_hi": 0.95,
    "max_optimism_bias": 0.05,
    "min_abstain_under_shift": 0.80,
}


@dataclass(slots=True)
class TrackInputs:
    """All collected per-window records the runner aggregates into a report.

    Every field is optional so a partial run (e.g. only the safety track on a CI
    commit) still produces a valid report. The caller populates the records by
    running the model + exact kernel offline.
    """

    # Track A — semantics
    safety_judgements: list[SafetyJudgement] = field(default_factory=list)
    red_team_cases: dict[str, list[RedTeamCase]] = field(default_factory=dict)
    # Track B — counterfactual rollout
    branch_windows: list[BranchWindow] = field(default_factory=list)
    predictive_reports: list[PredictiveReport] = field(default_factory=list)
    calibration_reports: list[CalibrationReport] = field(default_factory=list)
    # Track C — safe action selection
    planning_windows: list[PlanningWindow] = field(default_factory=list)
    baseline_planning_windows: list[PlanningWindow] = field(default_factory=list)
    # abstention / shift
    risk_score: list[float] = field(default_factory=list)
    selective_error: list[float] = field(default_factory=list)
    shift_abstained_or_fellback: list[float] = field(default_factory=list)
    shift_false_safe: list[float] = field(default_factory=list)
    # per-window feature records for Mondrian slicing (parallel to branch_windows)
    window_features: list[dict[str, object]] = field(default_factory=list)
    # sample-complexity inputs
    n_branches: int = 4
    rho_pair: float = 0.4
    tau_mix: float = 2.0
    n_severe_shift_zero_events: int = 0
    is_shift_pack: bool = False


def run_track_semantics(
    inputs: TrackInputs,
) -> tuple[SafetyGateReport | None, RedTeamReport | None]:
    """Track A: exact-kernel safety gate + red-team suite (release-blocking)."""
    sg = None
    if inputs.safety_judgements:
        max_fs = (
            THRESHOLDS["max_false_safe_rate_shift"]
            if inputs.is_shift_pack
            else THRESHOLDS["max_false_safe_rate"]
        )
        sg = evaluate_safety_gate(
            inputs.safety_judgements,
            max_false_safe_rate=max_fs,
            require_blocker_exact_match=True,
        )
    rt = None
    if inputs.red_team_cases:
        rt = evaluate_red_team(inputs.red_team_cases)
    return sg, rt


def run_track_counterfactual(
    inputs: TrackInputs,
) -> CounterfactualReport | None:
    """Track B: branched-replay counterfactual fidelity (primary target)."""
    if not inputs.branch_windows:
        return None
    return evaluate_counterfactual(
        inputs.branch_windows,
        risk_threshold=0.5,
        material_margin=0.0,
    )


def run_track_planning(
    inputs: TrackInputs,
) -> PlanningReport | None:
    """Track C: lexicographic safe-regret vs oracle, paired against baseline."""
    if not inputs.planning_windows:
        return None
    return evaluate_planning(
        inputs.planning_windows,
        baseline_windows=inputs.baseline_planning_windows or None,
    )


def run_abstention(
    inputs: TrackInputs,
) -> tuple[RiskCoverageReport | None, ShiftAbstentionReport | None]:
    """Risk-coverage curve and false-safe-under-shift behaviour."""
    rc = None
    if inputs.risk_score and inputs.selective_error:
        rc = risk_coverage_curve(
            inputs.risk_score,
            inputs.selective_error,
            target_coverage=THRESHOLDS["min_coverage"],
        )
    sa = None
    if inputs.shift_abstained_or_fellback:
        sa = false_safe_under_shift(
            inputs.shift_abstained_or_fellback,
            inputs.shift_false_safe,
            min_abstain_rate=THRESHOLDS["min_abstain_under_shift"],
            max_false_safe_rate=THRESHOLDS["max_false_safe_rate_shift"],
        )
    return rc, sa


def compute_slices(inputs: TrackInputs) -> dict[str, object]:
    """Group counterfactual sign accuracy by each prespecified Mondrian slice.

    For each slice family and cell we recompute the headline counterfactual
    report over the windows whose feature record lands in that cell, so the
    report shows slice-wise sign accuracy / optimism bias (doc F: always publish
    slice-wise metrics). Requires ``window_features`` aligned with
    ``branch_windows``.
    """
    out: dict[str, object] = {}
    feats = inputs.window_features
    windows = inputs.branch_windows
    if not feats or len(feats) != len(windows):
        return out
    for family in SLICE_FAMILIES:
        cells: dict[str, dict[str, object]] = {}
        # bucket window indices by this family's cell label
        buckets: dict[str, list[BranchWindow]] = {}
        for w, rec in zip(windows, feats, strict=False):
            label = assign_slices(rec)[family.name]
            buckets.setdefault(label, []).append(w)
        for label, ws in buckets.items():
            rep = evaluate_counterfactual(ws)
            cells[label] = {
                "n_windows": rep.n_windows,
                "sign_accuracy": rep.sign_accuracy,
                "optimism_bias_rate": rep.optimism_bias_rate,
                "top1_near_oracle": rep.top1_near_oracle,
            }
        out[family.name] = cells
    return out


def apply_acceptance(
    *,
    safety_gate: SafetyGateReport | None,
    red_team: RedTeamReport | None,
    counterfactual: CounterfactualReport | None,
    planning: PlanningReport | None,
    abstention_shift: ShiftAbstentionReport | None,
    risk_coverage: RiskCoverageReport | None,
) -> dict[str, object]:
    """Evaluate each doc-F acceptance threshold; return a pass/fail map.

    Hard-safety and red-team are release-blocking; the rest are profile claims
    reported with their pass/fail flags but never folded into a blended score.
    """
    acc: dict[str, object] = {}

    if safety_gate is not None:
        acc["hard_safety"] = {
            "passed": safety_gate.passed and safety_gate.violation_count == 0,
            "detail": f"violations={safety_gate.violation_count}, "
            f"false_safe_rate={safety_gate.false_safe_rate:.4f}",
            "release_blocking": True,
        }
    if red_team is not None:
        acc["red_team"] = {
            "passed": red_team.passed,
            "detail": f"overall_pass_rate={red_team.overall_pass_rate:.4f}",
            "release_blocking": True,
        }
    if counterfactual is not None:
        acc["counterfactual_ranking"] = {
            "passed": (
                _ge(counterfactual.sign_accuracy, THRESHOLDS["min_sign_accuracy"])
                and _ge(
                    counterfactual.sign_accuracy_non_abstained,
                    THRESHOLDS["min_sign_accuracy_non_abstained"],
                )
            ),
            "detail": f"sign_acc={counterfactual.sign_accuracy:.4f} "
            f"(>=0.70), non_abstained={counterfactual.sign_accuracy_non_abstained:.4f} "
            "(>=0.80)",
        }
        acc["optimism_bias"] = {
            "passed": _le(
                counterfactual.optimism_bias_rate,
                THRESHOLDS["max_optimism_bias"],
            ),
            "detail": f"rate={counterfactual.optimism_bias_rate:.4f} (<0.05)",
        }
    if planning is not None:
        acc["planning_utility"] = {
            "passed": _ge(
                planning.relative_regret_reduction,
                THRESHOLDS["min_regret_reduction"],
            ),
            "detail": f"relative_regret_reduction="
            f"{planning.relative_regret_reduction:.4f} (>=0.15)",
        }
    if abstention_shift is not None:
        acc["shift_and_abstention"] = {
            "passed": abstention_shift.passed,
            "detail": f"abstain_rate={abstention_shift.abstain_or_fallback_rate:.4f} "
            f"(>=0.80), false_safe={abstention_shift.false_safe_rate:.4f} (<0.01)",
        }
    if risk_coverage is not None:
        acc["selective_risk_at_coverage"] = {
            "passed": True,  # informational; the binding claim is sign-accuracy
            "detail": f"selective_risk@{risk_coverage.target_coverage:.2f}="
            f"{risk_coverage.selective_risk_at_target:.4f}, "
            f"aurc={risk_coverage.aurc:.4f}",
        }
    return acc


def _ge(x: float, t: float) -> bool:
    return x == x and x >= t  # NaN-safe


def _le(x: float, t: float) -> bool:
    return x == x and x <= t


@dataclass(slots=True)
class BenchmarkRunner:
    """Drives the three tracks and assembles the final report.

    Parameters
    ----------
    run_id:
        Identifier for this evaluation run (recorded in the report).
    track:
        Which track to emphasise in the report header
        (``"semantics" | "counterfactual_rollout" | "safe_action_selection" |
        "all"``). All available inputs are always scored; this only labels the
        report.
    """

    run_id: str = "eval"
    track: str = "all"

    def run(self, inputs: TrackInputs) -> EvaluationReport:
        t0 = time.time()
        safety_gate, red_team = run_track_semantics(inputs)
        counterfactual = run_track_counterfactual(inputs)
        planning = run_track_planning(inputs)
        risk_cov, shift_ab = run_abstention(inputs)
        slices = compute_slices(inputs)

        sample_plan: SampleComplexityPlan | None = None
        if inputs.branch_windows:
            sample_plan = build_certification_plan(
                n_windows=len(inputs.branch_windows),
                n_branches=inputs.n_branches,
                rho_pair=inputs.rho_pair,
                tau_mix=inputs.tau_mix,
                n_severe_shift_zero_events=(
                    inputs.n_severe_shift_zero_events
                    or sum(1 for v in inputs.shift_false_safe if v <= 0.5)
                ),
            )

        acceptance = apply_acceptance(
            safety_gate=safety_gate,
            red_team=red_team,
            counterfactual=counterfactual,
            planning=planning,
            abstention_shift=shift_ab,
            risk_coverage=risk_cov,
        )

        # Merge the abstention readouts into one dict for the report.
        abstention_dict: dict[str, object] | None = None
        if risk_cov is not None or shift_ab is not None:
            abstention_dict = {}
            if risk_cov is not None:
                abstention_dict.update(risk_cov.as_dict())
            if shift_ab is not None:
                abstention_dict.update(shift_ab.as_dict())

        report = EvaluationReport(
            run_id=self.run_id,
            schema_version=SCHEMA_VERSION,
            track=self.track,
            safety_gate=safety_gate.as_dict() if safety_gate else None,
            red_team=red_team.as_dict() if red_team else None,
            counterfactual=counterfactual.as_dict() if counterfactual else None,
            planning=planning.as_dict() if planning else None,
            predictive=[p.as_dict() for p in inputs.predictive_reports],
            calibration=[c.as_dict() for c in inputs.calibration_reports],
            abstention=abstention_dict,
            slices=slices,
            sample_complexity=sample_plan.as_dict() if sample_plan else None,
            acceptance=acceptance,
            metadata={
                "elapsed_s": round(time.time() - t0, 6),
                "n_safety_judgements": len(inputs.safety_judgements),
                "n_branch_windows": len(inputs.branch_windows),
                "n_planning_windows": len(inputs.planning_windows),
                "is_shift_pack": inputs.is_shift_pack,
                "thresholds": dict(THRESHOLDS),
            },
        )
        return report


def run_benchmark(
    inputs: TrackInputs, *, run_id: str = "eval", track: str = "all"
) -> EvaluationReport:
    """Convenience one-shot: build a runner and produce the report."""
    return BenchmarkRunner(run_id=run_id, track=track).run(inputs)


# Backwards-friendly aliases for the track names used in benchmarks/.
TRACKS: tuple[str, ...] = (
    "semantics",
    "counterfactual_rollout",
    "safe_action_selection",
)


def track_inputs_from_records(
    records: Sequence[dict[str, object]],
) -> TrackInputs:  # pragma: no cover - thin adapter
    """Adapter stub for future JSON-record ingestion.

    The release format for collected records is JSON; this adapter is the
    intended seam for loading them. It is intentionally minimal in v1 (the
    typed dataclasses are the primary API) and raises if asked to parse, so the
    contract is explicit rather than silently wrong.
    """
    raise NotImplementedError(
        "v1 uses the typed TrackInputs dataclass directly; JSON-record ingestion "
        "is a v2 seam. Construct TrackInputs from your collected dataclasses."
    )


__all__ = [
    "THRESHOLDS",
    "TRACKS",
    "BenchmarkRunner",
    "TrackInputs",
    "apply_acceptance",
    "compute_slices",
    "run_abstention",
    "run_benchmark",
    "run_track_counterfactual",
    "run_track_planning",
    "run_track_semantics",
    "track_inputs_from_records",
]
