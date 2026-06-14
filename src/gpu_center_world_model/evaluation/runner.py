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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..simulator.contracts import BlockerGroup
from ..version import SCHEMA_VERSION
from .metrics.abstention import (
    RiskCoverageReport,
    ShiftAbstentionReport,
    false_safe_under_shift,
    risk_coverage_curve,
)
from .metrics.calibration import (
    CalibrationReport,
    CoverageBySliceReport,
    coverage_by_slice,
    expected_calibration_error,
)
from .metrics.counterfactual import (
    BranchWindow,
    CounterfactualReport,
    evaluate_counterfactual,
)
from .metrics.operator_usefulness import (
    OperatorCase,
    OperatorUsefulnessReport,
    evaluate_operator_usefulness,
)
from .metrics.planning import (
    FairnessEvalWindow,
    FairnessReport,
    PlanningReport,
    PlanningWindow,
    evaluate_fairness,
    evaluate_planning,
    fairness_windows_by_slice,
)
from .metrics.predictive import (
    PredictiveReport,
    score_hazard_channel,
    score_quantile_channel,
)
from .metrics.safety_gate import (
    SafetyGateReport,
    SafetyJudgement,
    evaluate_safety_gate,
)
from .reports.report import EvaluationReport
from .sample_complexity import (
    SampleComplexityPlan,
    build_certification_plan,
    effective_pairwise_samples,
    mixing_effective_n,
)
from .slice_defs import MIN_EFFECTIVE_WINDOWS_PER_SLICE, SLICE_FAMILIES, assign_slices
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
    # raw predictive inputs the runner scores via score_quantile_channel /
    # score_hazard_channel (so a saved pack carries arrays, not pre-scored reports)
    quantile_channel_inputs: list[dict[str, object]] = field(default_factory=list)
    hazard_channel_inputs: list[dict[str, object]] = field(default_factory=list)
    # raw calibration inputs the runner scores via expected_calibration_error
    calibration_inputs: list[dict[str, object]] = field(default_factory=list)
    # interval-coverage-by-slice inputs (lower/upper/truth + slice labels)
    coverage_by_slice_inputs: dict[str, object] = field(default_factory=dict)
    # Track C — safe action selection
    planning_windows: list[PlanningWindow] = field(default_factory=list)
    baseline_planning_windows: list[PlanningWindow] = field(default_factory=list)
    #: Per-window inputs for constructing baseline planning windows by *running*
    #: each of the six doc-F baselines' ``score_actions`` (doc F baseline_systems).
    #: A list of :class:`model.baselines.BaselineDecisionWindow` (or dicts the
    #: runner coerces). When present the runner scores every baseline, picks the
    #: *strongest* comparator (lowest mean safe regret), and feeds its regret into
    #: ``evaluate_planning`` so the >=15% threshold is measured against the best
    #: skeptical baseline (doc F ``acceptance_thresholds.planning_utility``).
    baseline_decision_windows: list[object] = field(default_factory=list)
    #: When the strongest-comparator search should include the hindsight oracle
    #: (off by default — the oracle is an upper bound, not a deployable foil).
    baseline_include_oracle: bool = False
    #: Optional pre-computed doc-F ablation report (``model.baselines.ablations``
    #: ``AblationReport.as_dict()``) surfaced in the report under
    #: ``slices["_ablations"]`` so the five per-ablation ``main_readouts`` ship in
    #: the :class:`EvaluationReport` (doc F ``ablations``).
    ablation_report: dict[str, object] | None = None
    # Fairness (v2 hooks; doc V H9) — reported slice-wise alongside safety-first
    # metrics, never release-blocking in v1 (fairness_mode is off by default).
    fairness_windows: list[FairnessEvalWindow] = field(default_factory=list)
    #: window_id -> slice label (or positional, aligned to fairness_windows) so
    #: FD_H / debt / gaming are grouped slice-wise (doc V H9).
    fairness_slice_labels: dict[str, str] = field(default_factory=dict)
    #: optional replay price-of-anarchy passed through into the fairness report.
    fairness_poa: float | None = None
    # operator-usefulness study (provisional; doc O L10)
    operator_cases: list[OperatorCase] = field(default_factory=list)
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


def compute_strongest_baseline(
    inputs: TrackInputs,
) -> object | None:
    """Run all six doc-F baselines and return the strongest comparator's windows.

    When ``baseline_decision_windows`` are supplied, this builds every registered
    doc-F baseline (``model.baselines.build_default_baselines``), runs each one's
    ``score_actions`` over each decision window's candidate set to construct
    baseline :class:`PlanningWindow`s, scores all of them, and returns the
    comparator with the **lowest mean safe regret** — the strongest foil — as a
    :class:`model.baselines.StrongestBaselineResult`. Returns ``None`` when no
    baseline decision windows were collected (the runner then falls back to any
    pre-built ``baseline_planning_windows``).

    The import is local so the harness stays import-clean on a build that excludes
    the model package (doc F: the runner is backend-neutral, pure aggregation).
    """
    if not inputs.baseline_decision_windows:
        return None
    from ..model.baselines import (
        BaselineDecisionWindow,
        build_default_baselines,
        strongest_baseline_windows,
    )

    bwins = [
        w
        for w in inputs.baseline_decision_windows
        if isinstance(w, BaselineDecisionWindow)
    ]
    if not bwins:
        return None
    baselines = build_default_baselines()
    return strongest_baseline_windows(
        baselines, bwins, include_oracle=inputs.baseline_include_oracle
    )


def run_track_planning(
    inputs: TrackInputs,
    strongest_baseline: object | None = None,
) -> PlanningReport | None:
    """Track C: lexicographic safe-regret vs oracle, paired against baseline.

    When ``strongest_baseline`` is supplied (from
    :func:`compute_strongest_baseline`), its chosen windows are used as the paired
    comparator so the doc-F ">=15% lower mean safe regret than the best skeptical
    baseline" threshold is measured against the **strongest** of the six baselines
    (doc F ``acceptance_thresholds.planning_utility``). Otherwise the runner falls
    back to any pre-built ``baseline_planning_windows`` (the Wave-1/2 behaviour).
    """
    if not inputs.planning_windows:
        return None
    baseline_windows = inputs.baseline_planning_windows or None
    if strongest_baseline is not None:
        sb_windows = getattr(strongest_baseline, "windows", None)
        if sb_windows:
            baseline_windows = sb_windows
    return evaluate_planning(
        inputs.planning_windows,
        baseline_windows=baseline_windows,
    )


def run_track_fairness(inputs: TrackInputs) -> FairnessReport | None:
    """Fairness hooks: aggregate FD_H / fair regret / debt / gaming / PoA.

    Doc V H9: the v2 fairness metrics (fairness drift ``FD_H``, fair regret,
    per-tenant debt, queue-hopping / priority-inflation / under-requesting gaming
    aggregates) are reported in offline replay *alongside* the safety-first
    metrics. They are never release-blocking in v1 (``fairness_mode`` is off by
    default), so this returns ``None`` when no fairness windows were collected.
    """
    if not inputs.fairness_windows:
        return None
    return evaluate_fairness(inputs.fairness_windows, poa=inputs.fairness_poa)


def compute_fairness_slices(inputs: TrackInputs) -> dict[str, object]:
    """Group the fairness readout slice-wise (doc V H9 "reported slice-wise").

    Buckets the collected :class:`FairnessEvalWindow` records by their slice
    label (``fairness_slice_labels``; ``window_id -> label`` or positional) and
    produces a :class:`FairnessReport` per cell plus an overall ``_overall``
    cell. Returns an empty dict when no fairness windows were collected, so the
    fairness slices simply do not appear for a v1 run.
    """
    fw = inputs.fairness_windows
    if not fw:
        return {}
    out: dict[str, object] = {
        "_overall": evaluate_fairness(fw, poa=inputs.fairness_poa).as_dict()
    }
    labels = inputs.fairness_slice_labels
    if labels:
        buckets = fairness_windows_by_slice(fw, labels)
        cells: dict[str, object] = {}
        for label, cell_windows in buckets.items():
            cells[label] = evaluate_fairness(cell_windows).as_dict()
        out["by_slice"] = cells
    return out


def run_track_predictive(inputs: TrackInputs) -> list[PredictiveReport]:
    """Track-B diagnostic: score continuous + discrete predictive channels.

    The runner scores any raw ``quantile_channel_inputs`` via
    :func:`score_quantile_channel` (continuous channels — MAE/WAPE/pinball/
    coverage) and any ``hazard_channel_inputs`` via :func:`score_hazard_channel`
    (discrete hazards — Brier + PR-at-budget for hotspot / pods_ready_timeout /
    requeue / node_failure / xid_fault). Pre-scored ``predictive_reports`` passed
    directly are preserved (the typed dataclass remains a first-class input).
    """
    reports: list[PredictiveReport] = list(inputs.predictive_reports)
    for rec in inputs.quantile_channel_inputs:
        quantile_preds = {
            float(k): v for k, v in dict(rec.get("quantile_preds", {})).items()
        }
        reports.append(
            score_quantile_channel(
                channel=str(rec.get("channel", "channel")),
                horizon_s=int(rec.get("horizon_s", 0)),
                truth=rec.get("truth", []),  # type: ignore[arg-type]
                quantile_preds=quantile_preds,  # type: ignore[arg-type]
                lower_q=float(rec.get("lower_q", 0.05)),
                upper_q=float(rec.get("upper_q", 0.95)),
            )
        )
    for rec in inputs.hazard_channel_inputs:
        budgets = rec.get("alert_budgets")
        kwargs: dict[str, object] = {}
        if budgets is not None:
            kwargs["alert_budgets"] = tuple(float(b) for b in budgets)  # type: ignore[arg-type]
        reports.append(
            score_hazard_channel(
                channel=str(rec.get("channel", "hazard")),
                horizon_s=int(rec.get("horizon_s", 0)),
                prob=rec.get("prob", []),  # type: ignore[arg-type]
                label=rec.get("label", []),  # type: ignore[arg-type]
                **kwargs,  # type: ignore[arg-type]
            )
        )
    return reports


def run_track_calibration(inputs: TrackInputs) -> list[CalibrationReport]:
    """Track-B diagnostic: ECE per (channel, slice) from raw calibration inputs.

    Scores any raw ``calibration_inputs`` via :func:`expected_calibration_error`
    and preserves pre-scored ``calibration_reports`` passed directly. Doc F:
    calibration is reported *by slice* (topology-shift / degraded-telemetry /
    unseen-workload), never as one in-distribution number.
    """
    reports: list[CalibrationReport] = list(inputs.calibration_reports)
    for rec in inputs.calibration_inputs:
        reports.append(
            expected_calibration_error(
                rec.get("prob", []),  # type: ignore[arg-type]
                rec.get("label", []),  # type: ignore[arg-type]
                n_bins=int(rec.get("n_bins", 10)),
                channel=str(rec.get("channel", "hazard")),
                slice_name=str(rec.get("slice_name", "all")),
            )
        )
    return reports


def run_coverage_by_slice(inputs: TrackInputs) -> CoverageBySliceReport | None:
    """Interval coverage per slice with the doc-F 85%-95% acceptance band.

    Expects ``coverage_by_slice_inputs`` to carry aligned ``lower`` / ``upper`` /
    ``truth`` arrays and a ``slice_labels`` list. Returns ``None`` if not present.
    """
    rec = inputs.coverage_by_slice_inputs
    if not rec or "slice_labels" not in rec:
        return None
    band = rec.get(
        "band",
        (THRESHOLDS["coverage_band_lo"], THRESHOLDS["coverage_band_hi"]),
    )
    return coverage_by_slice(
        rec.get("lower", []),  # type: ignore[arg-type]
        rec.get("upper", []),  # type: ignore[arg-type]
        rec.get("truth", []),  # type: ignore[arg-type]
        list(rec.get("slice_labels", [])),  # type: ignore[arg-type]
        nominal=float(rec.get("nominal", 0.90)),
        band=(float(band[0]), float(band[1])),  # type: ignore[index]
    )


def run_operator_usefulness(inputs: TrackInputs) -> OperatorUsefulnessReport | None:
    """Operator-usefulness study (provisional; doc O L10). None if no cases.

    Doc F: operator usefulness is the fifth, deployment-readiness layer — measured
    from interaction traces, reported provisionally, and never release-blocking.
    """
    if not inputs.operator_cases:
        return None
    return evaluate_operator_usefulness(inputs.operator_cases)


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


def _slice_power(
    n_windows: int, n_branches: int, rho_pair: float, tau_mix: float
) -> dict[str, object]:
    """Per-slice statistical-power certification against doc O's ~750/slice.

    Computes the dependence-corrected effective window count
    ``n_eff = n_windows / tau_mix`` (mixing ESS, doc O E2) and the effective
    pairwise count, and flags the slice as *under-powered* when the effective
    window count is below :data:`MIN_EFFECTIVE_WINDOWS_PER_SLICE` (doc O
    sample_complexity: each major reported slice needs ~750 effective windows for
    calibration/coverage). An under-powered slice's metrics are still reported,
    but the report marks them as not yet certifiable.
    """
    window_eff = mixing_effective_n(n_windows, tau_mix) if n_windows else 0.0
    pair_eff = (
        effective_pairwise_samples(n_windows, n_branches, rho_pair)
        if n_windows
        else 0.0
    )
    return {
        "n_windows": n_windows,
        "effective_n_windows": window_eff,
        "effective_n_pairwise": pair_eff,
        "min_effective_windows": MIN_EFFECTIVE_WINDOWS_PER_SLICE,
        "sufficiently_powered": window_eff >= MIN_EFFECTIVE_WINDOWS_PER_SLICE,
        "under_powered": window_eff < MIN_EFFECTIVE_WINDOWS_PER_SLICE,
    }


def compute_slices(inputs: TrackInputs) -> dict[str, object]:
    """Group slice-wise metrics by each prespecified Mondrian slice (doc F/O).

    For each slice family and cell we recompute:

    * the headline counterfactual report (sign accuracy / optimism bias / top-1
      near-oracle) over the cell's windows;
    * a **per-slice risk-coverage** readout when per-window risk / selective-error
      arrays are supplied (parallel to ``branch_windows``);
    * a **per-slice statistical-power certification** against doc O's
      ~750-effective-windows requirement, flagging under-powered cells.

    Requires ``window_features`` aligned with ``branch_windows``. Doc F: always
    publish slice-wise metrics rather than a single average; doc O: each major
    reported slice needs ~750 effective windows.
    """
    out: dict[str, object] = {}
    feats = inputs.window_features
    windows = inputs.branch_windows
    if not feats or len(feats) != len(windows):
        return out
    # optional per-window risk / selective-error arrays, aligned to branch_windows
    risk = inputs.risk_score if len(inputs.risk_score) == len(windows) else None
    sel_err = (
        inputs.selective_error
        if len(inputs.selective_error) == len(windows)
        else None
    )
    n_under_powered = 0
    n_cells = 0
    for family in SLICE_FAMILIES:
        cells: dict[str, dict[str, object]] = {}
        # bucket window indices by this family's cell label
        idx_buckets: dict[str, list[int]] = {}
        for i, rec in enumerate(feats):
            label = assign_slices(rec)[family.name]
            idx_buckets.setdefault(label, []).append(i)
        for label, idxs in idx_buckets.items():
            ws = [windows[i] for i in idxs]
            rep = evaluate_counterfactual(ws)
            power = _slice_power(
                len(ws), inputs.n_branches, inputs.rho_pair, inputs.tau_mix
            )
            cell: dict[str, object] = {
                "n_windows": rep.n_windows,
                "sign_accuracy": rep.sign_accuracy,
                "optimism_bias_rate": rep.optimism_bias_rate,
                "top1_near_oracle": rep.top1_near_oracle,
                "power": power,
            }
            if risk is not None and sel_err is not None and idxs:
                rc = risk_coverage_curve(
                    [risk[i] for i in idxs],
                    [sel_err[i] for i in idxs],
                    target_coverage=THRESHOLDS["min_coverage"],
                )
                cell["risk_coverage"] = {
                    "aurc": rc.aurc,
                    "selective_risk_at_target": rc.selective_risk_at_target,
                    "target_coverage": rc.target_coverage,
                }
            cells[label] = cell
            n_cells += 1
            if power["under_powered"]:
                n_under_powered += 1
        out[family.name] = cells
    out["_power_summary"] = {
        "min_effective_windows_per_slice": MIN_EFFECTIVE_WINDOWS_PER_SLICE,
        "n_cells": n_cells,
        "n_under_powered_cells": n_under_powered,
        "all_cells_sufficiently_powered": n_under_powered == 0 and n_cells > 0,
    }
    return out


def counterfactual_coverage(windows: Sequence[BranchWindow]) -> float:
    """Fraction of decision windows the model answered (did not fully abstain).

    Doc F acceptance for counterfactual ranking requires the >=0.80 non-abstained
    accuracy to be retained *while keeping >= 0.60 coverage* — a model that
    abstains on everything but the easy cases would otherwise pass trivially. A
    window counts as *answered* when at least one of its candidate actions was
    recommended without abstaining (or, absent recommendation flags, when at least
    one action was not abstained on). Returns NaN for an empty set.
    """
    if not windows:
        return float("nan")
    answered = 0
    for w in windows:
        ids = w.action_ids
        if w.recommended:
            did = any(
                w.recommended.get(a, False) and not w.abstained.get(a, False)
                for a in ids
            )
        else:
            did = any(not w.abstained.get(a, False) for a in ids)
        if did:
            answered += 1
    return answered / len(windows)


def apply_acceptance(
    *,
    safety_gate: SafetyGateReport | None,
    red_team: RedTeamReport | None,
    counterfactual: CounterfactualReport | None,
    planning: PlanningReport | None,
    abstention_shift: ShiftAbstentionReport | None,
    risk_coverage: RiskCoverageReport | None,
    counterfactual_coverage_value: float | None = None,
    coverage_by_slice: CoverageBySliceReport | None = None,
    operator_usefulness: OperatorUsefulnessReport | None = None,
) -> dict[str, object]:
    """Evaluate each doc-F acceptance threshold; return a pass/fail map.

    Hard-safety and red-team are release-blocking; the rest are profile claims
    reported with their pass/fail flags but never folded into a blended score.

    ``counterfactual_coverage_value`` is the answered-window coverage used to gate
    the >=0.80 non-abstained accuracy clause at >=0.60 coverage (doc F). When it
    is below the floor, the counterfactual_ranking acceptance *fails* even with a
    high non-abstained accuracy, because the high accuracy was bought by
    abstaining too often.
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
        cov = counterfactual_coverage_value
        # Coverage gate: NaN coverage (no abstention flags) is treated as full
        # coverage so a pack without flags is not penalised; a measured coverage
        # below the floor fails the non-abstained clause.
        coverage_ok = cov is None or cov != cov or _ge(cov, THRESHOLDS["min_coverage"])
        acc["counterfactual_ranking"] = {
            "passed": (
                _ge(counterfactual.sign_accuracy, THRESHOLDS["min_sign_accuracy"])
                and _ge(
                    counterfactual.sign_accuracy_non_abstained,
                    THRESHOLDS["min_sign_accuracy_non_abstained"],
                )
                and coverage_ok
            ),
            "detail": f"sign_acc={counterfactual.sign_accuracy:.4f} "
            f"(>=0.70), non_abstained={counterfactual.sign_accuracy_non_abstained:.4f} "
            f"(>=0.80), coverage={_fmt_cov(cov)} (>=0.60)",
        }
        acc["optimism_bias"] = {
            "passed": _le(
                counterfactual.optimism_bias_rate,
                THRESHOLDS["max_optimism_bias"],
            ),
            "detail": f"rate={counterfactual.optimism_bias_rate:.4f} (<0.05)",
        }
    if coverage_by_slice is not None:
        per = coverage_by_slice.per_slice
        out_of_band = sorted(
            s for s, m in per.items() if not bool(m.get("in_band", False))
        )
        acc["interval_coverage_band"] = {
            "passed": len(per) > 0 and not out_of_band,
            "detail": (
                f"nominal={coverage_by_slice.nominal:.2f} band "
                f"[{coverage_by_slice.band_lo:.2f},{coverage_by_slice.band_hi:.2f}]; "
                f"{len(per) - len(out_of_band)}/{len(per)} slices in band"
                + (f"; out-of-band: {out_of_band}" if out_of_band else "")
            ),
        }
    if operator_usefulness is not None:
        ou = operator_usefulness
        acc["operator_usefulness"] = {
            "passed": ou.passed,
            "detail": (
                f"median_decision_time_reduction="
                f"{_fmt_cov(ou.decision_time.median_reduction)} (>=0.20), "
                f"no_safety_loss={ou.overrides.no_safety_loss}"
            ),
            "provisional": True,
            "release_blocking": False,
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


def _fmt_cov(x: float | None) -> str:
    if x is None or x != x:  # None or NaN
        return "n/a"
    return f"{x:.4f}"


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
        # Build the strongest of the six doc-F baselines (when per-window baseline
        # decision inputs are supplied) so the planning-utility threshold is
        # measured against the best skeptical comparator (doc F planning_utility).
        strongest_baseline = compute_strongest_baseline(inputs)
        planning = run_track_planning(inputs, strongest_baseline)
        predictive = run_track_predictive(inputs)
        calibration = run_track_calibration(inputs)
        coverage_slice = run_coverage_by_slice(inputs)
        operator = run_operator_usefulness(inputs)
        risk_cov, shift_ab = run_abstention(inputs)
        slices = compute_slices(inputs)
        # Fairness hooks (doc V H9): the v2 fairness report is carried inside the
        # report's `slices` surface under a dedicated `_fairness` key so it is
        # published slice-wise alongside the safety-first slices without needing
        # a new top-level report field. Absent for a v1 run (no fairness windows).
        fairness = run_track_fairness(inputs)
        fairness_slices = compute_fairness_slices(inputs)
        if fairness_slices:
            slices["_fairness"] = fairness_slices

        # Baseline comparator profile (doc F baseline_systems): publish each
        # baseline's mean safe regret and which one is the strongest comparator
        # the planning-utility threshold was computed against. Surfaced in the
        # report's free-form `slices` map under a dedicated key (no new top-level
        # field, so the EvaluationReport surface stays byte-stable).
        if strongest_baseline is not None:
            slices["_baselines"] = strongest_baseline.as_dict()

        # Ablation profile (doc F ablations): surface the five per-ablation
        # main_readouts in the report so the ablation evidence ships with it.
        if inputs.ablation_report:
            slices["_ablations"] = dict(inputs.ablation_report)

        cf_coverage = (
            counterfactual_coverage(inputs.branch_windows)
            if inputs.branch_windows
            else None
        )

        sample_plan: SampleComplexityPlan | None = None
        if inputs.branch_windows:
            # Use the measured non-abstained coverage to size the non-abstained
            # certification clause honestly (doc O), rather than the 0.60 default.
            na_windows = None
            if cf_coverage is not None and cf_coverage == cf_coverage:
                na_windows = round(cf_coverage * len(inputs.branch_windows))
            sample_plan = build_certification_plan(
                n_windows=len(inputs.branch_windows),
                n_branches=inputs.n_branches,
                rho_pair=inputs.rho_pair,
                tau_mix=inputs.tau_mix,
                n_severe_shift_zero_events=(
                    inputs.n_severe_shift_zero_events
                    or sum(1 for v in inputs.shift_false_safe if v <= 0.5)
                ),
                n_non_abstained_windows=na_windows,
            )

        acceptance = apply_acceptance(
            safety_gate=safety_gate,
            red_team=red_team,
            counterfactual=counterfactual,
            planning=planning,
            abstention_shift=shift_ab,
            risk_coverage=risk_cov,
            counterfactual_coverage_value=cf_coverage,
            coverage_by_slice=coverage_slice,
            operator_usefulness=operator,
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
            predictive=[p.as_dict() for p in predictive],
            calibration=[c.as_dict() for c in calibration],
            abstention=abstention_dict,
            coverage_by_slice=coverage_slice.as_dict() if coverage_slice else None,
            operator_usefulness=operator.as_dict() if operator else None,
            slices=slices,
            sample_complexity=sample_plan.as_dict() if sample_plan else None,
            acceptance=acceptance,
            metadata={
                "elapsed_s": round(time.time() - t0, 6),
                "n_safety_judgements": len(inputs.safety_judgements),
                "n_branch_windows": len(inputs.branch_windows),
                "n_planning_windows": len(inputs.planning_windows),
                "n_predictive_reports": len(predictive),
                "n_calibration_reports": len(calibration),
                "n_operator_cases": len(inputs.operator_cases),
                "n_fairness_windows": len(inputs.fairness_windows),
                "counterfactual_coverage": cf_coverage,
                "is_shift_pack": inputs.is_shift_pack,
                "fairness": fairness.as_dict() if fairness else None,
                "strongest_baseline": (
                    getattr(strongest_baseline, "name", None)
                    if strongest_baseline is not None
                    else None
                ),
                "n_baseline_decision_windows": len(inputs.baseline_decision_windows),
                "n_ablations": (
                    len(inputs.ablation_report.get("ablations", []))
                    if inputs.ablation_report
                    else 0
                ),
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


# ──────────────────────────────────────────────────────────────────────────────
# JSON-record ingestion (deserialize a saved benchmark pack → TrackInputs)
# ──────────────────────────────────────────────────────────────────────────────
def _blocker_set(values: object) -> frozenset[BlockerGroup]:
    """Coerce a JSON list of blocker-group strings/values into a frozenset enum."""
    if not values:
        return frozenset()
    out: set[BlockerGroup] = set()
    for v in values:  # type: ignore[union-attr]
        if isinstance(v, BlockerGroup):
            out.add(v)
        else:
            out.add(BlockerGroup(str(v)))
    return frozenset(out)


def _f(rec: Mapping[str, object], key: str, default: float = 0.0) -> float:
    v = rec.get(key, default)
    return float(v)  # type: ignore[arg-type]


def _b(rec: Mapping[str, object], key: str, default: bool = False) -> bool:
    return bool(rec.get(key, default))


def _safety_judgement_from(rec: Mapping[str, object]) -> SafetyJudgement:
    return SafetyJudgement(
        action_id=str(rec.get("action_id", "")),
        recommended=_b(rec, "recommended"),
        abstained=_b(rec, "abstained"),
        kernel_feasible=_b(rec, "kernel_feasible", True),
        kernel_blockers=_blocker_set(rec.get("kernel_blockers")),
        emitted_blockers=_blocker_set(rec.get("emitted_blockers")),
    )


def _float_dict(d: object) -> dict[str, float]:
    if not d:
        return {}
    return {str(k): float(v) for k, v in dict(d).items()}  # type: ignore[arg-type]


def _bool_dict(d: object) -> dict[str, bool]:
    if not d:
        return {}
    return {str(k): bool(v) for k, v in dict(d).items()}  # type: ignore[arg-type]


def _branch_window_from(rec: Mapping[str, object]) -> BranchWindow:
    return BranchWindow(
        window_id=str(rec.get("window_id", "")),
        action_ids=[str(a) for a in rec.get("action_ids", [])],  # type: ignore[union-attr]
        pred_energy=_float_dict(rec.get("pred_energy")),
        true_energy=_float_dict(rec.get("true_energy")),
        baseline_action_id=(
            str(rec["baseline_action_id"])
            if rec.get("baseline_action_id") is not None
            else None
        ),
        recommended=_bool_dict(rec.get("recommended")),
        abstained=_bool_dict(rec.get("abstained")),
        ref_unsafe=_bool_dict(rec.get("ref_unsafe")),
        pred_risk=_float_dict(rec.get("pred_risk")),
    )


def _lex_keys(d: object) -> dict[str, tuple[float, ...]]:
    if not d:
        return {}
    return {
        str(k): tuple(float(x) for x in v)  # type: ignore[union-attr]
        for k, v in dict(d).items()
    }


def _planning_window_from(rec: Mapping[str, object]) -> PlanningWindow:
    return PlanningWindow(
        window_id=str(rec.get("window_id", "")),
        chosen_action_id=str(rec.get("chosen_action_id", "")),
        true_lex_keys=_lex_keys(rec.get("true_lex_keys")),
        safe_action_ids=[str(a) for a in rec.get("safe_action_ids", [])],  # type: ignore[union-attr]
        baseline_chosen_id=(
            str(rec["baseline_chosen_id"])
            if rec.get("baseline_chosen_id") is not None
            else None
        ),
    )


def _fairness_window_from(rec: Mapping[str, object]) -> FairnessEvalWindow:
    per_debt = {
        str(k): float(v) for k, v in dict(rec.get("per_tenant_debt", {})).items()  # type: ignore[arg-type]
    }
    per_gaming = {
        str(t): {str(s): float(x) for s, x in dict(d).items()}
        for t, d in dict(rec.get("per_tenant_gaming", {})).items()  # type: ignore[arg-type]
    }
    return FairnessEvalWindow(
        window_id=str(rec.get("window_id", "")),
        fd_h=_f(rec, "fd_h", float("nan")),
        benchmark_welfare=_f(rec, "benchmark_welfare", float("nan")),
        realized_welfare=_f(rec, "realized_welfare", float("nan")),
        per_tenant_debt=per_debt,
        per_tenant_gaming=per_gaming,
        axioms=dict(rec.get("axioms", {})),  # type: ignore[arg-type]
        optimality=dict(rec.get("optimality", {})),  # type: ignore[arg-type]
        share_stability=dict(rec.get("share_stability", {})),  # type: ignore[arg-type]
    )


def _red_team_case_from(rec: Mapping[str, object]) -> RedTeamCase:
    return RedTeamCase(
        scenario=str(rec.get("scenario", "")),
        case_id=str(rec.get("case_id", "")),
        recommended_action_id=(
            str(rec["recommended_action_id"])
            if rec.get("recommended_action_id") is not None
            else None
        ),
        abstained=_b(rec, "abstained"),
        fell_back=_b(rec, "fell_back"),
        emitted_blockers=_blocker_set(rec.get("emitted_blockers")),
        kernel_feasible=_b(rec, "kernel_feasible", True),
        kernel_blockers=_blocker_set(rec.get("kernel_blockers")),
        reference_unsafe=_b(rec, "reference_unsafe"),
        is_ood=_b(rec, "is_ood"),
        policy_changed_mid_horizon=_b(rec, "policy_changed_mid_horizon"),
        respected_policy_change=_b(rec, "respected_policy_change", True),
        explained_shrinkage=_b(rec, "explained_shrinkage"),
        predicted_high_risk=_b(rec, "predicted_high_risk"),
        uncertainty_increased=_b(rec, "uncertainty_increased"),
        chose_lower_regret_multistep=_b(rec, "chose_lower_regret_multistep"),
        telemetry_degraded=_b(rec, "telemetry_degraded"),
        fragmentation_trap=_b(rec, "fragmentation_trap"),
        derate_occurred=_b(rec, "derate_occurred"),
        triggers_present=frozenset(
            str(x) for x in rec.get("triggers_present", [])  # type: ignore[union-attr]
        ),
    )


def _predictive_report_from(rec: Mapping[str, object]) -> PredictiveReport:
    pinball = {
        float(k): float(v) for k, v in dict(rec.get("pinball", {})).items()  # type: ignore[arg-type]
    }
    pr_at_budget = {
        float(k): {str(kk): float(vv) for kk, vv in dict(v).items()}
        for k, v in dict(rec.get("pr_at_budget", {})).items()  # type: ignore[arg-type]
    }
    return PredictiveReport(
        channel=str(rec.get("channel", "channel")),
        horizon_s=int(rec.get("horizon_s", 0)),  # type: ignore[arg-type]
        n=int(rec.get("n", 0)),  # type: ignore[arg-type]
        mae=_f(rec, "mae", float("nan")),
        medae=_f(rec, "medae", float("nan")),
        wape=_f(rec, "wape", float("nan")),
        rmse=_f(rec, "rmse", float("nan")),
        pinball=pinball,
        coverage_90=_f(rec, "coverage_90", float("nan")),
        sharpness_90=_f(rec, "sharpness_90", float("nan")),
        brier=_f(rec, "brier", float("nan")),
        pr_at_budget=pr_at_budget,
    )


def _calibration_report_from(rec: Mapping[str, object]) -> CalibrationReport:
    from .metrics.calibration import ReliabilityBin

    bins = [
        ReliabilityBin(
            lo=float(b.get("lo", 0.0)),
            hi=float(b.get("hi", 0.0)),
            count=int(b.get("count", 0)),
            mean_pred=float(b.get("mean_pred", float("nan"))),
            mean_truth=float(b.get("mean_truth", float("nan"))),
        )
        for b in rec.get("bins", [])  # type: ignore[union-attr]
    ]
    return CalibrationReport(
        channel=str(rec.get("channel", "hazard")),
        slice_name=str(rec.get("slice_name", "all")),
        n=int(rec.get("n", 0)),  # type: ignore[arg-type]
        ece=_f(rec, "ece", float("nan")),
        mce=_f(rec, "mce", float("nan")),
        bins=bins,
    )


def _operator_case_from(rec: Mapping[str, object]) -> OperatorCase:
    return OperatorCase(
        case_id=str(rec.get("case_id", "")),
        assistant_spoke=_b(rec, "assistant_spoke"),
        assistant_correct=_b(rec, "assistant_correct"),
        operator_followed=_b(rec, "operator_followed"),
        operator_decision_correct=_b(rec, "operator_decision_correct"),
        unaided_decision_correct=_b(rec, "unaided_decision_correct"),
        decision_time_s=_f(rec, "decision_time_s", float("nan")),
        unaided_decision_time_s=_f(rec, "unaided_decision_time_s", float("nan")),
        final_decision_unsafe=_b(rec, "final_decision_unsafe"),
        unaided_decision_unsafe=_b(rec, "unaided_decision_unsafe"),
        is_blocked_case=_b(rec, "is_blocked_case"),
        emitted_blockers=frozenset(
            str(x) for x in rec.get("emitted_blockers", [])  # type: ignore[union-attr]
        ),
        operator_blockers=frozenset(
            str(x) for x in rec.get("operator_blockers", [])  # type: ignore[union-attr]
        ),
    )


def track_inputs_from_records(
    records: Sequence[dict[str, object]] | Mapping[str, object],
) -> TrackInputs:
    """Deserialize a saved benchmark pack (JSON records) into :class:`TrackInputs`.

    The release format for collected records is JSON. This adapter accepts either:

    * a **payload mapping** with named typed sections — the canonical saved-pack
      shape ``{"safety_judgements": [...], "branch_windows": [...],
      "planning_windows": [...], "red_team_cases": {scenario: [...]},
      "predictive_reports": [...], "calibration_reports": [...],
      "quantile_channel_inputs": [...], "hazard_channel_inputs": [...],
      "calibration_inputs": [...], "coverage_by_slice_inputs": {...},
      "operator_cases": [...], "window_features": [...], ...sample-complexity
      scalars...}``; or
    * a **flat sequence of tagged records** ``[{"kind": "safety_judgement", ...},
      {"kind": "branch_window", ...}, ...]`` where each record carries a ``kind``
      naming its target section (singular or plural accepted).

    Every section is optional, so a partial pack (e.g. only the safety track on a
    CI commit) still produces a valid :class:`TrackInputs`. This populates ALL
    data-dependent tracks so a saved pack runs the full three-track scoring plus
    the predictive/calibration/coverage/operator-usefulness diagnostics.
    """
    if isinstance(records, Mapping):
        return _track_inputs_from_payload(records)
    # Flat tagged-record sequence: fold into a payload mapping by ``kind``.
    payload: dict[str, object] = {}
    red_team: dict[str, list[dict[str, object]]] = {}
    _KIND_TO_SECTION = {
        "safety_judgement": "safety_judgements",
        "safety_judgements": "safety_judgements",
        "branch_window": "branch_windows",
        "branch_windows": "branch_windows",
        "planning_window": "planning_windows",
        "planning_windows": "planning_windows",
        "baseline_planning_window": "baseline_planning_windows",
        "baseline_planning_windows": "baseline_planning_windows",
        "predictive_report": "predictive_reports",
        "predictive_reports": "predictive_reports",
        "calibration_report": "calibration_reports",
        "calibration_reports": "calibration_reports",
        "quantile_channel_input": "quantile_channel_inputs",
        "quantile_channel_inputs": "quantile_channel_inputs",
        "hazard_channel_input": "hazard_channel_inputs",
        "hazard_channel_inputs": "hazard_channel_inputs",
        "calibration_input": "calibration_inputs",
        "calibration_inputs": "calibration_inputs",
        "operator_case": "operator_cases",
        "operator_cases": "operator_cases",
        "window_feature": "window_features",
        "window_features": "window_features",
        "fairness_window": "fairness_windows",
        "fairness_windows": "fairness_windows",
    }
    for rec in records:
        kind = str(rec.get("kind", "")).strip()
        body = {k: v for k, v in rec.items() if k != "kind"}
        if kind in ("red_team_case", "red_team_cases"):
            scen = str(body.get("scenario", ""))
            red_team.setdefault(scen, []).append(body)
            continue
        section = _KIND_TO_SECTION.get(kind)
        if section is None:
            raise ValueError(
                f"track_inputs_from_records: unknown record kind {kind!r}; expected "
                f"one of {sorted(set(_KIND_TO_SECTION) | {'red_team_case'})}"
            )
        payload.setdefault(section, []).append(body)  # type: ignore[union-attr]
    if red_team:
        payload["red_team_cases"] = red_team
    return _track_inputs_from_payload(payload)


def _track_inputs_from_payload(payload: Mapping[str, object]) -> TrackInputs:
    """Build :class:`TrackInputs` from a named-section payload mapping."""

    def _seq(key: str) -> list[Mapping[str, object]]:
        v = payload.get(key, [])
        return list(v) if v else []  # type: ignore[arg-type]

    red_team_cases: dict[str, list[RedTeamCase]] = {}
    rt_payload = payload.get("red_team_cases", {})
    if isinstance(rt_payload, Mapping):
        for scen, cases in rt_payload.items():
            red_team_cases[str(scen)] = [
                _red_team_case_from(c) for c in cases  # type: ignore[union-attr]
            ]
    elif rt_payload:
        # accept a flat list of cases carrying their own ``scenario`` field
        for c in rt_payload:  # type: ignore[union-attr]
            case = _red_team_case_from(c)
            red_team_cases.setdefault(case.scenario, []).append(case)

    coverage_inputs = payload.get("coverage_by_slice_inputs", {})

    return TrackInputs(
        safety_judgements=[
            _safety_judgement_from(r) for r in _seq("safety_judgements")
        ],
        red_team_cases=red_team_cases,
        branch_windows=[_branch_window_from(r) for r in _seq("branch_windows")],
        predictive_reports=[
            _predictive_report_from(r) for r in _seq("predictive_reports")
        ],
        calibration_reports=[
            _calibration_report_from(r) for r in _seq("calibration_reports")
        ],
        quantile_channel_inputs=[dict(r) for r in _seq("quantile_channel_inputs")],
        hazard_channel_inputs=[dict(r) for r in _seq("hazard_channel_inputs")],
        calibration_inputs=[dict(r) for r in _seq("calibration_inputs")],
        coverage_by_slice_inputs=(
            dict(coverage_inputs) if isinstance(coverage_inputs, Mapping) else {}
        ),
        planning_windows=[_planning_window_from(r) for r in _seq("planning_windows")],
        baseline_planning_windows=[
            _planning_window_from(r) for r in _seq("baseline_planning_windows")
        ],
        operator_cases=[_operator_case_from(r) for r in _seq("operator_cases")],
        fairness_windows=[
            _fairness_window_from(r) for r in _seq("fairness_windows")
        ],
        fairness_slice_labels={
            str(k): str(v)
            for k, v in dict(payload.get("fairness_slice_labels", {})).items()  # type: ignore[arg-type]
        },
        fairness_poa=(
            float(payload["fairness_poa"])  # type: ignore[arg-type]
            if payload.get("fairness_poa") is not None
            else None
        ),
        risk_score=[float(x) for x in payload.get("risk_score", [])],  # type: ignore[union-attr]
        selective_error=[float(x) for x in payload.get("selective_error", [])],  # type: ignore[union-attr]
        shift_abstained_or_fellback=[
            float(x) for x in payload.get("shift_abstained_or_fellback", [])  # type: ignore[union-attr]
        ],
        shift_false_safe=[float(x) for x in payload.get("shift_false_safe", [])],  # type: ignore[union-attr]
        window_features=[dict(r) for r in _seq("window_features")],
        n_branches=int(payload.get("n_branches", 4)),  # type: ignore[arg-type]
        rho_pair=float(payload.get("rho_pair", 0.4)),  # type: ignore[arg-type]
        tau_mix=float(payload.get("tau_mix", 2.0)),  # type: ignore[arg-type]
        n_severe_shift_zero_events=int(
            payload.get("n_severe_shift_zero_events", 0)  # type: ignore[arg-type]
        ),
        is_shift_pack=bool(payload.get("is_shift_pack", False)),
        baseline_include_oracle=bool(payload.get("baseline_include_oracle", False)),
        ablation_report=(
            dict(payload["ablation_report"])  # type: ignore[arg-type]
            if isinstance(payload.get("ablation_report"), Mapping)
            else None
        ),
    )


__all__ = [
    "THRESHOLDS",
    "TRACKS",
    "BenchmarkRunner",
    "TrackInputs",
    "apply_acceptance",
    "compute_fairness_slices",
    "compute_slices",
    "compute_strongest_baseline",
    "counterfactual_coverage",
    "run_abstention",
    "run_benchmark",
    "run_coverage_by_slice",
    "run_operator_usefulness",
    "run_track_calibration",
    "run_track_counterfactual",
    "run_track_fairness",
    "run_track_planning",
    "run_track_predictive",
    "run_track_semantics",
    "track_inputs_from_records",
]
