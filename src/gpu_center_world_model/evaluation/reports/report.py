# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Report emission: JSON + Markdown, *never a single blended score* (doc F).

Doc F: "I would also avoid publishing a single leaderboard score. Instead,
publish a safety gate and then a small profile." This module renders an
:class:`EvaluationReport` — a structured container of every metric family's
result plus the provable-vs-empirical ledger — into two artifacts:

* a machine-readable JSON document (the source of truth for downstream tooling),
* a human-readable Markdown report that leads with the **safety gate** (pass or
  release-blocked), then shows the small profile (counterfactual sign accuracy,
  safe regret, calibration, red-team pass rate), then the slice-wise tables, then
  the provable-vs-empirical ledger.

The Markdown deliberately puts a prominent RELEASE-BLOCKED banner at the top when
the safety gate or red-team suite fails, so a reader cannot miss it. Attribution
travels with both artifacts (project watermark).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..._watermark import AUTHOR, AUTHOR_URL, CREDIT, PROJECT
from .provable_vs_empirical import ledger_as_dict


@dataclass(slots=True)
class EvaluationReport:
    """A structured, multi-family evaluation report (never one blended number).

    Each field holds the ``as_dict``-style result of one metric family or suite.
    ``acceptance`` records pass/fail per doc-F threshold. The report is
    *release-blocked* whenever the safety gate or red-team suite fails.
    """

    run_id: str
    schema_version: str
    track: str = "all"
    safety_gate: dict[str, object] | None = None
    red_team: dict[str, object] | None = None
    counterfactual: dict[str, object] | None = None
    planning: dict[str, object] | None = None
    predictive: list[dict[str, object]] = field(default_factory=list)
    calibration: list[dict[str, object]] = field(default_factory=list)
    abstention: dict[str, object] | None = None
    # interval coverage-by-slice (doc F 85%-95% band, shift-aware)
    coverage_by_slice: dict[str, object] | None = None
    # operator-usefulness profile (provisional; doc O ledger L10)
    operator_usefulness: dict[str, object] | None = None
    slices: dict[str, object] = field(default_factory=dict)
    sample_complexity: dict[str, object] | None = None
    acceptance: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    # ── release gating ──────────────────────────────────────────────────────
    @property
    def safety_gate_passed(self) -> bool:
        return bool(self.safety_gate and self.safety_gate.get("passed"))

    @property
    def red_team_passed(self) -> bool:
        return bool(self.red_team and self.red_team.get("passed"))

    @property
    def release_blocked(self) -> bool:
        """A release is blocked iff a release-blocking gate fails (doc F)."""
        blocked = False
        if self.safety_gate is not None and not self.safety_gate_passed:
            blocked = True
        if self.red_team is not None and not self.red_team_passed:
            blocked = True
        return blocked

    # ── serialization ───────────────────────────────────────────────────────
    def as_dict(self) -> dict[str, object]:
        """Assemble the JSON-serialisable result dict (incl. the doc-O ledger)."""
        results_for_ledger = {
            "safety_gate": self.safety_gate,
            "counterfactual": self.counterfactual,
            "planning": self.planning,
            "calibration": self.calibration,
            "abstention": self.abstention,
            "operator_usefulness": self.operator_usefulness,
            "sample_complexity": self.sample_complexity,
        }
        return {
            "project": PROJECT,
            "author": AUTHOR,
            "author_url": AUTHOR_URL,
            "credit": CREDIT,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "track": self.track,
            "release_blocked": self.release_blocked,
            "safety_gate_passed": self.safety_gate_passed,
            "red_team_passed": self.red_team_passed,
            "safety_gate": self.safety_gate,
            "red_team": self.red_team,
            "counterfactual": self.counterfactual,
            "planning": self.planning,
            "predictive": self.predictive,
            "calibration": self.calibration,
            "abstention": self.abstention,
            "coverage_by_slice": self.coverage_by_slice,
            "operator_usefulness": self.operator_usefulness,
            "slices": self.slices,
            "sample_complexity": self.sample_complexity,
            "acceptance": self.acceptance,
            "provable_vs_empirical": ledger_as_dict(results_for_ledger),
            "metadata": self.metadata,
            "note": (
                "No single blended score is published (doc F). Safety is a gate, "
                "not a weighted metric."
            ),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        return render_markdown(self)


def _fmt(x: object) -> str:
    if isinstance(x, float):
        if x != x:  # NaN
            return "n/a"
        return f"{x:.4f}"
    return str(x)


def _acceptance_table(acceptance: dict[str, object]) -> list[str]:
    if not acceptance:
        return ["_No acceptance thresholds evaluated._", ""]
    lines = ["| Threshold | Result | Pass |", "|---|---|---|"]
    for name, val in acceptance.items():
        if isinstance(val, dict):
            ok = val.get("passed")
            detail = val.get("detail", val.get("value", ""))
            mark = "PASS" if ok else "FAIL"
            lines.append(f"| {name} | {_fmt(detail)} | {mark} |")
        else:
            lines.append(f"| {name} | {_fmt(val)} |  |")
    lines.append("")
    return lines


def render_markdown(report: EvaluationReport) -> str:
    """Render a human-readable Markdown report (safety gate first)."""
    lines: list[str] = []
    lines.append(f"# {PROJECT} — Evaluation Report")
    lines.append("")
    lines.append(
        f"_Run `{report.run_id}` · schema `{report.schema_version}` · "
        f"track `{report.track}`_"
    )
    lines.append("")
    lines.append(f"Author: {AUTHOR} · {AUTHOR_URL}")
    lines.append("")

    # 1. Safety gate banner — leads the report.
    if report.release_blocked:
        lines.append("> ## ⛔ RELEASE BLOCKED")
        lines.append(">")
        lines.append(
            "> A release-blocking safety gate failed. Per the project "
            "spine, safety is a gate, not a weighted metric — this run "
            "cannot be released regardless of any other metric."
        )
    else:
        lines.append("> ## ✅ Safety gate clear")
        lines.append(">")
        lines.append("> No release-blocking failures detected on the evaluated packs.")
    lines.append("")

    lines.append("## 1. Safety gate (release-blocking)")
    lines.append("")
    if report.safety_gate:
        sg = report.safety_gate
        lines.append(
            f"- Exact-kernel violation count: **{sg.get('violation_count')}** "
            "(spine: must be 0)"
        )
        lines.append(f"- False-safe rate: **{_fmt(sg.get('false_safe_rate'))}**")
        lines.append(
            f"- Blocker-set exact match: "
            f"**{sg.get('blocker_exact_match_count')}/"
            f"{sg.get('n_blocked_compared')}**"
        )
        lines.append(
            f"- Spurious blocks (feasible but blocked): "
            f"{sg.get('spurious_block_count')}"
        )
        lines.append(f"- Gate passed: **{sg.get('passed')}**")
        if sg.get("failures"):
            lines.append("")
            lines.append("Failures:")
            for f in sg["failures"]:  # type: ignore[index]
                lines.append(f"  - {f}")
    else:
        lines.append("_Safety gate not evaluated for this run._")
    lines.append("")

    # 2. Red team
    lines.append("## 2. Red-team suite (release-blocking)")
    lines.append("")
    if report.red_team:
        rt = report.red_team
        lines.append(f"- Overall pass rate: **{_fmt(rt.get('overall_pass_rate'))}**")
        lines.append(f"- Suite passed: **{rt.get('passed')}**")
        lines.append("")
        lines.append("| Scenario | Cases | Passed | Rate |")
        lines.append("|---|---|---|---|")
        for s in rt.get("scenarios", []):  # type: ignore[union-attr]
            lines.append(
                f"| {s['scenario']} | {s['n_cases']} | {s['n_passed']} | "
                f"{_fmt(s['pass_rate'])} |"
            )
    else:
        lines.append("_Red-team suite not evaluated for this run._")
    lines.append("")

    # 3. Profile (NOT one blended score)
    lines.append("## 3. Performance profile (separate objects, no blended score)")
    lines.append("")
    if report.counterfactual:
        cf = report.counterfactual
        lines.append("### Counterfactual fidelity (primary technical target)")
        lines.append(
            f"- Pairwise sign accuracy: **{_fmt(cf.get('sign_accuracy'))}** "
            f"({cf.get('sign_accuracy_pairs')} pairs)  — target ≥ 0.70"
        )
        lines.append(
            f"- Non-abstained sign accuracy: "
            f"**{_fmt(cf.get('sign_accuracy_non_abstained'))}**  — target ≥ 0.80"
        )
        lines.append(f"- Top-1 near-oracle: {_fmt(cf.get('top1_near_oracle'))}")
        lines.append(
            f"- Optimism-bias rate: **{_fmt(cf.get('optimism_bias_rate'))}** "
            "— target < 0.05"
        )
        lines.append(
            f"- Delta MAE / WAPE: {_fmt(cf.get('delta_mae'))} / "
            f"{_fmt(cf.get('delta_wape'))}"
        )
        lines.append("")
    if report.planning:
        pl = report.planning
        lines.append("### Planning utility (safe regret)")
        lines.append(f"- Mean safe regret: {_fmt(pl.get('mean_safe_regret'))}")
        lines.append(f"- p95 safe regret: {_fmt(pl.get('p95_safe_regret'))}")
        lines.append(f"- Oracle-match rate: {_fmt(pl.get('oracle_match_rate'))}")
        lines.append(
            f"- Regret reduction vs baseline: "
            f"**{_fmt(pl.get('relative_regret_reduction'))}** — target ≥ 0.15"
        )
        lines.append(f"- Unsafe choices: {pl.get('unsafe_choice_count')}")
        lines.append("")
    if report.abstention:
        ab = report.abstention
        lines.append("### Abstention / shift")
        if "aurc" in ab:
            lines.append(f"- Risk-coverage AURC: {_fmt(ab.get('aurc'))}")
            lines.append(
                f"- Selective risk @ {_fmt(ab.get('target_coverage'))} "
                f"coverage: {_fmt(ab.get('selective_risk_at_target'))}"
            )
        if "false_safe_rate" in ab:
            upper = _fmt(ab.get("false_safe_upper_95"))
            lines.append(
                f"- False-safe under shift: "
                f"**{_fmt(ab.get('false_safe_rate'))}** "
                f"(95% upper {upper}) — target < 0.01"
            )
            lines.append(
                f"- Abstain/fallback rate: "
                f"{_fmt(ab.get('abstain_or_fallback_rate'))} — target ≥ 0.80"
            )
        lines.append("")

    # 4. Predictive diagnostic
    if report.predictive:
        lines.append("## 4. Predictive accuracy (factual-replay diagnostic)")
        lines.append("")
        lines.append(
            "| Channel | Horizon (s) | n | MAE | WAPE | Cov90 | Sharp90 | Brier |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for p in report.predictive:
            lines.append(
                f"| {p.get('channel')} | {p.get('horizon_s')} | {p.get('n')} | "
                f"{_fmt(p.get('mae'))} | {_fmt(p.get('wape'))} | "
                f"{_fmt(p.get('coverage_90'))} | {_fmt(p.get('sharpness_90'))} | "
                f"{_fmt(p.get('brier'))} |"
            )
        lines.append("")

    # 5. Calibration by slice
    if report.calibration:
        lines.append("## 5. Calibration by slice (shift-aware)")
        lines.append("")
        lines.append("| Channel | Slice | n | ECE | MCE |")
        lines.append("|---|---|---|---|---|")
        for c in report.calibration:
            lines.append(
                f"| {c.get('channel')} | {c.get('slice_name')} | {c.get('n')} | "
                f"{_fmt(c.get('ece'))} | {_fmt(c.get('mce'))} |"
            )
        lines.append("")

    # 5b. Interval coverage by slice (85%-95% band)
    if report.coverage_by_slice:
        cbs = report.coverage_by_slice
        lines.append("### Interval coverage by slice (85%-95% band)")
        lines.append(
            f"- Nominal **{_fmt(cbs.get('nominal'))}** interval; acceptance band "
            f"[{_fmt(cbs.get('band_lo'))}, {_fmt(cbs.get('band_hi'))}]"
        )
        lines.append("")
        lines.append("| Slice | n | Coverage | In band |")
        lines.append("|---|---|---|---|")
        per = cbs.get("per_slice", {})
        if isinstance(per, dict):
            for s, m in per.items():
                lines.append(
                    f"| {s} | {m.get('n')} | {_fmt(m.get('coverage'))} | "
                    f"{m.get('in_band')} |"
                )
        lines.append("")

    # 5c. Operator usefulness (provisional; never release-blocking)
    if report.operator_usefulness:
        ou = report.operator_usefulness
        lines.append("### Operator usefulness (provisional — deployment-readiness)")
        cwr = ou.get("coverage_vs_win_rate", {})
        dt = ou.get("decision_time", {})
        ov = ou.get("overrides", {})
        rel = ou.get("appropriate_reliance", {})
        lines.append(
            f"- Coverage / win-rate: {_fmt(cwr.get('coverage'))} / "
            f"{_fmt(cwr.get('win_rate'))}"
        )
        lines.append(
            f"- Median decision-time reduction: "
            f"**{_fmt(dt.get('median_reduction'))}** "
            f"(target ≥ {_fmt(dt.get('target_reduction'))})"
        )
        lines.append(
            f"- Appropriate-reliance rate: "
            f"{_fmt(rel.get('appropriate_reliance_rate'))}"
        )
        lines.append(
            f"- Unsafe-acceptance rate: {_fmt(ov.get('unsafe_acceptance_rate'))} · "
            f"no safety loss: {ov.get('no_safety_loss')}"
        )
        lines.append(
            f"- Meets doc-F operator threshold: **{ou.get('passed')}** "
            "(provisional, never release-blocking)"
        )
        lines.append("")

    # 6. Slice-wise metrics
    if report.slices:
        lines.append("## 6. Slice-wise metrics (prespecified Mondrian slices)")
        lines.append("")
        for family, cells in report.slices.items():
            lines.append(f"### {family}")
            if isinstance(cells, dict):
                for cell, metrics in cells.items():
                    lines.append(f"- `{cell}`: {json.dumps(metrics, default=str)}")
            lines.append("")

    # 7. Sample-complexity certification
    if report.sample_complexity:
        sc = report.sample_complexity
        lines.append("## 7. Sample-complexity certification (doc O)")
        lines.append("")
        lines.append(
            f"- Windows: {sc.get('n_windows')} · branches: "
            f"{sc.get('n_branches')} · rho_pair: {sc.get('rho_pair')} · "
            f"tau_mix: {sc.get('tau_mix')}"
        )
        lines.append(
            f"- All claims sufficiently powered: **{sc.get('all_sufficient')}**"
        )
        lines.append("")
        lines.append("| Claim | Required n_eff | Available n_eff | Sufficient |")
        lines.append("|---|---|---|---|")
        for chk in sc.get("checks", []):  # type: ignore[union-attr]
            lines.append(
                f"| {chk['claim']} | {_fmt(chk['required_effective_n'])} | "
                f"{_fmt(chk['available_effective_n'])} | {chk['sufficient']} |"
            )
        lines.append("")

    # 8. Acceptance thresholds
    lines.append("## 8. Acceptance thresholds (doc F)")
    lines.append("")
    lines.extend(_acceptance_table(report.acceptance))

    # 9. Provable-vs-empirical ledger
    lines.append("## 9. Provable-vs-empirical ledger (doc O, L1-L12)")
    lines.append("")
    lines.append("| Claim | Headline | Status |")
    lines.append("|---|---|---|")
    for c in ledger_as_dict()["claims"]:  # type: ignore[index]
        lines.append(f"| {c['claim_id']} | {c['headline_claim']} | {c['status']} |")
    lines.append("")
    lines.append(f"---\n\n_{CREDIT}_")
    return "\n".join(lines)


def write_reports(
    report: EvaluationReport, out_dir: str | Path, *, stem: str = "report"
) -> dict[str, str]:
    """Write ``{stem}.json`` and ``{stem}.md`` into ``out_dir``; return paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{stem}.json"
    md_path = out / f"{stem}.md"
    json_path.write_text(report.to_json(), encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


__all__ = [
    "EvaluationReport",
    "render_markdown",
    "write_reports",
]
