# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""``run_suite`` — the CLI-facing convenience entrypoint for ``gcwm eval``.

This wraps the benchmark runner so a bare ``gcwm eval`` always produces an
*honest* report: the safety-gated three-track structure, the sample-complexity
certification plan, and the provable-vs-empirical ledger. Data-dependent tracks
(counterfactual rollout, planning) are populated only when a benchmark pack with
collected per-window records is supplied; otherwise they are reported as
``null`` rather than fabricated (doc F: never publish numbers you did not earn).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .runner import (
    TRACKS,
    TrackInputs,
    load_benchmark_config_from_yaml,
    run_benchmark,
    track_inputs_from_records,
)


def _load_inputs(data: str | None) -> tuple[TrackInputs, str]:
    """Load collected per-window records from a saved pack directory or JSON file.

    Returns ``(inputs, note)``. If ``data`` is missing or holds no evaluation
    records, returns empty ``TrackInputs`` (still a valid partial run) plus a
    note explaining what is needed for the full data-dependent tracks. When the
    pack carries records, ``track_inputs_from_records`` deserializes every typed
    section (safety_judgements / branch_windows / planning_windows /
    red_team_cases / predictive / calibration / coverage / operator_cases) so the
    full three-track scoring plus diagnostics run from the saved pack.

    The pack location is resolved flexibly: ``--data`` may point at a directory
    containing ``eval_inputs.json`` (the canonical pack file), or directly at a
    JSON file. The JSON may be either the named-section payload mapping or a flat
    list of ``{"kind": ...}`` records — both shapes ``track_inputs_from_records``
    accepts.
    """
    if not data:
        return TrackInputs(), (
            "no --data pack supplied: reporting the safety-gate structure, "
            "sample-complexity certification plan, and the provable-vs-empirical "
            "ledger only. Supply a benchmark pack with collected records "
            "(safety_judgements / branch_windows / planning_windows) for the "
            "counterfactual and planning tracks."
        )
    p = Path(data)
    records = p / "eval_inputs.json" if p.is_dir() else p
    if not records.exists():
        return TrackInputs(), (
            f"--data '{data}' has no eval_inputs.json with collected records; "
            "running the structural + certification report only."
        )
    payload: dict[str, Any] | list[dict[str, Any]] = json.loads(
        records.read_text(encoding="utf-8")
    )
    try:
        inputs = track_inputs_from_records(payload)
    except (ValueError, KeyError, TypeError) as exc:
        return TrackInputs(), (
            f"--data '{records}' could not be deserialized into TrackInputs "
            f"({exc}); running the structural + certification report only."
        )
    counts = (
        f"safety={len(inputs.safety_judgements)}, "
        f"branch_windows={len(inputs.branch_windows)}, "
        f"planning_windows={len(inputs.planning_windows)}, "
        f"red_team={sum(len(v) for v in inputs.red_team_cases.values())}, "
        f"operator_cases={len(inputs.operator_cases)}"
    )
    return inputs, f"loaded full benchmark pack from {records} ({counts})"


def run_suite(
    suite: str = "all",
    data: str | None = None,
    out: str = "./eval_report",
    config: str | None = None,
) -> str:
    """Run the offline evaluation suite and write JSON + Markdown reports.

    Parameters mirror the ``gcwm eval`` CLI: ``suite`` selects a track
    (``all`` or one of :data:`TRACKS`), ``data`` optionally points at a
    benchmark pack, and ``out`` is the report output directory. Returns the
    path to the written Markdown report.

    The acceptance thresholds, slice definitions, and skeptical baselines are
    read from ``configs/eval/benchmark.yaml`` (doc H configs/eval): the run
    *replaces the hardcoded threshold defaults* with the YAML's
    ``acceptance_thresholds`` so the gating can be re-parameterised without
    editing Python. ``config`` overrides that default path; a missing config
    degrades gracefully to the doc-F default thresholds (still a valid run).
    """
    from .reports.report import write_reports  # local import: keep load light

    track = suite if suite in TRACKS else "all"
    inputs, note = _load_inputs(data)
    bench_config = load_benchmark_config_from_yaml(config)
    report = run_benchmark(
        inputs, run_id="cli-eval", track=track, config=bench_config
    )
    paths = write_reports(report, out, stem="eval")
    cfg_id = bench_config.raw.get("config_id", "doc-F defaults")
    print(f"[gcwm eval] {note}")
    print(f"[gcwm eval] acceptance profile: {cfg_id}")
    print(f"[gcwm eval] safety-gate-first report written: {paths}")
    return paths.get("markdown") or paths.get("md") or str(Path(out) / "eval.md")


__all__ = ["run_suite"]
