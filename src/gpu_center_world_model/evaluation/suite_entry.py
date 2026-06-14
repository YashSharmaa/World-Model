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

from .runner import TRACKS, TrackInputs, run_benchmark


def _load_inputs(data: str | None) -> tuple[TrackInputs, str]:
    """Best-effort load of collected per-window records from a pack directory.

    Returns ``(inputs, note)``. If ``data`` is missing or holds no evaluation
    records, returns empty ``TrackInputs`` (still a valid partial run) plus a
    note explaining what is needed for the full data-dependent tracks.
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
    records = p / "eval_inputs.json"
    if not records.exists():
        return TrackInputs(), (
            f"--data '{data}' has no eval_inputs.json with collected records; "
            "running the structural + certification report only."
        )
    payload: dict[str, Any] = json.loads(records.read_text(encoding="utf-8"))
    inputs = TrackInputs(
        n_branches=int(payload.get("n_branches", 4)),
        rho_pair=float(payload.get("rho_pair", 0.4)),
        tau_mix=float(payload.get("tau_mix", 2.0)),
        is_shift_pack=bool(payload.get("is_shift_pack", False)),
        n_severe_shift_zero_events=int(payload.get("n_severe_shift_zero_events", 0)),
    )
    return inputs, f"loaded sample-complexity inputs from {records}"


def run_suite(
    suite: str = "all",
    data: str | None = None,
    out: str = "./eval_report",
) -> str:
    """Run the offline evaluation suite and write JSON + Markdown reports.

    Parameters mirror the ``gcwm eval`` CLI: ``suite`` selects a track
    (``all`` or one of :data:`TRACKS`), ``data`` optionally points at a
    benchmark pack, and ``out`` is the report output directory. Returns the
    path to the written Markdown report.
    """
    from .reports.report import write_reports  # local import: keep load light

    track = suite if suite in TRACKS else "all"
    inputs, note = _load_inputs(data)
    report = run_benchmark(inputs, run_id="cli-eval", track=track)
    paths = write_reports(report, out, stem="eval")
    print(f"[gcwm eval] {note}")
    print(f"[gcwm eval] safety-gate-first report written: {paths}")
    return paths.get("markdown") or paths.get("md") or str(Path(out) / "eval.md")


__all__ = ["run_suite"]
