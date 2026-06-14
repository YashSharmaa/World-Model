# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#
#  SPDX-FileCopyrightText: 2026 Yash Sharma <https://www.linkedin.com/in/yashsharmaa/>
#  SPDX-License-Identifier: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Shared driver for the three official benchmark scoring entrypoints.

Each ``benchmarks/<track>/score.py`` is a thin wrapper around :func:`run_track`.
This keeps the official scoring logic in ONE place while still presenting a
per-track entrypoint (doc F "official scoring scripts" artifact requirement), and
it routes every track through the in-tree authoritative
:class:`evaluation.runner.BenchmarkRunner` — it never re-implements or edits the
runner. Reports are written as JSON + Markdown so CI, docs, and readers consume
the same evaluation outputs (doc F / doc H).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gpu_center_world_model.evaluation import (  # noqa: E402
    BenchmarkRunner,
    track_inputs_from_records,
)
from gpu_center_world_model.evaluation.reports.report import (  # noqa: E402
    write_reports,
)


def _load_payload(pack_dir: Path) -> dict:
    """Load ``pack/eval_inputs.json`` from a track directory (the released pack)."""
    records = pack_dir / "pack" / "eval_inputs.json"
    if not records.exists():
        raise FileNotFoundError(
            f"no benchmark pack at {records}; run "
            "`python scripts/build_benchmark_packs.py` first."
        )
    return json.loads(records.read_text(encoding="utf-8"))


def _release_blocking_failures(acceptance: dict) -> list[str]:
    """Names of release-blocking acceptance checks that FAILED with present data."""
    failed: list[str] = []
    for name, val in acceptance.items():
        if not isinstance(val, dict):
            continue
        if val.get("release_blocking") and val.get("passed") is False:
            failed.append(f"{name}: {val.get('detail', '')}")
    return failed


def run_track(track: str, track_dir: Path, argv: list[str] | None = None) -> int:
    """Score one track from its released pack and write JSON + Markdown reports.

    Returns 0 in report-mode (always). With ``--enforce`` (full release packs),
    returns 1 if any release-blocking acceptance check failed on present data.
    """
    parser = argparse.ArgumentParser(description=f"Score the {track} benchmark track.")
    parser.add_argument(
        "--data",
        default=None,
        help="track directory holding pack/eval_inputs.json (default: this track)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="report output directory (default: <track>/reports)",
    )
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="exit non-zero if a release-blocking acceptance check fails",
    )
    args = parser.parse_args(argv)

    pack_dir = Path(args.data) if args.data else track_dir
    out_dir = Path(args.out) if args.out else (track_dir / "reports")

    payload = _load_payload(pack_dir)
    inputs = track_inputs_from_records(payload)
    report = BenchmarkRunner(run_id=f"{track}-official", track=track).run(inputs)

    paths = write_reports(report, out_dir, stem=track)

    print(f"[{track}] reports written: {paths}")
    print(
        f"[{track}] safety_gate_passed={report.safety_gate_passed} "
        f"red_team_passed={report.red_team_passed}"
    )
    acceptance = report.acceptance or {}
    for name, val in acceptance.items():
        if isinstance(val, dict):
            status = "PASS" if val.get("passed") else "fail"
            blocking = " [release-blocking]" if val.get("release_blocking") else ""
            print(f"[{track}]   {name}: {status}{blocking} — {val.get('detail', '')}")

    failures = _release_blocking_failures(acceptance)
    if args.enforce and failures:
        print(
            f"::error::{track}: release-blocking acceptance failed:\n  "
            + "\n  ".join(failures),
            file=sys.stderr,
        )
        return 1
    return 0


__all__ = ["run_track"]
