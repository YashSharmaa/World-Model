# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#
#  SPDX-FileCopyrightText: 2026 Yash Sharma <https://www.linkedin.com/in/yashsharmaa/>
#  SPDX-License-Identifier: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Official scoring entrypoint — Track A, Semantics (doc F / doc H).

The hard-safety track. Release-blocking. This entrypoint loads the released
semantics pack (``pack/eval_inputs.json``: kernel-anchored state-action records)
and scores it with the in-tree, authoritative
:class:`evaluation.runner.BenchmarkRunner` — the SAME evaluator the rest of the
project uses, so the benchmark and the library can never disagree.

It is intentionally a thin, NEW orchestration file under ``benchmarks/`` and does
NOT modify ``evaluation/runner.py`` (another lane owns the runner). It re-exports
the runner output verbatim.

Scored objects (doc F): feasibility violation count (**must be 0**), blocker-set
exact match on blocked actions (**zero discrepancy**), false-safe recommendation
rate, and the seven red-team scenarios. By design the released safety pack must
keep the safety gate green; the red-team gate additionally needs all seven
scenarios populated (the shipped demo pack carries the canonical "infeasible but
tempting" scenario, so a partial red-team coverage is reported honestly rather
than faked).

Usage:

    python benchmarks/semantics/score.py [--data DIR] [--out DIR] [--enforce]

``--enforce`` exits non-zero if any *release-blocking* acceptance check with
present data fails (use it for a full release pack); the default is report-mode,
which always writes the reports and returns 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``src/`` and the shared ``benchmarks/`` driver importable when run from a
# checkout without an install.
_TRACK_DIR = Path(__file__).resolve().parent
_BENCH_DIR = _TRACK_DIR.parent
_REPO_ROOT = _BENCH_DIR.parent
for _p in (_REPO_ROOT / "src", _BENCH_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _scoring_common import run_track  # noqa: E402

TRACK = "semantics"


def main(argv: list[str] | None = None) -> int:
    return run_track(TRACK, _TRACK_DIR, argv)


if __name__ == "__main__":
    raise SystemExit(main())
