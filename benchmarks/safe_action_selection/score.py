# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#
#  SPDX-FileCopyrightText: 2026 Yash Sharma <https://www.linkedin.com/in/yashsharmaa/>
#  SPDX-License-Identifier: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Official scoring entrypoint — Track C, Safe Action Selection (doc F / doc H).

Planning utility, scored by LEXICOGRAPHIC safety and regret (never a scalarized
blend). This entrypoint loads the released planning pack (``pack/eval_inputs.json``:
decision windows with the safe candidate set, the system's chosen action, the
reference-branch true lexicographic keys, and the strongest-baseline choice) and
scores it with the authoritative :class:`evaluation.runner.BenchmarkRunner`.

It is a NEW orchestration file under ``benchmarks/`` and does NOT modify
``evaluation/runner.py``.

Scored (doc F): mean / p95 safe regret vs the safe-set oracle, regret reduction
vs the strongest skeptical baseline (≥15% or a Pareto win at equal safety),
oracle-match rate, unsafe-choice count (**must be 0**, release-blocking via the
safety gate), and selective risk / abstention at 0.60 coverage. Regret is measured
over the kernel-certified safe set ONLY — the model may never expand it.

Usage:

    python benchmarks/safe_action_selection/score.py [--data DIR] [--out DIR]
    # add --enforce to fail on a release-blocking acceptance miss
"""

from __future__ import annotations

import sys
from pathlib import Path

_TRACK_DIR = Path(__file__).resolve().parent
_BENCH_DIR = _TRACK_DIR.parent
_REPO_ROOT = _BENCH_DIR.parent
for _p in (_REPO_ROOT / "src", _BENCH_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _scoring_common import run_track  # noqa: E402

TRACK = "safe_action_selection"


def main(argv: list[str] | None = None) -> int:
    return run_track(TRACK, _TRACK_DIR, argv)


if __name__ == "__main__":
    raise SystemExit(main())
