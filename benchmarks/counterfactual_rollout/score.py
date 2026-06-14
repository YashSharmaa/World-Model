# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#
#  SPDX-FileCopyrightText: 2026 Yash Sharma <https://www.linkedin.com/in/yashsharmaa/>
#  SPDX-License-Identifier: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Official scoring entrypoint — Track B, Counterfactual Rollout (doc F / doc H).

The primary technical target. This entrypoint loads the released counterfactual
pack (``pack/eval_inputs.json``: branched-replay windows from one pre-action
state hash + one shared seed bundle, each with ≥3 branched safe actions) and
scores it with the authoritative :class:`evaluation.runner.BenchmarkRunner`.

It is a NEW orchestration file under ``benchmarks/`` and does NOT modify
``evaluation/runner.py``.

Scored (doc F): pairwise sign accuracy (≥0.70 overall; ≥0.80 non-abstained at
≥0.60 coverage), top-1 near-oracle rate, delta error vs the reference branch,
optimism-bias rate (<0.05), and 90% interval coverage by slice. Pairwise sign
accuracy is the formal target of doc O's margin bound (GB4/T7).

Usage:

    python benchmarks/counterfactual_rollout/score.py [--data DIR] [--out DIR]
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

TRACK = "counterfactual_rollout"


def main(argv: list[str] | None = None) -> int:
    return run_track(TRACK, _TRACK_DIR, argv)


if __name__ == "__main__":
    raise SystemExit(main())
