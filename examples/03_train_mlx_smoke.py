#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Example 03 — a tiny smoke train of the residual world model (Research doc E/J).

Runs the staged warmup → rollout → rare-event → calibration curriculum on a
handful of self-contained toy decision windows, then writes a multi-file
checkpoint bundle (weights + feature stats + schema hash + calibration). The
whole thing finishes in well under a minute on plain CPU/NumPy — it is doc E's
"stage zero" tensor-pipeline smoke test, not an accuracy run.

Backend selection is automatic: MLX on Apple silicon (if the optional ``mlx``
wheel is installed) and NumPy everywhere else, including plain Linux CI. The
collapse diagnostics (effective rank, variance floor, action separation, doc J
D1/D2/D5) are reported so a smoke run that quietly collapsed is caught.

Run::

    python examples/03_train_mlx_smoke.py
    python examples/03_train_mlx_smoke.py --out /tmp/rwm_smoke --windows 24
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

# Example-local bootstrap (wires <repo>/src onto sys.path for a bare checkout).
from _bootstrap import CREDIT, ensure_importable

ensure_importable()

from gpu_center_world_model.model import (  # noqa: E402
    HAS_MLX,
    TrainConfig,
    get_backend,
    load_checkpoint,
    smoke_train,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("checkpoints") / "rwm_smoke",
        help="checkpoint output directory",
    )
    parser.add_argument(
        "--windows", type=int, default=16, help="number of smoke decision windows"
    )
    parser.add_argument("--seed", type=int, default=0, help="master seed")
    args = parser.parse_args()

    backend = get_backend()
    print("=" * 78)
    print("GPU-Center World Model — residual world-model smoke train")
    print("=" * 78)
    print(f"backend            : {backend.name}  (MLX available: {HAS_MLX})")
    print(f"smoke windows      : {args.windows}")
    print(f"checkpoint dir     : {args.out}")
    print("-" * 78)

    cfg = TrainConfig(seed=args.seed, verbose=False)
    t0 = time.perf_counter()
    report = smoke_train(out_dir=args.out, n_windows=args.windows, cfg=cfg)
    dt = time.perf_counter() - t0

    print("staged loss curve (mean per stage):")
    for stage, losses in report.stage_losses.items():
        if losses:
            mean = sum(losses) / len(losses)
            print(f"  {stage:<12} steps={len(losses):>3}  mean_loss={mean:>10.5f}")
    print("-" * 78)
    print(f"total steps        : {report.steps}")
    print(f"final loss         : {report.final_loss:.5f}")
    print(f"collapse alarm     : {report.collapsed()}  (doc J D1/D2/D5)")
    print(f"wall time          : {dt:.2f}s")

    # Round-trip the checkpoint to prove the bundle reloads (schema-gated).
    reloaded = load_checkpoint(args.out, backend=backend)
    print("-" * 78)
    print(
        f"checkpoint reloaded: model_version={reloaded.model_version!r} "
        f"schema_hash={reloaded.schema_hash!r}"
    )
    files = sorted(p.name for p in args.out.iterdir() if p.is_file())
    print(f"checkpoint files   : {files}")
    print("=" * 78)
    if report.collapsed():
        print("WARNING: a collapse alarm fired — inspect the diagnostics (doc J).")
    print(CREDIT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
