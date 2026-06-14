# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#
#  SPDX-FileCopyrightText: 2026 Yash Sharma <https://www.linkedin.com/in/yashsharmaa/>
#  SPDX-License-Identifier: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Compare-latency recorder (doc H MLX/serving gate).

Doc H ci_and_testing: *"Record p50 and p95 local compare latency on the scripted
three-action demo case."* This script records that measurement AND the companion
**enumerated-set** case the design calls for: the operator does not only compare
three branches — they can enumerate *every* kernel-safe placement domain for a
pending gang. That is a strictly larger, shape-unstable candidate set, and it must
still complete an interactive compare well under the human-perceptible **5-second**
budget at p95.

It is run from the ``macos-mlx`` CI lane (the Apple-silicon reference path) and:

* measures the **scripted three-action demo** (``hold`` + two placements) and
  records its p50/p95;
* builds one pending gang and enumerates ``hold`` + a placement onto **every**
  rack/topology domain — yielding a candidate set with **more than three**
  actions (the "enumerated set", >3) — and records its p50/p95;
* warms up first (excluding any first-call ``compile()`` cost, per the doc-H
  "second invocation avoids compile churn" criterion);
* emits the measured **p50 / p95** for both cases to the GitHub Actions *step
  summary* (``$GITHUB_STEP_SUMMARY``) so the latency is recorded on every run,
  not merely asserted; and
* FAILS (exit 1) if the enumerated-set **p95 ≥ 5 s** — the interactive-loop SLA.

It uses only the dependency-free reference engine + exact kernel, so it records a
meaningful latency floor on any runner; when MLX is present the same code path
exercises the compiled scoring graph.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

# Make ``src/`` importable when run from a checkout without an install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gpu_center_world_model.data_engine.synthetic import (  # noqa: E402
    PodSpec,
    generate_pod_state,
)
from gpu_center_world_model.schema import world_state_hash  # noqa: E402
from gpu_center_world_model.serving import (  # noqa: E402
    CandidateActionRequest,
    CompareRequest,
    EngineBundle,
    compare_actions,
)
from gpu_center_world_model.simulator import ExactSymbolicKernel  # noqa: E402

#: Interactive-loop SLA: an enumerated compare must finish under 5 s at p95.
P95_CEILING_MS = 5_000.0
_WARMUP = 5
_ITERS = 40


def _three_action_request(state, uid) -> CompareRequest:
    """The scripted doc-G three-action demo: ``hold`` + two placements."""
    cq = state.symbolic.workloads[uid].cluster_queue_id
    flavor = next(iter(state.symbolic.flavors))
    domains = list(state.symbolic.domains)
    second = domains[-1] if domains else "auto"
    return CompareRequest(
        state_hash=world_state_hash(state),
        candidate_actions=[
            CandidateActionRequest(
                action_id="hold", type="hold", params={"workload_uid": uid}
            ),
            CandidateActionRequest(
                action_id="place-0",
                type="admit_and_place",
                params={
                    "workload_uid": uid,
                    "cluster_queue_id": cq,
                    "resource_flavor_id": flavor,
                    "domain_policy": "preferred",
                },
            ),
            CandidateActionRequest(
                action_id="place-1",
                type="admit_and_place",
                params={
                    "workload_uid": uid,
                    "cluster_queue_id": cq,
                    "resource_flavor_id": flavor,
                    "domain_policy": "preferred",
                    "domain_id_or_set": [second],
                },
            ),
        ],
        horizon_s=1800.0,
        include_rollouts=True,
    )


def _enumerated_request(state, uid) -> CompareRequest:
    """``hold`` + a placement onto EVERY domain → a >3-action enumerated set."""
    cq = state.symbolic.workloads[uid].cluster_queue_id
    flavor = next(iter(state.symbolic.flavors))
    domains = list(state.symbolic.domains)
    actions = [
        CandidateActionRequest(
            action_id="hold", type="hold", params={"workload_uid": uid}
        )
    ]
    for i, dom in enumerate(domains):
        actions.append(
            CandidateActionRequest(
                action_id=f"place-{i}",
                type="admit_and_place",
                params={
                    "workload_uid": uid,
                    "cluster_queue_id": cq,
                    "resource_flavor_id": flavor,
                    "domain_policy": "preferred",
                    "domain_id_or_set": [dom],
                },
            )
        )
    return CompareRequest(
        state_hash=world_state_hash(state),
        candidate_actions=actions,
        horizon_s=1800.0,
        include_rollouts=True,
    )


def _write_summary(lines: list[str]) -> None:
    """Append a markdown block to the GitHub Actions step summary (if present)."""
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    block = "\n".join(lines) + "\n"
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(block)
    # Always echo to stdout too, so local runs and plain CI logs record it.
    print(block, end="")


def _measure(state, req: CompareRequest, engine: EngineBundle) -> tuple[float, float]:
    """Warm up, then return ``(p50_ms, p95_ms)`` over ``_ITERS`` compare calls."""
    for _ in range(_WARMUP):  # exclude first-call / compile cost
        compare_actions(state, req, engine)
    samples_ms: list[float] = []
    for _ in range(_ITERS):
        t0 = time.perf_counter()
        compare_actions(state, req, engine)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    samples_ms.sort()
    p50 = statistics.median(samples_ms)
    p95 = samples_ms[min(len(samples_ms) - 1, int(0.95 * len(samples_ms)))]
    return p50, p95


def main() -> int:
    # A genuinely enumerated topology: 3 racks ⇒ ≥4 placement domains, so the
    # candidate set is hold + ≥4 placements = ≥5 actions (> 3, "enumerated set").
    _pod, state, uid = generate_pod_state(
        PodSpec(n_racks=3, nodes_per_rack=2, gpus_per_node=4), seed=11, gpu_demand=4
    )
    engine = EngineBundle(kernel=ExactSymbolicKernel())

    # ── scripted three-action demo case (doc H "record p50/p95") ─────────────
    req3 = _three_action_request(state, uid)
    resp3 = compare_actions(state, req3, engine)
    assert len(resp3.candidates) >= 3
    p50_3, p95_3 = _measure(state, req3, engine)

    # ── enumerated-set case (>3 actions, <5 s p95 SLA) ───────────────────────
    req_n = _enumerated_request(state, uid)
    n_actions = len(req_n.candidate_actions)
    assert n_actions > 3, f"enumerated set must exceed 3 actions, got {n_actions}"
    resp_n = compare_actions(state, req_n, engine)  # every action scored
    assert len(resp_n.candidates) >= n_actions
    p50_n, p95_n = _measure(state, req_n, engine)

    enumerated_ok = p95_n < P95_CEILING_MS
    _write_summary(
        [
            "### Local compare latency — recorded (doc H)",
            "",
            "| case | candidate actions | p50 (ms) | p95 (ms) | SLA |",
            "|---|---:|---:|---:|---|",
            f"| scripted three-action demo | 3 | {p50_3:.3f} | {p95_3:.3f} | "
            "recorded |",
            f"| enumerated set (>3) | {n_actions} | {p50_n:.3f} | {p95_n:.3f} | "
            f"p95 < {P95_CEILING_MS / 1000:.0f} s → "
            f"{'✅ pass' if enumerated_ok else '❌ FAIL'} |",
            "",
            f"_{_ITERS} iterations each, after {_WARMUP} warm-up "
            "(first-call compile cost excluded)._",
        ]
    )

    if not enumerated_ok:
        print(
            f"::error::enumerated-set compare p95={p95_n:.2f}ms exceeds the "
            f"{P95_CEILING_MS:.0f}ms (5 s) interactive SLA",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
