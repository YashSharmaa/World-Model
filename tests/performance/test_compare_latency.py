# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Performance test: compare latency on the 3-action demo case (doc H).

Doc H's MLX/serving gate: "Record p50 and p95 local compare latency on the
scripted three-action demo case." The public promise is an interactive local
decision loop, so the compare hot path must stay fast *and stable*.

This test measures p50 / p95 wall-clock for the doc-G shape-stable three-branch
compare on the canonical demo pod, after warm-up (so a first-call cost — e.g. an
MLX ``compile()`` — is excluded, matching the "second invocation avoids compile
churn" criterion). The thresholds are intentionally generous so the assertion is
robust on slow / shared CI runners (the reference engine runs in well under a
millisecond locally) while still catching a pathological regression. It also
asserts *stability*: p95 must not be a wild multiple of p50.
"""

from __future__ import annotations

import statistics
import time

import pytest

from gpu_center_world_model.data_engine.synthetic import (
    PodSpec,
    generate_pod_state,
)
from gpu_center_world_model.schema import world_state_hash
from gpu_center_world_model.serving import (
    CandidateActionRequest,
    CompareRequest,
    EngineBundle,
    compare_actions,
)
from gpu_center_world_model.serving.dto import DEFAULT_BRANCHES
from gpu_center_world_model.simulator import ExactSymbolicKernel

# Generous CI-safe ceilings (ms). Local reference-engine latency is ~0.3 ms p50.
_P50_CEILING_MS = 250.0
_P95_CEILING_MS = 500.0
_WARMUP = 5
_ITERS = 60


def _three_action_request(state, uid) -> CompareRequest:
    """The doc-G scripted three-action demo: hold + two placements."""
    cq = state.symbolic.workloads[uid].cluster_queue_id
    flavor = next(iter(state.symbolic.flavors))
    domains = list(state.symbolic.domains)
    second = domains[-1] if domains else "auto"
    return CompareRequest(
        state_hash=world_state_hash(state),
        candidate_actions=[
            CandidateActionRequest(
                action_id="h", type="hold", params={"workload_uid": uid}
            ),
            CandidateActionRequest(
                action_id="p0",
                type="admit_and_place",
                params={
                    "workload_uid": uid,
                    "cluster_queue_id": cq,
                    "resource_flavor_id": flavor,
                    "domain_policy": "preferred",
                },
            ),
            CandidateActionRequest(
                action_id="p1",
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


def test_compare_latency_p50_p95_reasonable_and_stable():
    _pod, state, uid = generate_pod_state(
        PodSpec(n_racks=2, nodes_per_rack=2, gpus_per_node=4), seed=7, gpu_demand=4
    )
    engine = EngineBundle(kernel=ExactSymbolicKernel())
    req = _three_action_request(state, uid)

    # Correctness gate first: the demo really is a three-action, shape-stable case.
    resp = compare_actions(state, req, engine)
    assert len(resp.candidates) >= DEFAULT_BRANCHES

    # Warm up (exclude first-call / compile cost).
    for _ in range(_WARMUP):
        compare_actions(state, req, engine)

    samples_ms: list[float] = []
    for _ in range(_ITERS):
        t0 = time.perf_counter()
        compare_actions(state, req, engine)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    samples_ms.sort()
    p50 = statistics.median(samples_ms)
    p95 = samples_ms[min(len(samples_ms) - 1, int(0.95 * len(samples_ms)))]

    assert p50 < _P50_CEILING_MS, f"compare p50={p50:.2f}ms exceeds {_P50_CEILING_MS}ms"
    assert p95 < _P95_CEILING_MS, f"compare p95={p95:.2f}ms exceeds {_P95_CEILING_MS}ms"
    # Stability: tail should not be an extreme multiple of the median (jitter guard).
    # Only meaningful when the median is above timer granularity.
    if p50 > 0.05:
        assert p95 <= max(20.0 * p50, 50.0), (
            f"compare latency unstable: p95={p95:.2f}ms vs p50={p50:.2f}ms"
        )


@pytest.mark.mlx
def test_mlx_score_actions_smoke():
    """MLX learned hot-path smoke (skipped unless MLX is available).

    Doc H release gate: "run one tiny inference step on arm64 macOS" and "second
    invocation with the same shape buckets avoids compile churn". We build the
    real learned model and drive its compiled ``score_actions`` hot path twice on
    the same shape bucket, asserting both calls return one bundle per action.
    """
    mlx = pytest.importorskip("mlx", reason="MLX backend not installed")
    del mlx  # imported only to gate the test

    from gpu_center_world_model.model import ResidualWorldModel, get_backend
    from gpu_center_world_model.schema import HORIZONS_S, Hold

    backend = get_backend()
    if backend.name != "mlx":  # pragma: no cover - only on Apple silicon CI
        pytest.skip("active backend is not MLX")

    _pod, state, uid = generate_pod_state(
        PodSpec(n_racks=2, nodes_per_rack=2, gpus_per_node=4), seed=7, gpu_demand=4
    )
    model = ResidualWorldModel.build(n_ensemble=1)
    actions = [Hold(workload_uid=uid)]
    first = model.score_actions(state, actions, HORIZONS_S[1])
    second = model.score_actions(state, actions, HORIZONS_S[1])  # same bucket
    assert len(first) == len(actions)
    assert len(second) == len(actions)
