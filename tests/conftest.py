# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Shared pytest fixtures for the test pyramid (design doc H, ci_and_testing).

This conftest does two things:

1. **Import bootstrap.** The package uses a ``src/`` layout. When the package is
   *not* installed editable (e.g. a bare ``pytest`` run on a clean checkout), the
   ``src`` directory is put on ``sys.path`` so tests import the *real* on-disk
   public APIs rather than a stale wheel. When the package *is* installed this is
   a harmless no-op.

2. **Tiny, deterministic fixtures.** The whole pyramid is anchored on a single
   *demo pod* (2 racks, doc-H ``2-4 rack pod`` assumption) with one queued gang
   workload, built through the real synthetic generator. Because the generator is
   seeded and the schema state-hash is deterministic, every test that consumes
   these fixtures is reproducible across processes (the counterfactual alignment
   property of docs D/I/R that the whole project rests on).

All fixtures return *real* schema objects so tests exercise the genuine
public surface, never a mock of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Import bootstrap: make the src-layout package importable without install ──
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest  # noqa: E402  (must come after the path bootstrap)

from gpu_center_world_model.data_engine.synthetic import (  # noqa: E402
    PodSpec,
    generate_pod_state,
)
from gpu_center_world_model.schema import (  # noqa: E402
    AbstentionConfig,
    PolicyConfig,
    RiskThresholds,
    WorkloadStatus,
)
from gpu_center_world_model.simulator import ExactSymbolicKernel  # noqa: E402

#: The canonical demo-pod shape used across the whole pyramid: 2 racks, 2 nodes
#: each, 4 GPUs each = 16 GPUs. Small enough to be fast, large enough to exercise
#: multiple topology domains, gang packing, and quota arithmetic.
DEMO_POD = PodSpec(n_racks=2, nodes_per_rack=2, gpus_per_node=4)

#: A gang demand that fits comfortably in a single rack (4 of the 8 per-rack GPUs)
#: so admit_and_place has several feasible domains — required for ranking tests.
DEMO_GANG_GPU = 4

#: Seed for the deterministic pod build (CRN anchor for counterfactual branches).
DEMO_SEED = 7


@pytest.fixture
def pod_seed() -> int:
    return DEMO_SEED


@pytest.fixture
def demo_pod():
    """Return ``(pod, world_state, workload_uid)`` for the canonical demo pod.

    The world state carries exactly one PENDING gang workload (the planning
    object) wired with its workload-side relations, so the exact kernel can
    enumerate a real safe candidate set against it.
    """
    pod, state, uid = generate_pod_state(
        DEMO_POD, seed=DEMO_SEED, gpu_demand=DEMO_GANG_GPU
    )
    # Sanity: the fixture must actually present a queued gang.
    assert uid in state.symbolic.workloads
    assert state.symbolic.workloads[uid].status is WorkloadStatus.PENDING
    return pod, state, uid


@pytest.fixture
def demo_state(demo_pod):
    """Just the :class:`WorldState` of the demo pod."""
    _pod, state, _uid = demo_pod
    return state


@pytest.fixture
def queued_workload_uid(demo_pod) -> str:
    """The uid of the single queued gang workload."""
    _pod, _state, uid = demo_pod
    return uid


@pytest.fixture
def kernel() -> ExactSymbolicKernel:
    """A fresh exact symbolic kernel (the safety shield) per test."""
    return ExactSymbolicKernel()


@pytest.fixture
def permissive_policy() -> PolicyConfig:
    """A policy whose risk + abstention gates pass for the reference model.

    The dependency-free reference forecaster emits benign-but-incomplete hazard
    channels, so the *default* operator budgets would (correctly) abstain. For
    tests that need to exercise the *non-abstaining* recommend path end-to-end we
    relax every budget to its trivial bound; the hard-safety gate (the kernel) is
    untouched, so this never weakens the safety guarantee under test.
    """
    return PolicyConfig(
        risk=RiskThresholds(
            max_hotspot_prob=1.0,
            max_headroom_violation_prob=1.0,
            max_pods_ready_timeout_prob=1.0,
            max_requeue_prob=1.0,
            max_xid_fault_prob=1.0,
        ),
        abstention=AbstentionConfig(
            max_epistemic_disagreement=1.0,
            max_ood_score=1.0,
            max_source_health_bad_prob=1.0,
        ),
    )


@pytest.fixture
def default_policy() -> PolicyConfig:
    """The default operator policy (strict budgets; exercises abstention)."""
    return PolicyConfig()
