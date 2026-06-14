# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Permutation-equivariance of the tensorizer & its pooled aggregates (doc Q).

The world is a typed relational graph: entity *identity* is carried by relations,
not by row order. Relabelling (re-ordering) entities of a kind must therefore not
change any **permutation-invariant pooled summary** of that kind — the masked
sum / mean / max over real (non-padding) rows. This is the algebraic backbone of
the doc-Q hierarchical PNA pooling: the aggregator is a set function over masked
rows, so it is invariant to the row order the tensorizer happened to assign.

We assert this directly on the real :func:`tensorize` output (NumPy, backend-
free), permuting the entity dictionaries and confirming the masked pooled
aggregates are unchanged. ``hypothesis`` drives many pod shapes and permutations;
the module skips cleanly when hypothesis is unavailable.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

# Robust skip: handle both a clean absence and a partial/namespace-shell install
# of hypothesis (see test_kernel_properties for the rationale).
try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover - exercised on the numpy-only CI box
    pytest.skip(
        "hypothesis not available; property tests require it", allow_module_level=True
    )

from gpu_center_world_model.data_engine.synthetic import (
    PodSpec,
    generate_pod_state,
)
from gpu_center_world_model.data_engine.tensorize import tensorize

_POOLED_KINDS = ("workload", "node", "gpu", "rack", "queue")
_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _permute_entities(state, kind: str, perm: np.ndarray):
    """Return a copy of ``state`` with the given entity-kind dict re-ordered.

    Only the dict insertion order changes; the entity objects (and thus the
    physical content of the pod) are identical. The tensorizer assigns rows by
    iteration order, so this is exactly an entity *relabelling*.
    """
    sym = state.symbolic
    table_attr = {
        "workload": "workloads",
        "node": "nodes",
        "gpu": "gpus",
        "rack": "racks",
        "queue": "cluster_queues",
    }[kind]
    table = getattr(sym, table_attr)
    items = list(table.items())
    if len(items) <= 1:
        return state
    reordered = {items[i][0]: items[i][1] for i in perm}
    new_sym = replace(sym, **{table_attr: reordered})
    return replace(state, symbolic=new_sym)


def _pooled(tb, kind: str) -> dict[str, np.ndarray]:
    """Permutation-invariant masked aggregates for one entity kind."""
    arr = np.asarray(tb.arrays[kind])
    mask = np.asarray(tb.masks[kind]).astype(bool)
    rows = arr[mask]
    if rows.size == 0:
        zeros = np.zeros(arr.shape[-1])
        return {"sum": zeros, "max": zeros, "count": np.array(0)}
    return {
        "sum": rows.sum(axis=0),
        "max": rows.max(axis=0),
        "count": np.array(rows.shape[0]),
    }


@given(
    st.integers(1, 3),  # racks
    st.integers(1, 3),  # nodes/rack
    st.integers(2, 4),  # gpus/node
    st.integers(0, 2**16),  # seed
    st.integers(0, 2**31 - 1),  # permutation seed
    st.sampled_from(_POOLED_KINDS),
)
@_SETTINGS
def test_pooled_summary_invariant_under_relabelling(r, n, g, seed, pseed, kind):
    _pod, state, _uid = generate_pod_state(
        PodSpec(n_racks=r, nodes_per_rack=n, gpus_per_node=g), seed=seed, gpu_demand=2
    )
    tb_ref = tensorize(state)

    table_attr = {
        "workload": "workloads",
        "node": "nodes",
        "gpu": "gpus",
        "rack": "racks",
        "queue": "cluster_queues",
    }[kind]
    n_entities = len(getattr(state.symbolic, table_attr))
    perm = np.random.default_rng(pseed).permutation(max(n_entities, 1))

    state_perm = _permute_entities(state, kind, perm)
    tb_perm = tensorize(state_perm)

    ref = _pooled(tb_ref, kind)
    got = _pooled(tb_perm, kind)
    assert ref["count"] == got["count"]
    assert np.allclose(ref["sum"], got["sum"], atol=1e-5)
    assert np.allclose(ref["max"], got["max"], atol=1e-5)


@given(st.integers(1, 3), st.integers(1, 3), st.integers(2, 4), st.integers(0, 2**16))
@_SETTINGS
def test_mask_count_matches_real_entity_count(r, n, g, seed):
    # The aggregator's denominator (mask sum) must equal the true entity count, so
    # masked means are correct regardless of padding (doc Q pooling needs masks).
    _pod, state, _uid = generate_pod_state(
        PodSpec(n_racks=r, nodes_per_rack=n, gpus_per_node=g), seed=seed, gpu_demand=2
    )
    tb = tensorize(state)
    sym = state.symbolic
    expect = {
        "workload": len(sym.workloads),
        "node": len(sym.nodes),
        "gpu": len(sym.gpus),
        "rack": len(sym.racks),
        "queue": len(sym.cluster_queues),
    }
    for kind, count in expect.items():
        mask_sum = int(np.asarray(tb.masks[kind]).sum())
        assert mask_sum == min(count, tb.arrays[kind].shape[0])
