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
from gpu_center_world_model.model.architecture import (
    ArchConfig,
    FeatureBuilder,
    RelationalWorldModel,
    kernel_state_summary,
)

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


# ─────────────────────────────────────────────────────────────────────────────
#  Per-type ROW-PERMUTATION equivariance of the *learned* updates (doc Q G1–G2,
#  Th. 1–4). The tests above pin the algebraic backbone on the backend-free
#  tensorizer; these drive the REAL :class:`RelationalWorldModel` forward pass.
#
#  Doc Q G1: the symmetry group is 𝒢 = ∏_τ S_{n_τ} (independent within-type
#  permutations). G2 / Th. 1–3: under a per-type row permutation Π_τ of the input
#  features, masks, and every incidence/relation matrix that references that type
#  on either axis, **every learned latent update is equivariant**:
#  H_τ^(ℓ+1)' = Π_τ H_τ^(ℓ+1). Th. 4: any graph-level readout built from sums /
#  means / maxima over the entity axis (the ``global`` latent here) is *invariant*
#  because 1^⊤Π_τ = 1^⊤. Cross-type latents not touched by Π_τ are likewise
#  invariant (the permuted type enters them only through a symmetric reducer).
#
#  We assert all three on the real backbone:  the permuted type's latent permutes
#  exactly, untouched sibling types are unchanged, and the global readout is
#  invariant — exercising the full TemporalEncode → RelFuse → PNA → global path.
# ─────────────────────────────────────────────────────────────────────────────
def _np(be, x):
    return np.asarray(be.to_numpy(x))


def _set(ts, axis_key, arr, be):
    """Write ``arr`` into the matching incidence / hierarchy / relation slots."""
    a = be.from_numpy(arr)
    if axis_key in ("gpu->node", "node->rack", "rack->topology_domain"):
        pair = {
            "gpu->node": ("gpu", "node"),
            "node->rack": ("node", "rack"),
            "rack->topology_domain": ("rack", "topology_domain"),
        }[axis_key]
        ts.incidence[pair] = a
        ts.hierarchy[pair] = a
    elif axis_key == "workload->node":
        ts.incidence[("workload", "node")] = a
    ts.relations[axis_key] = a


def _permuted_perm(mask: np.ndarray, seed: int) -> np.ndarray:
    """A row permutation that shuffles only the *valid* rows of ``mask``."""
    idx = np.where(mask)[0]
    sub = np.random.default_rng(seed).permutation(idx.size)
    perm = np.arange(mask.shape[0])
    perm[idx] = idx[sub]
    return perm


def _build_model_state(seed: int, demand: int):
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    fb = FeatureBuilder()
    _pod, state, _uid = generate_pod_state(
        PodSpec(n_racks=2, nodes_per_rack=2, gpus_per_node=4),
        seed=seed,
        gpu_demand=demand,
    )
    return m, fb, state


@given(st.integers(0, 2**16), st.integers(2, 6), st.integers(0, 2**31 - 1))
@_SETTINGS
def test_gpu_row_permutation_equivariance_learned_updates(seed, demand, pseed):
    """Permuting GPU rows permutes the GPU latent identically (doc Q Th. 3).

    GPU is the source (column) axis of the ``gpu->node`` relation only, so this
    isolates the per-type GRU update + the typed message pass + the upward PNA
    pool from gpu→node. The valid GPU latent rows must permute exactly; the node,
    rack, and global latents must be invariant (Th. 4 readout invariance).
    """
    m, fb, state = _build_model_state(seed, demand)
    ts = fb.build(state)
    z0 = m.encode(ts, "hold")
    base = _np(m.be, z0["gpu"]).copy()
    mask = _np(m.be, ts.masks["gpu"]).astype(bool)
    if int(mask.sum()) < 2:
        return  # need ≥2 valid GPUs to permute

    perm = _permuted_perm(mask, pseed)
    ts.features["gpu"] = m.be.from_numpy(_np(m.be, ts.features["gpu"])[perm])
    ts.masks["gpu"] = m.be.from_numpy(_np(m.be, ts.masks["gpu"])[perm])
    # GPU is the *source* (columns) of gpu->node: permute the column axis.
    gn = _np(m.be, ts.relations["gpu->node"])[:, perm]
    _set(ts, "gpu->node", gn, m.be)

    z1 = m.encode(ts, "hold")
    got = _np(m.be, z1["gpu"])
    valid = mask[perm]
    # Th. 3: the learned GPU latent update is row-permutation equivariant.
    assert np.allclose(got[valid], base[perm][valid], atol=1e-5)
    # Th. 4 + cross-type invariance: untouched siblings and the global readout
    # are unchanged by a within-type GPU relabelling.
    assert np.allclose(_np(m.be, z1["node"]), _np(m.be, z0["node"]), atol=1e-5)
    assert np.allclose(_np(m.be, z1["rack"]), _np(m.be, z0["rack"]), atol=1e-5)
    assert np.allclose(_np(m.be, z1["global"]), _np(m.be, z0["global"]), atol=1e-5)


@given(st.integers(0, 2**16), st.integers(2, 6), st.integers(0, 2**31 - 1))
@_SETTINGS
def test_node_row_permutation_equivariance_learned_updates(seed, demand, pseed):
    """Permuting NODE rows permutes the node latent identically (doc Q Th. 1–3).

    NODE touches many typed relations (gpu→node, node→rack, workload→node, and
    the R5 node↔switch shortcuts), so this exercises the full G2 transform
    S_r↦Π_r S_r Π_σ^⊤ across several relation types at once. The permuted node
    latent rows must match the base latent permuted; the rack and global latents
    are invariant.
    """
    m, fb, state = _build_model_state(seed, demand)
    ts = fb.build(state)
    z0 = m.encode(ts, "hold")
    base = _np(m.be, z0["node"]).copy()
    mask = _np(m.be, ts.masks["node"]).astype(bool)
    if int(mask.sum()) < 2:
        return  # need ≥2 valid nodes to permute

    perm = _permuted_perm(mask, pseed)
    ts.features["node"] = m.be.from_numpy(_np(m.be, ts.features["node"])[perm])
    ts.masks["node"] = m.be.from_numpy(_np(m.be, ts.masks["node"])[perm])
    # node is DST(row) of gpu->node; SRC(col) of node->rack; DST(row) of
    # workload->node; and participates in both node<->switch shortcuts.
    _set(ts, "gpu->node", _np(m.be, ts.relations["gpu->node"])[perm, :], m.be)
    _set(ts, "node->rack", _np(m.be, ts.relations["node->rack"])[:, perm], m.be)
    _set(ts, "workload->node", _np(m.be, ts.relations["workload->node"])[perm, :], m.be)
    ts.relations["node->switch"] = m.be.from_numpy(
        _np(m.be, ts.relations["node->switch"])[:, perm]
    )
    ts.relations["switch->node"] = m.be.from_numpy(
        _np(m.be, ts.relations["switch->node"])[perm, :]
    )

    z1 = m.encode(ts, "hold")
    got = _np(m.be, z1["node"])
    valid = mask[perm]
    assert np.allclose(got[valid], base[perm][valid], atol=1e-5)
    assert np.allclose(_np(m.be, z1["rack"]), _np(m.be, z0["rack"]), atol=1e-5)
    assert np.allclose(_np(m.be, z1["global"]), _np(m.be, z0["global"]), atol=1e-5)


@given(st.integers(0, 2**16), st.integers(2, 6), st.integers(0, 2**31 - 1))
@_SETTINGS
def test_predictor_rollout_step_is_row_equivariant(seed, demand, pseed):
    """The free-running predictor step is per-type row-equivariant (doc Q Th. 3).

    The deployed recursive core ``ẑ_{h+1} = P_θ(ẑ_h, …)`` is a *row-shared* GRU
    per type, so permuting the GPU rows of the encoded latent (and its mask)
    permutes the rolled GPU latent identically over the whole H-step rollout —
    the action/kernel/exogenous conditioning is broadcast, never slot-specific.
    """
    m, fb, state = _build_model_state(seed, demand)
    ts = fb.build(state)
    z = m.encode(ts, "place")
    mask = _np(m.be, ts.masks["gpu"]).astype(bool)
    if int(mask.sum()) < 2:
        return
    psi = kernel_state_summary(state, m.be)
    traj = m.rollout(
        z, "place", ts.masks, kernel_states=[psi, psi], horizon=2, detach_every=0
    )
    base = _np(m.be, traj[-1]["gpu"]).copy()

    perm = _permuted_perm(mask, pseed)
    z_p = {
        t: m.be.from_numpy(_np(m.be, v)[perm] if t == "gpu" else _np(m.be, v))
        for t, v in z.items()
    }
    masks_p = {
        t: m.be.from_numpy(_np(m.be, v)[perm] if t == "gpu" else _np(m.be, v))
        for t, v in ts.masks.items()
    }
    traj_p = m.rollout(
        z_p, "place", masks_p, kernel_states=[psi, psi], horizon=2, detach_every=0
    )
    got = _np(m.be, traj_p[-1]["gpu"])
    valid = mask[perm]
    assert np.allclose(got[valid], base[perm][valid], atol=1e-5)
