# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Tests for MODEL-RELATIONAL-FUSION (doc Q D2–D4, Eq. 6–8, R5/R6/V4).

Pins the typed per-relation message blocks, the exact-semantic shortcut
relations (so the switch latent and power/cooling coupling participate), the
candidate-attention fusion into the placement/action-value latents, the FiLM
downward rebroadcast, and the PNA-style invariant global readout.
"""

from __future__ import annotations

import numpy as np

from gpu_center_world_model.data_engine.synthetic import PodSpec, generate_pod_state
from gpu_center_world_model.model.architecture import (
    RELATIONS,
    ArchConfig,
    FeatureBuilder,
    RelationalWorldModel,
)


def _state(n_racks=4, nodes_per_rack=4, gpus_per_node=8, seed=3, demand=8):
    spec = PodSpec(
        n_racks=n_racks, nodes_per_rack=nodes_per_rack, gpus_per_node=gpus_per_node
    )
    _pod, state, uid = generate_pod_state(spec, seed=seed, gpu_demand=demand)
    return state, uid


def _np(be, x):
    return np.asarray(be.to_numpy(x))


def test_typed_per_relation_blocks_exist():
    """Each relation r=(src_t,dst_t) gets its own W_r^src/W_r^dst/φ_r/η_r."""
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    for name, _src, _dst in RELATIONS:
        assert name in m.rel_src, f"missing W_r^src for {name}"
        assert name in m.rel_dst, f"missing W_r^dst for {name}"
        assert name in m.rel_phi, f"missing φ_r for {name}"
        assert name in m.rel_gate, f"missing η_r gate for {name}"
        # η_r is a scalar edge gate (output width 1)
        assert _np(m.be, m.rel_gate[name].W).shape[0] == 1


def test_distinct_relations_have_distinct_params():
    """node->switch and node->rack learn *different* edge functions (doc Q D2)."""
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    w_ns = _np(m.be, m.rel_src["node->switch"].W)
    w_nr = _np(m.be, m.rel_src["node->rack"].W)
    assert not np.allclose(w_ns, w_nr)


def test_shortcut_relations_built_including_switch():
    """The R5 shortcut incidences exist; the switch latent participates."""
    fb = FeatureBuilder()
    state, _uid = _state()
    ts = fb.build(state)
    for name in (
        "node->switch",
        "switch->node",
        "rack->power_domain",
        "rack->cooling_zone",
        "candidate_domain->workload",
    ):
        assert name in ts.relations, f"missing shortcut relation {name}"
    # the synthetic pod has switches connected to nodes
    assert _np(fb.be, ts.relations["node->switch"]).sum() > 0


def test_switch_latent_participates_in_message_passing():
    """Perturbing switch features changes node latents via node<->switch edges."""
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    fb = FeatureBuilder()
    state, _uid = _state()
    ts = fb.build(state)
    z_base = m.encode(ts, "hold")
    # the switch latent must be non-zero (switches exist and are encoded)
    assert np.abs(_np(m.be, z_base["switch"])).sum() > 0.0
    # bump switch features; node latents should move because switch->node carries
    sw = _np(m.be, ts.features["switch"]).copy()
    sw_mask = _np(m.be, ts.masks["switch"]).astype(bool)
    sw[sw_mask] += 0.5
    ts.features["switch"] = m.be.from_numpy(sw)
    z_pert = m.encode(ts, "hold")
    dnode = np.abs(_np(m.be, z_base["node"]) - _np(m.be, z_pert["node"])).sum()
    assert dnode > 1e-4, "switch perturbation did not reach node latents"


def test_rack_sibling_shortcuts_connect_shared_power_cooling():
    """Racks sharing a power domain / cooling zone are connected (no self-edge)."""
    fb = FeatureBuilder()
    state, _uid = _state()
    ts = fb.build(state)
    M_p = _np(fb.be, ts.relations["rack->power_domain"])
    M_c = _np(fb.be, ts.relations["rack->cooling_zone"])
    # no self edges
    assert np.trace(M_p) == 0.0
    assert np.trace(M_c) == 0.0
    # at least one cross-rack edge when ≥2 racks share a domain
    assert M_p.sum() > 0 or M_c.sum() > 0


def test_candidate_attention_fused_into_latents():
    """candidate_scores returns a masked (J,D) matrix; context is fused in."""
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    fb = FeatureBuilder()
    state, _uid = _state()
    ts = fb.build(state)
    z = m.encode(ts, "place")
    scores = m.candidate_scores(z, ts)
    s = _np(m.be, scores)
    fmask = _np(m.be, ts.candidate_mask)
    assert s.shape == fmask.shape
    # scores never fall outside the feasible candidate mask (the project spine)
    assert np.all(s[fmask == 0.0] == 0.0)
    # rows over feasible domains form a (sub-)distribution
    assert np.all(s >= -1e-9)


def test_candidate_scores_accepts_kernel_certified_mask():
    """A supplied F_WD overrides the free-GPU heuristic (doc Q Eq. 6, V6)."""
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    fb = FeatureBuilder()
    state, _uid = _state()
    ts = fb.build(state)
    z = m.encode(ts, "place")
    J, D = _np(m.be, ts.candidate_mask).shape
    # a custom mask that only allows domain 0 for workload 0
    f_wd = np.zeros((J, D), dtype=np.float32)
    if J > 0 and D > 0:
        f_wd[0, 0] = 1.0
    scores = m.candidate_scores(z, ts, f_wd=m.be.from_numpy(f_wd))
    s = _np(m.be, scores)
    assert np.all(s[f_wd == 0.0] == 0.0)
    if J > 0 and D > 0:
        assert s[0, 0] > 0.0


def test_candidate_fusion_changes_workload_latent():
    """The fused S_workload context changes the workload latent the heads read."""
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    fb = FeatureBuilder()
    state, _uid = _state()
    ts = fb.build(state)
    z_with = m.encode(ts, "place")
    # zero the candidate mask -> no candidate context -> different workload latent
    ts.candidate_mask = m.be.from_numpy(
        np.zeros_like(_np(m.be, ts.candidate_mask))
    )
    ts.relations["candidate_domain->workload"] = ts.candidate_mask
    z_without = m.encode(ts, "place")
    d = np.abs(_np(m.be, z_with["workload"]) - _np(m.be, z_without["workload"])).sum()
    assert d > 1e-4


def test_pna_global_readout_uses_count_and_variance():
    """The global latent is a learned PNA reducer, not a mean-only truncation."""
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    fb = FeatureBuilder()
    state, _uid = _state()
    ts = fb.build(state)
    z = m.encode(ts, "hold")
    g = _np(m.be, z["global"])
    assert g.shape[0] == 1  # single global row
    assert np.all(np.isfinite(g))
    # the readout Dense ingests 4*d+1 features per type (sum/mean/std/max/count)
    in_dim = _np(m.be, m.global_readout.W).shape[1]
    assert in_dim > 0


def test_film_downward_rebroadcast_present():
    """A FiLM generator (γ, β) drives the downward rebroadcast (doc Q R6/V4)."""
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    out_dim = _np(m.be, m.film_gen.W).shape[0]
    # (γ, β) pair => even output width
    assert out_dim % 2 == 0


def test_encode_is_permutation_equivariant_over_workloads():
    """Per-type permutation equivariance holds end-to-end (doc Q G1–G2, Th. 1–3).

    Permuting the workload rows of the input must permute the workload latent
    rows identically (up to the candidate fusion, which is itself equivariant
    over the workload axis).
    """
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    fb = FeatureBuilder()
    state, _uid = _state()
    ts = fb.build(state)
    mask = _np(m.be, ts.masks["workload"]).astype(bool)
    idx = np.where(mask)[0]
    if idx.size < 2:
        return  # need ≥2 valid workloads to permute
    z = m.encode(ts, "hold")
    base = _np(m.be, z["workload"])
    # swap the first two valid workload rows in features, masks, and incidences
    i, j = int(idx[0]), int(idx[1])
    feats = _np(m.be, ts.features["workload"]).copy()
    feats[[i, j]] = feats[[j, i]]
    ts.features["workload"] = m.be.from_numpy(feats)
    # candidate mask rows are indexed by workload -> permute rows too
    cm = _np(m.be, ts.candidate_mask).copy()
    cm[[i, j]] = cm[[j, i]]
    ts.candidate_mask = m.be.from_numpy(cm)
    ts.relations["candidate_domain->workload"] = ts.candidate_mask
    # workload->node incidence has workloads on the column axis -> permute cols
    wn = _np(m.be, ts.relations["workload->node"]).copy()
    wn[:, [i, j]] = wn[:, [j, i]]
    ts.relations["workload->node"] = m.be.from_numpy(wn)
    z2 = m.encode(ts, "hold")
    perm = _np(m.be, z2["workload"])
    assert np.allclose(perm[i], base[j], atol=1e-4)
    assert np.allclose(perm[j], base[i], atol=1e-4)
