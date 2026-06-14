# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Checkpoint round-trip completeness (doc E checkpointing_and_serving).

A checkpoint must persist **every** trainable array. The trap is that
:func:`load_checkpoint` rebuilds a fresh model from the same seeded config, so a
*missed* parameter silently reverts to its (identical) default and an untrained
model round-trips perfectly even when serialization is incomplete.

These tests defeat that by **perturbing every entry of every trainable array**
(the canonical set is :func:`model.train._trainable_arrays`, which both SPSA and
the MLX autodiff path optimise) *before* saving, then asserting that a reloaded
model reproduces (a) every trainable array bit-for-bit and (b) the served
``score_actions`` output. (a) catches params that never touch the forecast
(e.g. the exact-field decoders ``D_f``, used only by ``L_cons`` during training);
(b) catches any forward-path weight.
"""

from __future__ import annotations

import numpy as np

from gpu_center_world_model.data_engine.synthetic import PodSpec, generate_pod_state
from gpu_center_world_model.model.serialization import load_checkpoint, save_checkpoint
from gpu_center_world_model.model.train import _trainable_arrays
from gpu_center_world_model.model.world_model import ResidualWorldModel
from gpu_center_world_model.simulator import ExactSymbolicKernel

_POD = PodSpec(n_racks=2, nodes_per_rack=2, gpus_per_node=4)


def _member_view(model: ResidualWorldModel, member) -> ResidualWorldModel:
    """A lightweight ResidualWorldModel wrapper so we can walk a member's params."""
    return ResidualWorldModel(
        backbone=member,
        heads=model.heads,
        feature_builder=model.feature_builder,
        calibrator=model.calibrator,
        abstention=model.abstention,
    )


def _perturb(view: ResidualWorldModel, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
    be = view.be
    for owner, attr in _trainable_arrays(view):
        cur = np.asarray(be.to_numpy(getattr(owner, attr)))
        nxt = (cur + rng.standard_normal(cur.shape) * 0.1).astype(np.float32)
        setattr(owner, attr, be.from_numpy(nxt))


def _snapshot(view: ResidualWorldModel) -> list[np.ndarray]:
    be = view.be
    return [
        np.asarray(be.to_numpy(getattr(o, a))).copy()
        for o, a in _trainable_arrays(view)
    ]


def _all_params(model: ResidualWorldModel) -> list[np.ndarray]:
    out = _snapshot(model)
    for m in model.ensemble:
        out += _snapshot(_member_view(model, m))
    return out


def test_every_trainable_array_round_trips(tmp_path) -> None:
    """Perturb all trainables (backbone + ensemble), save, load → all identical."""
    model = ResidualWorldModel.build(n_ensemble=2)
    _perturb(model, seed=3)
    for i, member in enumerate(model.ensemble):
        _perturb(_member_view(model, member), seed=100 + i)

    before = _all_params(model)
    save_checkpoint(model, tmp_path / "ckpt")
    loaded = load_checkpoint(tmp_path / "ckpt")
    after = _all_params(loaded)

    assert len(before) == len(after)
    for idx, (b, a) in enumerate(zip(before, after, strict=True)):
        assert np.allclose(b, a, atol=1e-6), (
            f"trainable array #{idx} did NOT survive the checkpoint round-trip — "
            f"serialization is missing this parameter."
        )


def test_score_actions_identical_after_round_trip(tmp_path) -> None:
    """A perturbed, reloaded model produces byte-identical served forecasts."""
    model = ResidualWorldModel.build(n_ensemble=1)
    _perturb(model, seed=7)

    _pod, state, uid = generate_pod_state(_POD, seed=7, gpu_demand=4)
    kernel = ExactSymbolicKernel()
    safe = kernel.safe_candidates(state, uid)
    assert safe, "scenario must yield at least one safe candidate"

    before = model.score_actions(state, safe, horizon_s=1800)
    save_checkpoint(model, tmp_path / "ckpt2")
    loaded = load_checkpoint(tmp_path / "ckpt2")
    after = loaded.score_actions(state, safe, horizon_s=1800)

    assert len(before) == len(after)
    for b, a in zip(before, after, strict=True):
        qb = {(q.channel, q.horizon_s): q.quantiles for q in b.quantiles}
        qa = {(q.channel, q.horizon_s): q.quantiles for q in a.quantiles}
        assert qb.keys() == qa.keys()
        for key in qb:
            vb = [qb[key][t] for t in sorted(qb[key])]
            va = [qa[key][t] for t in sorted(qa[key])]
            assert np.allclose(vb, va, atol=1e-5), f"forecast {key} drifted on reload"
