# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Tests for MODEL-PREDICTOR-ROLLOUT: the P_θ transition predictor and the
free-running H-step rollout (doc J D6, Eqs. 8–10, B1–B4).

These pin the *deployed recursive core*: the predictor must produce genuinely
distinct latents ẑ_1..ẑ_H (no replication), be action-sensitive, condition on a
stop-gradient exact kernel state, honour truncated BPTT, and follow the horizon
curriculum. The serving contract (encode/forecast/score_actions) must stay
stable.
"""

from __future__ import annotations

import numpy as np

from gpu_center_world_model.data_engine.synthetic import PodSpec, generate_pod_state
from gpu_center_world_model.model.architecture import (
    ArchConfig,
    FeatureBuilder,
    RelationalWorldModel,
    WorldTransition,
    kernel_state_summary,
)
from gpu_center_world_model.model.jepa import GroupLogVars, JEPAConfig
from gpu_center_world_model.model.train import (
    TrainConfig,
    _curriculum_horizon,
    _exogenous_vector,
    make_smoke_windows,
    smoke_train,
    stage_loss,
)
from gpu_center_world_model.model.world_model import ResidualWorldModel
from gpu_center_world_model.schema.actions import Admit, Hold
from gpu_center_world_model.schema.kpis import HORIZONS_S


def _state():
    _pod, state, uid = generate_pod_state(
        PodSpec(n_racks=2, nodes_per_rack=2, gpus_per_node=4), seed=7, gpu_demand=4
    )
    return state, uid


def _np(be, x):
    return np.asarray(be.to_numpy(x))


def test_predictor_is_built_and_typed():
    """Every latent-bearing type gets its own transition GRU (doc J D6)."""
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    assert isinstance(m.predictor, WorldTransition)
    for t in ("workload", "gpu", "node", "rack", "switch", "topology_domain", "global"):
        assert t in m.predictor.cells, f"missing transition cell for {t}"
        assert t in m.predictor.psi_proj


def test_rollout_produces_distinct_latents():
    """ẑ_1..ẑ_H must be genuinely distinct — not the replicated one-step encoding.

    This is the core regression for the `[online for _ in range(horizon)]` bug
    (doc J B1/D6): a free-running recursion changes the latent at every step.
    """
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    fb = FeatureBuilder()
    state, _uid = _state()
    ts = fb.build(state)
    z0 = m.encode(ts, "admit")
    psi = kernel_state_summary(state, m.be)
    H = 6
    traj = m.rollout(z0, "admit", ts.masks, kernel_states=[psi] * H, horizon=H)
    assert len(traj) == H
    # z0 differs from z1 (the predictor actually moved the belief)
    d01 = np.abs(_np(m.be, z0["workload"]) - _np(m.be, traj[0]["workload"])).sum()
    assert d01 > 1e-3
    # every consecutive pair differs (distinct, not replicated)
    for i in range(H - 1):
        d = np.abs(
            _np(m.be, traj[i]["workload"]) - _np(m.be, traj[i + 1]["workload"])
        ).sum()
        assert d > 1e-4, f"steps {i},{i+1} are identical — rollout is degenerate"


def test_rollout_is_action_sensitive():
    """Different actions yield different rollouts (doc J D5/Eq. 9 action fidelity)."""
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    fb = FeatureBuilder()
    state, _uid = _state()
    ts = fb.build(state)
    z0 = m.encode(ts, "hold")
    psi = kernel_state_summary(state, m.be)
    H = 6
    t_hold = m.rollout(z0, "hold", ts.masks, kernel_states=[psi] * H, horizon=H)
    t_admit = m.rollout(z0, "admit", ts.masks, kernel_states=[psi] * H, horizon=H)
    sep = np.abs(
        _np(m.be, t_hold[-1]["workload"]) - _np(m.be, t_admit[-1]["workload"])
    ).sum()
    assert sep > 1e-2, "predictor ignores the action (action collapse)"


def test_kernel_state_summary_is_exact_and_fixed_width():
    """ψ(s^K) is a fixed-width exact summary (doc J D6/T4 constants)."""
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    state, _uid = _state()
    psi = kernel_state_summary(state, m.be)
    arr = _np(m.be, psi)
    assert arr.shape == (8,)
    assert np.all(np.isfinite(arr))
    # deterministic
    psi2 = kernel_state_summary(state, m.be)
    assert np.allclose(arr, _np(m.be, psi2))


def test_rollout_conditions_on_kernel_state():
    """Changing ψ(s^K) changes the rollout (the kernel state is actually used)."""
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    fb = FeatureBuilder()
    state, _uid = _state()
    ts = fb.build(state)
    z0 = m.encode(ts, "admit")
    psi = kernel_state_summary(state, m.be)
    psi2 = m.be.from_numpy(_np(m.be, psi) + 1.0)
    a = m.rollout(z0, "admit", ts.masks, kernel_states=[psi] * 4, horizon=4)
    b = m.rollout(z0, "admit", ts.masks, kernel_states=[psi2] * 4, horizon=4)
    d = np.abs(_np(m.be, a[-1]["workload"]) - _np(m.be, b[-1]["workload"])).sum()
    assert d > 1e-3


def test_horizon_curriculum_grows():
    """H grows from curriculum_h_start toward curriculum_h_end (doc J B2)."""
    cfg = TrainConfig(curriculum_h_start=2, curriculum_h_end=12, h_slow=12)
    h_first = _curriculum_horizon("rollout", 0, 100, cfg)
    h_last = _curriculum_horizon("rollout", 99, 100, cfg)
    assert h_first == 2
    assert h_last == 12
    assert _curriculum_horizon("warmup", 50, 100, cfg) == cfg.warmup_horizon
    # monotone non-decreasing
    seq = [_curriculum_horizon("rollout", s, 100, cfg) for s in range(0, 100, 10)]
    assert seq == sorted(seq)


def test_exogenous_vector_encodes_context():
    """u_h carries missingness/rare/real/branch indicators (doc J D6)."""
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    w = make_smoke_windows(1, seed=0)[0]
    u = _exogenous_vector(w, m.cfg.u_dim, m.be)
    arr = _np(m.be, u)
    assert arr.shape == (m.cfg.u_dim,)
    # the smoke window has a future_state, so the missing-future flag is 0
    assert arr[0] == 0.0


def test_truncated_bptt_detach_is_applied():
    """detach_every stop-gradients the carried latent without altering values.

    On the NumPy reference path stop_gradient is a no-op, so the *values* are
    identical with and without detaching — the detach only changes the gradient
    graph (doc J B3). This pins that the rollout values are detach-invariant.
    """
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    fb = FeatureBuilder()
    state, _uid = _state()
    ts = fb.build(state)
    z0 = m.encode(ts, "admit")
    psi = kernel_state_summary(state, m.be)
    ks = [psi] * 8
    a = m.rollout(z0, "admit", ts.masks, kernel_states=ks, horizon=8, detach_every=4)
    b = m.rollout(z0, "admit", ts.masks, kernel_states=ks, horizon=8, detach_every=0)
    for i in range(8):
        assert np.allclose(_np(m.be, a[i]["workload"]), _np(m.be, b[i]["workload"]))


def test_stage_loss_uses_free_running_rollout():
    """stage_loss in the rollout stage drives the predictor (no replication)."""
    m = ResidualWorldModel.build(n_ensemble=1)
    windows = make_smoke_windows(4, seed=0)
    jcfg = JEPAConfig()
    lv = GroupLogVars()
    total, groups = stage_loss(
        m, windows, "rollout", TrainConfig(), jcfg, lv, None, step=5, total_steps=10
    )
    assert np.isfinite(total)
    assert groups["rollout"] > 0.0  # the rollout group is exercised


def test_serving_contract_stable_and_one_bundle_per_action():
    """encode/score_actions stay contract-stable; one bundle per action."""
    m = ResidualWorldModel.build(n_ensemble=2)
    state, uid = _state()
    z = m.encode(state)
    assert set(z.keys()) == {
        "workload",
        "podset",
        "queue",
        "topology_domain",
        "node",
        "gpu",
        "rack",
        "switch",
        "global",
    }
    acts = [
        Hold(workload_uid=uid),
        Admit(workload_uid=uid, cluster_queue_id="cq-0", resource_flavor_id="flavor-0"),
    ]
    bundles = m.score_actions(state, acts, HORIZONS_S[2])
    assert len(bundles) == len(acts)
    assert bundles[0].action_id != bundles[1].action_id


def test_smoke_train_runs_with_predictor():
    """The end-to-end smoke train trains the predictor without collapse."""
    report = smoke_train(cfg=TrainConfig(seed=1))
    assert report.steps > 0
    assert not report.collapsed()
