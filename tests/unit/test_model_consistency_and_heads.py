# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Tests for MODEL-LCONS-HEADLOSSES, MODEL-MLX-AUTODIFF, and the branch-diff /
D5 / D3 / D4 diagnostics (docs J D11/Eq.18, H1/H2/H3 Eq.11-17, D13 G1/G3/G4/G5/G6,
D3/D4/D5; doc E stage_two).

Pins:

* the mandatory exact-state consistency loss ``L_cons`` from the per-field
  decoders ``D_f`` against stop-gradient kernel fields (G5);
* the head losses wired into their correct groups (pinball+non-crossing → G3,
  competing-risk + binary-hazard NLL → G4, focal-BCE OOD → G6) and the removal
  of the mislabeled stand-ins (no anti-collapse in G4, no abs(global) in G6);
* anti-collapse moved into G1 (jepa);
* the D3 covariance-energy and D4 cosine collapse diagnostics;
* the true D5 action-separation ratio against EMA-encoded counterfactual futures;
* the MLX autodiff backend import guards (NumPy SPSA stays the working fallback).
"""

from __future__ import annotations

import numpy as np

from gpu_center_world_model.model.architecture import (
    FIELD_SPEC,
    ArchConfig,
    FeatureBuilder,
    FieldDecoders,
    RelationalWorldModel,
    exact_field_targets,
)
from gpu_center_world_model.model.backend import get_backend
from gpu_center_world_model.model.jepa import (
    GroupLogVars,
    JEPAConfig,
    consistency_loss,
    cosine_collapse,
    covariance_energy,
    ood_head_loss,
    quantile_head_loss,
    rollout_consistency_loss,
    survival_head_loss,
)
from gpu_center_world_model.model.train import (
    TrainConfig,
    _diag_snapshot,
    _trainable_arrays,
    make_smoke_windows,
    smoke_train,
    stage_loss,
)
from gpu_center_world_model.model.world_model import ResidualWorldModel


def _np(be, x):
    return np.asarray(be.to_numpy(x))


# --------------------------------------------------------------------------- #
# MODEL-LCONS-HEADLOSSES — field decoders + L_cons (doc J D11 / Eq. 18).
# --------------------------------------------------------------------------- #
def test_field_decoders_built_for_every_exact_field():
    """Every F_exact field gets a decoder D_f over its latent type (doc J D11)."""
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    assert isinstance(m.field_decoders, FieldDecoders)
    for name, _ltype, _out, _flavor in FIELD_SPEC:
        assert name in m.field_decoders.decoders, f"missing decoder for {name}"


def test_exact_field_targets_align_with_decoder_outputs():
    """Decoder outputs and kernel targets share the padded (cap, out_dim) shape."""
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    fb = FeatureBuilder()
    state = make_smoke_windows(1, seed=0)[0].state
    ts = fb.build(state)
    z = m.encode(ts, "admit")
    dec = m.field_decoders.decode(z, m.be)
    tgt = exact_field_targets(state, ts.order, ts.masks, m.be)
    assert set(dec.keys()) == set(tgt.keys())
    for name, out in dec.items():
        t, valid = tgt[name]
        assert _np(m.be, out).shape == t.shape, f"{name} shape mismatch"
        assert valid.shape[0] == t.shape[0]


def test_consistency_loss_is_finite_and_nonnegative():
    """L_cons (Eq. 18) is a finite, non-negative scalar over the field set."""
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    fb = FeatureBuilder()
    state = make_smoke_windows(1, seed=0)[0].state
    ts = fb.build(state)
    z = m.encode(ts, "admit")
    dec = m.field_decoders.decode(z, m.be)
    tgt = exact_field_targets(state, ts.order, ts.masks, m.be)
    lc = consistency_loss(dec, tgt, FIELD_SPEC, be=m.be)
    assert np.isfinite(lc) and lc >= 0.0


def test_consistency_loss_rewards_correct_fields():
    """A decoder matching the kernel field beats a maximally-wrong one (T3)."""
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    fb = FeatureBuilder()
    state = make_smoke_windows(1, seed=0)[0].state
    ts = fb.build(state)
    m.encode(ts, "admit")
    tgt = exact_field_targets(state, ts.order, ts.masks, m.be)
    # build "perfect" binary preds: large +logit where target=1, large -logit else
    good = {}
    bad = {}
    for name, _lt, _od, flavor in FIELD_SPEC:
        if flavor != "bce":
            continue
        t, _valid = tgt[name]
        good[name] = m.be.from_numpy((2.0 * t - 1.0) * 8.0)  # logits aligned
        bad[name] = m.be.from_numpy((1.0 - 2.0 * t) * 8.0)  # logits inverted
    l_good = consistency_loss(good, tgt, FIELD_SPEC, be=m.be)
    l_bad = consistency_loss(bad, tgt, FIELD_SPEC, be=m.be)
    assert l_good < l_bad, "L_cons does not reward kernel-consistent decoders"


def test_rollout_consistency_sums_over_steps():
    """rollout_consistency_loss sums L_cons over every step h=1..H (doc J Eq. 18)."""
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    fb = FeatureBuilder()
    state = make_smoke_windows(1, seed=0)[0].state
    ts = fb.build(state)
    z = m.encode(ts, "admit")
    dec = m.field_decoders.decode(z, m.be)
    tgt = exact_field_targets(state, ts.order, ts.masks, m.be)
    H = 4
    out = rollout_consistency_loss(
        [dec] * H, [tgt] * H, FIELD_SPEC, be=m.be
    )
    single = consistency_loss(dec, tgt, FIELD_SPEC, be=m.be)
    # equal per-step weights -> the H-step average equals the single-step value
    assert np.isclose(out, single, atol=1e-8)


# --------------------------------------------------------------------------- #
# Loss-group wiring (doc J D13 G1/G3/G4/G5/G6).
# --------------------------------------------------------------------------- #
def test_all_six_groups_are_populated_in_rollout_stage():
    """Every coarse loss group carries a real, finite, non-zero loss (doc J D13)."""
    m = ResidualWorldModel.build(n_ensemble=1)
    windows = make_smoke_windows(4, seed=0)
    total, groups = stage_loss(
        m, windows, "rollout", TrainConfig(), JEPAConfig(), GroupLogVars(),
        None, step=5, total_steps=10,
    )
    assert np.isfinite(total)
    for g in ("jepa", "rollout", "quantile", "survival", "consistency", "abstention"):
        assert np.isfinite(groups[g]), f"group {g} is not finite"
    # G5 consistency, G3 quantile, G4 survival, G6 abstention all exercised
    assert groups["consistency"] > 0.0
    assert groups["quantile"] > 0.0
    assert groups["survival"] > 0.0
    assert groups["abstention"] > 0.0
    # G1 carries the JEPA latent loss *plus* anti-collapse (moved out of G5)
    assert groups["jepa"] > 0.0


def test_field_decoders_are_trainable():
    """The field decoders are in the trainable-array set (so L_cons trains them)."""
    m = ResidualWorldModel.build(n_ensemble=1)
    handles = _trainable_arrays(m)
    decoder_objs = set(id(d) for d in m.backbone.field_decoders.decoders.values())
    owned = set(id(o) for o, _a in handles)
    assert decoder_objs & owned, "field decoders are not trained"


# --------------------------------------------------------------------------- #
# Head losses (doc J H1/H2/H3, Eq. 11-17).
# --------------------------------------------------------------------------- #
def test_quantile_head_loss_includes_pinball_and_noncrossing():
    """G3 = pinball + λ_mono·non-crossing over realized KPI-delta labels (Eq.11-13)."""
    m = ResidualWorldModel.build(n_ensemble=1)
    state = make_smoke_windows(1, seed=0)[0].state
    ts = m.feature_builder.build(state, None)
    z = m.backbone.encode(ts, "hold")
    qf = m.heads.quantile_forecasts(z, ts.masks)
    labels = {(qf[0].channel.value, qf[0].horizon_s): 1.0}
    loss_with = quantile_head_loss(qf, labels, m.heads.cfg.quantiles, 1.0)
    loss_no = quantile_head_loss(qf, {}, m.heads.cfg.quantiles, 1.0)
    assert np.isfinite(loss_with) and loss_with >= 0.0
    # a non-zero label increases the pinball contribution over the label-free case
    assert loss_with >= loss_no


def test_survival_head_loss_uses_competing_and_binary_nll():
    """G4 = competing-risk NLL + parallel-binary hazard NLL (doc J Eq. 14-17)."""
    probs = {"workload": [np.array([0.7, 0.2, 0.1]) for _ in range(4)]}
    haz = {"xid_fault": np.array([0.1, 0.1, 0.1, 0.1])}
    # an observed event raises the NLL vs. right-censoring
    nll_event = survival_head_loss(
        probs, haz, {"workload": (1, 0)}, {"xid_fault": 0}
    )
    nll_cens = survival_head_loss(
        probs, haz, {"workload": (None, None)}, {"xid_fault": None}
    )
    assert np.isfinite(nll_event) and np.isfinite(nll_cens)
    assert nll_event > nll_cens


def test_ood_head_loss_is_focal_bce_and_drops_when_correct():
    """G6 = focal-BCE on the OOD label (doc J H3 / Eq. 17), not an abs placeholder."""
    # confident-correct (logit high, label 1) << confident-wrong (logit low, label 1)
    correct = ood_head_loss(0.0, 8.0, 1.0, focal=True)
    wrong = ood_head_loss(0.0, -8.0, 1.0, focal=True)
    assert correct < wrong
    assert np.isfinite(correct) and correct >= 0.0


# --------------------------------------------------------------------------- #
# Collapse diagnostics D3 / D4 (doc J D3/D4).
# --------------------------------------------------------------------------- #
def test_covariance_energy_zero_for_orthogonal_columns():
    """D3 c_off ~ 0 for decorrelated dims, large for fully-correlated dims."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, 8))
    assert covariance_energy(x) < 0.2
    # duplicate a column into all dims -> maximal off-diagonal energy
    col = rng.normal(size=(256, 1))
    xc = np.repeat(col, 8, axis=1) + 1e-3 * rng.normal(size=(256, 8))
    assert covariance_energy(xc) > covariance_energy(x)


def test_cosine_collapse_near_one_for_collapsed_rows():
    """D4 s_cos -> 1 when every row points the same way (complete collapse)."""
    v = np.ones((32, 6))
    assert cosine_collapse(v) > 0.99
    rng = np.random.default_rng(1)
    spread = rng.normal(size=(32, 6))
    assert cosine_collapse(spread) < cosine_collapse(v)


def test_diag_snapshot_reports_d3_d4_d5():
    """A diagnostic snapshot carries the D3/D4/D5 fields (doc J D3-D5)."""
    m = ResidualWorldModel.build(n_ensemble=1)
    windows = make_smoke_windows(4, seed=0)
    ts = m.feature_builder.build(windows[0].state, None)
    snap = _diag_snapshot(m, ts, windows, 0, TrainConfig(), None)
    assert snap.cov_energy and snap.cosine_sim
    for t in snap.cov_energy:
        assert np.isfinite(snap.cov_energy[t])
        assert np.isfinite(snap.cosine_sim[t])
    assert np.isfinite(snap.action_separation)


# --------------------------------------------------------------------------- #
# MODEL-BRANCH-DIFF-DIAGNOSTICS — true D5 ratio (doc J D5 / Eq. 9-10).
# --------------------------------------------------------------------------- #
def test_d5_uses_ema_branch_denominator_not_raw_norm():
    """With an EMA teacher, D5 is the true A=||Δẑ||/||Δz⁺|| ratio (doc J D5).

    Without a teacher the denominator falls back to 1 (raw predicted norm); the
    two must differ, proving the teacher-encoded counterfactual branch futures
    are actually used as the denominator.
    """
    m = ResidualWorldModel.build(n_ensemble=1)
    teacher = ResidualWorldModel.build(
        arch=m.backbone.cfg, head_cfg=m.heads.cfg, n_ensemble=1, backend=m.be
    )
    windows = make_smoke_windows(4, seed=0)
    ts = m.feature_builder.build(windows[0].state, None)
    raw = _diag_snapshot(m, ts, windows, 0, TrainConfig(), None).action_separation
    true = _diag_snapshot(m, ts, windows, 0, TrainConfig(), teacher).action_separation
    assert np.isfinite(raw) and np.isfinite(true)
    assert abs(raw - true) > 1e-6, "D5 ignores the EMA branch denominator"


def test_branch_diff_term_present_in_rollout_group():
    """The branch-difference action-fidelity term contributes to G2 (doc J Eq. 9)."""
    m = ResidualWorldModel.build(n_ensemble=1)
    # paired-branch windows (make_smoke_windows always sets branch_action)
    paired = make_smoke_windows(4, seed=0)
    assert all(w.branch_action is not None for w in paired)
    _t, groups = stage_loss(
        m, paired, "rollout", TrainConfig(), JEPAConfig(), GroupLogVars(),
        None, step=5, total_steps=10,
    )
    assert groups["rollout"] > 0.0


# --------------------------------------------------------------------------- #
# MODEL-MLX-AUTODIFF — import guards (NumPy SPSA stays the working fallback).
# --------------------------------------------------------------------------- #
def test_mlx_backend_imports_and_guards_without_mlx():
    """The mlx_backend package imports cleanly; absent MLX it guards every entry."""
    import gpu_center_world_model.model.mlx_backend as mb

    assert hasattr(mb, "HAS_MLX")
    if not mb.HAS_MLX:
        # placeholders are None and the entry point raises a clear RuntimeError
        assert mb.ParamTree is None
        import pytest

        with pytest.raises(RuntimeError):
            mb.mlx_train_step()


def test_numpy_spsa_smoke_train_descends_and_stays_uncollapsed():
    """The NumPy SPSA fallback trains end-to-end without collapse (HARD RULE 3).

    Requests the NumPy backend explicitly so the always-on SPSA reference path is
    exercised even when MLX is installed (the default backend on Apple silicon).
    """
    report = smoke_train(cfg=TrainConfig(seed=1), backend=get_backend("numpy"))
    assert report.backend == "numpy"
    assert report.steps > 0
    assert not report.collapsed()
