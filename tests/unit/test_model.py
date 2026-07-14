# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Model-lane unit tests (item TEST-MODEL-LANE).

Exercises the doc-J learning-objective primitives and the doc-E smoke-train
pipeline against the **real** :class:`RelationalWorldModel` / :class:`ForecastHeads`
(built, not mocked):

* **Non-crossing quantiles** — structural ``q_τ ≤ q_τ'`` from the real head and
  the ``noncrossing_penalty`` that scores it (doc J Eq. 13).
* **Rollout / branch-difference sign behavior over H steps** — ``L_roll`` =
  ``L_roll,abs + λ_Δ·L_roll,Δ`` and the action-fidelity term's sensitivity to the
  *sign* of the predicted branch difference (doc J Eq. 8–10, T2).
* **Combined objective six-group weighting** — ``Σ_m exp(-s_m)·G_m + s_m`` with
  the log-variance clamp (doc J Eq. 19/27).
* **EMA update** — ``ξ ← τ·ξ + (1-τ)·θ`` (doc J Eq. 1).
* **Competing-risk NLL censoring** — Eq. 16 (right-censoring keeps only the
  survived-interval terms) vs Eq. 15 (observed event).
* **Smoke-train collapse-alarm path** — a real ``smoke_train`` run plus the real
  ``_diag_snapshot`` D1–D5 alarm pipeline driven onto a collapsed latent.
"""

from __future__ import annotations

import itertools

import numpy as np

from gpu_center_world_model.data_engine.synthetic import PodSpec, generate_pod_state
from gpu_center_world_model.model.architecture import (
    ArchConfig,
    FeatureBuilder,
    RelationalWorldModel,
)
from gpu_center_world_model.model.backend import get_backend
from gpu_center_world_model.model.heads import ForecastHeads, latent_dims_of
from gpu_center_world_model.model.jepa import (
    LOSS_GROUPS,
    GroupLogVars,
    JEPAConfig,
    combined_objective,
    competing_risk_nll,
    ema_update,
    noncrossing_penalty,
    rollout_abs_loss,
    rollout_branch_diff_loss,
    rollout_loss,
)
from gpu_center_world_model.model.train import (
    DiagSnapshot,
    TrainConfig,
    TrainReport,
    _diag_snapshot,
    collapsed_dim_fraction,
    cosine_collapse,
    effective_rank,
    make_smoke_windows,
    smoke_train,
)
from gpu_center_world_model.model.world_model import ResidualWorldModel


def _np(be, x):
    return np.asarray(be.to_numpy(x))


def _state(seed: int = 7, demand: int = 4):
    _pod, state, uid = generate_pod_state(
        PodSpec(n_racks=2, nodes_per_rack=2, gpus_per_node=4),
        seed=seed,
        gpu_demand=demand,
    )
    return state, uid


# ───────────────────────────── non-crossing quantiles ──────────────────────────
def test_real_head_quantiles_are_noncrossing_by_construction():
    """Every real KPI forecast is monotone in τ (doc J Eq. 13 made structural).

    The KPI head is a median + cumulative-softplus offset parameterisation, so
    ``q_{τ} ≤ q_{τ'}`` for ``τ < τ'`` holds for *every* channel/horizon emitted by
    the real :meth:`ForecastHeads.quantile_forecasts` — never relying on a sort.
    """
    m = RelationalWorldModel.build(ArchConfig(seed=0))
    fb = FeatureBuilder()
    heads = ForecastHeads.build(latent_dims_of(m))
    state, _uid = _state()
    ts = fb.build(state)
    z = m.encode(ts, "place")
    forecasts = heads.quantile_forecasts(z, ts.masks)
    assert forecasts, "the head must emit at least one quantile forecast"
    for qf in forecasts:
        taus = sorted(qf.quantiles)
        vals = [qf.quantiles[t] for t in taus]
        # strict non-crossing q_τ ≤ q_τ' for τ < τ'
        for a, b in itertools.pairwise(vals):
            assert a <= b + 1e-6, f"{qf.channel}@{qf.horizon_s}s crosses: {vals}"
        # and the penalty that *measures* crossing is therefore zero
        assert noncrossing_penalty(qf.quantiles, tuple(taus)) <= 1e-6


def test_noncrossing_penalty_zero_on_sorted_positive_on_crossing():
    """``noncrossing_penalty`` = 0 iff non-decreasing, else the crossing mass."""
    taus = (0.1, 0.5, 0.9)
    assert noncrossing_penalty({0.1: 1.0, 0.5: 2.0, 0.9: 3.0}, taus) == 0.0
    # a single crossing of magnitude 1 between the first adjacent pair
    assert noncrossing_penalty({0.1: 2.0, 0.5: 1.0, 0.9: 3.0}, taus) == 1.0
    # fully reversed quantiles: 0.1>0.5 (1) + 0.5>0.9 (1) = 2
    assert noncrossing_penalty({0.1: 3.0, 0.5: 2.0, 0.9: 1.0}, taus) == 2.0


# ──────────────────────── rollout / branch-diff sign behavior ───────────────────
def _step(val: float, n: int = 1, d: int = 3) -> dict[str, np.ndarray]:
    return {"gpu": np.full((n, d), float(val))}


def test_branch_diff_loss_tracks_difference_and_penalizes_wrong_sign():
    """``L_roll,Δ`` matches Δẑ to sg[Δz⁺] — sign of the branch diff matters (Eq. 9).

    Doc J T2: paired branch-difference supervision is the minimal action-fidelity
    term. We hold the target branch difference fixed at ``+2`` and vary the
    *predicted* branch difference: matching it drives the loss to 0; a smaller
    same-sign diff gives a positive loss; a *flipped-sign* predicted diff is
    strictly worse than the same-magnitude correct-sign diff.
    """
    H = 3
    tgt_b, tgt_bp = [_step(2.0)] * H, [_step(0.0)] * H  # Δz⁺ = +2

    # predicted diff exactly +2 -> perfectly matched -> 0
    matched = rollout_branch_diff_loss(
        [_step(2.0)] * H, [_step(0.0)] * H, tgt_b, tgt_bp
    )
    assert matched == 0.0

    # predicted diff +1 (same sign, too small) -> |1 - 2| = 1
    same_sign = rollout_branch_diff_loss(
        [_step(1.0)] * H, [_step(0.0)] * H, tgt_b, tgt_bp
    )
    assert abs(same_sign - 1.0) < 1e-9

    # predicted diff -1 (FLIPPED sign) -> |-1 - 2| = 3 > 1
    wrong_sign = rollout_branch_diff_loss(
        [_step(0.0)] * H, [_step(1.0)] * H, tgt_b, tgt_bp
    )
    assert abs(wrong_sign - 3.0) < 1e-9
    assert wrong_sign > same_sign, "a wrong-sign branch diff must be worse"


def test_rollout_loss_adds_lambda_weighted_branch_term_over_h_steps():
    """``L_roll = L_roll,abs + λ_Δ·L_roll,Δ`` (doc J Eq. 10)."""
    H = 2
    pred = [_step(0.0)] * H
    tgt = [_step(1.0)] * H  # abs latent mismatch present

    # absolute term alone (layer-normed L1 of a constant row is 0): identical
    # constant rows layer-norm to zero, so L_roll,abs == 0 here.
    abs_only = rollout_loss(pred, tgt, branch_pairs=None)
    assert abs_only == rollout_abs_loss(pred, tgt)

    # add a branch pair whose |Δẑ - Δz⁺| = 1 per step, scaled by λ_Δ = 2 -> +2
    pb, pbp = [_step(1.0)] * H, [_step(0.0)] * H
    tb, tbp = [_step(2.0)] * H, [_step(0.0)] * H
    with_branch = rollout_loss(
        pred, tgt, branch_pairs=[(pb, pbp, tb, tbp)], cfg=JEPAConfig(lambda_delta=2.0)
    )
    assert with_branch > abs_only
    assert abs(with_branch - (abs_only + 2.0 * 1.0)) < 1e-9


def test_rollout_abs_step_weights_select_horizon_band():
    """``ρ_h`` step weights re-weight the per-step absolute rollout match (Eq. 8)."""
    # branch-distinct, non-constant rows so layer-norm yields a real distance
    pred = [{"gpu": np.array([[1.0, 2.0, 3.0]])}, {"gpu": np.array([[3.0, 2.0, 1.0]])}]
    tgt = [{"gpu": np.array([[3.0, 2.0, 1.0]])}, {"gpu": np.array([[1.0, 2.0, 3.0]])}]
    both = rollout_abs_loss(pred, tgt, step_weights=[1.0, 1.0])
    only_first = rollout_abs_loss(pred, tgt, step_weights=[1.0, 0.0])
    only_second = rollout_abs_loss(pred, tgt, step_weights=[0.0, 1.0])
    assert both > 0.0
    # zeroing a step's weight removes its contribution from the weighted mean
    assert only_first > 0.0 and only_second > 0.0
    # the weighted mean of the two equals the all-ones average
    assert abs(both - 0.5 * (only_first + only_second)) < 1e-9


# ─────────────────────── combined objective six-group weighting ────────────────
def test_combined_objective_reduces_to_plain_sum_at_zero_logvars():
    """With every ``s_m = 0`` the total is the plain sum of the six groups (Eq. 19)."""
    losses = {g: float(i + 1) for i, g in enumerate(LOSS_GROUPS)}
    total, parts = combined_objective(losses)
    assert set(parts) == set(LOSS_GROUPS)
    assert abs(total - sum(losses.values())) < 1e-9
    # missing groups contribute exactly zero
    total_partial, _ = combined_objective({"jepa": 5.0})
    assert abs(total_partial - 5.0) < 1e-9


def test_combined_objective_applies_exp_neg_s_weighting_and_clamps():
    """``Σ_m exp(-s_m)·G_m + s_m`` with the log-variance clamp to ``s_bounds``."""
    losses = {g: float(i + 1) for i, g in enumerate(LOSS_GROUPS)}
    lv = GroupLogVars()
    lv.s = {g: 1.0 for g in LOSS_GROUPS}
    total, parts = combined_objective(losses, lv)
    expected = sum(np.exp(-1.0) * losses[g] + 1.0 for g in LOSS_GROUPS)
    assert abs(total - expected) < 1e-9
    for g in LOSS_GROUPS:
        assert abs(parts[g] - (np.exp(-1.0) * losses[g] + 1.0)) < 1e-9

    # an out-of-range log-variance is clamped to s_bounds (Eq. 27 = (-3, 3))
    lv_hi = GroupLogVars()
    lv_hi.s = {g: 99.0 for g in LOSS_GROUPS}
    combined_objective(losses, lv_hi, JEPAConfig())
    assert all(v == 3.0 for v in lv_hi.s.values())
    lv_lo = GroupLogVars()
    lv_lo.s = {g: -99.0 for g in LOSS_GROUPS}
    combined_objective(losses, lv_lo, JEPAConfig())
    assert all(v == -3.0 for v in lv_lo.s.values())


def test_higher_logvar_downweights_its_group_loss():
    """A larger ``s_m`` shrinks the exp(-s_m) prefactor on that group's loss."""
    losses = {g: 10.0 for g in LOSS_GROUPS}
    lv_low = GroupLogVars()
    lv_low.s = {g: 0.0 for g in LOSS_GROUPS}
    lv_high = GroupLogVars()
    lv_high.s = {g: 2.0 for g in LOSS_GROUPS}
    _, p_low = combined_objective(losses, lv_low)
    _, p_high = combined_objective(losses, lv_high)
    # the *loss contribution* (weighted minus the additive s) falls as s grows
    for g in LOSS_GROUPS:
        contrib_low = p_low[g] - 0.0
        contrib_high = p_high[g] - 2.0
        assert contrib_high < contrib_low


# ──────────────────────────────── EMA update ──────────────────────────────────
def test_ema_update_is_convex_blend_of_target_and_online():
    """``ξ ← τ·ξ + (1-τ)·θ`` with ∇ξ = 0 (doc J Eq. 1)."""
    target = {"w": np.zeros(4), "b": np.full(2, 4.0)}
    online = {"w": np.ones(4), "b": np.zeros(2)}
    out = ema_update(target, online, decay=0.9)
    assert np.allclose(out["w"], 0.1)  # 0.9*0 + 0.1*1
    assert np.allclose(out["b"], 3.6)  # 0.9*4 + 0.1*0
    # τ = 1 freezes the target (the calibration-stage rule)
    frozen = ema_update(target, online, decay=1.0)
    assert np.allclose(frozen["w"], target["w"])
    assert np.allclose(frozen["b"], target["b"])
    # an online-only key (no prior target) passes through unchanged
    fresh = ema_update({}, {"w": np.full(3, 5.0)}, decay=0.99)
    assert np.allclose(fresh["w"], 5.0)


# ─────────────────────── competing-risk NLL censoring ──────────────────────────
def test_competing_risk_nll_censoring_keeps_only_survival_terms():
    """Right-censoring (Eq. 16) sums only survived-interval log-probs; event (Eq. 15)
    adds the realized class term."""
    # interval class dists [survive, risk_1, risk_2]
    ip = [
        np.array([0.7, 0.2, 0.1]),
        np.array([0.6, 0.3, 0.1]),
        np.array([0.5, 0.4, 0.1]),
    ]
    # right-censored (no event observed): -Σ_r log π_{r,0} over ALL intervals
    cens = competing_risk_nll(ip, event_class=None, event_interval=None)
    expected_cens = -(np.log(0.7) + np.log(0.6) + np.log(0.5))
    assert abs(cens - expected_cens) < 1e-9

    # observed event of class 1 at interval 1: survive intervals < 1, then -log π_{1,1}
    ev = competing_risk_nll(ip, event_class=1, event_interval=1)
    expected_ev = -(np.log(0.7) + np.log(ip[1][1]))
    assert abs(ev - expected_ev) < 1e-9

    # censoring at an earlier interval truncates the survival product (fewer terms)
    cens_early = competing_risk_nll(ip, event_class=None, event_interval=1)
    expected_early = -np.log(0.7)
    assert abs(cens_early - expected_early) < 1e-9
    assert cens_early < cens  # fewer survived intervals -> smaller NLL


def test_competing_risk_nll_penalizes_low_event_probability():
    """A sharper (higher-probability) realized event has lower NLL."""
    sharp = [np.array([0.1, 0.9])]
    flat = [np.array([0.5, 0.5])]
    nll_sharp = competing_risk_nll(sharp, event_class=1, event_interval=0)
    nll_flat = competing_risk_nll(flat, event_class=1, event_interval=0)
    assert nll_sharp < nll_flat


# ─────────────────────── smoke-train + collapse-alarm path ─────────────────────
def test_smoke_train_runs_and_reports_real_diagnostics():
    """A real ``smoke_train`` runs all stages and returns D1–D5 diagnostics (doc E).

    The healthy run must finish with a finite loss, a diagnostic per cadence step,
    and — for the well-conditioned default model — no collapse alarm. This case
    pins the always-on NumPy SPSA reference path (HARD RULE 3), so it requests the
    NumPy backend explicitly even when MLX is available (the MLX autodiff path has
    its own end-to-end smoke coverage in ``test_model_predictor_rollout``).
    """
    report = smoke_train(
        n_windows=8, cfg=TrainConfig(seed=0), backend=get_backend("numpy")
    )
    assert isinstance(report, TrainReport)
    assert report.backend == "numpy"
    assert report.steps > 0
    assert np.isfinite(report.final_loss)
    assert report.diagnostics, "expected at least one collapse-diagnostic snapshot"
    # every snapshot carries the real per-type D1 (rank) / D4 (cosine) readings
    for d in report.diagnostics:
        assert isinstance(d, DiagSnapshot)
        assert d.effective_rank and d.cosine_sim
    # the default model is well-conditioned: no sustained collapse alarm
    assert report.collapsed() is False


class _CollapsedBackbone:
    """Delegates everything to the real backbone but collapses ``encode`` output.

    Only the encoder *output* is rank-1 (all rows identical); every parameter and
    method is the real one, so :func:`_diag_snapshot` runs the genuine D1–D5
    pipeline (effective rank, cosine, variance fraction) over the collapsed latent.
    """

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):  # pragma: no cover - thin delegation
        return getattr(self._real, name)

    def encode(self, ts, action_name="hold"):
        z = self._real.encode(ts, action_name)
        out = {}
        for t, v in z.items():
            arr = _np(self._real.be, v)
            out[t] = self._real.be.from_numpy(np.full_like(arr, 0.5))
        return out


class _CollapsedModel:
    """A real :class:`ResidualWorldModel` with its backbone encode collapsed."""

    def __init__(self, real):
        self._real = real
        self.backbone = _CollapsedBackbone(real.backbone)

    def __getattr__(self, name):  # pragma: no cover - thin delegation
        return getattr(self._real, name)


def test_diag_snapshot_raises_collapse_alarm_on_collapsed_latent():
    """The real D1–D5 alarm pipeline fires when the latent collapses (doc J D1–D5).

    Built from the REAL model: a healthy encode raises no alarm, while a rank-1
    (all-rows-identical) latent drives the GPU effective rank → 0 and cosine → 1,
    tripping the alarm through the genuine ``_diag_snapshot`` code path that
    ``train`` uses every diagnostic cadence step.
    """
    model = ResidualWorldModel.build(arch=ArchConfig(seed=0), n_ensemble=1)
    fb = FeatureBuilder()
    windows = make_smoke_windows(4, seed=0)
    ts = fb.build(windows[0].state)
    cfg = TrainConfig(seed=0)

    healthy = _diag_snapshot(model, ts, windows, step=0, cfg=cfg)
    assert healthy.collapse_alarm is False

    collapsed = _diag_snapshot(
        _CollapsedModel(model), ts, windows, step=1, cfg=cfg
    )
    assert collapsed.collapse_alarm is True
    # the collapse is visible in the underlying D1/D4 diagnostics
    assert collapsed.effective_rank["gpu"] < 1.0 + 1e-9
    assert collapsed.cosine_sim["gpu"] > 0.98


def test_collapse_diagnostic_functions_separate_healthy_from_collapsed():
    """The raw D1/D3/D4 diagnostics distinguish a spread latent from a collapsed one."""
    rng = np.random.default_rng(0)
    healthy = rng.normal(size=(32, 16))
    collapsed = np.full((32, 16), 0.5)
    # D1 effective rank: healthy uses many dims, collapsed is rank-0
    assert effective_rank(healthy) > 2.0
    assert effective_rank(collapsed) == 0.0
    # D4 cosine: healthy near-orthogonal rows, collapsed identical rows -> 1
    assert cosine_collapse(collapsed) > 0.99
    assert cosine_collapse(healthy) < 0.5
    # D2 collapsed-dim fraction: a constant latent has every dim below the floor
    assert collapsed_dim_fraction(collapsed, var_floor=1.0) == 1.0
    assert collapsed_dim_fraction(healthy, var_floor=1.0) < 1.0
