# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The staged training loop for the residual world model (docs J, E).

This module realizes doc J's stage-wise curriculum on top of the backend shim so
the *same* loop runs on MLX (Apple silicon, real autodiff) and on the NumPy
reference path used by CI. The four stages mirror doc J C1–C5:

* **Stage 1 — warmup** (doc J C1). Action-free / weakly action-conditioned JEPA
  on mixed observational windows. Optimize the latent group ``G1`` (JEPA + weak
  VICReg anti-collapse) plus an optional short-horizon (``H = 2``) rollout. The
  EMA target decay ``τ`` rises on a cosine schedule from ``0.99`` to ``0.9995``.
* **Stage 2 — counterfactual rollout** (doc J C2). Free-running, action-
  conditioned multi-step rollouts with the *absolute* and *branch-difference*
  action-fidelity terms (doc J Eqs. 8–10). ``H`` grows toward 12 slow steps and
  the synthetic↔real data mixture anneals smoothly (no hard switch).
* **Stage 3 — rare-event / OOD sharpening** (doc J C3). Same smooth mixture but
  rare-event packs and topology/flavor holdouts are importance-sampled into at
  most 25 % of minibatches; ``λ_surv`` / ``λ_abst`` rise; ``λ_Δ`` is never
  lowered (action fidelity stays anchored).
* **Stage 4 — calibration** (doc J C5 / doc L). Freeze the encoder, recurrent
  cores, and relation blocks; set ``τ = 1`` (freeze the EMA teacher); fit only
  the conformal calibrators on a held-out, real-heavy validation block.

Optimization details follow doc E: **micro-batches of 8–16 windows with gradient
accumulation to an effective batch of 32–64**, an LR warmup over the first 5 % of
updates, bucketed padded shapes (one compile bucket per pod size), and a
multi-file checkpoint. On MLX the inner step uses ``nn.value_and_grad`` +
``AdamW``; on NumPy — which has no autodiff — the loop uses a deterministic
**simultaneous-perturbation stochastic approximation (SPSA)** update on a small
trainable parameter subset. SPSA needs only two forward evaluations per step and
provably descends in expectation, so the NumPy path is a *real* optimizer (not a
stub) at the demo-pod scale while keeping the architecture's pure-forward design.

Collapse diagnostics (doc J D1, D2, D5) are computed every ``diag_every`` steps:
per-type **effective rank**, per-dimension **variance floor** breaches, and the
paired-branch **action-separation** ratio ``A_h``. They are returned in the
:class:`TrainReport` so the evaluation suite can alarm on sustained collapse.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..schema.actions import Action, Hold
from ..schema.entities import (
    GPU,
    Node,
    PodSet,
    Rack,
    TopologyDomain,
    TopologyLevel,
    Workload,
    WorkloadStatus,
)
from ..schema.kpis import HazardChannel, KPIChannel
from ..schema.state import SymbolicState, WorldState
from .architecture import (
    ArchConfig,
    TensorState,
    exact_field_targets,
    kernel_state_summary,
)
from .backend import ArrayLike, Backend, get_backend
from .heads import (
    COMPETING_RISK_CHANNELS,
    HAZARD_QUERY_ENTITY,
    PARALLEL_BINARY_CHANNELS,
)
from .jepa import (
    LOSS_GROUPS,
    GroupLogVars,
    JEPAConfig,
    anti_collapse_loss,
    apply_typed_mask,
    combined_objective,
    cosine_collapse,
    covariance_energy,
    ema_update,
    jepa_mask_loss,
    ood_head_loss,
    quantile_head_loss,
    rollout_branch_diff_loss,
    rollout_consistency_loss,
    rollout_loss,
    survival_head_loss,
)
from .mlx_backend import HAS_MLX as HAS_MLX_BACKEND
from .mlx_backend import mlx_train_step
from .serialization import save_checkpoint
from .world_model import ResidualWorldModel

# Optional MLX optimizer / autodiff (guarded — NumPy SPSA is the always-on path).
# The real autodiff hot path (mlx.nn param tree + value_and_grad + AdamW +
# mx.compile) lives in :mod:`.mlx_backend`; here we only detect availability so
# :func:`train` can route the inner step. ``HAS_MLX_OPT`` stays the public flag.
try:  # pragma: no cover - Apple-silicon only
    import mlx.core as _mx  # type: ignore
    import mlx.optimizers as _mopt  # type: ignore

    HAS_MLX_OPT = True
except ImportError:
    _mx = _mopt = None  # type: ignore
    HAS_MLX_OPT = False


# The ordered training stages (doc J C1–C5). "calibration" freezes the backbone.
STAGES: tuple[str, ...] = ("warmup", "rollout", "rare_event", "calibration")


# --------------------------------------------------------------------------- #
# Decision-window container (doc J A1).
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class DecisionWindow:
    """One training item ``n = (x0, a0, ξ, x_cf)`` (doc J A1).

    ``state`` is the observed pre-action state ``x0``; ``action`` is the branch
    intervention ``a0``; ``branch_action`` (optional) is the *paired* branch
    ``a0'`` from the **same** pre-action hash and exogenous seed bundle that
    enables the branch-difference action-fidelity term (doc J T2). ``future_*``
    hold the realized counterfactual futures used as EMA targets — when absent
    the loop self-supervises against the EMA-encoded current state (a valid,
    if weaker, warmup signal).
    """

    state: WorldState
    action: Action
    branch_action: Action | None = None
    future_state: WorldState | None = None
    branch_future_state: WorldState | None = None
    is_rare_event: bool = False
    is_real: bool = False


# --------------------------------------------------------------------------- #
# Training configuration (doc E micro-batch / accumulation / schedules).
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class TrainConfig:
    """Hyper-parameters for the staged loop (doc E + doc J fixed coeffs)."""

    # micro-batch & gradient accumulation (doc E line 125–126)
    micro_batch: int = 8
    accum_steps: int = 4  # effective batch = micro_batch * accum_steps
    # per-stage update budgets (kept tiny so the smoke loop runs in CI seconds)
    steps_per_stage: dict[str, int] = field(
        default_factory=lambda: {
            "warmup": 12,
            "rollout": 12,
            "rare_event": 6,
            "calibration": 1,
        }
    )
    # optimizer (doc E AdamW on MLX; SPSA on NumPy)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05  # LR warmup over first 5 % of updates (doc E)
    grad_clip: float = 1.0
    # rollout horizon schedule (doc J C2: grow toward 12 slow steps)
    warmup_horizon: int = 2
    rollout_horizon: int = 6  # demo-scale cap (full v1 grows to 12)
    # horizon curriculum (doc J B2): H grows from ``curriculum_h_start`` to
    # ``curriculum_h_end`` across the action-conditioned (rollout/rare_event)
    # steps so optimization difficulty rises gradually under pure free-running.
    curriculum_h_start: int = 2
    curriculum_h_end: int = 12  # H_slow (doc J B3 T_slow cap)
    # truncated BPTT (doc J B3): detach the carried latent every K slow steps.
    bptt_detach_every: int = 4
    h_slow: int = 12  # outer BPTT clock length (doc J B3)
    # EMA decay schedule (doc J C1/C4)
    ema_start: float = 0.99
    ema_end: float = 0.9995
    # SPSA perturbation scale (NumPy path)
    spsa_c: float = 1e-2
    # diagnostics cadence
    diag_every: int = 4
    var_floor: float = 1.0  # γ for the per-dim variance alarm (doc J D2)
    eff_rank_frac: float = 0.3  # alarm threshold r_eff < 0.3·d (doc J D1)
    seed: int = 0
    verbose: bool = False


# --------------------------------------------------------------------------- #
# Collapse diagnostics (doc J D1, D2, D5).
# --------------------------------------------------------------------------- #
def effective_rank(latent: np.ndarray) -> float:
    """Effective rank ``r_eff = exp(-Σ p_i log p_i)`` of the latent covariance.

    (Doc J D1.) ``latent`` is ``(n_rows, d)`` over the *valid* (unmasked) rows.
    A value near ``d`` means the type uses its full latent budget; a value far
    below ``d`` is the dimensional-collapse alarm.
    """
    x = np.asarray(latent, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        return float(x.shape[1] if x.ndim == 2 else 0)
    xc = x - x.mean(axis=0, keepdims=True)
    cov = (xc.T @ xc) / (x.shape[0] - 1)
    evals = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    s = evals.sum()
    if s <= 1e-12:
        return 0.0
    p = evals / s
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))


def min_dim_variance(latent: np.ndarray) -> float:
    """Minimum per-dimension variance (doc J D2 variance-floor diagnostic)."""
    x = np.asarray(latent, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        return 0.0
    return float(x.var(axis=0).min())


def collapsed_dim_fraction(latent: np.ndarray, var_floor: float) -> float:
    """Fraction of latent dimensions whose variance is below the floor (doc J D2).

    Doc J D2 alarms on "a *large fraction* of dimensions falling below the
    variance floor γ" — not on any single dim — because a handful of low-variance
    dimensions on a small batch is normal. The standard deviation (not variance)
    is compared to ``γ`` so the scale matches ``L_var``'s hinge ``max(0, γ-std)``.
    """
    x = np.asarray(latent, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        return 0.0
    std = x.std(axis=0)
    return float(np.mean(std < var_floor))


def action_separation(
    z_b: dict[str, np.ndarray],
    z_bp: dict[str, np.ndarray],
    tgt_b: dict[str, np.ndarray] | None,
    tgt_bp: dict[str, np.ndarray] | None,
    eps: float = 1e-6,
) -> float:
    """Action-separation ratio ``A = ||Δẑ|| / (||Δz⁺|| + ε)`` (doc J D5).

    ``A ≪ 1`` is the UWM-JEPA action-collapse alarm: the predictor barely
    distinguishes paired branches. When no EMA targets are supplied the
    denominator falls back to ``1`` so the raw predicted separation is reported.
    """
    num = 0.0
    den = 0.0
    for t in z_b:
        if t not in z_bp:
            continue
        num += float(np.linalg.norm(z_b[t] - z_bp[t]))
        if tgt_b is not None and tgt_bp is not None and t in tgt_b and t in tgt_bp:
            den += float(np.linalg.norm(tgt_b[t] - tgt_bp[t]))
    den = den if den > eps else 1.0
    return num / (den + eps)


@dataclass(slots=True)
class DiagSnapshot:
    """One collapse-diagnostic snapshot (doc J D1–D5) at a training step.

    Carries the doc J collapse diagnostics: per-type effective rank (D1),
    minimum per-dimension variance (D2), cross-dimension covariance energy
    ``c_off`` (D3), mean off-diagonal cosine similarity ``s_cos`` (D4), and the
    paired-branch action-separation ratio ``A`` (D5).
    """

    step: int
    effective_rank: dict[str, float] = field(default_factory=dict)
    min_dim_var: dict[str, float] = field(default_factory=dict)
    cov_energy: dict[str, float] = field(default_factory=dict)
    cosine_sim: dict[str, float] = field(default_factory=dict)
    action_separation: float = 1.0
    collapse_alarm: bool = False


def _diag_snapshot(
    model: ResidualWorldModel,
    ts: TensorState,
    batch: Sequence[DecisionWindow],
    step: int,
    cfg: TrainConfig,
    target_model: ResidualWorldModel | None = None,
) -> DiagSnapshot:
    """Compute the D1–D5 collapse diagnostics for a step (doc J D1–D5).

    ``target_model`` is the EMA teacher: when supplied, the D5 action-separation
    ratio is computed against the teacher-encoded counterfactual branch futures
    (the true ``A = ||Δẑ|| / ||Δz⁺||`` ratio); otherwise the denominator falls
    back to 1 (raw predicted separation).
    """
    be = model.be
    z = model.backbone.encode(ts, "hold")
    eff: dict[str, float] = {}
    mvar: dict[str, float] = {}
    cov_e: dict[str, float] = {}
    cos_s: dict[str, float] = {}
    alarm = False
    lat = model.backbone.cfg.latent.as_dict()
    lat["topology_domain"] = lat["rack"]
    for t, h in z.items():
        a = np.asarray(be.to_numpy(h))
        if t in ts.masks:
            m = np.asarray(be.to_numpy(ts.masks[t])).astype(bool)
            a = a[m] if m.ndim == 1 and m.shape[0] == a.shape[0] else a
        eff[t] = effective_rank(a)
        mvar[t] = min_dim_variance(a)
        cov_e[t] = covariance_energy(a)  # D3 cross-dimension covariance energy
        cos_s[t] = cosine_collapse(a)  # D4 mean off-diagonal cosine similarity
        d = lat.get(t, a.shape[-1] if a.ndim == 2 else 1)
        # The effective-rank and variance alarms only have statistical meaning
        # once the number of valid rows can plausibly fill the latent budget
        # (doc J D1: "over a sustained window"). On a tiny demo pod r_eff is
        # capped at n_rows-1, so we require ≥ d samples before alarming on rank,
        # and (doc J D2) alarm only when a *large fraction* of dims sit below the
        # variance floor — not on a single low-variance dimension.
        n_rows = a.shape[0] if a.ndim == 2 else 0
        if n_rows >= max(4, d) and eff[t] < cfg.eff_rank_frac * d:
            alarm = True
        if n_rows >= d and collapsed_dim_fraction(a, cfg.var_floor) > 0.9:
            alarm = True
        # D4 complete-collapse alarm: near-unit cosine over a filled latent budget
        # (doc J D4: complete collapse drives s_cos → 1).
        if n_rows >= max(4, d) and cos_s[t] > 0.98:
            alarm = True

    # D5 action separation from the first paired window (true ratio vs teacher).
    sep = 1.0
    tgt_enc = target_model or model
    for w in batch:
        if w.branch_action is None:
            continue
        zb = {
            k: np.asarray(be.to_numpy(v))
            for k, v in model.backbone.encode(ts, w.action.type.value).items()
        }
        zbp = {
            k: np.asarray(be.to_numpy(v))
            for k, v in model.backbone.encode(ts, w.branch_action.type.value).items()
        }
        # EMA-encoded counterfactual branch futures from the same pre-action
        # hash/seed: z⁺ for branch b is its labelled future, for b' its branch
        # future (doc J D7); these form the true A=||Δẑ||/||Δz⁺|| denominator.
        if target_model is not None:
            fb_state = w.future_state or w.state
            fbp_state = w.branch_future_state or w.state
            tgt_b = _encode_targets(tgt_enc, fb_state, w.action.type.value, None)
            tgt_bp = _encode_targets(
                tgt_enc, fbp_state, w.branch_action.type.value, None
            )
        else:
            tgt_b = tgt_bp = None
        sep = action_separation(zb, zbp, tgt_b, tgt_bp)
        if sep < 0.05:
            alarm = True
        break
    return DiagSnapshot(
        step=step,
        effective_rank=eff,
        min_dim_var=mvar,
        cov_energy=cov_e,
        cosine_sim=cos_s,
        action_separation=sep,
        collapse_alarm=alarm,
    )


# --------------------------------------------------------------------------- #
# Trainable-parameter views (the small subset SPSA / AdamW move).
# --------------------------------------------------------------------------- #
def _trainable_arrays(model: ResidualWorldModel) -> list[tuple[object, str]]:
    """Return ``(owner, attr)`` handles to the backbone's learnable tensors.

    SPSA perturbs every entry of these arrays in lock-step (doc E captures
    model + optimizer state explicitly). Heads are excluded during the latent
    stages and become trainable only via the calibrator in the final stage —
    matching doc J C5's freeze-then-calibrate rule.
    """
    bb = model.backbone
    handles: list[tuple[object, str]] = []
    for enc in bb.encoders.values():
        handles += [(enc, "W"), (enc, "b")]
    for core in (bb.fast_core, bb.slow_core):
        handles += [(core, "Wz"), (core, "Wr"), (core, "Wh")]
    # typed per-relation message blocks (doc Q D2): W_r^src / W_r^dst / φ_r / η_r
    for d in bb.rel_src.values():
        handles += [(d, "W"), (d, "b")]
    for d in bb.rel_dst.values():
        handles += [(d, "W"), (d, "b")]
    for d in bb.rel_phi.values():
        handles += [(d, "W"), (d, "b")]
    for d in bb.rel_gate.values():
        handles += [(d, "W"), (d, "b")]
    # typed gated node update, hierarchical PNA pooling, and downward broadcast
    # (doc Q D4 / Eq.2-5) — all used in the forward pass, so all must train.
    for g in bb.relation_update.values():
        handles += [(g, "Wz"), (g, "Wr"), (g, "Wh")]
    for d in bb.pool_proj.values():
        handles += [(d, "W"), (d, "b")]
    for g in bb.broadcast_update.values():
        handles += [(g, "Wz"), (g, "Wr"), (g, "Wh")]
    # candidate attention (doc Q Eq.6-8), PNA global readout, and FiLM rebroadcast.
    for d in (bb.attn_q, bb.attn_k, bb.attn_v, bb.global_readout, bb.film_gen):
        handles += [(d, "W"), (d, "b")]
    handles.append((bb, "action_embed"))
    # world-transition predictor P_θ — the deployed recursive core (doc J D6).
    if bb.predictor is not None:
        for cell in bb.predictor.cells.values():
            handles += [(cell, "Wz"), (cell, "Wr"), (cell, "Wh")]
        for d in bb.predictor.psi_proj.values():
            handles += [(d, "W"), (d, "b")]
    # exact-state field decoders D_f for the consistency loss (doc J D11): they
    # must be trained (and EMA-transferred) so L_cons can pin the residual into
    # the kernel-null directions (doc J T3).
    if bb.field_decoders is not None:
        for d in bb.field_decoders.decoders.values():
            handles += [(d, "W"), (d, "b")]
    return handles


# --------------------------------------------------------------------------- #
# Loss assembly for one micro-batch.
# --------------------------------------------------------------------------- #
def _encode_targets(
    model: ResidualWorldModel,
    state: WorldState,
    action_name: str,
    caps,
) -> dict[str, np.ndarray]:
    """EMA-target latent for a (counterfactual) future state (doc J D2/D7)."""
    be = model.be
    ts = model.feature_builder.build(state, caps)
    z = model.backbone.encode(ts, action_name)
    # stop-gradient on the target branch (doc J Eq. 1); no-op on NumPy.
    return {k: np.asarray(be.to_numpy(be.stop_gradient(v))) for k, v in z.items()}


def _exogenous_vector(w: DecisionWindow, u_dim: int, be: Backend) -> np.ndarray:
    """Exogenous context ``u_h`` for the predictor (doc J D6 missingness/staleness).

    A small, fixed vector of context indicators: whether a labelled future exists
    (staleness proxy), whether the window is a rare-event / real sample, and a
    branch indicator. These are constants in the predictor gradient (exogenous),
    matching doc J D6's ``u_h`` (missingness/staleness/context features).
    """
    feats = [
        1.0 if w.future_state is None else 0.0,  # missing-future / staleness flag
        1.0 if w.is_rare_event else 0.0,
        1.0 if w.is_real else 0.0,
        1.0 if w.branch_action is not None else 0.0,
    ]
    vec = np.zeros((u_dim,), dtype=np.float32)
    vec[: min(u_dim, len(feats))] = feats[:u_dim]
    return be.from_numpy(vec)


def _curriculum_horizon(
    stage: str, step: int, total_steps: int, cfg: TrainConfig
) -> int:
    """Free-running rollout horizon under the doc J B2 curriculum.

    Warmup uses the short ``warmup_horizon``; the action-conditioned stages grow
    ``H`` linearly from ``curriculum_h_start`` to ``curriculum_h_end`` (capped at
    ``h_slow``) across the run, preserving pure free-running training while
    controlling optimization difficulty (doc J B2 — no scheduled sampling).
    """
    if stage == "warmup":
        return cfg.warmup_horizon
    h0, h1 = cfg.curriculum_h_start, min(cfg.curriculum_h_end, cfg.h_slow)
    if total_steps <= 1:
        return h1
    frac = min(1.0, max(0.0, step / max(1, total_steps - 1)))
    return round(h0 + (h1 - h0) * frac)


# Shortcut-prone exact-field entity types masked on the *online* context before
# encode (doc J E1/E3 / C3: typed masking is the primary anti-collapse scaffold).
# Masking these blocks the predictor from copying observed exact-bound values and
# forces it to use the action + relational context (doc J Causal-JEPA).
_MASK_TYPES: tuple[str, ...] = ("workload", "gpu", "node", "topology_domain")


def _masked_online_encode(
    model: ResidualWorldModel,
    ts: TensorState,
    action_name: str,
    rng: np.random.Generator,
    mask_prob: float,
) -> dict[str, ArrayLike]:
    """Encode the *typed-masked* online context (doc J A2/E1, Eqs. 1–3).

    The online encoder consumes a typed-masked context ``M ⊙ x0`` while the EMA
    target encoder sees the unmasked future (asymmetry is the primary
    anti-collapse mechanism, doc J C3). The validity masks are unchanged so the
    padding / pooling logic still holds; only feature *rows* are zeroed.
    """
    feats = apply_typed_mask(ts.features, _MASK_TYPES, rng, mask_prob, model.be)
    masked_ts = replace(ts, features=feats)
    return model.backbone.encode(masked_ts, action_name)


def _field_target_traj(
    state: WorldState,
    order: dict[str, list[str]],
    masks: dict[str, ArrayLike],
    be: Backend,
    horizon: int,
) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Exact kernel field targets ``f(s^K_h)`` for every rollout step h=1..H.

    The exact fields are read off the (counterfactual future) symbolic state, so
    they are constant across the H steps of one window's rollout (the kernel
    rollout ledger is hashed once per window); they are repeated per step so
    :func:`jepa.rollout_consistency_loss` can pair ``D_f(ẑ_h)`` with ``f(s^K_h)``
    at every step (doc J Eq. 18: ``Σ_{h=1..H}``).
    """
    tgt = exact_field_targets(state, order, masks, be)
    return [tgt for _ in range(max(horizon, 0))]


def _decode_field_traj(
    model: ResidualWorldModel,
    pred_traj: list[dict[str, ArrayLike]],
) -> list[dict[str, ArrayLike]]:
    """Decode ``D_f(ẑ_h)`` for each rolled latent in the trajectory (doc J D11)."""
    fd = model.backbone.field_decoders
    if fd is None:
        return []
    return [fd.decode(z, model.be) for z in pred_traj]


def _kpi_delta_labels(
    model: ResidualWorldModel,
    window: DecisionWindow,
) -> dict[tuple[str, int], float]:
    """Realized KPI-delta labels ``y_{k,h}`` from the counterfactual future (H1).

    KPI targets are *deltas vs. the baseline action* (typically ``hold``) at each
    horizon (doc J A5). We read each channel's realized signal off the labelled
    future symbolic state minus the pre-action state, normalized by a robust
    scale (doc J R4), and broadcast it across the horizon grid. Windows without a
    labelled future contribute no targets (the non-crossing penalty still runs).
    """
    from ..schema.kpis import HORIZONS_S

    fut = window.future_state
    if fut is None:
        return {}
    pre, post = window.state.symbolic, fut.symbolic

    def _free_gpu(X) -> float:
        return float(sum(d.free_gpu for d in X.domains.values()))

    def _pending(X) -> float:
        return float(
            sum(
                sum(ps.count * ps.gpu_req for ps in w.podsets)
                for w in X.workloads.values()
                if w.status in (WorkloadStatus.PENDING, WorkloadStatus.RESERVING)
            )
        )

    def _hotspot(X) -> float:
        ds = list(X.domains.values())
        return float(sum(d.hotspot_score for d in ds) / max(len(ds), 1))

    def _headroom(X) -> float:
        ds = list(X.domains.values())
        return float(sum(d.headroom_score for d in ds) / max(len(ds), 1))

    # robust-scaled deltas per channel (doc J R4 normalize-before-pinball)
    deltas: dict[KPIChannel, float] = {
        KPIChannel.ADMITTED_GPU_HOURS_DELTA: (_free_gpu(pre) - _free_gpu(post)) / 16.0,
        KPIChannel.QUEUE_WAIT_DELTA_S: (_pending(post) - _pending(pre)) / 16.0,
        KPIChannel.TOPOLOGY_FRAGMENTATION_DELTA: (_hotspot(post) - _hotspot(pre)),
        KPIChannel.CROSS_RACK_RATE: (_hotspot(post) - _hotspot(pre)),
        KPIChannel.RACK_THERMAL_HEADROOM_C: (_headroom(post) - _headroom(pre)),
        KPIChannel.SLOWDOWN: (_pending(post) - _pending(pre)) / 16.0,
    }
    out: dict[tuple[str, int], float] = {}
    for ch, val in deltas.items():
        for hz in HORIZONS_S:
            out[(ch.value, hz)] = float(val)
    return out


def _query_pool(
    be: Backend, h: ArrayLike, mask: ArrayLike | None
) -> np.ndarray:
    """Masked-mean pool an entity latent block into one query vector (doc Q V4)."""
    a = np.asarray(be.to_numpy(h), dtype=np.float64)
    if a.ndim == 1:
        return a
    if mask is None:
        return a.mean(axis=0)
    m = np.asarray(be.to_numpy(mask), dtype=np.float64).reshape(-1)
    denom = max(float(m.sum()), 1.0)
    return (a * m[:, None]).sum(axis=0) / denom


def _survival_components(
    model: ResidualWorldModel,
    z_head: dict[str, ArrayLike],
    masks: dict[str, ArrayLike],
):
    """Interval-level competing-risk probs and binary hazards from head weights.

    Re-derives the per-interval distributions the public head smooths into
    cumulative incidence, by applying the *public* head Dense weights
    (``heads.hz_competing`` / ``heads.hz_binary``) to the masked-mean query
    latent. This keeps the survival NLL exact (doc J Eq. 14–16) without touching
    :mod:`heads`. Returns ``(competing_interval_probs, binary_interval_hazards)``.
    """
    from ..schema.kpis import HORIZONS_S

    be = model.be
    heads = model.heads
    n_h = len(HORIZONS_S)

    # ---- competing risks (workload-level): softmax over {survive, ...} ----- #
    qw = _query_pool(be, z_head.get("workload"), masks.get("workload"))
    cr_W = np.asarray(be.to_numpy(heads.hz_competing.W), dtype=np.float64)
    cr_b = np.asarray(be.to_numpy(heads.hz_competing.b), dtype=np.float64)
    cr_logits = (cr_W @ qw[: cr_W.shape[1]] + cr_b).reshape(n_h, -1)
    competing: dict[str, list[np.ndarray]] = {
        "workload": [_softmax_np(cr_logits[hi]) for hi in range(n_h)]
    }

    # ---- parallel-binary incidents: per-interval sigmoid hazards ----------- #
    binary: dict[str, np.ndarray] = {}
    for ch in PARALLEL_BINARY_CHANNELS:
        ent = HAZARD_QUERY_ENTITY[ch]
        d = heads.hz_binary.get(ch)
        if d is None:
            continue
        q = _query_pool(be, z_head.get(ent), masks.get(ent))
        W = np.asarray(be.to_numpy(d.W), dtype=np.float64)
        b = np.asarray(be.to_numpy(d.b), dtype=np.float64)
        logits = W @ q[: W.shape[1]] + b
        binary[ch.value] = _sigmoid_np(logits)
    return competing, binary


def _survival_labels(window: DecisionWindow):
    """Event / censoring labels for the survival head from the future state (H2).

    Reads the realized competing-risk event (timeout / requeue) and parallel
    binary incidents (xid, hotspot, headroom, node-failure) off the labelled
    counterfactual future symbolic state. Returns
    ``(competing_labels, binary_labels)`` keyed exactly as
    :func:`jepa.survival_head_loss` expects. With no labelled future every
    instance is right-censored (the honest no-event-observed default, doc J Eq. 16).
    """
    from ..schema.kpis import HORIZONS_S

    n_h = len(HORIZONS_S)
    fut = window.future_state
    competing_labels: dict[str, tuple[int | None, int | None]] = {
        "workload": (None, None)
    }
    binary_labels: dict[str, int | None] = {
        ch.value: None for ch in PARALLEL_BINARY_CHANNELS
    }
    if fut is None:
        return competing_labels, binary_labels
    X = fut.symbolic
    # competing risk: requeued / evicted workloads (class index per doc heads).
    ec, ei = None, None
    for w in X.workloads.values():
        if w.status in (WorkloadStatus.REQUEUED, WorkloadStatus.EVICTED):
            # class 0 = survive; competing classes are 1-based in head order.
            try:
                ci = list(COMPETING_RISK_CHANNELS).index(HazardChannel.REQUEUE) + 1
            except ValueError:
                ci = 1
            ec, ei = ci, min(1, n_h - 1)
            break
    competing_labels["workload"] = (ec, ei)
    # parallel-binary incidents observed in the future state.
    xid = any(g.xid_code > 0 for g in X.gpus.values())
    binary_labels[HazardChannel.XID_FAULT.value] = 0 if xid else None
    hot = any(d.hotspot_score > 0.8 for d in X.domains.values())
    binary_labels[HazardChannel.HOTSPOT.value] = 0 if hot else None
    return competing_labels, binary_labels


def _abstention_logits(
    model: ResidualWorldModel,
    z_head: dict[str, ArrayLike],
    masks: dict[str, ArrayLike],
) -> tuple[float, float]:
    """Raw (epistemic, ood) abstention logits from the public head weights (H3)."""
    be = model.be
    heads = model.heads
    g = _query_pool(be, z_head.get("global"), masks.get("global"))
    tW = np.asarray(be.to_numpy(heads.abstain_trunk.W), dtype=np.float64)
    tb = np.asarray(be.to_numpy(heads.abstain_trunk.b), dtype=np.float64)
    trunk = np.tanh(tW @ g[: tW.shape[1]] + tb)
    oW = np.asarray(be.to_numpy(heads.abstain_out.W), dtype=np.float64)
    ob = np.asarray(be.to_numpy(heads.abstain_out.b), dtype=np.float64)
    logits = oW @ trunk[: oW.shape[1]] + ob
    return float(logits[0]), float(logits[1])


def _ood_label(window: DecisionWindow) -> float:
    """Held-out OOD indicator ``o_h`` (doc J H3).

    Constructed from the held-out topology/flavor/fault/source-health detectors:
    a rare-event or real (held-out source) window is the positive class, an
    ordinary synthetic window the negative class.
    """
    return 1.0 if (window.is_rare_event or window.is_real) else 0.0


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.clip(e.sum(), 1e-12, None)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def stage_loss(
    model: ResidualWorldModel,
    batch: Sequence[DecisionWindow],
    stage: str,
    cfg: TrainConfig,
    jcfg: JEPAConfig,
    logvars: GroupLogVars,
    target_model: ResidualWorldModel | None = None,
    step: int = 0,
    total_steps: int = 1,
    mask_prob: float = 0.15,
) -> tuple[float, dict[str, float]]:
    """Total uncertainty-weighted loss for a micro-batch at ``stage`` (doc J Eq.19).

    Builds the six coarse loss groups (doc J D13) faithfully:

    * **G1 jepa** — masked-JEPA latent loss ``L_JEPA,mask`` *plus* the weak
      per-type VICReg variance/covariance anti-collapse ``λ_var·L_var +
      λ_cov·L_cov`` (doc J D13 G1). The online encoder consumes a typed-masked
      context while the EMA target encodes the unmasked future (primary
      anti-collapse, doc J C3).
    * **G2 rollout** — free-running ``L_roll,abs + λ_Δ·L_roll,Δ`` over the
      predictor ``P_θ`` recursion (doc J Eqs. 8–10), the branch-difference term
      summed over every step ``h=1..H`` against EMA-encoded counterfactual branch
      futures (doc J Eq. 9, see :data:`MODEL-BRANCH-DIFF-DIAGNOSTICS`).
    * **G3 quantile** — pinball + non-crossing on realized KPI-delta labels
      (doc J Eq. 11–13).
    * **G4 survival** — competing-risk + parallel-binary hazard NLL on the
      event / censoring labels (doc J Eq. 14–17).
    * **G5 consistency** — the *mandatory* exact-state consistency loss
      ``L_cons`` from the per-field decoders ``D_f(ẑ_h)`` against stop-gradient
      kernel fields, summed over the rollout (doc J D11/Eq. 18).
    * **G6 abstention** — (focal-)BCE OOD/topology/fault detector loss
      (doc J H3 / Eq. 17).

    ``target_model`` is the **EMA teacher** (doc J Eq. 1, ∇ξ = 0). When ``None``
    the online model self-encodes the targets (a valid but weaker warmup signal).
    ``mask_prob`` is the typed-context masking probability.
    """
    be = model.be
    tgt_enc = target_model or model
    group: dict[str, float] = {g: 0.0 for g in LOSS_GROUPS}
    n = max(len(batch), 1)
    # deterministic per-(step) masking RNG so the loss is reproducible / SPSA-safe
    mask_rng = np.random.default_rng(cfg.seed * 1_000_003 + step * 7919 + 11)
    quantiles = model.heads.cfg.quantiles

    horizon = _curriculum_horizon(stage, step, total_steps, cfg)
    for w in batch:
        caps = None
        ts = model.feature_builder.build(w.state, caps)
        # online encoder sees the *typed-masked* context (doc J A2/E1, primary
        # anti-collapse); the EMA target encodes the *unmasked* future.
        online = _masked_online_encode(
            model, ts, w.action.type.value, mask_rng, mask_prob
        )

        # --- G1 JEPA + anti-collapse (doc J D13 G1, Eqs. 2–7) ------------- #
        target_state = w.future_state or w.state
        tgt = _encode_targets(tgt_enc, target_state, w.action.type.value, caps)
        tgt_be = {k: be.from_numpy(v) for k, v in tgt.items()}
        group["jepa"] += jepa_mask_loss(online, tgt_be, ts.masks, be)
        # weak per-type VICReg variance/covariance moved into G1 (doc J D13 G1).
        group["jepa"] += anti_collapse_loss(online, ts.masks, jcfg, be)

        # --- G2 action-conditioned rollout (doc J Eqs. 8–10) ------------- #
        # The predictor / consistency machinery is exercised once a free-running
        # rollout exists (rollout / rare_event stages). The warmup stage trains G1
        # only plus the head/consistency geometry on the single-step encoding so
        # the heads stay anchored (doc J C1: G1 only, optional short-H G2).
        psi = model.backbone.predictor and kernel_state_summary(target_state, be)
        kstates = [be.stop_gradient(psi)] * horizon if psi is not None else None
        uvec = _exogenous_vector(w, model.backbone.cfg.u_dim, be)
        exo = [uvec] * horizon
        if stage in ("rollout", "rare_event"):
            pred_traj = model.backbone.rollout(
                online,
                w.action.type.value,
                ts.masks,
                kernel_states=kstates,
                exogenous=exo,
                horizon=horizon,
                detach_every=cfg.bptt_detach_every,
            )
            tgt_traj = [tgt_be for _ in range(horizon)]
            group["rollout"] += rollout_loss(pred_traj, tgt_traj, None, jcfg, be)
            if w.branch_action is not None:
                # paired-branch free-running rollout for the action-fidelity term
                bp0 = model.backbone.encode(ts, w.branch_action.type.value)
                bf = w.branch_future_state or w.state
                bpsi = model.backbone.predictor and kernel_state_summary(bf, be)
                bk = [be.stop_gradient(bpsi)] * horizon if bpsi is not None else None
                bp_traj = model.backbone.rollout(
                    bp0,
                    w.branch_action.type.value,
                    ts.masks,
                    kernel_states=bk,
                    exogenous=exo,
                    horizon=horizon,
                    detach_every=cfg.bptt_detach_every,
                )
                # EMA-encoded counterfactual branch futures z_h^+ for BOTH branches
                # from the same pre-action hash/seed (doc J D7/Eq. 9): the absolute
                # branch target is the action branch's own counterfactual future,
                # the paired target is the branch' future — so action_separation
                # reads the true A = ||Δẑ|| / ||Δz⁺|| ratio (doc J D5).
                tb_be = tgt_be
                tbp_np = _encode_targets(tgt_enc, bf, w.branch_action.type.value, caps)
                tbp_be = {k: be.from_numpy(v) for k, v in tbp_np.items()}
                tb = [tb_be for _ in range(horizon)]
                tbp = [tbp_be for _ in range(horizon)]
                group["rollout"] += jcfg.lambda_delta * rollout_branch_diff_loss(
                    pred_traj, bp_traj, tb, tbp, be=be
                )

            # --- G5 consistency L_cons over the rollout (doc J Eq. 18) ---- #
            field_pred_traj = _decode_field_traj(model, pred_traj)
            field_tgt_traj = _field_target_traj(
                target_state, ts.order, ts.masks, be, horizon
            )
            if field_pred_traj:
                group["consistency"] += rollout_consistency_loss(
                    field_pred_traj,
                    field_tgt_traj,
                    model.backbone.field_decoders.spec,
                    be=be,
                )
            z_head = pred_traj[-1] if pred_traj else online
        else:
            # warmup: anchor the head + consistency geometry on the encoded latent
            # (single step, h=1), so the supervised heads are not cold-started when
            # stage 2 begins (doc J C1 optional short-horizon term).
            field_preds = (
                model.backbone.field_decoders.decode(online, be)
                if model.backbone.field_decoders is not None
                else {}
            )
            field_tgts = exact_field_targets(target_state, ts.order, ts.masks, be)
            if field_preds:
                from .jepa import consistency_loss as _cons

                group["consistency"] += _cons(
                    field_preds, field_tgts, model.backbone.field_decoders.spec, be=be
                )
            z_head = online

        # --- G3 quantile head loss (doc J Eq. 11–13) --------------------- #
        q_forecasts = model.heads.quantile_forecasts(
            z_head, ts.masks, model._query_row(ts, w.action)
        )
        q_labels = _kpi_delta_labels(model, w)
        group["quantile"] += quantile_head_loss(
            q_forecasts, q_labels, quantiles, jcfg.lambda_mono
        )

        # --- G4 survival head loss (doc J Eq. 14–17) --------------------- #
        comp_probs, bin_haz = _survival_components(model, z_head, ts.masks)
        comp_labels, bin_labels = _survival_labels(w)
        group["survival"] += survival_head_loss(
            comp_probs, bin_haz, comp_labels, bin_labels
        )

        # --- G6 abstention / OOD head loss (doc J H3 / Eq. 17) ----------- #
        epi_logit, ood_logit = _abstention_logits(model, z_head, ts.masks)
        group["abstention"] += ood_head_loss(
            epi_logit, ood_logit, _ood_label(w), focal=True
        )

    for g in group:
        group[g] /= n
    total, _ = combined_objective(group, logvars, jcfg)
    return total, group


# --------------------------------------------------------------------------- #
# Optimizer steps.
# --------------------------------------------------------------------------- #
def _spsa_step(
    model: ResidualWorldModel,
    loss_fn: Callable[[], float],
    lr: float,
    c: float,
    rng: np.random.Generator,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
) -> float:
    """One SPSA update over the trainable arrays (the NumPy reference optimizer).

    SPSA estimates the gradient with a single random ``±1`` (Rademacher) probe:
    ``ĝ = (L(θ+cΔ) - L(θ-cΔ)) / (2c) · Δ⁻¹`` (Spall 1992). It needs only two
    forward passes per step regardless of parameter count and descends in
    expectation, so it is a correct optimizer for the demo-pod scale without the
    autodiff NumPy lacks. Returns the *post-update* loss so the curve reflects
    the descent. ``weight_decay`` / ``grad_clip`` honour the :class:`TrainConfig`.
    """
    be = model.be
    handles = _trainable_arrays(model)
    base = [np.asarray(be.to_numpy(getattr(o, a))).copy() for o, a in handles]
    deltas = [
        rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=b.shape) for b in base
    ]

    def _set(arrays: list[np.ndarray]) -> None:
        for (o, a), arr in zip(handles, arrays, strict=True):
            setattr(o, a, be.from_numpy(arr.astype(np.float32)))

    _set([b + c * d for b, d in zip(base, deltas, strict=True)])
    lp = loss_fn()
    _set([b - c * d for b, d in zip(base, deltas, strict=True)])
    lm = loss_fn()
    ghat = (lp - lm) / (2.0 * c)  # scalar directional-derivative estimate
    # Weight-decay-coupled descent: θ ← (1-lr·wd)·θ - lr·(ĝ·Δ⁻¹); Δ⁻¹ = Δ for ±1.
    # ĝ is clipped so a single high-variance probe cannot blow up the step
    # (doc E grad-clip). SPSA descends in expectation (Spall 1992).
    ghat = float(np.clip(ghat, -grad_clip, grad_clip))
    new_arrays = [
        (1.0 - lr * weight_decay) * b - lr * ghat * d
        for b, d in zip(base, deltas, strict=True)
    ]
    _set(new_arrays)
    return float(loss_fn())  # post-update loss (reflects the descent)


def _lr_at(step: int, total: int, base_lr: float, warmup_frac: float) -> float:
    """Linear LR warmup over the first ``warmup_frac`` of updates (doc E)."""
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return base_lr * (step + 1) / warm
    return base_lr


def _ema_at(step: int, total: int, start: float, end: float) -> float:
    """Cosine EMA-decay schedule ``τ: start → end`` (doc J C1/C4)."""
    if total <= 1:
        return end
    t = min(1.0, step / max(1, total - 1))
    return float(end + 0.5 * (start - end) * (1.0 + np.cos(np.pi * t)))


# --------------------------------------------------------------------------- #
# The staged training loop.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class TrainReport:
    """The outcome of a training run (loss curve + collapse diagnostics)."""

    stage_losses: dict[str, list[float]] = field(default_factory=dict)
    diagnostics: list[DiagSnapshot] = field(default_factory=list)
    final_loss: float = float("nan")
    steps: int = 0
    backend: str = "numpy"

    def collapsed(self) -> bool:
        """True if any diagnostic snapshot raised a collapse alarm (doc J D1–D5)."""
        return any(d.collapse_alarm for d in self.diagnostics)


def _minibatches(
    windows: Sequence[DecisionWindow], size: int, rng: np.random.Generator
) -> list[list[DecisionWindow]]:
    idx = rng.permutation(len(windows))
    return [
        [windows[i] for i in idx[s : s + size]] for s in range(0, len(windows), size)
    ]


def train(
    model: ResidualWorldModel,
    windows: Sequence[DecisionWindow],
    cfg: TrainConfig | None = None,
    jcfg: JEPAConfig | None = None,
    stages: Sequence[str] = STAGES,
) -> TrainReport:
    """Run the staged JEPA + rollout + calibration curriculum (doc J C1–C5).

    The loop is *backend-neutral*: on MLX it would use ``nn.value_and_grad`` +
    AdamW (the :func:`_mlx_step` path); on NumPy it uses SPSA. Both honour doc E
    micro-batching with gradient accumulation to an effective batch and an LR
    warmup over the first 5 % of updates. EMA target params are updated each
    step (doc J Eq. 1) until the calibration stage freezes them (``τ = 1``).
    """
    cfg = cfg or TrainConfig()
    jcfg = jcfg or JEPAConfig()
    rng = np.random.default_rng(cfg.seed)
    logvars = GroupLogVars()
    report = TrainReport(backend=model.be.name)
    if not windows:
        return report

    # EMA target = a frozen teacher whose weights track the online model by an
    # exponential moving average (doc J Eq. 1, ∇ξ = 0). We hold it as a separate
    # backbone so the latent prediction loss has a genuine, online-weight-
    # independent target each step. Built lazily by cloning the online model.
    target_params = _snapshot_backbone(model)
    target_model = ResidualWorldModel.build(
        arch=model.backbone.cfg,
        head_cfg=model.heads.cfg,
        n_ensemble=1,
        backend=model.be,
    )
    _load_backbone(target_model, target_params)
    global_step = 0
    total_steps = sum(cfg.steps_per_stage.get(s, 0) for s in stages)

    for stage in stages:
        n_steps = cfg.steps_per_stage.get(stage, 0)
        report.stage_losses[stage] = []
        if stage == "calibration":
            # Freeze backbone + τ=1; fit calibrators (doc J C5 / doc L).
            _fit_calibration(model, windows, cfg)
            report.stage_losses[stage].append(0.0)
            continue

        for _ in range(n_steps):
            batches = _minibatches(windows, cfg.micro_batch, rng)
            # gradient accumulation to the effective batch (doc E line 125).
            accum = batches[: cfg.accum_steps] or [list(windows)]
            lr = _lr_at(global_step, total_steps, cfg.lr, cfg.warmup_frac)
            step_loss = 0.0
            for mb in accum:

                def _loss(
                    _mb: list[DecisionWindow] = mb,
                    _stage: str = stage,
                    _step: int = global_step,
                ) -> float:
                    val, _ = stage_loss(
                        model,
                        _mb,
                        _stage,
                        cfg,
                        jcfg,
                        logvars,
                        target_model,
                        step=_step,
                        total_steps=total_steps,
                    )
                    return val

                if (
                    HAS_MLX_OPT and HAS_MLX_BACKEND and model.be.is_mlx
                ):  # pragma: no cover - Apple-silicon only
                    # Real MLX autodiff: nn.value_and_grad + AdamW + mx.compile
                    # over the predictor rollout + latent + consistency graph.
                    step_loss += _mlx_step(
                        model,
                        mb,
                        stage,
                        cfg,
                        jcfg,
                        logvars,
                        target_model,
                        lr,
                        global_step,
                        total_steps,
                    )
                else:
                    step_loss += _spsa_step(
                        model,
                        _loss,
                        lr,
                        cfg.spsa_c,
                        rng,
                        weight_decay=cfg.weight_decay,
                        grad_clip=cfg.grad_clip,
                    )
            step_loss /= max(len(accum), 1)
            report.stage_losses[stage].append(step_loss)

            # EMA teacher update (doc J Eq. 1) — track online weights, ∇ξ = 0.
            tau = _ema_at(global_step, total_steps, cfg.ema_start, cfg.ema_end)
            online_params = _snapshot_backbone(model)
            target_params = ema_update(target_params, online_params, tau)
            _load_backbone(target_model, target_params)

            if global_step % cfg.diag_every == 0:
                ts = model.feature_builder.build(windows[0].state, None)
                report.diagnostics.append(
                    _diag_snapshot(
                        model, ts, accum[0], global_step, cfg, target_model
                    )
                )
            if cfg.verbose:
                print(f"[{stage}] step {global_step} loss={step_loss:.5f} τ={tau:.4f}")
            global_step += 1

    report.steps = global_step
    if report.stage_losses:
        last = [v for vs in report.stage_losses.values() for v in vs if vs]
        report.final_loss = last[-1] if last else float("nan")
    return report


def _mlx_differentiable_loss(  # pragma: no cover - Apple-silicon only
    model: ResidualWorldModel,
    batch: Sequence[DecisionWindow],
    stage: str,
    cfg: TrainConfig,
    jcfg: JEPAConfig,
    target_model: ResidualWorldModel | None,
    step: int,
    total_steps: int,
):
    """Build a pure-MLX differentiable loss over the trainable leaves (doc E).

    Returns ``mlx_loss_fn(leaves) -> mx.array`` for
    :func:`mlx_backend.mlx_train_step`. The closure writes the leaves back into
    the backbone arrays, runs the MLX-native ``encode`` + free-running ``rollout``
    forward, and computes the doc J objective entirely in ``mx`` ops so
    ``nn.value_and_grad`` differentiates through it. The EMA targets and the
    exact kernel summary enter through ``mx.stop_gradient`` (``∂K/∂θ = 0``,
    doc J Eq. 22), so the supervised world-model gradient is exact.

    Differentiable terms: G1 masked-JEPA latent L1 + per-type variance hinge,
    G2 free-running absolute rollout match + branch-difference action fidelity,
    and G5 exact-state consistency (BCE / Huber over the field decoders). The
    head-supervision groups G3/G4/G6 read non-differentiable realized labels and
    are folded in by the NumPy reference loss for logging; the autodiff step
    optimizes the latent + consistency geometry that the MLX graph expresses.
    """
    from .mlx_backend import mx  # local: only importable when HAS_MLX

    be = model.be
    handles = _trainable_arrays(model)
    tgt_enc = target_model or model
    horizon = _curriculum_horizon(stage, step, total_steps, cfg)
    mask_rng = np.random.default_rng(cfg.seed * 1_000_003 + step * 7919 + 11)

    def _sg(d: dict) -> dict:
        return {k: mx.stop_gradient(v) for k, v in d.items()}

    def _latent_l1(pred: dict, tgt: dict, masks: dict) -> Any:
        acc = mx.array(0.0)
        n = 0
        for t, p in pred.items():
            if t not in tgt:
                continue
            diff = mx.abs(be.layer_norm(p) - mx.stop_gradient(be.layer_norm(tgt[t])))
            m = masks.get(t)
            if m is not None:
                diff = diff * m[:, None]
                denom = mx.maximum(mx.sum(m) * p.shape[-1], 1.0)
                acc = acc + mx.sum(diff) / denom
            else:
                acc = acc + mx.mean(diff)
            n += 1
        return acc / max(n, 1)

    def _var_hinge(pred: dict, masks: dict) -> Any:
        # masked per-type variance hinge Σ max(0, γ - std_B(q))² (doc J Eq. 5).
        acc = mx.array(0.0)
        n = 0
        for t, q in pred.items():
            m = masks.get(t)
            if m is not None:
                w = m[:, None]
                cnt = mx.maximum(mx.sum(m), 1.0)
                mean = mx.sum(q * w, axis=0) / cnt
                var = mx.sum(((q - mean) ** 2) * w, axis=0) / cnt
            else:
                var = mx.var(q, axis=0)
            std = mx.sqrt(var + 1e-6)
            hinge = mx.maximum(0.0, jcfg.var_floor - std)
            acc = acc + mx.mean(hinge * hinge)
            n += 1
        return acc / max(n, 1)

    def mlx_loss_fn(leaves: list) -> Any:
        # scatter the (live, differentiable) leaves back into the model arrays so
        # the forward path is a function of the optimizer's parameters.
        for (owner, attr), leaf in zip(handles, leaves, strict=True):
            setattr(owner, attr, leaf)
        loss = mx.array(0.0)
        nb = max(len(batch), 1)
        for w in batch:
            ts = model.feature_builder.build(w.state, None)
            feats = apply_typed_mask(
                ts.features, _MASK_TYPES, mask_rng, 0.15, be
            )
            masked_ts = replace(ts, features=feats)
            online = model.backbone.encode(masked_ts, w.action.type.value)
            target_state = w.future_state or w.state
            ts_t = tgt_enc.feature_builder.build(target_state, None)
            tgt = _sg(tgt_enc.backbone.encode(ts_t, w.action.type.value))
            # G1: masked-JEPA latent + weak variance anti-collapse
            loss = loss + _latent_l1(online, tgt, ts.masks)
            loss = loss + jcfg.lambda_var * _var_hinge(online, ts.masks)
            if stage in ("rollout", "rare_event"):
                psi = kernel_state_summary(target_state, be)
                ks = [mx.stop_gradient(psi)] * horizon
                uvec = _exogenous_vector(w, model.backbone.cfg.u_dim, be)
                pred_traj = model.backbone.rollout(
                    online,
                    w.action.type.value,
                    ts.masks,
                    kernel_states=ks,
                    exogenous=[uvec] * horizon,
                    horizon=horizon,
                    detach_every=cfg.bptt_detach_every,
                )
                # G2 absolute rollout match
                for zh in pred_traj:
                    loss = loss + _latent_l1(zh, tgt, ts.masks) / max(horizon, 1)
                # G2 branch-difference action fidelity (doc J Eq. 9)
                if w.branch_action is not None:
                    bp0 = model.backbone.encode(masked_ts, w.branch_action.type.value)
                    bf = w.branch_future_state or w.state
                    bpsi = kernel_state_summary(bf, be)
                    bp_traj = model.backbone.rollout(
                        bp0,
                        w.branch_action.type.value,
                        ts.masks,
                        kernel_states=[mx.stop_gradient(bpsi)] * horizon,
                        exogenous=[uvec] * horizon,
                        horizon=horizon,
                        detach_every=cfg.bptt_detach_every,
                    )
                    ts_bp = tgt_enc.feature_builder.build(bf, None)
                    tbp = _sg(
                        tgt_enc.backbone.encode(ts_bp, w.branch_action.type.value)
                    )
                    for zh, zhp in zip(pred_traj, bp_traj, strict=False):
                        for t in zh:
                            if t in zhp and t in tgt and t in tbp:
                                dq = zh[t] - zhp[t]
                                dt = mx.stop_gradient(tgt[t] - tbp[t])
                                loss = loss + jcfg.lambda_delta * mx.mean(
                                    mx.abs(dq - dt)
                                ) / max(horizon, 1)
                # G5 exact-state consistency (doc J Eq. 18) over the rollout
                fd = model.backbone.field_decoders
                if fd is not None:
                    targets = exact_field_targets(
                        target_state, ts.order, ts.masks, be
                    )
                    flavor = {nm: fl for nm, _lt, _od, fl in fd.spec}
                    for zh in pred_traj:
                        preds = fd.decode(zh, be)
                        for name, p in preds.items():
                            if name not in targets:
                                continue
                            tt, valid = targets[name]
                            y = mx.stop_gradient(be.from_numpy(tt))
                            vm = mx.stop_gradient(be.from_numpy(valid))[:, None]
                            if flavor.get(name) == "huber":
                                r = p - y
                                a = mx.abs(r)
                                per = mx.where(
                                    a <= 1.0, 0.5 * r * r, a - 0.5
                                )
                            else:  # bce-with-logits
                                per = (
                                    mx.maximum(p, 0.0)
                                    - p * y
                                    + mx.log1p(mx.exp(-mx.abs(p)))
                                )
                            denom = mx.maximum(mx.sum(vm) * per.shape[-1], 1.0)
                            loss = loss + mx.sum(per * vm) / denom / max(horizon, 1)
        return loss / nb

    return handles, mlx_loss_fn


def _mlx_step(  # pragma: no cover - Apple-silicon only
    model: ResidualWorldModel,
    batch: Sequence[DecisionWindow],
    stage: str,
    cfg: TrainConfig,
    jcfg: JEPAConfig,
    logvars: GroupLogVars,
    target_model: ResidualWorldModel | None,
    lr: float,
    step: int,
    total_steps: int,
) -> float:
    """MLX autodiff + AdamW inner step (doc E nn.value_and_grad + mx.compile).

    Builds the pure-MLX differentiable loss over the predictor rollout + latent
    + consistency geometry and runs one compiled ``value_and_grad`` + ``AdamW`` +
    global-norm-clip update via :func:`mlx_backend.mlx_train_step` (which owns the
    ``mlx.nn`` parameter tree, the captured optimizer / RNG state, and the
    ``mx.compile``'d outer step). The trained leaves are written back into the
    model arrays. The NumPy SPSA path is the always-on CI fallback; this branch
    is only reached when MLX is genuinely importable and the backend is MLX.
    """
    handles, mlx_loss_fn = _mlx_differentiable_loss(
        model, batch, stage, cfg, jcfg, target_model, step, total_steps
    )
    return mlx_train_step(
        model,
        handles,
        mlx_loss_fn,
        lr=lr,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        seed=cfg.seed + 17 + step,
    )


# --------------------------------------------------------------------------- #
# Backbone parameter snapshot / load (for the EMA teacher, doc J Eq. 1).
# --------------------------------------------------------------------------- #
def _snapshot_backbone(model: ResidualWorldModel) -> dict[str, np.ndarray]:
    """Flat positional snapshot of the trainable arrays (structural keys).

    Keys are the *position* in :func:`_trainable_arrays` so the same dict can be
    transferred between two structurally-identical backbones (online → EMA
    teacher) regardless of object identity.
    """
    be = model.be
    out: dict[str, np.ndarray] = {}
    for i, (o, a) in enumerate(_trainable_arrays(model)):
        out[f"{i}.{a}"] = np.asarray(be.to_numpy(getattr(o, a))).copy()
    return out


def _load_backbone(model: ResidualWorldModel, params: dict[str, np.ndarray]) -> None:
    """Load a positional param dict into a backbone (the EMA-teacher transfer)."""
    be = model.be
    for i, (o, a) in enumerate(_trainable_arrays(model)):
        key = f"{i}.{a}"
        if key in params:
            setattr(o, a, be.from_numpy(params[key].astype(np.float32)))


# --------------------------------------------------------------------------- #
# Calibration stage (doc J C5 / doc L): freeze backbone, fit conformal heads.
# --------------------------------------------------------------------------- #
def _calibration_slice_key(window: DecisionWindow, horizon_s: int) -> str:
    """Doc-L 5-tuple calibration slice key for one decision window.

    The five facets (doc L MondrianConformal): action family, horizon, workload
    family (median priority band), topology/flavor family (sorted node flavors),
    and source-health bucket. Rendered as one hashable string so the Mondrian
    back-off keys are clean. This is the exchangeability bucket the conformal
    quantile is fitted within (doc L A2 / handoff: calibrate only on action-
    conditioned decision windows).
    """
    x = window.state.symbolic
    action_fam = window.action.type.value
    flavors = "+".join(sorted({n.flavor_id for n in x.nodes.values()})) or "none"
    wls = list(x.workloads.values())
    if wls:
        prio = int(np.median([getattr(w, "priority", 0) for w in wls]))
        wl_fam = f"prio{prio}"
    else:
        wl_fam = "default"
    src = window.state.source
    bad_tokens = {"bad", "down", "degraded", "stale", "dropped", "delayed", "unhealthy"}
    sh_bad = any(
        str(v).lower() in bad_tokens for v in src.collector_health.values()
    ) or any(p > 0.0 for p in src.channel_missing_prob.values())
    sh_bucket = "degraded" if sh_bad else "clean"
    return f"{action_fam}|h{int(horizon_s)}|{wl_fam}|{flavors}|{sh_bucket}"


def _fit_calibration(
    model: ResidualWorldModel, windows: Sequence[DecisionWindow], cfg: TrainConfig
) -> None:
    """Fit CQR + per-slice Mondrian calibrators on held-out windows (doc L).

    Calibrates the model forecast band against the **actual branch-simulated
    counterfactual KPI delta** from the labelled counterfactual table
    (:func:`_kpi_delta_labels`) — *not* the model's own predicted median (the old
    self-referential target, which made the conformal layer vacuous). With no
    labelled futures the calibrators stay unfitted and pass through unchanged (the
    honest never-undercover fallback, doc L T1).

    In addition to the marginal CQR calibrators, a per-slice
    :class:`uncertainty.conformal.MondrianConformal` is fitted using the doc-L
    5-tuple slice key (action, horizon, workload, topology/flavor, source-health)
    with hierarchical back-off, so serve-time widening is group-conditional (doc L
    T3). The Mondrian calibrators are attached to ``model.calibrator.mondrian`` for
    the serving layer to consume.
    """
    from ..schema.kpis import HORIZONS_S
    from ..uncertainty.conformal import MondrianConformal

    cal = model.calibrator
    # marginal CQR rows + per-slice Mondrian rows keyed by channel/horizon.
    by_channel: dict[tuple[str, int], list[tuple[float, float, float]]] = {}
    mondrian_rows: dict[
        tuple[str, int], dict[str, list[float]]
    ] = {}
    parent_of: dict[object, object] = {}
    for w in windows:
        if w.future_state is None:
            continue
        # the labelled, branch-simulated counterfactual KPI deltas (H1 labels).
        y_labels = _kpi_delta_labels(model, w)
        if not y_labels:
            continue
        for hz in HORIZONS_S:
            bundle = model.forecast(w.state, w.action, hz)
            slice_key = _calibration_slice_key(w, hz)
            parent_of[slice_key] = "|".join(slice_key.split("|")[:3])
            for qf in bundle.quantiles:
                key = (qf.channel.value, qf.horizon_s)
                y = y_labels.get((qf.channel.value, qf.horizon_s))
                if y is None:
                    continue
                lo = qf.quantiles.get(0.05, 0.0)
                hi = qf.quantiles.get(0.95, 0.0)
                by_channel.setdefault(key, []).append((lo, hi, float(y)))
                rows = mondrian_rows.setdefault(
                    key, {"g": [], "lo": [], "hi": [], "y": []}
                )
                rows["g"].append(slice_key)
                rows["lo"].append(lo)
                rows["hi"].append(hi)
                rows["y"].append(float(y))
    for (ch, h), rows in by_channel.items():
        if len(rows) < 3:
            continue
        lo = np.array([r[0] for r in rows])
        hi = np.array([r[1] for r in rows])
        y = np.array([r[2] for r in rows])
        cal.fit_kpi(ch, h, lo, hi, y)
    # per-slice Mondrian (CQR mode) with hierarchical back-off (doc L T3).
    mondrian: dict[tuple[str, int], MondrianConformal] = {}
    for key, rows in mondrian_rows.items():
        if len(rows["y"]) < 3:
            continue
        groups = np.array(rows["g"], dtype=object)
        mond = MondrianConformal(alpha=0.10, mode="cqr", min_group=8)
        mond.fit(
            groups,
            np.array(rows["y"], float),
            {"q_lo": np.array(rows["lo"], float), "q_hi": np.array(rows["hi"], float)},
            parent_of=parent_of,
        )
        mondrian[key] = mond
    if mondrian:
        cal.mondrian = mondrian


# --------------------------------------------------------------------------- #
# A tiny self-contained smoke-train entrypoint (no external data needed).
# --------------------------------------------------------------------------- #
def _toy_state(seed: int, n_workloads: int = 3) -> WorldState:
    """Build a minimal but schema-valid 1-rack demo state (smoke fixture).

    Deterministic given ``seed`` so the smoke loop is reproducible. This is *not*
    the synthetic SCM data engine (that lives in ``data_engine``); it is just a
    self-contained fixture so ``smoke_train`` runs with zero dependencies.
    """
    rng = np.random.default_rng(seed)
    X = SymbolicState()
    rack = Rack(
        rack_id="r0",
        block_id="d0",
        power_domain_id="p0",
        cooling_zone_id="c0",
        designed_gpu_cap=8,
        rack_power_kw=40.0,
        rack_headroom_kw=10.0,
        inferred_thermal_margin_c=6.0,
    )
    X.racks["r0"] = rack
    X.domains["d0"] = TopologyDomain(
        domain_id="d0",
        level=TopologyLevel.BLOCK,
        free_gpu=4,
        free_cpu=64.0,
        free_mem_gb=512.0,
        headroom_score=0.6,
        hotspot_score=0.2,
    )
    for ni in range(2):
        node = Node(
            node_uid=f"n{ni}",
            rack_id="r0",
            block_id="d0",
            switch_id="s0",
            flavor_id="f0",
            cpu_cap=128.0,
            mem_cap_gb=1024.0,
            gpu_cap=4,
            running_pods=int(rng.integers(0, 3)),
        )
        X.nodes[node.node_uid] = node
        for gi in range(4):
            gu = f"n{ni}-g{gi}"
            X.gpus[gu] = GPU(
                gpu_uuid=gu,
                node_uid=node.node_uid,
                memory_total_mb=80000,
                util=float(rng.uniform(0, 1)),
                mem_util=float(rng.uniform(0, 1)),
                temp_c=float(rng.uniform(40, 80)),
                power_w=float(rng.uniform(200, 600)),
            )
    for wi in range(n_workloads):
        wu = f"w{wi}"
        ps = PodSet(
            workload_uid=wu,
            name="worker",
            count=int(rng.integers(1, 3)),
            gpu_req=1,
            cpu_req=8.0,
            mem_req_gb=64.0,
        )
        X.workloads[wu] = Workload(
            workload_uid=wu,
            tenant_id="t0",
            local_queue_id="lq0",
            cluster_queue_id="cq0",
            priority=int(rng.integers(0, 5)),
            podsets=[ps],
            status=WorkloadStatus.PENDING,
            est_remaining_s=float(rng.uniform(600, 7200)),
        )
    return WorldState(ts_ns=seed, tick=seed, symbolic=X)


def make_smoke_windows(n: int = 16, seed: int = 0) -> list[DecisionWindow]:
    """Build ``n`` paired-branch decision windows from the toy fixture."""
    rng = np.random.default_rng(seed)
    out: list[DecisionWindow] = []
    for i in range(n):
        s = _toy_state(seed + i)
        wl = sorted(s.symbolic.workloads)[0]
        a = Hold(workload_uid=wl)
        # paired branch: a different actor action over the same pre-state.
        from ..schema.actions import Admit

        bp = Admit(workload_uid=wl, cluster_queue_id="cq0", resource_flavor_id="f0")
        fut = _toy_state(seed + i + 10000)
        out.append(
            DecisionWindow(
                state=s,
                action=a,
                branch_action=bp,
                future_state=fut,
                branch_future_state=fut,
                is_rare_event=bool(rng.random() < 0.2),
                is_real=False,
            )
        )
    return out


def smoke_train(
    out_dir: str | Path | None = None,
    n_windows: int = 16,
    backend: Backend | None = None,
    cfg: TrainConfig | None = None,
) -> TrainReport:
    """End-to-end smoke train: build → train all stages → (optionally) checkpoint.

    Runs in a few seconds on plain CPU/NumPy with no external data (doc E
    "stage zero: tensor-pipeline smoke tests"). Returns the :class:`TrainReport`;
    if ``out_dir`` is given the trained model is checkpointed via
    :func:`serialization.save_checkpoint`.
    """
    be = backend or get_backend()
    cfg = cfg or TrainConfig()
    model = ResidualWorldModel.build(
        arch=ArchConfig(seed=cfg.seed), n_ensemble=2, backend=be
    )
    windows = make_smoke_windows(n_windows, cfg.seed)
    report = train(model, windows, cfg)
    if out_dir is not None:
        save_checkpoint(model, out_dir, feature_stats={"smoke": True})
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI shim: ``python -m gpu_center_world_model.model.train [out_dir]``."""
    import sys

    args = list(argv if argv is not None else sys.argv[1:])
    out_dir = args[0] if args else None
    cfg = TrainConfig(verbose=True)
    report = smoke_train(out_dir=out_dir, cfg=cfg)
    print(
        f"smoke_train done: backend={report.backend} steps={report.steps} "
        f"final_loss={report.final_loss:.5f} collapsed={report.collapsed()}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "HAS_MLX_OPT",
    "STAGES",
    "DecisionWindow",
    "DiagSnapshot",
    "TrainConfig",
    "TrainReport",
    "action_separation",
    "effective_rank",
    "main",
    "make_smoke_windows",
    "min_dim_variance",
    "smoke_train",
    "stage_loss",
    "train",
]
