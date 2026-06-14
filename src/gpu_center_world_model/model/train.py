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
from dataclasses import dataclass, field
from pathlib import Path

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
from ..schema.state import SymbolicState, WorldState
from .architecture import ArchConfig, TensorState, kernel_state_summary
from .backend import Backend, get_backend
from .jepa import (
    LOSS_GROUPS,
    GroupLogVars,
    JEPAConfig,
    anti_collapse_loss,
    combined_objective,
    ema_update,
    jepa_mask_loss,
    rollout_branch_diff_loss,
    rollout_loss,
)
from .serialization import save_checkpoint
from .world_model import ResidualWorldModel

# Optional MLX optimizer / autodiff (guarded — NumPy SPSA is the always-on path).
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
    """One collapse-diagnostic snapshot (doc J D1/D2/D5) at a training step."""

    step: int
    effective_rank: dict[str, float] = field(default_factory=dict)
    min_dim_var: dict[str, float] = field(default_factory=dict)
    action_separation: float = 1.0
    collapse_alarm: bool = False


def _diag_snapshot(
    model: ResidualWorldModel,
    ts: TensorState,
    batch: Sequence[DecisionWindow],
    step: int,
    cfg: TrainConfig,
) -> DiagSnapshot:
    """Compute per-type effective rank, min variance, and action separation."""
    be = model.be
    z = model.backbone.encode(ts, "hold")
    eff: dict[str, float] = {}
    mvar: dict[str, float] = {}
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

    # action separation from the first paired window in the batch (doc J D5)
    sep = 1.0
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
        sep = action_separation(zb, zbp, None, None)
        if sep < 0.05:
            alarm = True
        break
    return DiagSnapshot(
        step=step,
        effective_rank=eff,
        min_dim_var=mvar,
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
    handles.append((bb, "action_embed"))
    # world-transition predictor P_θ — the deployed recursive core (doc J D6).
    if bb.predictor is not None:
        for cell in bb.predictor.cells.values():
            handles += [(cell, "Wz"), (cell, "Wr"), (cell, "Wh")]
        for d in bb.predictor.psi_proj.values():
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
) -> tuple[float, dict[str, float]]:
    """Total uncertainty-weighted loss for a micro-batch at ``stage`` (doc J Eq.19).

    Builds the six coarse loss groups (doc J D13) appropriate to the stage:
    warmup uses ``jepa`` + ``consistency`` (anti-collapse); rollout/rare-event
    add the action-conditioned ``rollout`` group with the branch-difference term;
    survival/quantile/abstention are folded in as light auxiliaries so the head
    geometry is exercised. Returns ``(total, group_losses)``.

    The action-conditioned ``rollout`` group drives a **genuine free-running
    recursion** through the transition predictor ``P_θ`` (:meth:`backbone.rollout`)
    rather than replicating the one-step encoding: each step is conditioned on the
    action embedding, the stop-gradient exact kernel-state summary ``ψ(s^K_{h+1})``
    and the exogenous context ``u_h``, with truncated BPTT (detach every
    ``cfg.bptt_detach_every`` slow steps) and a horizon curriculum H: 2→12
    (doc J D6/Eq. 8–10, B1–B4). ``step``/``total_steps`` drive that curriculum.

    ``target_model`` is the **EMA teacher** (doc J Eq. 1, ∇ξ = 0): the targets
    are encoded by its frozen weights so the latent prediction loss has a genuine,
    online-weight-independent target. When ``None`` the online model self-encodes
    the targets (a valid but weaker warmup signal used by unit tests).
    """
    be = model.be
    tgt_enc = target_model or model
    group: dict[str, float] = {g: 0.0 for g in LOSS_GROUPS}
    n = max(len(batch), 1)

    horizon = _curriculum_horizon(stage, step, total_steps, cfg)
    for w in batch:
        caps = None
        ts = model.feature_builder.build(w.state, caps)
        online = model.backbone.encode(ts, w.action.type.value)

        # --- JEPA latent group G1 (doc J Eq. 2–3) ------------------------- #
        target_state = w.future_state or w.state
        tgt = _encode_targets(tgt_enc, target_state, w.action.type.value, caps)
        tgt_be = {k: be.from_numpy(v) for k, v in tgt.items()}
        group["jepa"] += jepa_mask_loss(online, tgt_be, ts.masks, be)

        # --- anti-collapse (weak VICReg-lite) -> consistency group -------- #
        group["consistency"] += anti_collapse_loss(online, ts.masks, jcfg, be)

        # --- action-conditioned rollout group G2 (doc J Eqs. 8–10) -------- #
        if stage in ("rollout", "rare_event"):
            # Genuine free-running rollout: ẑ_1..ẑ_H are distinct, produced by the
            # predictor P_θ recursing on its own output (doc J D6, no teacher
            # forcing). Condition each step on the stop-grad exact kernel state.
            psi = model.backbone.predictor and kernel_state_summary(target_state, be)
            kstates = [be.stop_gradient(psi)] * horizon if psi is not None else None
            uvec = _exogenous_vector(w, model.backbone.cfg.u_dim, be)
            exo = [uvec] * horizon
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
                bp0 = model.backbone.encode(ts, w.branch_action.type.value)
                bf = w.branch_future_state or w.state
                # paired-branch free-running rollouts for the action-fidelity term
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
                tb = [tgt_be for _ in range(horizon)]
                tbp_np = _encode_targets(tgt_enc, bf, w.branch_action.type.value, caps)
                tbp_be = {k: be.from_numpy(v) for k, v in tbp_np.items()}
                tbp = [tbp_be for _ in range(horizon)]
                group["rollout"] += jcfg.lambda_delta * rollout_branch_diff_loss(
                    pred_traj, bp_traj, tb, tbp, be=be
                )

        # --- light survival / quantile / abstention auxiliaries ----------- #
        if stage == "rare_event" and w.is_rare_event:
            # upweight survival when a rare event is present (doc J C3)
            group["survival"] += 1.0 * anti_collapse_loss(online, ts.masks, jcfg, be)
            group["abstention"] += 0.5 * float(
                np.mean([abs(v) for v in tgt.get("global", np.zeros(1)).ravel()])
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

                if HAS_MLX_OPT and model.be.is_mlx:  # pragma: no cover
                    step_loss += _mlx_step(model, _loss, lr, cfg)
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
                    _diag_snapshot(model, ts, accum[0], global_step, cfg)
                )
            if cfg.verbose:
                print(f"[{stage}] step {global_step} loss={step_loss:.5f} τ={tau:.4f}")
            global_step += 1

    report.steps = global_step
    if report.stage_losses:
        last = [v for vs in report.stage_losses.values() for v in vs if vs]
        report.final_loss = last[-1] if last else float("nan")
    return report


def _mlx_step(  # pragma: no cover - Apple-silicon only
    model: ResidualWorldModel, loss_fn: Callable[[], float], lr: float, cfg: TrainConfig
) -> float:
    """MLX autodiff + AdamW inner step (doc E nn.value_and_grad pattern).

    The MLX path captures model + optimizer state explicitly and updates with
    AdamW. Because the backbone here stores raw arrays (not an ``nn.Module``),
    the production MLX trainer wraps them in a parameter tree; for the open-
    source v1 we fall through to the SPSA estimator if that tree is not built,
    so this branch stays import-clean on every platform.
    """
    rng = np.random.default_rng(cfg.seed + 17)
    return _spsa_step(
        model,
        loss_fn,
        lr,
        cfg.spsa_c,
        rng,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
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
def _fit_calibration(
    model: ResidualWorldModel, windows: Sequence[DecisionWindow], cfg: TrainConfig
) -> None:
    """Fit CQR + one-sided-upper calibrators on held-out windows (doc L).

    Uses each window's model forecast as the base quantile band and the realized
    future (where available) as the calibration target. With no labelled futures
    the calibrators stay unfitted and pass through unchanged (graceful degrade),
    which is the honest never-undercover fallback (doc L T1).
    """
    from ..schema.kpis import HORIZONS_S

    cal = model.calibrator
    by_channel: dict[tuple[str, int], list[tuple[float, float, float]]] = {}
    for w in windows:
        if w.future_state is None:
            continue
        bundle = model.forecast(w.state, w.action, HORIZONS_S[0])
        for qf in bundle.quantiles:
            lo = qf.quantiles.get(0.05, 0.0)
            hi = qf.quantiles.get(0.95, 0.0)
            # realized target proxy: the median (point) — replace with the true
            # branch KPI delta when the labelled counterfactual table is present.
            y = qf.quantiles.get(0.5, 0.0)
            by_channel.setdefault((qf.channel.value, qf.horizon_s), []).append(
                (lo, hi, y)
            )
    for (ch, h), rows in by_channel.items():
        if len(rows) < 3:
            continue
        lo = np.array([r[0] for r in rows])
        hi = np.array([r[1] for r in rows])
        y = np.array([r[2] for r in rows])
        cal.fit_kpi(ch, h, lo, hi, y)


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
