# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The latent JEPA objective and counterfactual rollout loss (doc J).

This module implements the learning objective from doc J as pure functions over
the backend shim (NumPy by default, so the tests run without MLX):

* **EMA target encoder + stop-gradient + typed masking** (doc J E1–E2, Eqs. 1–3).
  The online encoder sees a *typed-masked* context; the EMA target encoder sees
  the full future state; the latent prediction loss ``d(·,·)`` is a per-type
  layer-normed L1 distance with stop-gradient on the target (doc J D4).
* **Counterfactual rollout loss** (doc J Eqs. 8–10). Absolute latent match
  ``L_roll,abs`` plus the **branch-difference action-fidelity term**
  ``L_roll,Δ`` — the minimal action-fidelity supervision the paired
  counterfactual packs uniquely enable (doc J T2). This is what stops the model
  from "predicting well while ignoring the action" (doc J E1 / UWM-JEPA).
* **Weak per-type variance/covariance anti-collapse** (doc J Eqs. 5–7). VICReg-
  lite, applied *only* to the online projector outputs as an auxiliary term, not
  a second invariance term (doc J C3).
* **Combined objective with group-wise homoscedastic uncertainty weighting**
  (doc J Eq. 19 / Eqs. 26–27): six learned log-variances ``s_1..s_6`` over the
  six coarse loss groups, bounded to ``[-3, 3]``.
* **Stop-gradient kernel conditioning** (doc J Eqs. 20–22, T4): the exact kernel
  outputs are constants in the gradient — there is no ``∂K/∂θ`` term — so the
  supervised world-model gradient is exact, not approximate.

The functions take backend arrays and a :class:`Backend`; the NumPy path is the
reference used by tests, the MLX path is identical code under ``mx``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backend import ArrayLike, Backend, get_backend

# The six coarse loss groups (doc J D13): G1 jepa+vc, G2 rollout, G3 quantile,
# G4 survival, G5 consistency, G6 abstention.
LOSS_GROUPS: tuple[str, ...] = (
    "jepa",
    "rollout",
    "quantile",
    "survival",
    "consistency",
    "abstention",
)


@dataclass(slots=True)
class JEPAConfig:
    """JEPA / rollout / anti-collapse hyper-parameters (doc J fixed coeffs)."""

    ema_decay: float = 0.99  # τ start (rises to ~0.9995 in train, doc J C1)
    lambda_var: float = 1.0  # L_var weight (doc J Eq. 7)
    lambda_cov: float = 0.04  # L_cov weight
    lambda_delta: float = 1.0  # branch-difference action-fidelity (doc J Eq. 10)
    lambda_mono: float = 1.0  # quantile non-crossing penalty (doc J Eq. 13)
    var_floor: float = 1.0  # γ variance floor per type (doc J Eq. 5)
    s_bounds: tuple[float, float] = (-3.0, 3.0)  # log-variance clamp (Eq. 27)


# --------------------------------------------------------------------------- #
# Core distances.
# --------------------------------------------------------------------------- #
def _layer_norm_np(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sd + eps)


def latent_distance(
    pred: dict[str, ArrayLike],
    target: dict[str, ArrayLike],
    masks: dict[str, ArrayLike] | None = None,
    be: Backend | None = None,
    type_weights: dict[str, float] | None = None,
) -> float:
    """Per-type layer-normed L1 latent distance ``d(ẑ, sg[z⁺])`` (doc J D4).

    Stop-gradient is applied to the *target* (the EMA branch) — in the NumPy
    reference path that is a no-op, but it documents the gradient contract and
    is honored on the MLX path via ``be.stop_gradient``.
    """
    be = be or get_backend()
    total = 0.0
    n_types = 0
    for t, p in pred.items():
        if t not in target:
            continue
        tgt = be.stop_gradient(target[t])
        pn = _layer_norm_np(np.asarray(be.to_numpy(p)))
        tn = _layer_norm_np(np.asarray(be.to_numpy(tgt)))
        diff = np.abs(pn - tn)
        if masks and t in masks:
            m = np.asarray(be.to_numpy(masks[t]))[..., None]
            diff = diff * m
            denom = max(float(m.sum()) * pn.shape[-1], 1.0)
            d = float(diff.sum()) / denom
        else:
            d = float(diff.mean())
        w = (type_weights or {}).get(t, 1.0)
        total += w * d
        n_types += 1
    return total / max(n_types, 1)


def jepa_mask_loss(
    online: dict[str, ArrayLike],
    target: dict[str, ArrayLike],
    masks: dict[str, ArrayLike] | None = None,
    be: Backend | None = None,
) -> float:
    """Masked action-free JEPA warmup loss ``L_JEPA,mask`` (doc J Eq. 3)."""
    return latent_distance(online, target, masks, be)


# --------------------------------------------------------------------------- #
# EMA target encoder update.
# --------------------------------------------------------------------------- #
def ema_update(
    target_params: dict[str, np.ndarray],
    online_params: dict[str, np.ndarray],
    decay: float,
) -> dict[str, np.ndarray]:
    """In-place-style EMA: ``ξ ← τ·ξ + (1-τ)·θ`` (doc J Eq. 1, ∇ξ=0).

    Operates on flat NumPy parameter dicts so it works for either backend's
    checkpointed weights. ``decay`` is τ (close to 1).
    """
    out: dict[str, np.ndarray] = {}
    for k, theta in online_params.items():
        xi = target_params.get(k, theta)
        out[k] = decay * xi + (1.0 - decay) * theta
    return out


# --------------------------------------------------------------------------- #
# Counterfactual rollout loss (absolute + branch-difference).
# --------------------------------------------------------------------------- #
def rollout_abs_loss(
    pred_traj: list[dict[str, ArrayLike]],
    target_traj: list[dict[str, ArrayLike]],
    step_weights: list[float] | None = None,
    masks: dict[str, ArrayLike] | None = None,
    be: Backend | None = None,
) -> float:
    """Absolute free-running rollout match ``L_roll,abs`` (doc J Eq. 8).

    ``pred_traj[h]`` and ``target_traj[h]`` are per-type latent dicts at rollout
    step ``h``. Targets are EMA-encoded counterfactual futures (stop-gradient).
    """
    be = be or get_backend()
    n = min(len(pred_traj), len(target_traj))
    if n == 0:
        return 0.0
    weights = step_weights or [1.0] * n
    total = 0.0
    wsum = 0.0
    for h in range(n):
        rho = weights[h] if h < len(weights) else 1.0
        total += rho * latent_distance(pred_traj[h], target_traj[h], masks, be)
        wsum += rho
    return total / max(wsum, 1e-8)


def rollout_branch_diff_loss(
    pred_b: list[dict[str, ArrayLike]],
    pred_bp: list[dict[str, ArrayLike]],
    tgt_b: list[dict[str, ArrayLike]],
    tgt_bp: list[dict[str, ArrayLike]],
    step_weights: list[float] | None = None,
    be: Backend | None = None,
) -> float:
    """Branch-difference action-fidelity term ``L_roll,Δ`` (doc J Eq. 9).

    For two branches ``b, b'`` from the *same* pre-action state hash and the
    *same* exogenous seed bundle, match the predicted latent *difference* to the
    target difference: ``|| Δẑ - sg[Δz⁺] ||₁``. This is the minimal supervision
    that keeps the predictor sharply action-sensitive (doc J T2 / collapse D5).
    """
    be = be or get_backend()
    n = min(len(pred_b), len(pred_bp), len(tgt_b), len(tgt_bp))
    if n == 0:
        return 0.0
    weights = step_weights or [1.0] * n
    total = 0.0
    wsum = 0.0
    for h in range(n):
        nu = weights[h] if h < len(weights) else 1.0
        acc = 0.0
        ntypes = 0
        for t in pred_b[h]:
            if t not in pred_bp[h] or t not in tgt_b[h] or t not in tgt_bp[h]:
                continue
            dq = np.asarray(be.to_numpy(pred_b[h][t])) - np.asarray(
                be.to_numpy(pred_bp[h][t])
            )
            dt = np.asarray(be.to_numpy(be.stop_gradient(tgt_b[h][t]))) - np.asarray(
                be.to_numpy(be.stop_gradient(tgt_bp[h][t]))
            )
            acc += float(np.abs(dq - dt).mean())
            ntypes += 1
        total += nu * acc / max(ntypes, 1)
        wsum += nu
    return total / max(wsum, 1e-8)


def rollout_loss(
    pred_traj: list[dict[str, ArrayLike]],
    target_traj: list[dict[str, ArrayLike]],
    branch_pairs: list[tuple] | None = None,
    cfg: JEPAConfig | None = None,
    be: Backend | None = None,
) -> float:
    """Combined rollout loss ``L_roll = L_roll,abs + λ_Δ·L_roll,Δ`` (doc J Eq. 10).

    ``branch_pairs`` is a list of ``(pred_b, pred_bp, tgt_b, tgt_bp)`` rollout
    tuples; pass an empty list / ``None`` to use only the absolute term.
    """
    cfg = cfg or JEPAConfig()
    be = be or get_backend()
    abs_l = rollout_abs_loss(pred_traj, target_traj, be=be)
    delta_l = 0.0
    if branch_pairs:
        for pb, pbp, tb, tbp in branch_pairs:
            delta_l += rollout_branch_diff_loss(pb, pbp, tb, tbp, be=be)
        delta_l /= len(branch_pairs)
    return abs_l + cfg.lambda_delta * delta_l


# --------------------------------------------------------------------------- #
# Anti-collapse: weak per-type variance / covariance (VICReg-lite).
# --------------------------------------------------------------------------- #
def variance_loss(
    proj: dict[str, ArrayLike],
    masks: dict[str, ArrayLike] | None = None,
    var_floor: float = 1.0,
    be: Backend | None = None,
) -> float:
    """Per-type variance hinge ``Σ max(0, γ - std_B(q))²`` (doc J Eq. 5)."""
    be = be or get_backend()
    total = 0.0
    n = 0
    for t, q in proj.items():
        x = np.asarray(be.to_numpy(q))
        if masks and t in masks:
            m = np.asarray(be.to_numpy(masks[t])).astype(bool)
            if m.sum() < 2:
                continue
            x = x[m]
        if x.shape[0] < 2:
            continue
        std = x.std(axis=0)
        hinge = np.maximum(0.0, var_floor - std)
        total += float((hinge * hinge).mean())
        n += 1
    return total / max(n, 1)


def covariance_loss(
    proj: dict[str, ArrayLike],
    masks: dict[str, ArrayLike] | None = None,
    be: Backend | None = None,
) -> float:
    """Off-diagonal covariance energy ``||offdiag(Cov(q))||²_F / d`` (doc J Eq. 6)."""
    be = be or get_backend()
    total = 0.0
    n = 0
    for t, q in proj.items():
        x = np.asarray(be.to_numpy(q))
        if masks and t in masks:
            m = np.asarray(be.to_numpy(masks[t])).astype(bool)
            if m.sum() < 2:
                continue
            x = x[m]
        if x.shape[0] < 2 or x.shape[1] < 2:
            continue
        xc = x - x.mean(axis=0, keepdims=True)
        cov = (xc.T @ xc) / (x.shape[0] - 1)
        d = cov.shape[0]
        off = cov - np.diag(np.diag(cov))
        total += float((off * off).sum()) / d
        n += 1
    return total / max(n, 1)


def anti_collapse_loss(
    proj: dict[str, ArrayLike],
    masks: dict[str, ArrayLike] | None = None,
    cfg: JEPAConfig | None = None,
    be: Backend | None = None,
) -> float:
    """``L_collapse = λ_var·L_var + λ_cov·L_cov`` (doc J Eq. 7)."""
    cfg = cfg or JEPAConfig()
    be = be or get_backend()
    return cfg.lambda_var * variance_loss(
        proj, masks, cfg.var_floor, be
    ) + cfg.lambda_cov * covariance_loss(proj, masks, be)


# --------------------------------------------------------------------------- #
# Supervised head losses (pinball + non-crossing, survival NLL) — doc J H1/H2.
# --------------------------------------------------------------------------- #
def pinball_loss(
    pred_quantiles: dict[float, float], target: float, quantiles: tuple[float, ...]
) -> float:
    """Pinball / quantile loss ``Σ_τ ρ_τ(y - q_τ)`` (doc J Eq. 11–12)."""
    total = 0.0
    for tau in quantiles:
        u = target - pred_quantiles.get(tau, 0.0)
        total += u * (tau - (1.0 if u < 0 else 0.0))
    return total


def noncrossing_penalty(
    pred_quantiles: dict[float, float], quantiles: tuple[float, ...]
) -> float:
    """``Σ_{τ<τ'} max(0, q_τ - q_τ')`` quantile-crossing penalty (doc J Eq. 13)."""
    qs = sorted(quantiles)
    pen = 0.0
    for i in range(len(qs) - 1):
        pen += max(
            0.0, pred_quantiles.get(qs[i], 0.0) - pred_quantiles.get(qs[i + 1], 0.0)
        )
    return pen


def competing_risk_nll(
    interval_probs: list[np.ndarray],
    event_class: int | None,
    event_interval: int | None,
) -> float:
    """Discrete-time competing-risk NLL (doc J Eq. 14–16, doc L E7).

    ``interval_probs[h]`` is the per-interval class distribution
    ``[survive, risk_1, ...]``. ``event_class`` is the 1-based risk index of the
    realized event (or ``None`` for right-censoring); ``event_interval`` is the
    interval it occurred in (or the censoring interval).
    """
    nll = 0.0
    h_end = event_interval if event_interval is not None else len(interval_probs)
    for r in range(min(h_end, len(interval_probs))):
        nll -= float(np.log(max(interval_probs[r][0], 1e-8)))  # survived interval r
    if (
        event_class is not None
        and event_interval is not None
        and 0 <= event_interval < len(interval_probs)
    ):
        p = interval_probs[event_interval][event_class]
        nll -= float(np.log(max(p, 1e-8)))
    return nll


def binary_hazard_nll(
    interval_hazards: np.ndarray, event_interval: int | None
) -> float:
    """Parallel-binary survival NLL for one incident channel (doc L H2)."""
    nll = 0.0
    h_end = event_interval if event_interval is not None else len(interval_hazards)
    for r in range(min(h_end, len(interval_hazards))):
        nll -= float(np.log(max(1.0 - interval_hazards[r], 1e-8)))
    if event_interval is not None and 0 <= event_interval < len(interval_hazards):
        nll -= float(np.log(max(interval_hazards[event_interval], 1e-8)))
    return nll


def quantile_head_loss(
    forecasts,
    targets: dict[tuple[str, int], float],
    quantiles: tuple[float, ...],
    lambda_mono: float = 1.0,
) -> float:
    """``L_quant + λ_mono·L_qmono`` over a quantile forecast list (doc J Eq. 11–13).

    ``forecasts`` is a list of ``QuantileForecast`` (each carries a channel,
    horizon, and ``{τ: q_τ}`` map). ``targets`` maps ``(channel_value, horizon_s)``
    to the realized KPI-delta label ``y_{k,h}``; channels without a label
    contribute only the non-crossing penalty so the head geometry is still
    exercised. Returns the per-forecast mean.
    """
    if not forecasts:
        return 0.0
    total = 0.0
    for qf in forecasts:
        key = (qf.channel.value, qf.horizon_s)
        pen = lambda_mono * noncrossing_penalty(qf.quantiles, quantiles)
        if key in targets:
            total += pinball_loss(qf.quantiles, targets[key], quantiles) + pen
        else:
            total += pen
    return total / max(len(forecasts), 1)


def survival_head_loss(
    competing_interval_probs: dict[str, list[np.ndarray]],
    binary_interval_hazards: dict[str, np.ndarray],
    competing_labels: dict[str, tuple[int | None, int | None]],
    binary_labels: dict[str, int | None],
) -> float:
    """``L_surv`` over competing-risk + parallel-binary channels (doc J Eq. 14–17).

    ``competing_interval_probs[entity]`` is the per-interval class distribution
    ``[survive, risk_1, ...]`` for that entity's competing-risk head;
    ``competing_labels[entity]`` is ``(event_class, event_interval)`` (both
    ``None`` for right-censoring). ``binary_interval_hazards[ch]`` is the
    per-interval hazard for parallel-binary channel ``ch`` with label
    ``binary_labels[ch]`` (event interval or ``None``). Returns the mean
    per-instance NLL so the scale stays bounded across the group.
    """
    total = 0.0
    n = 0
    for ent, probs in competing_interval_probs.items():
        ec, ei = competing_labels.get(ent, (None, None))
        total += competing_risk_nll(probs, ec, ei)
        n += 1
    for ch, haz in binary_interval_hazards.items():
        total += binary_hazard_nll(haz, binary_labels.get(ch))
        n += 1
    return total / max(n, 1)


def ood_head_loss(
    epistemic_logit: float,
    ood_logit: float,
    ood_label: float,
    focal: bool = True,
    gamma: float = 2.0,
) -> float:
    """``L_abst`` = (focal-)BCE on the OOD/topology/fault detector label (doc J H3).

    ``ood_label`` is the held-out OOD indicator ``o_h`` (1 = OOD / held-out
    topology-flavor-fault, 0 = in-distribution). The epistemic logit is included
    as a weak BCE against the same label so the disagreement head is anchored;
    focal-BCE is used by default because positives are rare (doc J H3 / Eq. 17).
    """
    logit = np.array([ood_logit], dtype=np.float64)
    epi = np.array([epistemic_logit], dtype=np.float64)
    y = np.array([ood_label], dtype=np.float64)
    if focal:
        l_ood = float(_focal_bce_with_logits(logit, y, gamma)[0])
    else:
        l_ood = float(_bce_with_logits(logit, y)[0])
    l_epi = float(_bce_with_logits(epi, y)[0])
    return l_ood + 0.5 * l_epi


# --------------------------------------------------------------------------- #
# Mandatory exact-state consistency loss L_cons (doc J D11/Eq. 18).
# --------------------------------------------------------------------------- #
def _bce_with_logits(logit: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Numerically-stable per-element BCE ``-[y·logσ + (1-y)·log(1-σ)]``."""
    # = max(x,0) - x·y + log(1 + exp(-|x|))
    return np.maximum(logit, 0.0) - logit * target + np.log1p(np.exp(-np.abs(logit)))


def _focal_bce_with_logits(
    logit: np.ndarray, target: np.ndarray, gamma: float = 2.0
) -> np.ndarray:
    """Focal-BCE ``(1 - p_t)^γ · BCE`` for rare positives (doc J H3 / Eq. 17)."""
    # p_t = σ(x) if y=1 else 1-σ(x); stable logistic.
    z = np.clip(logit, -30.0, 30.0)
    p = 1.0 / (1.0 + np.exp(-z))
    p_t = np.where(target > 0.5, p, 1.0 - p)
    mod = np.power(np.clip(1.0 - p_t, 1e-8, 1.0), gamma)
    return mod * _bce_with_logits(logit, target)


def _huber(pred: np.ndarray, target: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """Per-element Huber (smooth-L1) for exact arithmetic counters (doc J Eq. 18)."""
    r = pred - target
    a = np.abs(r)
    quad = 0.5 * r * r
    lin = delta * (a - 0.5 * delta)
    return np.where(a <= delta, quad, lin)


def consistency_loss(
    field_preds: dict[str, ArrayLike],
    field_targets: dict[str, tuple[np.ndarray, np.ndarray]],
    spec: tuple[tuple[str, str, int, str], ...],
    field_weights: dict[str, float] | None = None,
    be: Backend | None = None,
) -> float:
    """Exact-state consistency loss ``L_cons`` at one rollout step (doc J Eq. 18).

    ``L_cons = Σ_f κ_f ℓ_f(D_f(ẑ_h), sg[f(s^K)])`` where ``ℓ_f`` is BCE for
    binary masks / blocker bits and Huber for exact arithmetic counters
    (categorical-id consistency is realized as presence BCE, see
    :data:`architecture.FIELD_SPEC`). The kernel targets are stop-gradient (the
    exact fields are constants in the gradient, ``∂K/∂θ = 0``, doc J T4), so this
    only constrains the learned residual — it cannot improve loss by violating
    exact symbolic semantics (doc J T3). Only valid (non-padding) rows count.

    ``field_targets[name]`` is ``(target, valid_mask)`` from
    :func:`architecture.exact_field_targets`; ``field_weights`` are the optional
    per-field ``κ_f`` (default 1).
    """
    be = be or get_backend()
    flavor = {name: fl for name, _lt, _od, fl in spec}
    kappa = field_weights or {}
    total = 0.0
    n = 0
    for name, pred in field_preds.items():
        if name not in field_targets:
            continue
        target, valid = field_targets[name]
        p = np.asarray(be.to_numpy(be.stop_gradient(pred)), dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        if p.shape != t.shape:
            continue
        fl = flavor.get(name, "bce")
        if fl == "huber":
            per = _huber(p, t)
        elif fl == "focal":
            per = _focal_bce_with_logits(p, t)
        else:  # "bce"
            per = _bce_with_logits(p, t)
        vmask = np.asarray(valid, dtype=np.float64).reshape(-1)
        m2 = vmask[:, None] if per.ndim == 2 else vmask
        denom = max(float(m2.sum()) * (per.shape[-1] if per.ndim == 2 else 1), 1.0)
        field_l = float((per * m2).sum()) / denom
        total += kappa.get(name, 1.0) * field_l
        n += 1
    return total / max(n, 1)


def rollout_consistency_loss(
    field_pred_traj: list[dict[str, ArrayLike]],
    field_target_traj: list[dict[str, tuple[np.ndarray, np.ndarray]]],
    spec: tuple[tuple[str, str, int, str], ...],
    step_weights: list[float] | None = None,
    field_weights: dict[str, float] | None = None,
    be: Backend | None = None,
) -> float:
    """Sum ``L_cons`` over every rollout step ``h = 1..H`` (doc J Eq. 18).

    ``field_pred_traj[h]`` are the decoded fields ``D_f(ẑ_h)`` and
    ``field_target_traj[h]`` the matched exact kernel targets at step ``h``.
    """
    be = be or get_backend()
    n = min(len(field_pred_traj), len(field_target_traj))
    if n == 0:
        return 0.0
    weights = step_weights or [1.0] * n
    total = 0.0
    wsum = 0.0
    for h in range(n):
        w = weights[h] if h < len(weights) else 1.0
        total += w * consistency_loss(
            field_pred_traj[h],
            field_target_traj[h],
            spec,
            field_weights=field_weights,
            be=be,
        )
        wsum += w
    return total / max(wsum, 1e-8)


# --------------------------------------------------------------------------- #
# Combined objective with group-wise homoscedastic uncertainty weighting.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class GroupLogVars:
    """Learned per-group log-variances ``s_1..s_6`` (doc J Eq. 19/26)."""

    s: dict[str, float] = field(default_factory=lambda: {g: 0.0 for g in LOSS_GROUPS})

    def clamp(self, lo: float, hi: float) -> None:
        for k in self.s:
            self.s[k] = float(min(hi, max(lo, self.s[k])))


def combined_objective(
    group_losses: dict[str, float],
    logvars: GroupLogVars | None = None,
    cfg: JEPAConfig | None = None,
) -> tuple[float, dict[str, float]]:
    """Group-wise uncertainty-weighted total ``Σ_m exp(-s_m)·G_m + s_m`` (Eq. 19).

    Returns ``(total, per_group_weighted)``. Missing groups contribute zero.
    With all ``s_m = 0`` this reduces to the plain sum of group losses, which is
    the fixed-weight ablation (doc J: ``s_m = log λ_m^{-1}``).
    """
    cfg = cfg or JEPAConfig()
    lv = logvars or GroupLogVars()
    lv.clamp(*cfg.s_bounds)
    total = 0.0
    parts: dict[str, float] = {}
    for g in LOSS_GROUPS:
        Gm = float(group_losses.get(g, 0.0))
        sm = lv.s[g]
        weighted = float(np.exp(-sm)) * Gm + sm
        parts[g] = weighted
        total += weighted
    return total, parts


# --------------------------------------------------------------------------- #
# Collapse diagnostics D3 (covariance energy) and D4 (cosine) — doc J D3/D4.
# --------------------------------------------------------------------------- #
def covariance_energy(latent: np.ndarray, eps: float = 1e-8) -> float:
    """Cross-dimension covariance energy ``c_off`` (doc J D3).

    ``c_off = ||offdiag(Cov(norm(q)))||_F² / ||diag(Cov(norm(q)))||_F²``. High
    ``c_off`` indicates dimensional collapse or redundant predictors. ``latent``
    is ``(n_rows, d)`` over the valid (unmasked) rows; the rows are per-dimension
    normalized first so the energy is scale-free.
    """
    x = np.asarray(latent, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        return 0.0
    sd = x.std(axis=0, keepdims=True)
    xn = (x - x.mean(axis=0, keepdims=True)) / (sd + eps)
    cov = (xn.T @ xn) / (x.shape[0] - 1)
    diag = np.diag(cov)
    off = cov - np.diag(diag)
    off_e = float((off * off).sum())
    diag_e = float((diag * diag).sum())
    return off_e / (diag_e + eps)


def cosine_collapse(latent: np.ndarray, eps: float = 1e-8) -> float:
    """Mean off-diagonal cosine similarity ``s_cos`` (doc J D4).

    ``s_cos = mean_{i≠j} cos(q_i, q_j)``. Complete collapse drives ``s_cos`` → 1;
    partial collapse shows high within-batch cosine concentration. ``latent`` is
    ``(n_rows, d)`` over the valid rows.
    """
    x = np.asarray(latent, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        return 0.0
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    xn = x / (norm + eps)
    sim = xn @ xn.T
    n = sim.shape[0]
    off_sum = float(sim.sum() - np.trace(sim))
    return off_sum / max(n * (n - 1), 1)


# --------------------------------------------------------------------------- #
# Typed masking (the JEPA context masking) — doc J E1/A2, doc P shortcut block.
# --------------------------------------------------------------------------- #
def apply_typed_mask(
    features: dict[str, ArrayLike],
    mask_types: tuple[str, ...],
    rng: np.random.Generator,
    mask_prob: float,
    be: Backend | None = None,
) -> dict[str, ArrayLike]:
    """Zero a random fraction of rows for the named types (doc J typed masking).

    Masking acts as a latent intervention that blocks shortcut solutions and
    forces the predictor to *use the action and relational context* rather than
    copying observed values (doc J E3 / Causal-JEPA). Returns a new feature dict;
    the validity masks themselves are unchanged so padding logic still holds.
    """
    be = be or get_backend()
    out: dict[str, ArrayLike] = {}
    for t, f in features.items():
        if t in mask_types:
            x = np.asarray(be.to_numpy(f)).copy()
            keep = rng.random(x.shape[0]) >= mask_prob
            x = x * keep[:, None]
            out[t] = be.from_numpy(x.astype(np.float32))
        else:
            out[t] = f
    return out


__all__ = [
    "LOSS_GROUPS",
    "GroupLogVars",
    "JEPAConfig",
    "anti_collapse_loss",
    "apply_typed_mask",
    "binary_hazard_nll",
    "combined_objective",
    "competing_risk_nll",
    "consistency_loss",
    "cosine_collapse",
    "covariance_energy",
    "covariance_loss",
    "ema_update",
    "jepa_mask_loss",
    "latent_distance",
    "noncrossing_penalty",
    "ood_head_loss",
    "pinball_loss",
    "quantile_head_loss",
    "rollout_abs_loss",
    "rollout_branch_diff_loss",
    "rollout_consistency_loss",
    "rollout_loss",
    "survival_head_loss",
    "variance_loss",
]
