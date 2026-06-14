# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Aleatoric vs epistemic uncertainty separation (doc L, §uq_decomposition).

The project's safety stack is deliberately *two-channel*: it never collapses
uncertainty to a single confidence scalar. Following doc L (E1) we use the law
of total variance

    Var(Y | X, D) = E_theta[Var(Y | X, theta)]   (aleatoric)
                  + Var_theta(E[Y | X, theta])    (epistemic).

* **Aleatoric** uncertainty is irreducible observation/dynamics noise. We read
  it from the *quantile spread* of a single ensemble member's KPI-delta head
  (E2) and from the *hazard probability* of the survival head (E5/E6). It is the
  width of an individual member's predictive distribution.
* **Epistemic** uncertainty is model disagreement that more data could reduce.
  We read it from *BatchEnsemble-style disagreement* across the ``M`` members
  (E4, E9, E10): quantile variance for continuous channels and Jensen-Shannon
  divergence across hazard vectors for event channels.

The two are returned as a typed :class:`UncertaintyDecomposition`, never summed.
Per doc L's epistemic_method, disagreement is used *only* for abstention, OOD
fusion, and slice diagnostics — it is **not** used to heuristically widen
intervals once the conformal layer (``conformal.py``) is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Numerical floor used everywhere a probability enters a log.
_EPS = 1e-12


@dataclass(slots=True)
class UncertaintyDecomposition:
    """Separated aleatoric and epistemic uncertainty for one (x, a, H) query.

    Attributes are kept distinct on purpose; ``total`` is provided only as a
    convenience for logging and must never be used as the trust gate (doc L).
    """

    aleatoric: float
    epistemic: float
    #: Per-channel breakdown so slice diagnostics can attribute disagreement.
    per_channel_aleatoric: dict[str, float] = field(default_factory=dict)
    per_channel_epistemic: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        """Total predictive variance (law of total variance, E1).

        Convenience only — the planner gate consumes the two components and the
        conformal upper bounds separately, never this scalar.
        """
        return self.aleatoric + self.epistemic


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> np.ndarray:
    """Per-sample pinball / check loss ``rho_tau(u)`` (doc L, E2).

    ``rho_tau(u) = u * (tau - 1{u < 0})`` with ``u = y_true - y_pred``. Returns
    the element-wise loss (caller decides how to reduce).
    """
    if not 0.0 < tau < 1.0:
        raise ValueError(f"quantile tau must be in (0, 1); got {tau}")
    u = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return np.where(u >= 0.0, tau * u, (tau - 1.0) * u)


def non_crossing_penalty(quantiles: dict[float, np.ndarray]) -> np.ndarray:
    """Quantile non-crossing penalty ``L_nc`` (doc L, E8).

    Sums ``max(0, q_tau - q_tau')`` over every ordered pair ``tau < tau'`` so a
    well-trained head produces monotone quantiles. Returned per-sample.
    """
    taus = sorted(quantiles)
    arrs = [np.asarray(quantiles[t], dtype=float) for t in taus]
    if not arrs:
        return np.zeros(0)
    penalty = np.zeros_like(arrs[0], dtype=float)
    for lo in range(len(taus)):
        for hi in range(lo + 1, len(taus)):
            penalty = penalty + np.maximum(0.0, arrs[lo] - arrs[hi])
    return penalty


def enforce_monotone_quantiles(
    quantiles: dict[float, float],
) -> dict[float, float]:
    """Repair crossing quantiles by a running max along increasing ``tau``.

    A cheap, order-preserving rearrangement that guarantees a valid (monotone)
    quantile function for downstream CQR scoring without retraining the head.
    """
    taus = sorted(quantiles)
    out: dict[float, float] = {}
    running = -np.inf
    for t in taus:
        running = max(running, float(quantiles[t]))
        out[t] = running
    return out


def quantile_interval_width(
    quantiles: dict[float, float], lo: float = 0.05, hi: float = 0.95
) -> float:
    """Aleatoric width = ``q_hi - q_lo`` from a single member's quantile head.

    The spread between two predictive quantiles of *one* member is the member's
    own (aleatoric) predictive uncertainty (doc L E2): it is irreducible noise,
    not model disagreement.
    """
    q = enforce_monotone_quantiles(quantiles)
    taus = sorted(q)
    lo_key = min(taus, key=lambda t: abs(t - lo))
    hi_key = min(taus, key=lambda t: abs(t - hi))
    return float(q[hi_key] - q[lo_key])


def ensemble_mean_quantiles(
    member_quantiles: list[dict[float, float]],
) -> dict[float, float]:
    """Average quantile head over ensemble members (doc L, E3).

    ``q_bar_tau(x) = (1/M) sum_m q_hat_{m,tau}(x)``.
    """
    if not member_quantiles:
        return {}
    taus = sorted(member_quantiles[0])
    out: dict[float, float] = {}
    for t in taus:
        out[t] = float(np.mean([m[t] for m in member_quantiles]))
    return out


def quantile_disagreement(
    member_quantiles: list[dict[float, float]],
) -> float:
    """Epistemic disagreement of a continuous channel (doc L, E4).

    ``D_quant(x) = sum_tau Var_m(q_hat_{m,tau}(x))`` — variance *across members*
    of each quantile, summed over the quantile grid. This is genuine model
    disagreement (epistemic), reducible with more data.
    """
    if len(member_quantiles) < 2:
        return 0.0
    taus = sorted(member_quantiles[0])
    total = 0.0
    for t in taus:
        vals = np.array([m[t] for m in member_quantiles], dtype=float)
        total += float(np.var(vals, ddof=1))
    return total


def scalar_epistemic_variance(member_means: np.ndarray) -> float:
    """Ensemble variance of a scalar point prediction (doc L, E10).

    ``sigma_epi^2 = (1/(M-1)) sum_m (mu_hat_m - mu_bar)^2``.
    """
    arr = np.asarray(member_means, dtype=float).ravel()
    if arr.size < 2:
        return 0.0
    return float(np.var(arr, ddof=1))


def _normalize_hazard_vector(p: np.ndarray) -> np.ndarray:
    """Coerce a hazard / probability vector to a valid categorical distribution.

    Binary parallel hazards are turned into ``[p, 1 - p]`` two-class vectors;
    competing-risk vectors are clipped and renormalised.
    """
    p = np.asarray(p, dtype=float).ravel()
    p = np.clip(p, _EPS, 1.0)
    s = p.sum()
    if p.size == 1:  # single binary hazard -> [p, 1-p]
        h = float(np.clip(p[0], _EPS, 1.0 - _EPS))
        return np.array([h, 1.0 - h])
    if s <= 0.0:
        return np.full_like(p, 1.0 / p.size)
    return p / s


def jensen_shannon_divergence(distributions: list[np.ndarray]) -> float:
    """Jensen-Shannon divergence across hazard vectors (doc L, E9 ``JS`` term).

    JS = H(mean) - mean(H(member)), using natural log; bounded in ``[0, ln k]``
    for ``k`` classes. This is the epistemic disagreement primitive for event /
    hazard channels (mutually-exclusive competing risks or binary incidents).
    """
    if len(distributions) < 2:
        return 0.0
    dists = [_normalize_hazard_vector(d) for d in distributions]
    k = max(d.size for d in dists)
    padded = []
    for d in dists:
        if d.size < k:  # pad shorter competing-risk vectors with eps mass
            d = np.concatenate([d, np.full(k - d.size, _EPS)])
            d = d / d.sum()
        padded.append(d)
    stacked = np.stack(padded, axis=0)
    mean = stacked.mean(axis=0)

    def _entropy(p: np.ndarray) -> float:
        p = np.clip(p, _EPS, 1.0)
        return float(-np.sum(p * np.log(p)))

    mean_entropy = _entropy(mean)
    expected_entropy = float(np.mean([_entropy(p) for p in padded]))
    return max(0.0, mean_entropy - expected_entropy)


def aleatoric_hazard(member_hazards: list[float]) -> float:
    """Aleatoric component of a binary hazard channel.

    For a Bernoulli event the irreducible (aleatoric) uncertainty is the mean
    Bernoulli variance ``E_m[p_m (1 - p_m)]`` (the ``E_theta[Var(Y|theta)]``
    term of E1), distinct from the across-member disagreement (epistemic).
    """
    if not member_hazards:
        return 0.0
    p = np.clip(np.asarray(member_hazards, dtype=float), 0.0, 1.0)
    return float(np.mean(p * (1.0 - p)))


def decompose(
    *,
    member_quantiles: dict[str, list[dict[float, float]]] | None = None,
    member_hazards: dict[str, list[float]] | None = None,
    member_hazard_vectors: dict[str, list[np.ndarray]] | None = None,
    lambda_q: float = 1.0,
    lambda_h: float = 1.0,
    aleatoric_lo: float = 0.05,
    aleatoric_hi: float = 0.95,
) -> UncertaintyDecomposition:
    """Decompose ensemble forecasts into aleatoric and epistemic uncertainty.

    Implements doc L E1/E4/E9/E10 jointly across continuous KPI channels and
    event-hazard channels. Aleatoric and epistemic are accumulated separately
    and never summed into the gate.

    Parameters
    ----------
    member_quantiles:
        ``{channel: [member_0_quantiles, ...]}`` — per-member quantile dicts
        for each continuous KPI-delta channel.
    member_hazards:
        ``{channel: [p_member_0, ...]}`` — per-member *binary* hazard probs.
    member_hazard_vectors:
        ``{channel: [vec_member_0, ...]}`` — per-member competing-risk vectors
        (use this instead of ``member_hazards`` for mutually-exclusive events).
    lambda_q, lambda_h:
        Channel-group weights for the fused epistemic score (doc L, E9).
    """
    per_a: dict[str, float] = {}
    per_e: dict[str, float] = {}

    if member_quantiles:
        for ch, members in member_quantiles.items():
            if not members:
                continue
            # Aleatoric: average single-member spread.
            widths = [
                quantile_interval_width(m, aleatoric_lo, aleatoric_hi) for m in members
            ]
            per_a[ch] = float(np.mean(widths))
            # Epistemic: across-member quantile variance.
            per_e[ch] = lambda_q * quantile_disagreement(members)

    if member_hazard_vectors:
        for ch, vecs in member_hazard_vectors.items():
            if not vecs:
                continue
            means = [_normalize_hazard_vector(v) for v in vecs]
            mean = np.mean(np.stack(_pad_to_common(means), axis=0), axis=0)
            # Aleatoric for a categorical: expected per-member entropy proxy via
            # mean Gini impurity = 1 - sum p^2.
            per_a[ch] = float(1.0 - np.sum(mean**2))
            per_e[ch] = lambda_h * jensen_shannon_divergence(vecs)

    if member_hazards:
        for ch, ps in member_hazards.items():
            if not ps:
                continue
            per_a[ch] = aleatoric_hazard(ps)
            vecs = [np.array([p]) for p in ps]
            per_e[ch] = lambda_h * jensen_shannon_divergence(vecs)

    aleatoric = float(sum(per_a.values()))
    epistemic = float(sum(per_e.values()))
    return UncertaintyDecomposition(
        aleatoric=aleatoric,
        epistemic=epistemic,
        per_channel_aleatoric=per_a,
        per_channel_epistemic=per_e,
    )


def _pad_to_common(vectors: list[np.ndarray]) -> list[np.ndarray]:
    """Pad categorical vectors to a common length with eps mass + renormalise."""
    k = max(v.size for v in vectors)
    out = []
    for v in vectors:
        if v.size < k:
            v = np.concatenate([v, np.full(k - v.size, _EPS)])
            v = v / v.sum()
        out.append(v)
    return out


__all__ = [
    "UncertaintyDecomposition",
    "aleatoric_hazard",
    "decompose",
    "enforce_monotone_quantiles",
    "ensemble_mean_quantiles",
    "jensen_shannon_divergence",
    "non_crossing_penalty",
    "pinball_loss",
    "quantile_disagreement",
    "quantile_interval_width",
    "scalar_epistemic_variance",
]
