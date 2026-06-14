# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Importance-sampling rare-event estimation with likelihood-ratio accounting.

Rare events (fault bursts, pods-ready timeouts, cooling derates, power caps,
queue bursts) must be sampled with **statistical correction**, never naive
oversampling (doc R, rare_event_simulation; doc D risks #4). We tilt *parent*
hazards — not isolated observables — and carry the exact point-process likelihood
ratio so the rare-event probability estimate stays **unbiased** (doc R,
EQUATIONS Eq21-Eq25):

    Q-tilt          λ^Q(t) = λ^P(t) · exp{ η · g(t, S_t) }          (Eq22)
    log-LR          log L = Σ_i log(λ^P_{m_i}/λ^Q_{m_i})
                            − ∫₀^H (Λ^P(t) − Λ^Q(t)) dt              (Eq23)
    estimator       p̂_F = (1/N) Σ_i 1{ω_i ∈ F_H} · L(ω_i)          (Eq24)

with ``ω_i ~ Q_η``. THEOREM T3: if ``Q_η`` dominates ``P`` on the rare set ``F_H``
and the full likelihood ratio is computed exactly, the estimator is unbiased.

This module is generator-agnostic: it accepts the *nominal* and *tilted* hazard
functions, the realised event times, and the indicator, and returns the unbiased
weighted estimate. It reuses the same tilt semantics the simulator's
``rare_events`` layer uses (the tilt is a multiplier passed to
:func:`arrivals.simulate_arrivals` and :func:`faults.sample_fault_cluster`), so a
counterfactual pack can request a rare-event-tilted branch and still report a
calibrated (unbiased) hazard.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

#: The rare-event families v1 ships (doc D rare_event_library).
RARE_EVENT_FAMILIES: tuple[str, ...] = (
    "gpu_fault_burst",
    "pods_ready_timeout",
    "cooling_derate",
    "power_derate",
    "queue_burst",
    "maintenance_mask",
    "telemetry_degradation",
)


@dataclass(slots=True)
class ExponentialTilt:
    """An exponential hazard tilt ``λ^Q = λ^P · exp(η · g)`` (doc R, Eq22).

    ``eta`` is the tilt strength; ``g`` is a *parent-level* statistic such as a
    queue-burst score, zone cooling slack, rack power slack, or health-precursor
    score (doc R, Eq22 requires parent-level g, not isolated observables). The
    realised multiplier ``exp(eta * g)`` is what gets passed to the generators.
    """

    eta: float
    g_value: float = 1.0

    @property
    def multiplier(self) -> float:
        return float(np.exp(self.eta * self.g_value))


def point_process_log_lr(
    event_times: list[float],
    nominal_intensity: Callable[[float], float],
    tilted_intensity: Callable[[float], float],
    horizon_s: float,
    *,
    n_quad: int = 256,
) -> float:
    """Exact point-process log-likelihood ratio ``log(dP/dQ)`` (doc R, Eq23).

    Sum-of-logs over realised marks minus the difference of compensators
    ``∫(Λ^P − Λ^Q)``, the compensator integral evaluated by the midpoint rule
    (fine enough for the smooth hazards of the demo pod). Returns ``log L`` such
    that ``L = exp(log L) = dP/dQ`` is the weight applied to a Q-sampled path.
    """
    log_lr = 0.0
    for ti in event_times:
        lam_p = max(nominal_intensity(ti), 1e-300)
        lam_q = max(tilted_intensity(ti), 1e-300)
        log_lr += np.log(lam_p) - np.log(lam_q)
    # − ∫ (Λ^P − Λ^Q) dt  via midpoint quadrature
    grid = (np.arange(n_quad) + 0.5) * (horizon_s / n_quad)
    h = horizon_s / n_quad
    comp_p = sum(nominal_intensity(float(t)) for t in grid) * h
    comp_q = sum(tilted_intensity(float(t)) for t in grid) * h
    log_lr -= comp_p - comp_q
    return float(log_lr)


@dataclass(slots=True)
class RareEventEstimate:
    """Unbiased importance-sampling rare-event estimate (doc R, Eq24)."""

    family: str
    probability: float  # p̂_F
    std_error: float  # MC standard error of p̂_F
    n_samples: int
    ess: float  # effective sample size of the weights
    hit_rate_q: float  # fraction of Q-samples in F_H (sanity check)


def importance_estimate(
    indicators: np.ndarray, log_weights: np.ndarray, family: str
) -> RareEventEstimate:
    """Compute ``p̂_F = mean(1{F_H} · L)`` and its standard error (doc R, Eq24).

    Works in log-space for the weights for numerical stability, then exponentiates
    against a shared offset. The effective sample size guards against weight
    degeneracy (doc R support_overlap_and_provenance_test / ESS).
    """
    indicators = np.asarray(indicators, dtype=float)
    log_weights = np.asarray(log_weights, dtype=float)
    n = indicators.shape[0]
    if n == 0:
        return RareEventEstimate(family, 0.0, 0.0, 0, 0.0, 0.0)
    # Stabilise: subtract max log-weight, exponentiate, rescale back.
    offset = float(np.max(log_weights))
    w = np.exp(log_weights - offset)
    contributions = indicators * w
    mean_c = float(contributions.mean())
    p_hat = mean_c * np.exp(offset)
    # MC variance of the (rescaled) contributions, mapped back through offset.
    var_c = float(contributions.var(ddof=1)) if n > 1 else 0.0
    se = np.sqrt(var_c / n) * np.exp(offset)
    ess = float((w.sum() ** 2) / (np.square(w).sum() + 1e-300))
    hit = float((indicators > 0).mean())
    return RareEventEstimate(
        family=family,
        probability=float(p_hat),
        std_error=float(se),
        n_samples=n,
        ess=ess,
        hit_rate_q=hit,
    )


@dataclass(slots=True)
class SubsetLevel:
    """One nested level of subset simulation / splitting (doc R, Eq25)."""

    threshold: float
    conditional_prob: float
    n_samples: int


def subset_simulation_probability(levels: list[SubsetLevel]) -> float:
    """Product estimator ``p_F = Π_k P(F_k | F_{k−1})`` (doc R, Eq25 / T4).

    For extremely small probabilities, nested events ``F_1 ⊃ … ⊃ F_K = F_H`` are
    defined by increasing thresholds on a severity score; the overall probability
    is the product of conditional probabilities. Consistency holds when the
    conditional samplers are ergodic (doc R, T4).
    """
    p = 1.0
    for lvl in levels:
        p *= max(lvl.conditional_prob, 0.0)
    return float(p)


def build_tilt(family: str, severity: float = 1.0) -> ExponentialTilt:
    """Construct a calibrated tilt for a rare-event family.

    The tilt strength is chosen so the proposal *dominates* the nominal process
    on the rare set (the support-inclusion condition of doc R, T3) without
    collapsing the ESS. ``severity`` scales the parent-level statistic ``g``.
    """
    eta_by_family = {
        "gpu_fault_burst": 2.5,
        "pods_ready_timeout": 1.5,
        "cooling_derate": 2.0,
        "power_derate": 1.5,
        "queue_burst": 1.2,
        "maintenance_mask": 0.0,  # deterministic guardrail, no hazard tilt
        "telemetry_degradation": 1.5,
    }
    return ExponentialTilt(eta=eta_by_family.get(family, 1.0), g_value=severity)


__all__ = [
    "RARE_EVENT_FAMILIES",
    "ExponentialTilt",
    "RareEventEstimate",
    "SubsetLevel",
    "build_tilt",
    "importance_estimate",
    "point_process_log_lr",
    "subset_simulation_probability",
]
