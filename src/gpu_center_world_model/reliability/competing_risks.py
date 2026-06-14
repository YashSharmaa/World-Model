# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Discrete-time cause-specific competing-risks hazards (doc U §1).

The planner needs *intervention-faithful first-event probabilities* conditional
on still being at risk after a candidate action — not a single undifferentiated
"failure score". We therefore model, on a planner-aligned discrete grid
``0 = tau_0 < tau_1 < ... < tau_M`` (doc U A1), a **multinomial law over
``{no_event} union K(u)``** for the *first* adverse event in each bucket, where
``K(u)`` is the entity-specific admissible cause set (doc U A2).

Design choices (doc U MODEL_CHOICE):

* **Cause-specific hazards with a softmax-over-{no-event}+causes link** (Eq1).
  The no-event logit is pinned to ``0``, guaranteeing simplex coherence at every
  step. We deliberately reject Fine-Gray subdistribution targets as the *control*
  head because their risk sets retain units after competing events, which makes
  them poor for "am I still at risk after this action?" reasoning (doc U
  why_rejected). Subdistribution-style CIFs are still produced — but derived from
  the cause-specific hazards via the exact discrete Aalen-Johansen recursion
  (Eq3), which is the correct reporting object.
* **Right-censored likelihood** (Eq4) under conditionally-independent censoring
  (A4): completion / drain / horizon truncation censor the open hazards.
* **Entity-specific heads** (doc U I1): GPU devices carry ``xid_fault``;
  workloads carry ``pods_ready_timeout`` and ``requeue``. The *algebra* is shared
  (one ``CauseSpecificHazardModel`` class), the *active causes* differ and are
  enforced by a structural mask.

Everything here is pure NumPy and vectorised over a batch of prediction units so
the same code serves both per-entity scoring at decision time and bulk
likelihood evaluation during offline replay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schema.entities import DeviceHealth
from ..schema.kpis import HazardChannel

# Entity-specific admissible cause sets (doc U A2 / I1). The order fixes the
# column order of every hazard array produced for that entity family.
GPU_CAUSES: tuple[HazardChannel, ...] = (HazardChannel.XID_FAULT,)
WORKLOAD_CAUSES: tuple[HazardChannel, ...] = (
    HazardChannel.PODS_READY_TIMEOUT,
    HazardChannel.REQUEUE,
)
#: Registry consumed by ``CauseSpecificHazardModel.for_entity``.
ENTITY_CAUSES: dict[str, tuple[HazardChannel, ...]] = {
    "gpu": GPU_CAUSES,
    "workload": WORKLOAD_CAUSES,
}


# ──────────────────────────────────────────────────────────────────────────────
# Link function (doc U Eq1)
# ──────────────────────────────────────────────────────────────────────────────
def hazards_from_logits(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Softmax-over-{no-event}+causes link (doc U Eq1).

    Given cause logits ``eta`` of shape ``(..., M, K)`` (the no-event baseline
    logit is pinned to ``0`` and not stored), return the per-bucket cause hazards
    ``h`` of shape ``(..., M, K)`` and the no-event survival-step probabilities
    ``q`` of shape ``(..., M)`` with the simplex identity
    ``q + sum_j h_j = 1`` holding exactly at every step.

    The computation is numerically stabilised by subtracting the row max across
    the augmented logit vector ``[0, eta_1, ..., eta_K]``.
    """
    logits = np.asarray(logits, dtype=np.float64)
    # Augment with the pinned no-event logit (0) as the first column.
    zeros = np.zeros((*logits.shape[:-1], 1), dtype=np.float64)
    aug = np.concatenate([zeros, logits], axis=-1)  # (..., M, K+1)
    aug = aug - np.max(aug, axis=-1, keepdims=True)
    exp = np.exp(aug)
    denom = np.sum(exp, axis=-1, keepdims=True)
    probs = exp / denom  # (..., M, K+1); column 0 is no-event
    q = probs[..., 0]
    h = probs[..., 1:]
    return h, q


def logits_from_hazards(
    h: np.ndarray, q: np.ndarray, *, eps: float = 1e-12
) -> np.ndarray:
    """Inverse link: recover cause logits ``eta`` from ``(h, q)`` (for testing).

    ``eta_j = log(h_j) - log(q)`` because the no-event logit is pinned to 0.
    """
    h = np.clip(np.asarray(h, dtype=np.float64), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, 1.0)
    return np.log(h) - np.log(q)[..., None]


# ──────────────────────────────────────────────────────────────────────────────
# Survival & cumulative incidence (doc U Eq2, Eq3, Eq5)
# ──────────────────────────────────────────────────────────────────────────────
def survival_from_q(q: np.ndarray) -> np.ndarray:
    """Discrete survival ``S(m) = prod_{r<=m} q_r`` (doc U Eq2).

    ``q`` has shape ``(..., M)``; the return has shape ``(..., M)`` with
    ``S(m) = prod_{r=1}^{m} q_r`` (so ``S`` is the survival *after* bucket m).
    """
    q = np.asarray(q, dtype=np.float64)
    return np.cumprod(q, axis=-1)


def cumulative_incidence(h: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Discrete Aalen-Johansen CIF ``F_j(m)=sum_{r<=m} S(r-1) h_{j,r}`` (Eq3).

    ``h`` has shape ``(..., M, K)`` and ``q`` has shape ``(..., M)``. The return
    has shape ``(..., M, K)``: the cumulative incidence of each cause up to and
    including bucket m. Because the modelled causes are mutually-exclusive first
    events, ``sum_j F_j(M) + S(M) = 1`` (doc U Eq5).
    """
    h = np.asarray(h, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    s_full = survival_from_q(q)  # S(r), r=1..M
    # S(r-1): prepend S(0)=1 and drop the last entry.
    ones = np.ones((*s_full.shape[:-1], 1), dtype=np.float64)
    s_prev = np.concatenate([ones, s_full[..., :-1]], axis=-1)  # (..., M)
    incremental = s_prev[..., None] * h  # (..., M, K)
    return np.cumsum(incremental, axis=-2)


def all_cause_risk(q: np.ndarray) -> np.ndarray:
    """All-cause cumulative risk ``1 - S(m)`` (doc U Eq5)."""
    return 1.0 - survival_from_q(q)


# ──────────────────────────────────────────────────────────────────────────────
# Right-censored likelihood (doc U Eq4)
# ──────────────────────────────────────────────────────────────────────────────
def censored_log_likelihood(
    h: np.ndarray,
    q: np.ndarray,
    *,
    event_bucket: np.ndarray,
    cause_index: np.ndarray,
    observed: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-unit discrete competing-risks log-likelihood (doc U Eq4).

    For unit ``u`` with observed-or-censored bucket ``Y_u = event_bucket`` (a
    1-based bucket index in ``1..M``), cause ``J_u = cause_index`` (ignored when
    censored), and event indicator ``Delta_u = observed``::

        L_u = [prod_{r < Y_u} q_r] * ( h_{J_u, Y_u}   if Delta_u = 1
                                       q_{Y_u}         if Delta_u = 0 )

    The leading product is the survival of *every fully-survived prior bucket*;
    an event then contributes its cause hazard, a (right-)censoring contributes
    the no-event step probability of the final at-risk bucket (A4: censoring is
    conditionally independent given history + actions).

    Parameters
    ----------
    h, q
        Hazards ``(B, M, K)`` and no-event steps ``(B, M)`` from the link.
    event_bucket
        1-based final bucket per unit, shape ``(B,)``. A value of ``M`` with
        ``observed=False`` denotes administrative censoring at the horizon.
    cause_index
        0-based cause column per unit, shape ``(B,)``. Unused for censored units.
    observed
        Boolean event indicator per unit, shape ``(B,)``.

    Returns
    -------
    np.ndarray
        Log-likelihood per unit, shape ``(B,)``.
    """
    h = np.clip(np.asarray(h, dtype=np.float64), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, 1.0)
    event_bucket = np.asarray(event_bucket, dtype=np.int64)
    cause_index = np.asarray(cause_index, dtype=np.int64)
    observed = np.asarray(observed, dtype=bool)

    b, m = q.shape
    log_q = np.log(q)  # (B, M)
    # Mask of buckets strictly before Y_u (fully survived): r in 1..Y_u-1.
    bucket_ix = np.arange(1, m + 1)[None, :]  # 1-based, (1, M)
    survived = bucket_ix < event_bucket[:, None]  # (B, M)
    ll = np.sum(np.where(survived, log_q, 0.0), axis=-1)  # (B,)

    rows = np.arange(b)
    last = np.clip(event_bucket - 1, 0, m - 1)  # 0-based final bucket
    # Final-bucket contribution: hazard if event, else no-event step prob.
    event_term = np.log(h[rows, last, np.clip(cause_index, 0, h.shape[-1] - 1)])
    censor_term = log_q[rows, last]
    ll = ll + np.where(observed, event_term, censor_term)
    return ll


# ──────────────────────────────────────────────────────────────────────────────
# Model container & entity heads
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class CauseSpecificForecast:
    """A horizon-resolved competing-risks forecast for one prediction unit.

    All arrays are indexed by the discrete bucket grid (length ``M``). Causes
    follow ``causes`` column order.
    """

    causes: tuple[HazardChannel, ...]
    hazard: np.ndarray  # (M, K) per-bucket cause hazards h_{j,m}
    no_event: np.ndarray  # (M,) per-bucket no-event step prob q_m
    survival: np.ndarray  # (M,) S(m) = prod q
    cif: np.ndarray  # (M, K) cumulative incidence F_j(m)

    def cause_risk_at(self, cause: HazardChannel, bucket: int) -> float:
        """Cause-specific horizon risk ``F_j(H)`` (doc U Eq5)."""
        j = self.causes.index(cause)
        b = int(np.clip(bucket, 1, self.cif.shape[0])) - 1
        return float(self.cif[b, j])

    def all_cause_risk_at(self, bucket: int) -> float:
        """All-cause horizon risk ``1 - S(H)`` (doc U Eq5)."""
        b = int(np.clip(bucket, 1, self.survival.shape[0])) - 1
        return float(1.0 - self.survival[b])


@dataclass(slots=True)
class CauseSpecificHazardModel:
    """Shared-algebra, entity-specific discrete competing-risks model (doc U I1).

    A *linear* cause-logit head ``eta_{j,m} = w_j · phi_m + b_j`` over a
    per-bucket feature matrix ``phi`` (shape ``(M, d)``) realises a correct,
    well-documented exact discrete-time competing-risks model that runs at demo
    pod scale. The same class instantiated with ``GPU_CAUSES`` / ``WORKLOAD_CAUSES``
    gives the GPU and workload heads; cross-entity coupling enters through the
    feature vector (graph/latent context), exactly as doc U I1 prescribes.

    The head is intentionally simple and inspectable rather than a black box —
    the heavy learned dynamics live in the residual world model; this module owns
    the *coherent survival algebra* around whatever logits are supplied. ``coef``
    and ``intercept`` are public so a trained backend may overwrite them, and
    :func:`from_logits` lets the model wrap externally-produced logits directly.
    """

    causes: tuple[HazardChannel, ...]
    coef: np.ndarray  # (K, d) cause-logit weights
    intercept: np.ndarray  # (K,) cause-logit biases

    @property
    def n_causes(self) -> int:
        return len(self.causes)

    # -- constructors --------------------------------------------------------
    @classmethod
    def for_entity(
        cls,
        entity_kind: str,
        n_features: int,
        *,
        coef: np.ndarray | None = None,
        intercept: np.ndarray | None = None,
    ) -> CauseSpecificHazardModel:
        """Build the head for ``"gpu"`` or ``"workload"`` (doc U I1)."""
        if entity_kind not in ENTITY_CAUSES:
            raise ValueError(
                f"no competing-risks head for entity {entity_kind!r}; "
                f"known: {sorted(ENTITY_CAUSES)}"
            )
        causes = ENTITY_CAUSES[entity_kind]
        k = len(causes)
        c = (
            np.zeros((k, n_features), dtype=np.float64)
            if coef is None
            else np.asarray(coef, dtype=np.float64)
        )
        b = (
            np.zeros(k, dtype=np.float64)
            if intercept is None
            else np.asarray(intercept, dtype=np.float64)
        )
        if c.shape != (k, n_features):
            raise ValueError(f"coef must be {(k, n_features)}, got {c.shape}")
        if b.shape != (k,):
            raise ValueError(f"intercept must be {(k,)}, got {b.shape}")
        return cls(causes=causes, coef=c, intercept=b)

    # -- forward -------------------------------------------------------------
    def logits(self, features: np.ndarray) -> np.ndarray:
        """Cause logits ``eta`` from a per-bucket feature matrix.

        ``features`` has shape ``(..., M, d)``; the return is ``(..., M, K)``.
        """
        features = np.asarray(features, dtype=np.float64)
        return features @ self.coef.T + self.intercept

    def forecast(self, features: np.ndarray) -> CauseSpecificForecast:
        """Full coherent forecast (hazards, survival, CIF) for one unit.

        ``features`` has shape ``(M, d)``.
        """
        eta = self.logits(features)  # (M, K)
        h, q = hazards_from_logits(eta)
        s = survival_from_q(q)
        f = cumulative_incidence(h, q)
        return CauseSpecificForecast(
            causes=self.causes, hazard=h, no_event=q, survival=s, cif=f
        )

    def forecast_from_logits(self, logits: np.ndarray) -> CauseSpecificForecast:
        """Wrap externally-produced logits ``(M, K)`` into a coherent forecast."""
        logits = np.asarray(logits, dtype=np.float64)
        if logits.shape[-1] != self.n_causes:
            raise ValueError(
                f"expected {self.n_causes} cause logits, got {logits.shape[-1]}"
            )
        h, q = hazards_from_logits(logits)
        return CauseSpecificForecast(
            causes=self.causes,
            hazard=h,
            no_event=q,
            survival=survival_from_q(q),
            cif=cumulative_incidence(h, q),
        )

    # -- likelihood ----------------------------------------------------------
    def log_likelihood(
        self,
        features: np.ndarray,
        *,
        event_bucket: np.ndarray,
        cause_index: np.ndarray,
        observed: np.ndarray,
    ) -> np.ndarray:
        """Right-censored batch log-likelihood (doc U Eq4).

        ``features`` has shape ``(B, M, d)``; the labels are per-unit arrays.
        """
        eta = self.logits(features)  # (B, M, K)
        h, q = hazards_from_logits(eta)
        return censored_log_likelihood(
            h,
            q,
            event_bucket=event_bucket,
            cause_index=cause_index,
            observed=observed,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Health-belief → device-hazard prior (bridges the HSMM to the GPU head)
# ──────────────────────────────────────────────────────────────────────────────
#: Monotone latent-health → per-bucket xid-fault prior (doc U I1 coupling). The
#: HSMM belief over :class:`DeviceHealth` supplies an interpretable hazard floor
#: that the competing-risks head conditions on. Values are *per-bucket* hazards.
DEVICE_HEALTH_XID_PRIOR: dict[DeviceHealth, float] = {
    DeviceHealth.HEALTHY: 1e-4,
    DeviceHealth.DEGRADED: 5e-3,
    DeviceHealth.INTERMITTENTLY_FAULTY: 3e-2,
    DeviceHealth.RESET_RECOVERY: 1e-2,
    DeviceHealth.MASKED: 0.0,  # exact-kernel masked: out of the at-risk set
}


def device_health_hazard_prior(belief: dict[DeviceHealth, float]) -> float:
    """Expected per-bucket xid hazard under a device-health belief (doc U I1).

    ``belief`` is a posterior over :class:`DeviceHealth` (need not be normalised);
    returns the belief-weighted per-bucket xid hazard used as the GPU head's
    intercept floor so quiet degradation raises forecast risk *before* any Xid
    fires.
    """
    total = sum(belief.values()) or 1.0
    return sum(
        DEVICE_HEALTH_XID_PRIOR.get(state, 0.0) * w / total
        for state, w in belief.items()
    )


__all__ = [
    "DEVICE_HEALTH_XID_PRIOR",
    "ENTITY_CAUSES",
    "GPU_CAUSES",
    "WORKLOAD_CAUSES",
    "CauseSpecificForecast",
    "CauseSpecificHazardModel",
    "all_cause_risk",
    "censored_log_likelihood",
    "cumulative_incidence",
    "device_health_hazard_prior",
    "hazards_from_logits",
    "logits_from_hazards",
    "survival_from_q",
]
