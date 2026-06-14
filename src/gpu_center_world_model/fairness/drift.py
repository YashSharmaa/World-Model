# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The fairness-drift KPI from Kueue-native observables (doc V, drift section).

The drift KPI measures how far the scheduler has drifted from its *own*
weighted-share target, at the two layers Kueue exposes:

* **Borrowing layer (cohort)** — ``ClusterQueue.weighted_share`` (Kueue
  ``kueue_cluster_queue_weighted_share``): the max over resources of usage above
  nominal quota, divided by lendable cohort resources and by the queue weight;
  zero means below nominal quota (doc V, D21). Drift is the spread of weighted
  shares across the ClusterQueues that are *borrowing / contending* in a cohort
  (Eq12):

      Delta_CQ_c(tau) = max_{q in B_c} s_q(tau) - min_{q in B_c} s_q(tau).

* **Admission layer (LocalQueue)** — Kueue Admission Fair Sharing orders
  workloads by decayed historical LocalQueue usage; drift is the spread of those
  history shares across LocalQueues actively contending inside a ClusterQueue
  (D22, Eq13):

      Delta_LQ_k(tau) = max_{l in A_k} h_l(tau) - min_{l in A_k} h_l(tau).

The horizon-integrated KPI (Eq14) blends the two layers with weight ``alpha``
(``alpha = 1`` if Admission Fair Sharing is disabled):

    FD_H = (1/H) ∫_0^H [ alpha · mean_c Delta_CQ_c + (1-alpha) · mean_k Delta_LQ_k ] dτ.

This is preferred over abstract Gini-only metrics because it is grounded in
Kueue's quota / borrowing / decayed-history semantics, and over instantaneous
envy gaps because those are too brittle for hold/admit/place decisions and
ignore the platform's explicitly historical fair-sharing (doc V rejected
alternatives). The KPI is reported as mean / p95 / max, and the planner consumes
the **action-conditioned delta vs hold** (D23):

    delta_FD(a | s) = E[FD_H | s, a] - E[FD_H | s, hold].
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..schema.state import SymbolicState, WorldState

# ``np.trapezoid`` is the NumPy 2.x name; ``np.trapz`` is the 1.x name. Bind once
# so the module is import-clean on either line of NumPy in CI.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]


@dataclass(slots=True)
class DriftSample:
    """A single-timestep drift observation across both fairness layers (Eq12-13).

    Attributes
    ----------
    ts_s:
        Sample wall-clock time in seconds (used for horizon integration).
    cohort_spreads:
        ``cohort_id -> Delta_CQ_c`` — borrowing-layer weighted-share spread.
    cluster_queue_spreads:
        ``cluster_queue_id -> Delta_LQ_k`` — admission-layer history spread.
    """

    ts_s: float
    cohort_spreads: dict[str, float] = field(default_factory=dict)
    cluster_queue_spreads: dict[str, float] = field(default_factory=dict)

    def blended(self, alpha: float) -> float:
        """Eq14 integrand here: ``alpha·mean Delta_CQ + (1-alpha)·mean Delta_LQ``."""
        cq = (
            float(np.mean(list(self.cohort_spreads.values())))
            if self.cohort_spreads
            else 0.0
        )
        lq = (
            float(np.mean(list(self.cluster_queue_spreads.values())))
            if self.cluster_queue_spreads
            else 0.0
        )
        return alpha * cq + (1.0 - alpha) * lq


@dataclass(slots=True)
class FairnessDrift:
    """Horizon-integrated fairness-drift KPI ``FD_H`` (doc V, Eq14).

    Reports the time-averaged blended drift plus its p95 and max over the
    horizon, matching the project's KPI-distribution convention (never a bare
    point estimate).
    """

    horizon_s: float
    alpha: float
    mean: float
    p95: float
    max: float
    n_samples: int
    cohort_mean: float = 0.0
    cluster_queue_mean: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "fd_mean": self.mean,
            "fd_p95": self.p95,
            "fd_max": self.max,
            "fd_cohort_mean": self.cohort_mean,
            "fd_local_queue_mean": self.cluster_queue_mean,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Instantaneous spreads from observables (Eq12 / Eq13)
# ──────────────────────────────────────────────────────────────────────────────
def cohort_drift(
    weighted_share_by_cq: Mapping[str, float],
    cohort_of_cq: Mapping[str, str],
    *,
    contending: Sequence[str] | None = None,
) -> dict[str, float]:
    """Borrowing-layer drift ``Delta_CQ_c`` per cohort (doc V, Eq12).

    Parameters
    ----------
    weighted_share_by_cq:
        ``cluster_queue_id -> s_q`` (Kueue ``kueue_cluster_queue_weighted_share``).
    cohort_of_cq:
        ``cluster_queue_id -> cohort_id`` membership.
    contending:
        Restrict to ClusterQueues that are borrowing / contending for borrowed
        quota ``B_c(tau)``. ``None`` ⇒ use every queue with a recorded share.

    Returns
    -------
    ``cohort_id -> (max share - min share)`` over the contending members.
    """
    contend = set(contending) if contending is not None else None
    buckets: dict[str, list[float]] = {}
    for cq, share in weighted_share_by_cq.items():
        if contend is not None and cq not in contend:
            continue
        cohort = cohort_of_cq.get(cq)
        if cohort is None:
            continue
        buckets.setdefault(cohort, []).append(float(share))
    return {
        cohort: (max(vals) - min(vals)) if len(vals) > 1 else 0.0
        for cohort, vals in buckets.items()
    }


def local_queue_drift(
    history_share_by_lq: Mapping[str, float],
    cluster_queue_of_lq: Mapping[str, str],
    *,
    contending: Sequence[str] | None = None,
) -> dict[str, float]:
    """Admission-layer drift ``Delta_LQ_k`` per ClusterQueue (doc V, Eq13).

    Parameters
    ----------
    history_share_by_lq:
        ``local_queue_id -> h_l`` — the LocalQueue Admission Fair Sharing
        weightedShare (``status.fairSharing.weightedShare`` or its decayed
        consumed-resources equivalent).
    cluster_queue_of_lq:
        ``local_queue_id -> cluster_queue_id`` membership.
    contending:
        Restrict to LocalQueues actively contending inside the ClusterQueue
        ``A_k(tau)``. ``None`` ⇒ use every LocalQueue with a recorded share.
    """
    contend = set(contending) if contending is not None else None
    buckets: dict[str, list[float]] = {}
    for lq, share in history_share_by_lq.items():
        if contend is not None and lq not in contend:
            continue
        cq = cluster_queue_of_lq.get(lq)
        if cq is None:
            continue
        buckets.setdefault(cq, []).append(float(share))
    return {
        cq: (max(vals) - min(vals)) if len(vals) > 1 else 0.0
        for cq, vals in buckets.items()
    }


def drift_sample_from_state(
    state: SymbolicState | WorldState,
    *,
    ts_s: float | None = None,
    cohort_contending: Sequence[str] | None = None,
    lq_contending: Sequence[str] | None = None,
) -> DriftSample:
    """Read a :class:`DriftSample` directly from the symbolic world state.

    Uses the Kueue observables retained on the schema (doc V, H2):
    :attr:`ClusterQueue.weighted_share` for the borrowing layer and
    :attr:`LocalQueue.fairshare_weighted_share` for the admission layer. A
    ClusterQueue is treated as borrowing/contending when ``borrowed_gpu > 0`` or
    ``pending_gpu > 0`` (override via ``cohort_contending``); a LocalQueue when
    ``pending_ct > 0`` (override via ``lq_contending``).
    """
    sym = state.symbolic if isinstance(state, WorldState) else state
    if ts_s is None:
        ts_s = float(state.ts_ns) / 1e9 if isinstance(state, WorldState) else 0.0

    ws_by_cq: dict[str, float] = {}
    cohort_of_cq: dict[str, str] = {}
    auto_cohort_contend: list[str] = []
    for cq in sym.cluster_queues.values():
        ws_by_cq[cq.name] = float(cq.weighted_share)
        if cq.cohort_id is not None:
            cohort_of_cq[cq.name] = cq.cohort_id
        if cq.borrowed_gpu > 0 or cq.pending_gpu > 0:
            auto_cohort_contend.append(cq.name)

    h_by_lq: dict[str, float] = {}
    cq_of_lq: dict[str, str] = {}
    auto_lq_contend: list[str] = []
    for lq in sym.local_queues.values():
        h_by_lq[lq.name] = float(lq.fairshare_weighted_share)
        cq_of_lq[lq.name] = lq.cluster_queue_ref
        if lq.pending_ct > 0:
            auto_lq_contend.append(lq.name)

    cohort_spreads = cohort_drift(
        ws_by_cq,
        cohort_of_cq,
        contending=cohort_contending
        if cohort_contending is not None
        else (auto_cohort_contend or None),
    )
    lq_spreads = local_queue_drift(
        h_by_lq,
        cq_of_lq,
        contending=lq_contending
        if lq_contending is not None
        else (auto_lq_contend or None),
    )
    return DriftSample(
        ts_s=ts_s,
        cohort_spreads=cohort_spreads,
        cluster_queue_spreads=lq_spreads,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Horizon integration (Eq14)
# ──────────────────────────────────────────────────────────────────────────────
def fairness_drift_kpi(
    samples: Sequence[DriftSample],
    *,
    horizon_s: float | None = None,
    alpha: float = 1.0,
) -> FairnessDrift:
    """Integrate samples into the horizon fairness-drift KPI ``FD_H`` (Eq14).

    The time average is computed with the trapezoidal rule over the sample times
    so unevenly-spaced telemetry is handled correctly; if a single sample is
    given (or all share one timestamp) the instantaneous blended value is used.
    ``alpha`` blends borrowing-layer vs admission-layer drift; set ``alpha = 1``
    when Admission Fair Sharing is disabled (doc V, Eq14).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if not samples:
        return FairnessDrift(
            horizon_s=horizon_s or 0.0,
            alpha=alpha,
            mean=0.0,
            p95=0.0,
            max=0.0,
            n_samples=0,
        )

    ordered = sorted(samples, key=lambda s: s.ts_s)
    ts = np.array([s.ts_s for s in ordered], dtype=float)
    blended = np.array([s.blended(alpha) for s in ordered], dtype=float)
    cohort_vals = np.array(
        [
            float(np.mean(list(s.cohort_spreads.values()))) if s.cohort_spreads else 0.0
            for s in ordered
        ]
    )
    lq_vals = np.array(
        [
            float(np.mean(list(s.cluster_queue_spreads.values())))
            if s.cluster_queue_spreads
            else 0.0
            for s in ordered
        ]
    )

    span = float(ts[-1] - ts[0])
    if span > 0:
        mean = float(_trapezoid(blended, ts) / span)
        cohort_mean = float(_trapezoid(cohort_vals, ts) / span)
        lq_mean = float(_trapezoid(lq_vals, ts) / span)
    else:
        mean = float(np.mean(blended))
        cohort_mean = float(np.mean(cohort_vals))
        lq_mean = float(np.mean(lq_vals))

    return FairnessDrift(
        horizon_s=horizon_s if horizon_s is not None else (span or 0.0),
        alpha=alpha,
        mean=mean,
        p95=float(np.percentile(blended, 95)),
        max=float(np.max(blended)),
        n_samples=len(ordered),
        cohort_mean=cohort_mean,
        cluster_queue_mean=lq_mean,
    )


def action_conditioned_drift_delta(
    fd_action: FairnessDrift,
    fd_hold: FairnessDrift,
) -> float:
    """Action-conditioned drift delta vs hold ``delta_FD(a|s)`` (doc V, D23).

    Positive ⇒ the action *worsens* fairness drift relative to holding; negative
    ⇒ it repairs drift. The planner uses this as the budgeted fairness term, not
    as a weighted-sum penalty (doc V, Eq11 / rejected scalarization).
    """
    return float(fd_action.mean - fd_hold.mean)


__all__ = [
    "DriftSample",
    "FairnessDrift",
    "action_conditioned_drift_delta",
    "cohort_drift",
    "drift_sample_from_state",
    "fairness_drift_kpi",
    "local_queue_drift",
]
