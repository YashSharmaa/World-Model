# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Multi-component novelty / OOD detection with conformal FDR control (doc L).

Doc L (ood_detection) specifies a *four-component* novelty score fused into one
conformal novelty test. The components measure different ways a decision window
can be out-of-support:

* ``N1`` — **latent kNN distance**: distance of the action-conditioned latent
  embedding to its ``k`` nearest training embeddings.
* ``N2`` — **topology-signature distance**: distance over rack/domain/flavor
  graph fingerprints (a structural fingerprint of the pod the candidate targets).
* ``N3`` — **workload-mixture novelty**: novelty of the regime / workload-mixture
  embedding (arrival burst, comm-class mix, priority spread).
* ``N4`` — **epistemic + staleness**: ensemble disagreement plus telemetry
  staleness / source-health degradation (an incident often *is* missing data).

They are standardised against the ID calibration set and fused by ``max`` so any
single strongly-novel axis triggers caution. The fused score is calibrated into a
**conformal p-value** ``p = (1 + #{S_cal >= S}) / (n_cal + 1)`` which controls
the in-support false-alarm rate at level ``alpha_ood`` under exchangeability.
When many candidates / windows are screened at once, p-values are passed through
**Benjamini-Hochberg** FDR control (doc L: convert to q-values, control FDR).

Safety rule (doc L): an OOD flag feeds **abstention / conservative fallback**,
never a more aggressive recommendation. What is calibrated: in-support false
alarms and batched FDR. What remains empirical: recall on genuinely unseen
regimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Component novelty scores
# ─────────────────────────────────────────────────────────────────────────────
def knn_distance(query: np.ndarray, reference: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean distance to the ``k`` nearest reference embeddings (``N1``).

    ``query`` is ``(n_q, d)``, ``reference`` is ``(n_r, d)``. Returns ``(n_q,)``.
    Uses the squared-Euclidean expansion ``|x|^2 + |y|^2 - 2 x.y`` for an
    O(n_q n_r d) computation that is fine at demo-pod scale (a few thousand
    reference embeddings).
    """
    q = np.atleast_2d(np.asarray(query, float))
    r = np.atleast_2d(np.asarray(reference, float))
    if r.shape[0] == 0:
        return np.full(q.shape[0], np.inf)
    k = min(k, r.shape[0])
    qn = np.sum(q * q, axis=1, keepdims=True)
    rn = np.sum(r * r, axis=1, keepdims=True).T
    d2 = np.maximum(qn + rn - 2.0 * q @ r.T, 0.0)
    # partial sort: k smallest squared distances per row.
    part = np.partition(d2, kth=k - 1, axis=1)[:, :k]
    return np.sqrt(part).mean(axis=1)


def topology_signature_distance(
    query_sig: np.ndarray, reference_sigs: np.ndarray
) -> np.ndarray:
    """Min distance of a topology fingerprint to known fingerprints (``N2``).

    A topology signature is a fixed-length fingerprint of the rack/domain/flavor
    graph (e.g. sorted free-capacity histogram, fanout counts). Novelty is the
    distance to the *closest* previously-seen signature.
    """
    q = np.atleast_2d(np.asarray(query_sig, float))
    r = np.atleast_2d(np.asarray(reference_sigs, float))
    if r.shape[0] == 0:
        return np.full(q.shape[0], np.inf)
    qn = np.sum(q * q, axis=1, keepdims=True)
    rn = np.sum(r * r, axis=1, keepdims=True).T
    d2 = np.maximum(qn + rn - 2.0 * q @ r.T, 0.0)
    return np.sqrt(d2.min(axis=1))


def workload_mixture_novelty(
    query_mix: np.ndarray, reference_mix: np.ndarray, k: int = 5
) -> np.ndarray:
    """Workload / regime-mixture novelty (``N3``) — kNN in mixture space.

    Mixture vectors are regime embeddings (arrival-burst class, comm-class mix,
    priority spread, workload-family proportions). Reuses the kNN primitive.
    """
    return knn_distance(query_mix, reference_mix, k=k)


def staleness_epistemic_score(
    epistemic: np.ndarray | float,
    staleness_age_s: np.ndarray | float,
    source_health_bad_prob: np.ndarray | float,
    age_scale_s: float = 60.0,
) -> np.ndarray:
    """Epistemic + staleness + source-health novelty (``N4``).

    Combines ensemble disagreement with a saturating staleness term
    ``1 - exp(-age / age_scale)`` and the source-health "bad" probability. A
    blackout / degraded-telemetry incident therefore *raises* novelty even when
    the mean forecast barely moves (doc T C15: missingness widens risk bands).
    """
    epi = np.asarray(epistemic, float)
    age = np.asarray(staleness_age_s, float)
    bad = np.asarray(source_health_bad_prob, float)
    stale_term = 1.0 - np.exp(-np.maximum(age, 0.0) / max(age_scale_s, _EPS))
    return epi + stale_term + bad


# ─────────────────────────────────────────────────────────────────────────────
# Fusion + conformal calibration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class _Standardizer:
    """Robust per-component standardiser (median / MAD) fit on the ID set."""

    center: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scale: np.ndarray = field(default_factory=lambda: np.ones(0))

    def fit(self, x: np.ndarray) -> _Standardizer:
        x = np.atleast_2d(np.asarray(x, float))
        self.center = np.median(x, axis=0)
        mad = np.median(np.abs(x - self.center), axis=0)
        # 1.4826 makes MAD a consistent sd estimator under normality.
        self.scale = np.where(mad > _EPS, 1.4826 * mad, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, float))
        return (x - self.center) / self.scale


def fuse_components(components: np.ndarray, standardizer: _Standardizer) -> np.ndarray:
    """Fuse standardised components by ``max`` (doc L ``S_ood``).

    ``components`` is ``(n, n_components)``; returns ``(n,)`` fused scores. The
    ``max`` makes the detector trip if *any* axis is strongly novel, which is the
    conservative, safety-first choice for a gate that only ever abstains.
    """
    z = standardizer.transform(components)
    return z.max(axis=1)


@dataclass(slots=True)
class OODDetector:
    """Conformal novelty detector over the fused multi-component score (doc L).

    Fit on an ID calibration set of raw component vectors; at decision time
    :meth:`p_value` returns the conformal p-value and :meth:`is_novel` thresholds
    it at ``alpha_ood``. For batched screening use :meth:`flag_batch` which adds
    Benjamini-Hochberg FDR control.
    """

    alpha_ood: float = 0.05
    standardizer: _Standardizer = field(default_factory=_Standardizer)
    cal_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def fit(self, cal_components: np.ndarray) -> OODDetector:
        comp = np.atleast_2d(np.asarray(cal_components, float))
        self.standardizer = _Standardizer().fit(comp)
        self.cal_scores = fuse_components(comp, self.standardizer)
        return self

    def score(self, components: np.ndarray) -> np.ndarray:
        return fuse_components(
            np.atleast_2d(np.asarray(components, float)), self.standardizer
        )

    def p_value(self, components: np.ndarray) -> np.ndarray:
        """Conformal novelty p-value ``(1 + #{S_cal >= S}) / (n_cal + 1)``.

        Under exchangeability of ID points this controls in-support false alarms
        at any threshold ``alpha_ood`` (doc L). Larger fused score -> smaller p.
        """
        s = self.score(components)
        n = self.cal_scores.size
        if n == 0:
            return np.ones_like(s)
        # vectorised: count cal scores >= each test score.
        cal_sorted = np.sort(self.cal_scores)
        # number of cal >= s  == n - searchsorted(cal_sorted, s, 'left')
        ge = n - np.searchsorted(cal_sorted, s, side="left")
        return (1.0 + ge) / (n + 1.0)

    def is_novel(self, components: np.ndarray) -> np.ndarray:
        """Flag ``p <= alpha_ood`` (single / per-candidate test)."""
        return self.p_value(components) <= self.alpha_ood

    def flag_batch(
        self, components: np.ndarray, fdr: float | None = None
    ) -> np.ndarray:
        """Batched novelty flags with Benjamini-Hochberg FDR control (doc L).

        When evaluating many candidate actions / windows simultaneously, naive
        per-test thresholding inflates false discoveries. BH at level ``fdr``
        (defaults to ``alpha_ood``) controls the expected proportion of false
        novelty flags. Returns a boolean flag array.
        """
        p = self.p_value(components)
        return benjamini_hochberg(p, fdr if fdr is not None else self.alpha_ood)


def benjamini_hochberg(pvalues: np.ndarray, fdr: float) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR control; returns a rejection mask.

    Reject the hypotheses with the ``k`` smallest p-values where ``k`` is the
    largest ``i`` such that ``p_(i) <= i/m * fdr``. Controls FDR at ``fdr`` under
    independence / PRDS. Conformal p-values are valid inputs (doc L).
    """
    p = np.asarray(pvalues, float).ravel()
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, m + 1) / m) * fdr
    below = ranked <= thresh
    if not below.any():
        return np.zeros(m, dtype=bool)
    k = np.max(np.nonzero(below)[0])  # largest index passing (0-based)
    reject = np.zeros(m, dtype=bool)
    reject[order[: k + 1]] = True
    return reject


def build_components(
    *,
    latent_knn: np.ndarray,
    topology_dist: np.ndarray,
    mixture_novelty: np.ndarray,
    staleness_epistemic: np.ndarray,
) -> np.ndarray:
    """Stack the four novelty components into an ``(n, 4)`` matrix (N1-N4).

    Convenience assembler so callers feed named component arrays and get the
    matrix :class:`OODDetector` expects, in the canonical doc L order.
    """
    cols = [
        np.asarray(latent_knn, float).ravel(),
        np.asarray(topology_dist, float).ravel(),
        np.asarray(mixture_novelty, float).ravel(),
        np.asarray(staleness_epistemic, float).ravel(),
    ]
    n = max(c.size for c in cols)
    cols = [np.full(n, c.item()) if c.size == 1 else c for c in cols]
    return np.column_stack(cols)


__all__ = [
    "OODDetector",
    "benjamini_hochberg",
    "build_components",
    "fuse_components",
    "knn_distance",
    "staleness_epistemic_score",
    "topology_signature_distance",
    "workload_mixture_novelty",
]
