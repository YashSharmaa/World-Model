# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Fidelity-certification + poisoning-detection suite (design doc R).

A synthetic corpus is only trustworthy once it passes a *multi-metric* battery —
a single discriminator is not enough (doc R, energy_distance_test evidence). This
module ships pure-NumPy implementations of the doc R suite so it runs on plain
Linux CI without sklearn:

Certification:
* :func:`exact_kernel_conformance` — deterministic-zero audit: synthetic
  feasibility/blockers must match the exact kernel (doc R, T2 / kernel_conformance).
* :func:`mmd2_test` — kernel two-sample MMD² on window embeddings with a
  permutation null (doc R, T6).
* :func:`energy_distance_test` — energy-distance two-sample test (doc R, T7).
* :func:`classifier_two_sample_test` — a logistic origin classifier; AUC ≈ 0.5
  under ``P_real = P_syn`` (doc R, T8).
* :func:`tstr_trts` — train-on-synthetic/test-on-real utility hook (doc R).

Poisoning detection (the failure modes of doc D risks):
* :func:`lag_structure_test` — phase→util / power→temp lag preservation.
* :func:`incident_bundle_test` — fault incidents must be coherent bundles.
* :func:`predecision_leakage_test` — no post-action features leak pre-t0.

All tests return a typed result with a pass/fail flag and the statistic so the
data card can publish them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ...schema.actions import Action
from ...schema.state import WorldState
from ...simulator.contracts import FeasibilityResult


@dataclass(slots=True)
class TestResult:
    """A single fidelity / poisoning test outcome."""

    name: str
    passed: bool
    statistic: float
    p_value: float | None = None
    detail: dict[str, float] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Window embeddings
# ──────────────────────────────────────────────────────────────────────────────
def embed_windows(features: list[dict[str, float]], keys: list[str]) -> np.ndarray:
    """Stack window feature dicts into a dense embedding matrix ``φ(W)`` (doc R).

    A robust, fixed-order projection over the named keys; missing keys are 0 but
    that is a *feature absence*, not zero-imputed telemetry (these are derived
    summaries, not raw channels).
    """
    if not features:
        return np.zeros((0, len(keys)))
    return np.array(
        [[float(f.get(k, 0.0)) for k in keys] for f in features], dtype=float
    )


def _standardize(x: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardise both samples by the pooled mean/std (shared embedding space)."""
    pooled = np.concatenate([x, ref], axis=0) if ref.size else x
    mu = pooled.mean(axis=0, keepdims=True)
    sd = pooled.std(axis=0, keepdims=True) + 1e-9
    return (x - mu) / sd, (ref - mu) / sd


# ──────────────────────────────────────────────────────────────────────────────
# Certification: exact-kernel conformance (doc R, T2)
# ──────────────────────────────────────────────────────────────────────────────
def exact_kernel_conformance(
    cases: list[tuple[WorldState, Action]],
    generator_feasible: Callable[[WorldState, Action], FeasibilityResult],
    oracle_feasible: Callable[[WorldState, Action], FeasibilityResult],
) -> TestResult:
    """Deterministic-zero audit of feasibility + blocker sets (doc R, T2).

    Because the exact kernel is deterministic and topology/quota/gang invariants
    are not randomised, any nonzero violation is a hard failure, never modelled
    uncertainty. Compares the boolean feasibility and the blocker-group set.
    """
    violations = 0
    for state, action in cases:
        g = generator_feasible(state, action)
        o = oracle_feasible(state, action)
        if g.feasible != o.feasible:
            violations += 1
            continue
        gset = {b.group.value for b in g.blockers}
        oset = {b.group.value for b in o.blockers}
        if gset != oset:
            violations += 1
    return TestResult(
        name="exact_kernel_conformance",
        passed=violations == 0,
        statistic=float(violations),
        detail={"n_cases": float(len(cases))},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Certification: MMD² two-sample test (doc R, T6)
# ──────────────────────────────────────────────────────────────────────────────
def _rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    d2 = np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None, :] - 2.0 * x @ y.T
    return np.exp(-gamma * np.maximum(d2, 0.0))


def _median_heuristic_gamma(z: np.ndarray) -> float:
    if z.shape[0] < 2:
        return 1.0
    # subsample for speed on larger corpora
    m = min(z.shape[0], 200)
    idx = np.linspace(0, z.shape[0] - 1, m).astype(int)
    s = z[idx]
    d2 = np.sum(s * s, axis=1)[:, None] + np.sum(s * s, axis=1)[None, :] - 2.0 * s @ s.T
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    return 1.0 / (med + 1e-9)


def mmd2_unbiased(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    """Unbiased MMD² estimate with an RBF (characteristic) kernel (doc R, T6)."""
    if x.shape[0] < 2 or y.shape[0] < 2:
        return 0.0
    pooled = np.concatenate([x, y], axis=0)
    g = gamma if gamma is not None else _median_heuristic_gamma(pooled)
    kxx = _rbf_kernel(x, x, g)
    kyy = _rbf_kernel(y, y, g)
    kxy = _rbf_kernel(x, y, g)
    m, n = x.shape[0], y.shape[0]
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    term_x = kxx.sum() / (m * (m - 1))
    term_y = kyy.sum() / (n * (n - 1))
    term_xy = 2.0 * kxy.mean()
    return float(term_x + term_y - term_xy)


def mmd2_test(
    real: np.ndarray,
    syn: np.ndarray,
    *,
    n_perm: int = 200,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> TestResult:
    """MMD² permutation two-sample test (doc R, T6 — exact finite-sample level).

    Under exchangeability (``P_real = P_syn``) the permutation distribution gives
    an exact level. ``passed`` means we *fail to reject* equality (high fidelity);
    a low p-value flags a distribution mismatch.
    """
    rng = rng or np.random.default_rng(0)
    xs, ys = _standardize(real, syn)
    if xs.shape[0] < 2 or ys.shape[0] < 2:
        return TestResult("mmd2", True, 0.0, 1.0)
    pooled = np.concatenate([xs, ys], axis=0)
    g = _median_heuristic_gamma(pooled)
    observed = mmd2_unbiased(xs, ys, g)
    m = xs.shape[0]
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled.shape[0])
        a = pooled[perm[:m]]
        b = pooled[perm[m:]]
        if mmd2_unbiased(a, b, g) >= observed:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return TestResult("mmd2", passed=p >= alpha, statistic=observed, p_value=p)


# ──────────────────────────────────────────────────────────────────────────────
# Certification: energy-distance two-sample test (doc R, T7)
# ──────────────────────────────────────────────────────────────────────────────
def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Empirical energy distance (a proper metric on the embedding; doc R, T7)."""
    if x.shape[0] == 0 or y.shape[0] == 0:
        return 0.0

    def pdist(a: np.ndarray, b: np.ndarray) -> float:
        d = np.sqrt(
            np.maximum(
                np.sum(a * a, axis=1)[:, None]
                + np.sum(b * b, axis=1)[None, :]
                - 2.0 * a @ b.T,
                0.0,
            )
        )
        return float(d.mean())

    return 2.0 * pdist(x, y) - pdist(x, x) - pdist(y, y)


def energy_distance_test(
    real: np.ndarray,
    syn: np.ndarray,
    *,
    n_perm: int = 200,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> TestResult:
    """Energy-distance permutation test (doc R, T7)."""
    rng = rng or np.random.default_rng(0)
    xs, ys = _standardize(real, syn)
    if xs.shape[0] == 0 or ys.shape[0] == 0:
        return TestResult("energy_distance", True, 0.0, 1.0)
    observed = energy_distance(xs, ys)
    pooled = np.concatenate([xs, ys], axis=0)
    m = xs.shape[0]
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled.shape[0])
        if energy_distance(pooled[perm[:m]], pooled[perm[m:]]) >= observed:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return TestResult(
        "energy_distance", passed=p >= alpha, statistic=observed, p_value=p
    )


# ──────────────────────────────────────────────────────────────────────────────
# Certification: classifier two-sample test (doc R, T8)
# ──────────────────────────────────────────────────────────────────────────────
def _logistic_fit(
    X: np.ndarray, y: np.ndarray, *, epochs: int = 200, lr: float = 0.1
) -> np.ndarray:
    """Tiny L2-regularised logistic regression (pure NumPy, doc R fallback)."""
    n, d = X.shape
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
    w = np.zeros(d + 1)
    for _ in range(epochs):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = Xb.T @ (p - y) / n + 1e-3 * w
        w -= lr * grad
    return w


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the rank statistic (Mann-Whitney U)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, order.size + 1)
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def classifier_two_sample_test(
    real: np.ndarray,
    syn: np.ndarray,
    *,
    auc_tol: float = 0.65,
    rng: np.random.Generator | None = None,
) -> TestResult:
    """C2ST: held-out origin-classifier AUC should revert to chance (doc R, T8).

    Trains a logistic classifier on a 50/50 split to distinguish real vs.
    synthetic; ``passed`` when held-out AUC ≤ ``auc_tol`` (close to 0.5 ⇒ the two
    samples are indistinguishable ⇒ high fidelity). Uses scikit-learn if present,
    otherwise the NumPy logistic fallback above so CI stays dependency-light.
    """
    rng = rng or np.random.default_rng(0)
    xs, ys = _standardize(real, syn)
    if xs.shape[0] < 4 or ys.shape[0] < 4:
        return TestResult("classifier_two_sample", True, 0.5)
    X = np.concatenate([xs, ys], axis=0)
    y = np.concatenate([np.ones(xs.shape[0]), np.zeros(ys.shape[0])])
    idx = rng.permutation(X.shape[0])
    X, y = X[idx], y[idx]
    cut = X.shape[0] // 2
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    auc = _classifier_auc(Xtr, ytr, Xte, yte)
    return TestResult(
        "classifier_two_sample",
        passed=auc <= auc_tol,
        statistic=auc,
        detail={"auc": auc},
    )


def _classifier_auc(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray
) -> float:
    try:  # optional sklearn for a stronger classifier
        from sklearn.linear_model import LogisticRegression  # type: ignore

        clf = LogisticRegression(max_iter=500)
        clf.fit(Xtr, ytr)
        scores = clf.predict_proba(Xte)[:, 1]
    except ImportError:
        w = _logistic_fit(Xtr, ytr)
        Xb = np.concatenate([Xte, np.ones((Xte.shape[0], 1))], axis=1)
        scores = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
    return _auc(scores, yte)


def tstr_trts(
    train_fn: Callable[[np.ndarray, np.ndarray], object],
    score_fn: Callable[[object, np.ndarray, np.ndarray], float],
    real_X: np.ndarray,
    real_y: np.ndarray,
    syn_X: np.ndarray,
    syn_y: np.ndarray,
) -> TestResult:
    """TSTR/TRTS predictive-utility diagnostic (doc R; external-validity hook).

    ΔTSTR = Perf(real-trained on real-test) − Perf(synthetic-trained on real-test).
    Caller supplies a learner ``train_fn`` and a metric ``score_fn``; this stays a
    *diagnostic*, not a hypothesis test (doc R guarantee).
    """
    m_real = train_fn(real_X, real_y)
    m_syn = train_fn(syn_X, syn_y)
    perf_rr = score_fn(m_real, real_X, real_y)
    perf_sr = score_fn(m_syn, real_X, real_y)
    delta_tstr = perf_rr - perf_sr
    perf_ss = score_fn(m_syn, syn_X, syn_y)
    perf_rs = score_fn(m_real, syn_X, syn_y)
    delta_trts = perf_ss - perf_rs
    return TestResult(
        "tstr_trts",
        passed=abs(delta_tstr) <= 0.15,
        statistic=delta_tstr,
        detail={"delta_tstr": delta_tstr, "delta_trts": delta_trts},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Poisoning detection
# ──────────────────────────────────────────────────────────────────────────────
def _lagged_xcorr(parent: np.ndarray, child: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised cross-correlation of child vs. lagged parent, lags 0..max_lag."""
    p = (parent - parent.mean()) / (parent.std() + 1e-9)
    c = (child - child.mean()) / (child.std() + 1e-9)
    out = np.zeros(max_lag + 1)
    n = len(p)
    for lag in range(max_lag + 1):
        if n - lag <= 1:
            continue
        out[lag] = float(np.mean(p[: n - lag] * c[lag:]))
    return out


def lag_structure_test(
    real_pairs: list[tuple[np.ndarray, np.ndarray]],
    syn_pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    max_lag: int = 8,
    tol: float = 0.3,
) -> TestResult:
    """Lag-signature preservation for parent→child bundles (doc R, lag_structure).

    Targets the "independent channel randomization breaks lag structure" failure
    (doc D risks #1). For each (parent, child) series pair (e.g. power→temp,
    phase→util) it compares the lag-correlation signature and the *argmax lag*
    (the causal delay). Poisoning is flagged when the synthetic peak lag reverses
    or the signature deviates beyond ``tol``.
    """
    if not real_pairs or not syn_pairs:
        return TestResult("lag_structure", True, 0.0)
    real_sig = np.mean([_lagged_xcorr(p, c, max_lag) for p, c in real_pairs], axis=0)
    syn_sig = np.mean([_lagged_xcorr(p, c, max_lag) for p, c in syn_pairs], axis=0)
    dev = float(np.abs(real_sig - syn_sig).mean())
    lag_real = int(np.argmax(real_sig))
    lag_syn = int(np.argmax(syn_sig))
    reversed_lag = (lag_real > 0 and lag_syn == 0) or abs(
        lag_real - lag_syn
    ) > max_lag // 2
    passed = dev <= tol and not reversed_lag
    return TestResult(
        "lag_structure",
        passed=passed,
        statistic=dev,
        detail={"peak_lag_real": float(lag_real), "peak_lag_syn": float(lag_syn)},
    )


def incident_bundle_test(
    real_bundles: list[dict[str, float]],
    syn_bundles: list[dict[str, float]],
    required_descendants: list[str],
    *,
    min_cooccurrence: float = 0.5,
    auc_tol: float = 0.8,
) -> TestResult:
    """Incident-bundle coherence (doc R, incident_bundle_coherence_test).

    Targets "cartoon anomalies as isolated spikes" (doc D risks #4). For each
    fault family the synthetic bundle must (a) contain the required descendants
    with adequate co-occurrence and (b) NOT be trivially separable from the real
    bundle by a classifier. Returns the worst co-occurrence and the separability
    AUC.
    """
    if not syn_bundles:
        return TestResult("incident_bundle", False, 0.0)
    # co-occurrence rate of each required descendant in synthetic bundles
    worst = 1.0
    for d in required_descendants:
        rate = float(np.mean([1.0 if b.get(d, 0.0) > 0 else 0.0 for b in syn_bundles]))
        worst = min(worst, rate)
    keys = sorted({k for b in real_bundles + syn_bundles for k in b})
    sep_auc = 0.5
    if real_bundles and keys:
        real_X = embed_windows(real_bundles, keys)
        syn_X = embed_windows(syn_bundles, keys)
        res = classifier_two_sample_test(real_X, syn_X, auc_tol=auc_tol)
        sep_auc = res.statistic
    passed = worst >= min_cooccurrence and sep_auc <= auc_tol
    return TestResult(
        "incident_bundle",
        passed=passed,
        statistic=worst,
        detail={"worst_cooccurrence": worst, "separability_auc": sep_auc},
    )


def predecision_leakage_test(
    pre_only_X: np.ndarray,
    pre_plus_post_X: np.ndarray,
    labels: np.ndarray,
    *,
    leak_tol: float = 0.05,
    rng: np.random.Generator | None = None,
) -> TestResult:
    """Pre-decision leakage test (doc R, predecision_window_leakage_test).

    Targets teacher-forced leakage of post-action observations into pre-t0
    features (doc D risks #7). Train the same predictor on pre-only ``W⁻`` and on
    ``W⁺ = W⁻ ∪ post`` features; ``Leak = Perf(W⁺) − Perf(W⁻)``. A materially
    positive leak on purportedly pre-decision data flags poisoning — features that
    should be measurable w.r.t. ``≤ τ⁻`` only (doc R, [A7]).
    """
    rng = rng or np.random.default_rng(0)
    if pre_only_X.shape[0] < 4:
        return TestResult("predecision_leakage", True, 0.0)
    y = np.asarray(labels, dtype=float)
    idx = rng.permutation(pre_only_X.shape[0])
    cut = pre_only_X.shape[0] // 2

    def auc_for(X: np.ndarray) -> float:
        Xp = (X - X.mean(0)) / (X.std(0) + 1e-9)
        Xp = Xp[idx]
        yy = y[idx]
        return _classifier_auc(Xp[:cut], yy[:cut], Xp[cut:], yy[cut:])

    auc_pre = auc_for(pre_only_X)
    auc_post = auc_for(pre_plus_post_X)
    leak = auc_post - auc_pre
    return TestResult(
        "predecision_leakage",
        passed=leak <= leak_tol,
        statistic=leak,
        detail={"auc_pre": auc_pre, "auc_post": auc_post},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Suite aggregation
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class FidelityReport:
    """Aggregated certification + poisoning results for a data release."""

    results: list[TestResult] = field(default_factory=list)

    @property
    def certified(self) -> bool:
        """A release is certified only if *every* test passes (doc R gating)."""
        return all(r.passed for r in self.results)

    def as_dict(self) -> dict[str, dict]:
        return {
            r.name: {
                "passed": r.passed,
                "statistic": r.statistic,
                "p_value": r.p_value,
                **r.detail,
            }
            for r in self.results
        }


__all__ = [
    "FidelityReport",
    "TestResult",
    "classifier_two_sample_test",
    "embed_windows",
    "energy_distance",
    "energy_distance_test",
    "exact_kernel_conformance",
    "incident_bundle_test",
    "lag_structure_test",
    "mmd2_test",
    "mmd2_unbiased",
    "predecision_leakage_test",
    "tstr_trts",
]
