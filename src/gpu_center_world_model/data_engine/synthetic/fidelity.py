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

* :func:`counterfactual_fidelity_test` — doubly-robust action-conditioned
  marginal fidelity against held-out *real* interventions (doc R, T9 / Eq11-12).

Poisoning detection (the seven failure modes of doc R poisoning_detection_tests):
* :func:`lag_structure_test` — phase→util / power→temp lag preservation.
* :func:`incident_bundle_test` — fault incidents must be coherent bundles.
* :func:`predecision_leakage_test` — no post-action features leak pre-t0.
* :func:`paired_branch_action_sensitivity_test` — branch contrasts must not
  collapse / go deterministic / flip sign vs. real held-out DR contrasts.
* :func:`generator_agreement_trap_test` — synthetic-only leaderboards must track
  real-only leaderboards (rank correlation across seeded generator variants).
* :func:`support_overlap_and_provenance_test` — density-ratio ``r(x)``, clipped
  weight tails, and weighted ESS per stratum (doc R Eq27/Eq28).
* :func:`missingness_semantics_test` — missingness↔incident conditional mutual
  information + a missingness-conditioned C2ST (doc R Eq20).

The whole battery is driven by :func:`run_fidelity_suite`, which embeds windows,
runs every certification + poisoning test, and returns a populated
:class:`FidelityReport`. :func:`synthetic_mixing_weight` realises the Eq27
importance-reweighting term ``ω_g(x)=clip(r_g(x),0,ω_max)·c_g`` whose gate
``c_g`` is *exactly* this report's certification status.

All tests return a typed result with a pass/fail flag and the statistic so the
data card can publish them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ...schema.actions import Action
from ...schema.kpis import HazardChannel, KPIChannel
from ...schema.observations import SourceHealth
from ...schema.state import WorldState
from ...simulator.contracts import FeasibilityResult
from .counterfactual import BranchOutcome, CounterfactualPack


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
# The doc-L label channels (the targets every certification test ranges over)
# ──────────────────────────────────────────────────────────────────────────────
#: The KPI-delta + virtual-queue channels the embedding/TSTR harness regress on.
#: Reconciles 1:1 with :class:`KPIChannel` (doc I kpi_map) plus the collapsed
#: slow-latent ``vq:*`` targets (doc K T14 EqL1-EqL5) the labels carry.
KPI_LABEL_CHANNELS: tuple[str, ...] = tuple(c.value for c in KPIChannel)
VQ_LABEL_CHANNELS: tuple[str, ...] = (
    "W_bottleneck",
    "D_fairness",
    "Z_headroom",
    "Z_hotspot",
    "F_k",
    "mean_wait_s",
    "link_util_max",
)
#: The event-hazard channels (competing risks + parallel binaries; doc L/U).
HAZARD_LABEL_CHANNELS: tuple[str, ...] = tuple(c.value for c in HazardChannel)


def branch_feature_dict(branch: BranchOutcome) -> dict[str, float]:
    """Project one :class:`BranchOutcome` onto the doc-L label embedding ``φ(W)``.

    The embedding concatenates the static feasibility bit, the doc-I KPI deltas,
    the collapsed slow-latent virtual-queue coordinates (doc K T14), the realised
    event hazards (doc L), and a telemetry-missingness summary (doc R Eq20) — the
    ``[φstatic, φevent, φtelemetry, φrelational]`` block the kernel two-sample test
    consumes (doc R kernel_two_sample_test_on_path_space). Missing channels are a
    *feature absence* (these are derived summaries, not zero-imputed raw channels).
    """
    feat: dict[str, float] = {"feasible": 1.0 if branch.feasible else 0.0}
    for ch in KPI_LABEL_CHANNELS:
        feat[f"kpi:{ch}"] = float(branch.kpis.get(ch, 0.0))
    for ch in VQ_LABEL_CHANNELS:
        feat[f"vq:{ch}"] = float(branch.vq_features.get(ch, 0.0))
    for ch in HAZARD_LABEL_CHANNELS:
        feat[f"haz:{ch}"] = float(branch.hazards.get(ch, 0.0))
    mr, missing = _missingness_summary(branch.observations)
    feat["telemetry_missing_rate"] = mr
    feat["telemetry_n_missing"] = float(missing)
    return feat


def _missingness_summary(observations: Sequence[object]) -> tuple[float, int]:
    """Missing-channel rate + count over a branch's emitted telemetry (Eq20).

    Counts a channel as *missing* iff its integrity flag (``available``) is False
    or its :class:`SourceHealth` is degraded to ``STALE``/``DROPPED`` — never by
    reading a zero value, since missingness is first-class state, not a zero.
    """
    total = 0
    missing = 0
    for obs in observations:
        chans = getattr(obs, "channels", None)
        if not chans:
            continue
        for ch in chans.values():
            total += 1
            avail = getattr(ch, "available", True)
            sh = getattr(ch, "source_health", SourceHealth.OBSERVED)
            if (not avail) or sh in (SourceHealth.STALE, SourceHealth.DROPPED):
                missing += 1
    return (missing / total if total else 0.0), missing


def pack_features(packs: Sequence[CounterfactualPack]) -> list[dict[str, float]]:
    """Flatten a corpus of packs into per-branch embedding feature dicts."""
    return [branch_feature_dict(b) for p in packs for b in p.branches]


# ──────────────────────────────────────────────────────────────────────────────
# Certification: counterfactual fidelity vs. held-out real interventions (T9)
# ──────────────────────────────────────────────────────────────────────────────
def _branch_contrast(
    pack: CounterfactualPack, functional: str
) -> dict[tuple[str, str], float]:
    """Synthetic action contrasts Δ_g(a,a′)=g(Y^a)−g(Y^a′) (doc R Eq12).

    ``g`` selects a scalar outcome functional from the realised branch: a KPI
    channel, a ``vq:`` slow-latent target, or a ``haz:`` hazard probability.
    Returns ``{(a, a′): Δ}`` for every ordered branch pair sharing the pack's
    pre-action hash + seed bundle (common random numbers).
    """

    def g(b: BranchOutcome) -> float:
        if functional.startswith("haz:"):
            return float(b.hazards.get(functional[4:], 0.0))
        if functional.startswith("vq:"):
            return float(b.vq_features.get(functional[3:], 0.0))
        return float(b.kpis.get(functional, 0.0))

    vals = {b.action_id: g(b) for b in pack.branches if b.feasible}
    out: dict[tuple[str, str], float] = {}
    ids = list(vals)
    for i, a in enumerate(ids):
        for ap in ids[i + 1 :]:
            out[(a, ap)] = vals[a] - vals[ap]
    return out


def _dr_real_contrast(
    real_outcomes: Sequence[tuple[str, float, float]], a: str, ap: str
) -> tuple[float, float]:
    """Doubly-robust mean-effect estimate of a real action contrast (doc R T9).

    ``real_outcomes`` is a held-out log of ``(action_id, outcome, propensity)``
    triples within one stratum. The AIPW (doubly-robust) estimator combines an
    outcome regression (the per-action empirical mean ``μ̂_a``) with an
    inverse-propensity correction on the residual:

        ψ̂_a = μ̂_a + (1/n) Σ_i 1{A_i=a}/ê(a) · (Y_i − μ̂_a)

    so it is consistent if *either* the outcome mean or the propensity is correct
    (the T9 double-robustness guarantee). The contrast is ψ̂_a − ψ̂_{a′}; the
    second return value is a pooled standard error for the contrast.
    """
    ys = np.array([y for _, y, _ in real_outcomes], dtype=float)
    acts = [aid for aid, _, _ in real_outcomes]
    props = np.array([max(p, 1e-3) for _, _, p in real_outcomes], dtype=float)
    n = len(ys)

    def psi(act: str) -> tuple[float, float]:
        mask = np.array([aid == act for aid in acts], dtype=float)
        if mask.sum() == 0:
            return 0.0, float("inf")
        mu = float((ys * mask).sum() / mask.sum())  # outcome regression μ̂_a
        # AIPW residual correction (inverse-propensity weighted)
        corr = float(np.mean(mask / props * (ys - mu)))
        infl = mu + mask / props * (ys - mu)  # per-sample influence
        se = float(np.std(infl) / np.sqrt(n)) if n > 1 else float("inf")
        return mu + corr, se

    psi_a, se_a = psi(a)
    psi_ap, se_ap = psi(ap)
    pooled_se = float(np.sqrt(se_a**2 + se_ap**2)) if np.isfinite(se_a + se_ap) else 1.0
    return psi_a - psi_ap, max(pooled_se, 1e-6)


def counterfactual_fidelity_test(
    syn_packs: Sequence[CounterfactualPack],
    real_log: dict[str, Sequence[tuple[str, float, float]]],
    *,
    functionals: Sequence[str] = (
        KPIChannel.QUEUE_WAIT_DELTA_S.value,
        KPIChannel.ADMITTED_GPU_HOURS_DELTA.value,
        "haz:" + HazardChannel.PODS_READY_TIMEOUT.value,
    ),
    z_tol: float = 3.0,
) -> TestResult:
    """DR counterfactual-fidelity test on outcome functionals (doc R T9 / Eq11-12).

    For each outcome functional ``g`` we form the synthetic action contrast
    ``Δ_syn(a,a′)=g(Y^a)−g(Y^a′)`` averaged over packs (Eq12, common random
    numbers) and compare it to a doubly-robust estimate of the *real* contrast
    on the held-out intervention log, keyed by the same action ids. The statistic
    is the largest standardised DR mean-effect error ``|Δ_syn−Δ_real|/SE``; the
    release passes when every functional's contrast lies within ``z_tol`` of the
    real DR target (vanishing discrepancy ⇒ action-conditioned marginal fidelity).

    With no usable held-out real interventions the test abstains as *passing* but
    records ``n_compared=0`` so the data card shows the certification was vacuous.
    """
    syn_contrasts: dict[str, dict[tuple[str, str], list[float]]] = {}
    for fn in functionals:
        agg: dict[tuple[str, str], list[float]] = {}
        for pack in syn_packs:
            for pair, d in _branch_contrast(pack, fn).items():
                agg.setdefault(pair, []).append(d)
        syn_contrasts[fn] = agg

    worst_z = 0.0
    n_compared = 0
    worst_detail: dict[str, float] = {}
    for fn in functionals:
        outcomes = real_log.get(fn) or real_log.get(_strip_fn(fn))
        if not outcomes:
            continue
        for (a, ap), ds in syn_contrasts[fn].items():
            real_acts = {aid for aid, _, _ in outcomes}
            if a not in real_acts or ap not in real_acts:
                continue
            delta_syn = float(np.mean(ds))
            delta_real, se = _dr_real_contrast(outcomes, a, ap)
            z = abs(delta_syn - delta_real) / se
            n_compared += 1
            if z > worst_z:
                worst_z = z
                worst_detail = {
                    "functional_hash": float(abs(hash(fn)) % 1000),
                    "delta_syn": delta_syn,
                    "delta_real": delta_real,
                    "se": se,
                }
    passed = n_compared == 0 or worst_z <= z_tol
    detail = {"n_compared": float(n_compared), **worst_detail}
    return TestResult(
        "counterfactual_fidelity",
        passed=passed,
        statistic=worst_z,
        detail=detail,
    )


def _strip_fn(fn: str) -> str:
    for pre in ("haz:", "vq:"):
        if fn.startswith(pre):
            return fn[len(pre):]
    return fn


# ──────────────────────────────────────────────────────────────────────────────
# Poisoning: paired-branch action sensitivity (doc R poisoning #4)
# ──────────────────────────────────────────────────────────────────────────────
def paired_branch_action_sensitivity_test(
    syn_packs: Sequence[CounterfactualPack],
    real_log: dict[str, Sequence[tuple[str, float, float]]] | None = None,
    *,
    functionals: Sequence[str] = (
        KPIChannel.QUEUE_WAIT_DELTA_S.value,
        KPIChannel.ADMITTED_GPU_HOURS_DELTA.value,
        KPIChannel.TOPOLOGY_FRAGMENTATION_DELTA.value,
    ),
    min_spread: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> TestResult:
    """Branch contrasts must respond to the action (doc R paired_branch test).

    On the counterfactual harness (≥ 3 actions from one pre-action hash + common
    seed bundle) we form Δ_g(a,a′)=g(Y^a)−g(Y^a′). Poisoning — action-ignorance
    or shortcut learning from teacher forcing/weak branching — is flagged when the
    contrast distribution (a) **collapses toward zero** (no action sensitivity),
    (b) is **implausibly deterministic** (zero spread across packs sharing a pair),
    or (c) **flips sign** relative to a held-out real DR contrast.

    The statistic is the mean absolute branch contrast magnitude across
    functionals (larger ⇒ more action-sensitive). Determinism is checked as the
    per-pair contrast variance across packs; a sign flip requires ``real_log``.
    """
    rng = rng or np.random.default_rng(0)
    if not syn_packs:
        return TestResult("paired_branch_action_sensitivity", False, 0.0)
    mags: list[float] = []
    pair_series: dict[tuple[str, str, str], list[float]] = {}
    # Determinism is meaningful only *within one pre-action hash*: multiple Monte-
    # Carlo draws over Û of the SAME hash should give a contrast with non-zero
    # variance (Eq11/Eq12). A contrast that is stable across *different* hashes is
    # correctly stable, not poisoned — so the per-hash series is keyed separately.
    hash_series: dict[tuple[str, str, str, str], list[float]] = {}
    n_pairs = 0
    for fn in functionals:
        for pack in syn_packs:
            contrasts = _branch_contrast(pack, fn)
            n_pairs += len(contrasts)
            for (a, ap), d in contrasts.items():
                mags.append(abs(d))
                pair_series.setdefault((fn, a, ap), []).append(d)
                hash_series.setdefault((fn, pack.state_hash, a, ap), []).append(d)
    # Abstain (pass, recorded) when the harness produced no comparable branch
    # *pairs* — e.g. every pack had a single feasible branch — since the contrast
    # distribution is then undefined, not collapsed (doc R needs ≥3 actions from a
    # shared hash; an infeasible-placement pre-state legitimately yields one).
    if n_pairs == 0:
        return TestResult(
            "paired_branch_action_sensitivity",
            True,
            0.0,
            detail={"n_pairs": 0.0},
        )
    # per-hash spreads over repeated Û draws of the same pre-action hash
    repeat_spreads = [
        float(np.std(s)) for s in hash_series.values() if len(s) >= 2
    ]
    mean_mag = float(np.mean(mags)) if mags else 0.0
    # collapse-toward-zero: contrasts are everywhere ~0 (no action effect at all)
    collapsed = mean_mag <= min_spread
    # implausible determinism: a nonzero effect whose *every* repeated-Û draw of a
    # shared hash has zero variance (a shortcut that ignores Û). Only assessable
    # when at least one hash has ≥2 realisations.
    has_repeats = len(repeat_spreads) > 0
    deterministic = (
        has_repeats and mean_mag > min_spread and max(repeat_spreads) <= min_spread
    )
    # sign-flip vs. real held-out DR contrasts
    sign_flips = 0
    n_sign = 0
    if real_log:
        for fn in functionals:
            outcomes = real_log.get(fn) or real_log.get(_strip_fn(fn))
            if not outcomes:
                continue
            real_acts = {aid for aid, _, _ in outcomes}
            for (a, ap), ds in {
                k[1:]: v for k, v in pair_series.items() if k[0] == fn
            }.items():
                if a not in real_acts or ap not in real_acts:
                    continue
                delta_syn = float(np.mean(ds))
                delta_real, _ = _dr_real_contrast(outcomes, a, ap)
                if abs(delta_real) > min_spread and abs(delta_syn) > min_spread:
                    n_sign += 1
                    if np.sign(delta_syn) != np.sign(delta_real):
                        sign_flips += 1
    passed = (not collapsed) and (not deterministic) and sign_flips == 0
    return TestResult(
        "paired_branch_action_sensitivity",
        passed=passed,
        statistic=mean_mag,
        detail={
            "mean_abs_contrast": mean_mag,
            "max_per_hash_spread": (
                float(max(repeat_spreads)) if repeat_spreads else 0.0
            ),
            "n_repeat_hashes": float(len(repeat_spreads)),
            "collapsed": 1.0 if collapsed else 0.0,
            "deterministic": 1.0 if deterministic else 0.0,
            "sign_flips": float(sign_flips),
            "n_sign_compared": float(n_sign),
            "n_pairs": float(n_pairs),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Poisoning: generator-agreement trap (doc R poisoning #5)
# ──────────────────────────────────────────────────────────────────────────────
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two score vectors (ties via avg rank)."""
    if a.size < 2 or b.size < 2:
        return 1.0

    def rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(1, x.size + 1, dtype=float)
        # average ranks for ties
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(counts.size)
        np.add.at(sums, inv, r)
        return (sums / counts)[inv]

    ra, rb = rank(a), rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 1.0


def generator_agreement_trap_test(
    syn_only_scores: dict[str, float],
    real_only_scores: dict[str, float],
    *,
    rho_tol: float = 0.5,
) -> TestResult:
    """Synthetic-only leaderboards must track real-only ones (doc R poisoning #5).

    Targets synthetic-only benchmarks that measure agreement with *one* generator
    rather than real reliability. Given ``K`` independently seeded generator
    variants (or scenario packs), each scored on a synthetic-only test and on a
    real-only held-out test, compute Spearman rank correlation ``ρ`` between the
    two leaderboards (keyed by the same variant ids). Poisoning is flagged when
    synthetic-only rankings are poorly aligned with real-only rankings
    (``ρ < rho_tol``). With < 2 shared variants the anchor cannot be computed and
    the test abstains as passing (recording ``n_variants``).
    """
    keys = [k for k in syn_only_scores if k in real_only_scores]
    if len(keys) < 2:
        return TestResult(
            "generator_agreement_trap",
            True,
            1.0,
            detail={"n_variants": float(len(keys))},
        )
    s = np.array([syn_only_scores[k] for k in keys], dtype=float)
    r = np.array([real_only_scores[k] for k in keys], dtype=float)
    rho = _spearman(s, r)
    return TestResult(
        "generator_agreement_trap",
        passed=rho >= rho_tol,
        statistic=rho,
        detail={"rank_correlation": rho, "n_variants": float(len(keys))},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Poisoning: support overlap + provenance — r(x), ESS, clipped weights (Eq27/28)
# ──────────────────────────────────────────────────────────────────────────────
def density_ratio_weights(
    real: np.ndarray,
    syn: np.ndarray,
    *,
    omega_max: float = 10.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Estimate clipped density ratios ``r(x)=p_real/q_syn`` on the synthetic set.

    Fits an origin classifier ``ĉ(x)=P(real|x)`` (doc R support_overlap test) and
    converts its calibrated probability into the density ratio via
    ``r(x)=ĉ(x)/(1−ĉ(x))`` (correcting for the class-balance odds), then clips to
    ``[0, ω_max]`` (the Eq27 clip). Returns the per-synthetic-sample weights
    ``ω(x)=clip(r(x),0,ω_max)`` and the held-out classifier AUC (a support-overlap
    diagnostic: AUC near 0.5 ⇒ strong overlap ⇒ ratios near 1).
    """
    rng = rng or np.random.default_rng(0)
    xs, ys = _standardize(real, syn)
    if xs.shape[0] < 2 or ys.shape[0] < 2:
        return np.ones(syn.shape[0]), 0.5
    X = np.concatenate([xs, ys], axis=0)
    y = np.concatenate([np.ones(xs.shape[0]), np.zeros(ys.shape[0])])
    w = _logistic_fit(X, y, epochs=300)
    # class-balance correction: training prior is n_real/(n_real+n_syn); the
    # density ratio is the likelihood ratio = odds / prior_odds.
    n_r, n_s = xs.shape[0], ys.shape[0]
    prior_odds = n_r / max(n_s, 1)
    Xb = np.concatenate([ys, np.ones((ys.shape[0], 1))], axis=1)
    p_real = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
    p_real = np.clip(p_real, 1e-6, 1 - 1e-6)
    r = (p_real / (1.0 - p_real)) / prior_odds
    weights = np.clip(r, 0.0, omega_max)
    # held-out AUC as overlap diagnostic
    idx = rng.permutation(X.shape[0])
    cut = X.shape[0] // 2
    auc = _classifier_auc(X[idx][:cut], y[idx][:cut], X[idx][cut:], y[idx][cut:])
    return weights, auc


def effective_sample_size(weights: np.ndarray) -> float:
    """Weighted effective sample size ``ESS=(Σω)²/Σω²`` (doc R Eq28)."""
    s1 = float(np.sum(weights))
    s2 = float(np.sum(weights * weights))
    return (s1 * s1 / s2) if s2 > 0 else 0.0


def support_overlap_and_provenance_test(
    real: np.ndarray,
    syn: np.ndarray,
    *,
    omega_max: float = 10.0,
    min_ess_frac: float = 0.1,
    overlap_auc_tol: float = 0.95,
    rng: np.random.Generator | None = None,
) -> TestResult:
    """Support-overlap + provenance test (doc R poisoning #6; Eq27/Eq28).

    Fits ``r(x)=p_real(x)/q_syn(x)`` via an origin classifier, then computes the
    clipped weight tail and the weighted effective sample size
    ``ESS=(Σω)²/Σω²`` (Eq28). Poisoning is flagged when support overlap is too
    small (origin AUC ≥ ``overlap_auc_tol`` ⇒ disjoint supports, a positivity
    violation) or the ESS fraction ``ESS/|syn|`` collapses below ``min_ess_frac``
    (large synthetic regions carry almost no real-supported information). Such
    strata receive zero synthetic weight (the Eq27 gate ``c_g=0``).
    """
    rng = rng or np.random.default_rng(0)
    if syn.shape[0] < 4 or real.shape[0] < 4:
        return TestResult(
            "support_overlap_and_provenance",
            True,
            1.0,
            detail={"n_syn": float(syn.shape[0]), "n_real": float(real.shape[0])},
        )
    weights, auc = density_ratio_weights(real, syn, omega_max=omega_max, rng=rng)
    ess = effective_sample_size(weights)
    ess_frac = ess / syn.shape[0]
    clipped_frac = float(np.mean(weights >= omega_max - 1e-9))
    overlap_ok = auc <= overlap_auc_tol
    ess_ok = ess_frac >= min_ess_frac
    passed = overlap_ok and ess_ok
    return TestResult(
        "support_overlap_and_provenance",
        passed=passed,
        statistic=ess_frac,
        detail={
            "ess": ess,
            "ess_frac": ess_frac,
            "origin_auc": auc,
            "clipped_weight_frac": clipped_frac,
            "omega_max": float(omega_max),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Poisoning: missingness semantics — CMI + missingness-conditioned C2ST (Eq20)
# ──────────────────────────────────────────────────────────────────────────────
def _binary_cmi(m: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Conditional mutual information ``I(M;Y|Z)`` for binary M,Y over strata Z.

    Plug-in CMI in nats: ``Σ_z p(z) Σ_{m,y} p(m,y|z) log[p(m,y|z)/(p(m|z)p(y|z))]``.
    ``z`` is an integer stratum code (e.g. a coarse incident-context bucket). With
    a single stratum this reduces to the unconditional mutual information.
    """
    m = m.astype(int)
    y = y.astype(int)
    z = z.astype(int)
    n = m.size
    if n == 0:
        return 0.0
    cmi = 0.0
    for zv in np.unique(z):
        sel = z == zv
        nz = int(sel.sum())
        if nz == 0:
            continue
        pz = nz / n
        mm, yy = m[sel], y[sel]
        local = 0.0
        for mv in (0, 1):
            for yv in (0, 1):
                p_my = np.mean((mm == mv) & (yy == yv))
                p_m = np.mean(mm == mv)
                p_y = np.mean(yy == yv)
                if p_my > 0 and p_m > 0 and p_y > 0:
                    local += p_my * np.log(p_my / (p_m * p_y))
        cmi += pz * local
    return float(max(cmi, 0.0))


def missingness_semantics_test(
    real_records: Sequence[tuple[int, int, int]],
    syn_records: Sequence[tuple[int, int, int]],
    *,
    cmi_floor: float = 0.01,
    cmi_gap_tol: float = 0.10,
    rng: np.random.Generator | None = None,
) -> TestResult:
    """Missingness must stay coupled to incident state (doc R poisoning #7; Eq20).

    Each record is ``(missing, incident, stratum)``: ``missing`` is a telemetry
    missingness indicator, ``incident`` a fault/requeue indicator, ``stratum`` a
    coarse context code. We compute the conditional mutual information
    ``I(missing; incident | stratum)`` on the real and synthetic corpora. Poisoning
    — treating incident-time missingness as benign cleaning noise — is flagged
    when (a) the *synthetic* CMI collapses below ``cmi_floor`` while the real CMI
    is materially positive (missingness made ~independent of fault/requeue state),
    or (b) the real-vs-synthetic CMI gap exceeds ``cmi_gap_tol``. A
    missingness-conditioned C2ST (origin AUC within the missing-only slice) backs
    the CMI when both corpora have missing samples.

    The statistic is the absolute CMI gap; the detail carries both CMIs and the
    conditioned C2ST AUC.
    """
    rng = rng or np.random.default_rng(0)
    if not real_records or not syn_records:
        return TestResult("missingness_semantics", True, 0.0)
    rm = np.array([r[0] for r in real_records])
    ri = np.array([r[1] for r in real_records])
    rz = np.array([r[2] for r in real_records])
    sm = np.array([r[0] for r in syn_records])
    si = np.array([r[1] for r in syn_records])
    sz = np.array([r[2] for r in syn_records])
    cmi_real = _binary_cmi(rm, ri, rz)
    cmi_syn = _binary_cmi(sm, si, sz)
    gap = abs(cmi_real - cmi_syn)
    # synthetic independence failure: real coupling exists but synthetic is flat.
    collapsed = cmi_real >= cmi_floor and cmi_syn < cmi_floor
    # missingness-conditioned C2ST on the (incident,stratum) signature within the
    # *missing* slice: if a classifier separates real vs syn given M=1, the
    # missingness mechanism differs (doc R Eq20 MNAR conditioning).
    cond_auc = 0.5
    rsel = rm == 1
    ssel = sm == 1
    if int(rsel.sum()) >= 4 and int(ssel.sum()) >= 4:
        real_X = np.stack([ri[rsel], rz[rsel]], axis=1).astype(float)
        syn_X = np.stack([si[ssel], sz[ssel]], axis=1).astype(float)
        cond = classifier_two_sample_test(real_X, syn_X, rng=rng)
        cond_auc = cond.statistic
    # The CMI signals are primary (they target the doc-R independence failure);
    # the missingness-conditioned C2ST is a backstop that only fails on *strong*
    # separability (the missingness mechanism differs materially), so its tol is
    # looser than the unconditioned C2ST to absorb small-sample slice noise.
    passed = (not collapsed) and gap <= cmi_gap_tol and cond_auc <= 0.85
    return TestResult(
        "missingness_semantics",
        passed=passed,
        statistic=gap,
        detail={
            "cmi_real": cmi_real,
            "cmi_syn": cmi_syn,
            "cmi_gap": gap,
            "conditioned_c2st_auc": cond_auc,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# TSTR/TRTS task harness over the doc-L label channels
# ──────────────────────────────────────────────────────────────────────────────
def _ridge_fit(X: np.ndarray, y: np.ndarray, *, l2: float = 1.0) -> np.ndarray:
    """Closed-form ridge regression ``(XᵀX+λI)⁻¹Xᵀy`` with a bias column."""
    n = X.shape[0]
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
    d = Xb.shape[1]
    A = Xb.T @ Xb + l2 * np.eye(d)
    return np.linalg.solve(A, Xb.T @ y)


def _ridge_r2(model: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Out-of-sample R² of a ridge model (clamped to ``[-1, 1]`` for stability)."""
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    pred = Xb @ model
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return float(np.clip(1.0 - ss_res / ss_tot, -1.0, 1.0))


def make_label_task(
    feature_keys: Sequence[str], target_channel: str
) -> tuple[Callable, Callable]:
    """A concrete TSTR/TRTS regression task over the doc-L label channels.

    Returns ``(train_fn, score_fn)`` for :func:`tstr_trts` where the learner is a
    ridge regressor predicting one doc-L target (a KPI delta such as
    ``queue_wait_delta_s``, a ``vq:`` slow-latent target, or a ``haz:`` hazard
    probability) from the remaining embedding coordinates, scored by held-out R².
    The two callables ignore their target channel via closure, so the same
    embedding matrices the two-sample tests use drive the predictive-utility
    diagnostic without a second featurisation pass.
    """

    def train_fn(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return _ridge_fit(X, y)

    def score_fn(model: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        return _ridge_r2(model, X, y)

    return train_fn, score_fn


def tstr_trts_over_labels(
    real_features: list[dict[str, float]],
    syn_features: list[dict[str, float]],
    *,
    target_channels: Sequence[str] = (
        "kpi:" + KPIChannel.QUEUE_WAIT_DELTA_S.value,
        "kpi:" + KPIChannel.ADMITTED_GPU_HOURS_DELTA.value,
        "haz:" + HazardChannel.PODS_READY_TIMEOUT.value,
    ),
    delta_tol: float = 0.25,
) -> TestResult:
    """Concrete TSTR/TRTS harness over the doc-L label channels (doc R).

    For each target channel ``g`` (a doc-L KPI/vq/hazard label) we predict ``g``
    from the remaining embedding coordinates with a ridge learner and compute
    ``ΔTSTR(g)=R²(real→real)−R²(syn→real)`` and the asymmetry ``ΔTRTS(g)``. The
    release passes when the worst-task ``|ΔTSTR|`` stays within ``delta_tol`` — a
    synthetic model trained on the doc-L targets predicts real held-out targets
    about as well as a real-trained one (external-validity diagnostic, doc R).
    """
    keys = sorted(
        {k for f in real_features + syn_features for k in f}
    )
    n_real, n_syn = len(real_features), len(syn_features)
    # A held-out real test split makes Perf(real→real) *also* out-of-sample, so
    # ΔTSTR is a fair train-on-X / test-on-real gap (doc R), not an in-sample vs
    # out-of-sample artefact. Below ~8 real windows the split is too small to be
    # informative and the diagnostic abstains (doc R: external-validity *hook*).
    if n_real < 8 or n_syn < 6 or not keys:
        return TestResult("tstr_trts", True, 0.0, detail={"n_tasks": 0.0})
    cut = max(2, n_real // 3)  # ~1/3 held out for the real test set
    train_feats = real_features[: n_real - cut]
    test_feats = real_features[n_real - cut :]
    worst = 0.0
    worst_trts = 0.0
    n_tasks = 0
    for tgt in target_channels:
        if tgt not in keys:
            continue
        feat_keys = [k for k in keys if k != tgt]
        # cap feature dimension to the train size to avoid degenerate overfitting
        # on tiny corpora (ridge with d ≫ n is uninformative, not unfaithful).
        if len(feat_keys) > max(2, len(train_feats) - 1):
            feat_keys = feat_keys[: max(2, len(train_feats) - 1)]
        if not feat_keys:
            continue
        rtr_X = embed_windows(train_feats, feat_keys)
        rtr_y = embed_windows(train_feats, [tgt]).ravel()
        rte_X = embed_windows(test_feats, feat_keys)
        rte_y = embed_windows(test_feats, [tgt]).ravel()
        syn_X = embed_windows(syn_features, feat_keys)
        syn_y = embed_windows(syn_features, [tgt]).ravel()
        if np.std(rte_y) < 1e-9:
            continue  # a constant real target carries no predictive signal
        m_real = _ridge_fit(rtr_X, rtr_y)
        m_syn = _ridge_fit(syn_X, syn_y)
        # ΔTSTR = R²(real-trained on real-test) − R²(syn-trained on real-test)
        delta_tstr = _ridge_r2(m_real, rte_X, rte_y) - _ridge_r2(m_syn, rte_X, rte_y)
        # ΔTRTS = R²(syn-trained on syn) − R²(real-trained on syn)  (asymmetry)
        if np.std(syn_y) >= 1e-9:
            r2_ss = _ridge_r2(m_syn, syn_X, syn_y)
            r2_rs = _ridge_r2(m_real, syn_X, syn_y)
            delta_trts = r2_ss - r2_rs
        else:
            delta_trts = 0.0
        n_tasks += 1
        worst = max(worst, abs(delta_tstr))
        worst_trts = max(worst_trts, abs(delta_trts))
    passed = n_tasks == 0 or worst <= delta_tol
    return TestResult(
        "tstr_trts",
        passed=passed,
        statistic=worst,
        detail={
            "worst_delta_tstr": worst,
            "worst_delta_trts": worst_trts,
            "n_tasks": float(n_tasks),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Suite aggregation
# ──────────────────────────────────────────────────────────────────────────────
#: The six certification + seven poisoning test names the suite is expected to
#: cover (doc R fidelity_certification_suite + poisoning_detection_tests). Used
#: to record certification *coverage* on the data card so a vacuously-passing
#: (all-abstained) report is distinguishable from a genuinely-exercised one.
CERTIFICATION_TESTS: tuple[str, ...] = (
    "exact_kernel_conformance",
    "mmd2",
    "energy_distance",
    "classifier_two_sample",
    "tstr_trts",
    "counterfactual_fidelity",
)
POISONING_TESTS: tuple[str, ...] = (
    "lag_structure",
    "incident_bundle",
    "predecision_leakage",
    "paired_branch_action_sensitivity",
    "generator_agreement_trap",
    "support_overlap_and_provenance",
    "missingness_semantics",
)


@dataclass(slots=True)
class FidelityReport:
    """Aggregated certification + poisoning results for a data release."""

    results: list[TestResult] = field(default_factory=list)
    #: Whether the suite actually ran on real windows (else it is uncertified).
    #: Additive field (doc R gating): a report with ``ran=False`` represents the
    #: legacy ``fidelity=None`` case so the data card can mark it explicitly.
    ran: bool = True
    #: Names of tests that abstained (no usable real anchor) — recorded so the
    #: data card shows which certifications were vacuous vs. genuinely exercised.
    abstained: list[str] = field(default_factory=list)

    @property
    def certified(self) -> bool:
        """A release is certified only if *every* test passes (doc R gating)."""
        return self.ran and all(r.passed for r in self.results)

    @property
    def coverage(self) -> dict[str, bool]:
        """Which doc-R certification + poisoning tests were exercised non-vacuously."""
        run = {r.name for r in self.results}
        abst = set(self.abstained)
        return {
            name: (name in run and name not in abst)
            for name in CERTIFICATION_TESTS + POISONING_TESTS
        }

    def gate(self) -> float:
        """The Eq27 stratum certification gate ``c_g∈{0,1}`` (1 iff certified)."""
        return 1.0 if self.certified else 0.0

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


def run_fidelity_suite(
    real_packs: Sequence[CounterfactualPack],
    syn_packs: Sequence[CounterfactualPack],
    *,
    conformance_cases: Sequence[tuple[WorldState, Action]] | None = None,
    generator_feasible: Callable[[WorldState, Action], FeasibilityResult] | None = None,
    oracle_feasible: Callable[[WorldState, Action], FeasibilityResult] | None = None,
    real_intervention_log: dict[str, Sequence[tuple[str, float, float]]] | None = None,
    syn_only_scores: dict[str, float] | None = None,
    real_only_scores: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> FidelityReport:
    """Run the full doc-R fidelity + poisoning battery and return a report.

    This is the single orchestrator the release path invokes (doc R gating). It
    embeds both corpora once into the doc-L label space (:func:`branch_feature_dict`)
    and runs *every* certification + poisoning test on the shared embedding:

    Certification (doc R fidelity_certification_suite, 6 tests):
        ``exact_kernel_conformance`` (when an oracle is supplied), ``mmd2``,
        ``energy_distance``, ``classifier_two_sample``, ``tstr_trts`` over the
        doc-L channels, and ``counterfactual_fidelity`` (DR, when a real
        intervention log is supplied).

    Poisoning (doc R poisoning_detection_tests, 7 tests):
        ``lag_structure``, ``incident_bundle``, ``predecision_leakage``,
        ``paired_branch_action_sensitivity``, ``generator_agreement_trap``,
        ``support_overlap_and_provenance``, and ``missingness_semantics``.

    Tests with no usable real anchor (e.g. no held-out interventions, no second
    generator variant) **abstain** as passing and are recorded in
    :attr:`FidelityReport.abstained`, so a synthetic-only demo release is reported
    as *certified-but-vacuous* rather than silently strong. The returned report's
    :meth:`FidelityReport.gate` is precisely the Eq27 ``c_g`` certification gate.
    """
    rng = rng or np.random.default_rng(0)
    real_feats = pack_features(real_packs)
    syn_feats = pack_features(syn_packs)
    keys = sorted({k for f in real_feats + syn_feats for k in f})
    real_X = embed_windows(real_feats, keys)
    syn_X = embed_windows(syn_feats, keys)

    results: list[TestResult] = []
    abstained: list[str] = []

    # ---- Certification ------------------------------------------------------
    if conformance_cases and generator_feasible and oracle_feasible:
        results.append(
            exact_kernel_conformance(
                list(conformance_cases), generator_feasible, oracle_feasible
            )
        )
    else:
        abstained.append("exact_kernel_conformance")

    results.append(mmd2_test(real_X, syn_X, rng=rng))
    results.append(energy_distance_test(real_X, syn_X, rng=rng))
    results.append(classifier_two_sample_test(real_X, syn_X, rng=rng))

    tstr = tstr_trts_over_labels(real_feats, syn_feats)
    results.append(tstr)
    if tstr.detail.get("n_tasks", 0.0) == 0.0:
        abstained.append("tstr_trts")

    cf = counterfactual_fidelity_test(syn_packs, real_intervention_log or {})
    results.append(cf)
    if cf.detail.get("n_compared", 0.0) == 0.0:
        abstained.append("counterfactual_fidelity")

    # ---- Poisoning ----------------------------------------------------------
    real_lag = _lag_pairs_from_packs(real_packs)
    syn_lag = _lag_pairs_from_packs(syn_packs)
    lag = lag_structure_test(real_lag, syn_lag)
    results.append(lag)
    if not real_lag or not syn_lag:
        abstained.append("lag_structure")

    real_bundles = _incident_bundles_from_packs(real_packs)
    syn_bundles = _incident_bundles_from_packs(syn_packs)
    # Required descendants = the incident hazards the corpora actually realise, so
    # the coherence test certifies *present* descendants co-occur rather than
    # demanding a fixed family the demo state never triggers (doc R incident test).
    realised = {
        k[4:]
        for bset in (real_bundles, syn_bundles)
        for b in bset
        for k, v in b.items()
        if k.startswith("haz:") and v > 0.0
    }
    required = sorted(f"haz:{c}" for c in realised) or [
        "haz:" + HazardChannel.XID_FAULT.value
    ]
    if syn_bundles and real_bundles:
        results.append(incident_bundle_test(real_bundles, syn_bundles, required))
    else:
        # No paired real-vs-syn incident bundles to anchor against ⇒ abstain.
        results.append(
            TestResult(
                "incident_bundle",
                True,
                0.0,
                detail={
                    "n_real_bundles": float(len(real_bundles)),
                    "n_syn_bundles": float(len(syn_bundles)),
                },
            )
        )
        abstained.append("incident_bundle")

    pre_X, prepost_X, leak_y = _leakage_design(syn_packs)
    if pre_X.shape[0] >= 4:
        results.append(predecision_leakage_test(pre_X, prepost_X, leak_y, rng=rng))
    else:
        results.append(TestResult("predecision_leakage", True, 0.0))
        abstained.append("predecision_leakage")

    pbas = paired_branch_action_sensitivity_test(
        syn_packs, real_intervention_log, rng=rng
    )
    results.append(pbas)
    if pbas.detail.get("n_pairs", 0.0) == 0.0:
        abstained.append("paired_branch_action_sensitivity")

    gat = generator_agreement_trap_test(
        syn_only_scores or {}, real_only_scores or {}
    )
    results.append(gat)
    if gat.detail.get("n_variants", 0.0) < 2:
        abstained.append("generator_agreement_trap")

    sov = support_overlap_and_provenance_test(real_X, syn_X, rng=rng)
    results.append(sov)
    if real_X.shape[0] < 4 or syn_X.shape[0] < 4:
        abstained.append("support_overlap_and_provenance")

    real_miss = _missingness_records_from_packs(real_packs)
    syn_miss = _missingness_records_from_packs(syn_packs)
    mss = missingness_semantics_test(real_miss, syn_miss, rng=rng)
    results.append(mss)
    if not real_miss or not syn_miss:
        abstained.append("missingness_semantics")

    return FidelityReport(results=results, ran=True, abstained=abstained)


def _lag_pairs_from_packs(
    packs: Sequence[CounterfactualPack],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Power→temperature parent→child series per feasible branch (doc R lag test).

    Reads the realised per-GPU DLC telemetry stream emitted during each branch
    rollout (doc R Eq17) and extracts the (power_usage, gpu_temp) pair so the
    lag-signature test can certify the thermal-transfer lag is preserved.
    """
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for pack in packs:
        for b in pack.branches:
            if not b.feasible or not b.observations:
                continue
            power: list[float] = []
            temp: list[float] = []
            for obs in b.observations:
                ch_p = _present_value(obs, "DCGM_FI_DEV_POWER_USAGE")
                ch_t = _present_value(obs, "DCGM_FI_DEV_GPU_TEMP")
                if ch_p is not None and ch_t is not None:
                    power.append(ch_p)
                    temp.append(ch_t)
            if len(power) >= 4:
                out.append((np.asarray(power), np.asarray(temp)))
    return out


def _present_value(obs: object, name: str) -> float | None:
    """Read a channel's value iff it is genuinely present (never zero-impute)."""
    chans = getattr(obs, "channels", None)
    if not chans:
        return None
    ch = chans.get(name)
    if ch is None or not getattr(ch, "available", True):
        return None
    return float(getattr(ch, "value", 0.0))


def _incident_bundles_from_packs(
    packs: Sequence[CounterfactualPack],
) -> list[dict[str, float]]:
    """Per-branch incident-descendant bundle vectors (doc R incident_bundle test).

    Builds ``B_f`` from each branch's realised hazards + missingness rate so the
    coherence test can require the expected fault descendants (Xid, requeue,
    pods-ready-timeout, hotspot, missingness) co-occur rather than appear as
    isolated spikes. Only incident-bearing branches contribute a bundle.
    """
    out: list[dict[str, float]] = []
    for pack in packs:
        for b in pack.branches:
            if not b.feasible:
                continue
            mr, _ = _missingness_summary(b.observations)
            haz = {f"haz:{k}": float(v) for k, v in b.hazards.items()}
            if max([0.0, *haz.values()]) <= 0.0 and mr <= 0.0:
                continue  # not an incident branch
            bundle = dict(haz)
            bundle["missingness_rate"] = mr
            out.append(bundle)
    return out


def _leakage_design(
    packs: Sequence[CounterfactualPack],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-only vs. pre+post feature designs for the leakage test (doc R #3).

    The pre-action design uses the static branch coordinates (feasibility + KPI
    deltas + slow-latent targets) measurable w.r.t. ``≤ τ⁻``; the augmented design
    appends realised post-action hazards/missingness — observations that should
    *not* improve a feasibility predictor if windows are leak-free. The label is
    the branch feasibility bit.
    """
    pre_rows: list[list[float]] = []
    post_rows: list[list[float]] = []
    labels: list[float] = []
    for pack in packs:
        for b in pack.branches:
            feat = branch_feature_dict(b)
            pre = [0.0] + [feat.get(f"kpi:{c}", 0.0) for c in KPI_LABEL_CHANNELS]
            pre += [feat.get(f"vq:{c}", 0.0) for c in VQ_LABEL_CHANNELS]
            post = pre + [feat.get(f"haz:{c}", 0.0) for c in HAZARD_LABEL_CHANNELS]
            post += [feat.get("telemetry_missing_rate", 0.0)]
            pre_rows.append(pre)
            post_rows.append(post)
            labels.append(1.0 if b.feasible else 0.0)
    if not pre_rows:
        return np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0,))
    return (
        np.asarray(pre_rows, dtype=float),
        np.asarray(post_rows, dtype=float),
        np.asarray(labels, dtype=float),
    )


def _missingness_records_from_packs(
    packs: Sequence[CounterfactualPack],
) -> list[tuple[int, int, int]]:
    """``(missing, incident, stratum)`` records per emitted telemetry frame (Eq20).

    For every per-GPU telemetry frame a branch emitted, ``missing`` is the
    channel-level missingness indicator (any degraded channel in the frame),
    ``incident`` is whether the branch realised a fault/requeue incident, and the
    stratum codes the branch's operator-caution band — so the CMI test can certify
    missingness stays MNAR-coupled to incident state rather than benign noise.
    """
    out: list[tuple[int, int, int]] = []
    for pack in packs:
        for b in pack.branches:
            if not b.feasible or not b.observations:
                continue
            incident = int(
                b.hazards.get(HazardChannel.XID_FAULT.value, 0.0) > 0.0
                or b.hazards.get(HazardChannel.REQUEUE.value, 0.0) > 0.0
                or b.hazards.get(HazardChannel.NODE_FAILURE.value, 0.0) > 0.0
            )
            caution = b.vq_features.get("operator_caution", 0.0)
            stratum = 1 if caution >= 0.5 else 0
            for obs in b.observations:
                chans = getattr(obs, "channels", None)
                if not chans:
                    continue
                miss = 0
                for ch in chans.values():
                    avail = getattr(ch, "available", True)
                    sh = getattr(ch, "source_health", SourceHealth.OBSERVED)
                    if (not avail) or sh in (
                        SourceHealth.STALE,
                        SourceHealth.DROPPED,
                    ):
                        miss = 1
                        break
                out.append((miss, incident, stratum))
    return out


def synthetic_mixing_weight(
    real: np.ndarray,
    syn: np.ndarray,
    report: FidelityReport | None,
    *,
    omega_max: float = 10.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Eq27 importance-reweighting term ``ω_g(x)=clip(r_g(x),0,ω_max)·c_g``.

    Realises the doc-R real/synthetic mixing weight: the per-synthetic-sample
    density ratio ``r_g(x)≈p_real(x|g)/q_syn(x|g)`` (estimated via the origin
    classifier of :func:`density_ratio_weights`) clipped to ``[0,ω_max]`` and
    multiplied by the stratum certification gate ``c_g∈[0,1]`` taken from
    ``report.gate()``. If the stratum fails symbolic fidelity, poisoning, or
    counterfactual fidelity (``report`` uncertified/None), ``c_g=0`` and every
    weight is zeroed — the procedure reverts toward real-only training (T10).
    """
    rng = rng or np.random.default_rng(0)
    c_g = report.gate() if report is not None else 0.0
    if c_g == 0.0 or syn.shape[0] == 0:
        return np.zeros(syn.shape[0])
    weights, _ = density_ratio_weights(real, syn, omega_max=omega_max, rng=rng)
    return weights * c_g


__all__ = [
    "CERTIFICATION_TESTS",
    "HAZARD_LABEL_CHANNELS",
    "KPI_LABEL_CHANNELS",
    "POISONING_TESTS",
    "VQ_LABEL_CHANNELS",
    "FidelityReport",
    "TestResult",
    "branch_feature_dict",
    "classifier_two_sample_test",
    "counterfactual_fidelity_test",
    "density_ratio_weights",
    "effective_sample_size",
    "embed_windows",
    "energy_distance",
    "energy_distance_test",
    "exact_kernel_conformance",
    "generator_agreement_trap_test",
    "incident_bundle_test",
    "lag_structure_test",
    "make_label_task",
    "missingness_semantics_test",
    "mmd2_test",
    "mmd2_unbiased",
    "pack_features",
    "paired_branch_action_sensitivity_test",
    "predecision_leakage_test",
    "run_fidelity_suite",
    "support_overlap_and_provenance_test",
    "synthetic_mixing_weight",
    "tstr_trts",
    "tstr_trts_over_labels",
]
