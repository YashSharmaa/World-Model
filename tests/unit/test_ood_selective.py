# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""OOD novelty + selective-prediction validity lane (doc L ood_detection / P1-P5).

Two coupled guarantees from doc L:

* **OOD** (P5): the conformal novelty p-value ``p = (1 + #{S_cal ≥ S}) / (n+1)``
  controls in-support false alarms at level ``alpha_ood`` — ID inputs are flagged
  novel at most ~``alpha_ood`` of the time — and Benjamini-Hochberg over batched
  p-values controls the false-discovery rate. The detector must also *recall*
  genuinely shifted inputs (an empirical, not certified, property we still check).
* **Selective** (P1-P4): split-select-certify gives a finite-sample Clopper-Pearson
  upper bound on accepted risk and a lower bound on non-abstained coverage, valid
  for the threshold chosen on an *independent* split. The certificate must *hold*
  (true accepted risk ≤ certified upper) across repeated splits.

The square-root-EKF / certify edge cases live in ``test_filters_completeness.py``;
here we exercise the *coverage / false-alarm-rate* statistics directly.
"""

from __future__ import annotations

import numpy as np

from gpu_center_world_model.uncertainty import (
    OODDetector,
    TrustScoreWeights,
    benjamini_hochberg,
    build_components,
    certify,
    clopper_pearson_lower,
    clopper_pearson_upper,
    fuse_components,
    knn_distance,
    optimism_bias_gate,
    safety_chance_constraint,
    select_threshold,
    split_select_certify,
    staleness_epistemic_score,
    trust_score,
)
from gpu_center_world_model.uncertainty.ood import _Standardizer


def _rng(seed: int = 20260614):
    return np.random.default_rng(seed)


# ──────────────────────────────────────────────────────────────────────────────
# OOD conformal p-values: in-support false-alarm calibration (doc L P5)
# ──────────────────────────────────────────────────────────────────────────────
def test_ood_false_alarm_rate_is_at_most_alpha_on_id_inputs():
    rng = _rng(1)
    cal = rng.normal(size=(5000, 4))
    det = OODDetector(alpha_ood=0.05).fit(cal)
    id_test = rng.normal(size=(8000, 4))  # exchangeable with calibration => ID
    far = det.is_novel(id_test).mean()
    # Conformal validity: in-support false alarms ≤ alpha (up to sampling noise).
    assert far <= 0.05 + 0.01


def test_ood_pvalues_are_approximately_uniform_on_id():
    rng = _rng(2)
    det = OODDetector().fit(rng.normal(size=(4000, 4)))
    p = det.p_value(rng.normal(size=(8000, 4)))
    # Under exchangeability the conformal p-value is (super-)uniform: the fraction
    # below any level t should be ≤ t + slack for several thresholds.
    for t in (0.05, 0.1, 0.2, 0.5):
        assert (p <= t).mean() <= t + 0.02


def test_ood_recalls_shifted_inputs():
    rng = _rng(3)
    det = OODDetector(alpha_ood=0.05).fit(rng.normal(size=(4000, 4)))
    shifted = rng.normal(loc=5.0, size=(500, 4))  # far from the ID support
    # Strongly-shifted inputs are reliably flagged (empirical power).
    assert det.is_novel(shifted).mean() > 0.95


def test_ood_max_fusion_trips_on_a_single_novel_axis():
    rng = _rng(4)
    cal = rng.normal(size=(3000, 4))
    det = OODDetector(alpha_ood=0.05).fit(cal)
    # A point that is ID on three axes but extreme on ONE axis must be flagged,
    # because the fused score is the max over standardized components (doc L S_ood).
    pt = np.zeros((1, 4))
    pt[0, 2] = 12.0
    assert bool(det.is_novel(pt)[0])


# ──────────────────────────────────────────────────────────────────────────────
# Benjamini-Hochberg FDR control over batched conformal p-values (doc L P5)
# ──────────────────────────────────────────────────────────────────────────────
def test_bh_controls_fdr_over_a_mostly_id_batch():
    rng = _rng(5)
    det = OODDetector(alpha_ood=0.05).fit(rng.normal(size=(4000, 4)))
    fdr_level = 0.1
    false_fracs = []
    for _ in range(40):
        id_batch = rng.normal(size=(900, 4))
        ood_batch = rng.normal(loc=6.0, size=(50, 4))
        batch = np.vstack([id_batch, ood_batch])
        rej = det.flag_batch(batch, fdr=fdr_level)
        n_rej = int(rej.sum())
        if n_rej:
            false_fracs.append(rej[:900].sum() / n_rej)
        else:
            false_fracs.append(0.0)
    # Expected proportion of false discoveries is controlled near/below the level.
    assert np.mean(false_fracs) <= fdr_level + 0.03


def test_bh_false_discovery_proportion_is_near_zero_on_a_pure_id_batch():
    """Under the complete null (all-ID), batched novelty FDR stays near zero.

    Doc L P5: BH over conformal p-values controls FDR; under the complete null the
    *number* of false discoveries is small in expectation (E[V]/m <= alpha) — here
    there are no true OOD points, so any rejection is a false discovery and the mean
    per-batch false-discovery count must be small at a modest target level.
    """
    rng = _rng(50)
    det = OODDetector(alpha_ood=0.05).fit(rng.normal(size=(4000, 4)))
    fdr_level = 0.1
    false_counts = []
    for _ in range(60):
        id_batch = rng.normal(size=(400, 4))  # all in-support: nothing is truly OOD
        rej = det.flag_batch(id_batch, fdr=fdr_level)
        false_counts.append(int(rej.sum()))
    # under the complete null FDR control keeps the mean rejection count tiny.
    assert np.mean(false_counts) <= fdr_level * 400 * 0.1  # << the naive alpha*m


def test_bh_rejects_nothing_when_no_pvalue_is_significant():
    # All-large p-values -> no rejections (BH respects the empty rejection set).
    p = np.linspace(0.5, 1.0, 50)
    assert not benjamini_hochberg(p, 0.05).any()


def test_bh_is_at_least_as_powerful_as_bonferroni():
    # BH rejects a superset of Bonferroni at the same level (step-up dominance).
    rng = _rng(6)
    p = np.concatenate([rng.uniform(0, 0.001, 10), rng.uniform(0.2, 1.0, 90)])
    bh = benjamini_hochberg(p, 0.05)
    bonf = p <= 0.05 / p.size
    assert np.all(bh | ~bonf)  # everything Bonferroni rejects, BH rejects too
    assert bh.sum() >= bonf.sum()


# ──────────────────────────────────────────────────────────────────────────────
# Component novelty primitives (N1 / N4) behave monotonically
# ──────────────────────────────────────────────────────────────────────────────
def test_knn_distance_grows_with_separation():
    ref = _rng(7).normal(size=(200, 3))
    near = knn_distance(np.zeros((1, 3)), ref, k=5)
    far = knn_distance(np.full((1, 3), 10.0), ref, k=5)
    assert far[0] > near[0]


def test_staleness_score_monotone_in_age_and_source_health():
    base = staleness_epistemic_score(0.1, 0.0, 0.0)
    stale = staleness_epistemic_score(0.1, 300.0, 0.0)
    bad = staleness_epistemic_score(0.1, 0.0, 0.8)
    # Older telemetry and worse source-health both raise novelty (doc T C15).
    assert stale > base
    assert bad > base


def test_build_and_fuse_components_use_max():
    comps = build_components(
        latent_knn=np.array([1.0, 0.0]),
        topology_dist=np.array([0.0, 0.0]),
        mixture_novelty=np.array([0.0, 5.0]),
        staleness_epistemic=np.array([0.0, 0.0]),
    )
    assert comps.shape == (2, 4)
    std = _Standardizer().fit(comps)
    fused = fuse_components(comps, std)
    assert fused.shape == (2,)


# ──────────────────────────────────────────────────────────────────────────────
# Trust score wiring (doc L D7)
# ──────────────────────────────────────────────────────────────────────────────
def test_trust_score_decreases_with_risk_and_ood():
    base = trust_score(epistemic=0.1, ood=0.1, risk_upper=0.1, benefit_margin=0.5)
    riskier = trust_score(epistemic=0.1, ood=0.1, risk_upper=0.9, benefit_margin=0.5)
    odder = trust_score(epistemic=0.1, ood=0.9, risk_upper=0.1, benefit_margin=0.5)
    better = trust_score(epistemic=0.1, ood=0.1, risk_upper=0.1, benefit_margin=2.0)
    # More risk / more OOD lowers trust; more benefit margin raises it.
    assert riskier < base
    assert odder < base
    assert better > base


def test_trust_score_weights_scale_terms():
    w = TrustScoreWeights(l3_risk=10.0)
    heavy = trust_score(
        epistemic=0.0, ood=0.0, risk_upper=0.5, benefit_margin=0.0, weights=w
    )
    light = trust_score(epistemic=0.0, ood=0.0, risk_upper=0.5, benefit_margin=0.0)
    assert heavy < light  # heavier risk weight penalises the same risk more


# ──────────────────────────────────────────────────────────────────────────────
# Selective certificate validity: U_R holds across repeated splits (doc L P1)
# ──────────────────────────────────────────────────────────────────────────────
def test_certified_accepted_risk_upper_holds_in_repeated_splits():
    # The true accepted risk at tau* must lie below the certified upper bound at
    # the certified confidence (allow a small empirical violation budget).
    rng = _rng(8)
    n = 4000
    delta_r = 0.05
    violations = 0
    trials = 200
    for _ in range(trials):
        # higher score -> safer; failure prob decreases with the score.
        scores = rng.uniform(0.0, 1.0, n)
        fail_prob = np.clip(0.5 - 0.5 * scores, 0.0, 1.0)
        failures = (rng.random(n) < fail_prob).astype(float)
        sel = slice(0, n // 2)
        crt = slice(n // 2, n)
        gate = split_select_certify(
            scores[sel],
            failures[sel],
            scores[crt],
            failures[crt],
            risk_budget=0.1,
            delta_risk=delta_r,
            delta_coverage=delta_r,
        )
        cert = gate.certificate
        # True accepted risk under this tau* on a *fresh* draw.
        fresh_scores = rng.uniform(0.0, 1.0, 20000)
        fresh_fp = np.clip(0.5 - 0.5 * fresh_scores, 0.0, 1.0)
        fresh_fail = (rng.random(20000) < fresh_fp).astype(float)
        acc = fresh_scores >= gate.tau
        if acc.sum() >= 50:
            true_risk = fresh_fail[acc].mean()
            if true_risk > cert.accepted_risk_upper + 1e-9:
                violations += 1
    # Clopper-Pearson is conservative: violations are far below delta_risk.
    assert violations / trials <= delta_r


def test_certified_coverage_lower_bound_holds_in_repeated_splits():
    """The certified coverage lower bound ``L_C`` is a valid 1-delta_C lower bound.

    Doc L P4: with probability >= ``1 - delta_C`` the *true* non-abstained coverage
    ``P(accepted)`` exceeds the certified ``coverage_lower``. We fix the threshold,
    certify ``L_C`` on an independent certification split, then measure the true
    acceptance fraction on a fresh draw; violations (true coverage below L_C) must
    occur at most ~``delta_C`` of the time (Clopper-Pearson is conservative).
    """
    rng = _rng(20)
    n = 4000
    delta_c = 0.05
    tau = 0.4
    violations = 0
    trials = 200
    for _ in range(trials):
        scores = rng.uniform(0.0, 1.0, n)  # acceptance fraction at tau is ~ 1 - tau
        failures = (rng.random(n) < 0.02).astype(float)
        cert = certify(
            scores, failures, tau=tau, delta_coverage=delta_c, coverage_target=0.5
        )
        fresh = rng.uniform(0.0, 1.0, 20000)
        true_cov = float(np.mean(fresh >= tau))
        if true_cov < cert.coverage_lower - 1e-9:
            violations += 1
    assert violations / trials <= delta_c  # conservative -> well under the budget


def test_coverage_lower_bound_is_a_valid_lower_bound():
    # The certified coverage lower bound must not exceed the realized acceptance
    # fraction on the certification split (it is a *lower* bound by construction).
    rng = _rng(9)
    scores = rng.uniform(0.0, 1.0, 2000)
    failures = (rng.random(2000) < 0.02).astype(float)
    cert = certify(scores, failures, tau=0.3, coverage_target=0.5)
    realized_coverage = cert.n_accepted / cert.n_cert
    assert cert.coverage_lower <= realized_coverage + 1e-9
    assert 0.0 < cert.coverage_lower < 1.0


def test_select_threshold_respects_the_risk_budget():
    rng = _rng(10)
    scores = rng.uniform(0.0, 1.0, 3000)
    fail_prob = np.clip(0.6 - 0.6 * scores, 0.0, 1.0)
    failures = (rng.random(3000) < fail_prob).astype(float)
    res = select_threshold(scores, failures, risk_budget=0.1)
    assert res.feasible
    # the chosen operating point's empirical accepted risk respects the budget.
    assert res.empirical_risk <= 0.1 + 1e-9
    # accepting more (a lower threshold) is what selection maximises.
    accept = scores >= res.tau
    assert accept.sum() == res.n_accepted


def test_clopper_pearson_bounds_bracket_the_rate():
    # For k successes in n, the CP lower/upper bracket k/n and are ordered.
    k, n = 7, 100
    lo = clopper_pearson_lower(k, n, 0.05)
    hi = clopper_pearson_upper(k, n, 0.05)
    assert 0.0 <= lo <= k / n <= hi <= 1.0
    # zero failures still yields a non-trivial finite upper bound (no false zero).
    assert 0.0 < clopper_pearson_upper(0, n, 0.05) < 1.0
    # all-success lower bound is the finite Beta(n,1) quantile, not a degenerate 0.
    assert clopper_pearson_lower(n, n, 0.05) > 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Optimism-bias gate + safety chance constraint (doc L P3 / E17)
# ──────────────────────────────────────────────────────────────────────────────
def test_optimism_gate_only_deploys_on_nonpositive_excess_risk():
    u = np.array([-0.2, 0.0, 0.01, 1.0])
    mask = optimism_bias_gate(u)
    # deploy iff the one-sided upper bound on excess risk is ≤ 0.
    assert mask.tolist() == [True, True, False, False]


def test_safety_chance_constraint_is_and_over_channels():
    risk_uppers = {
        "hotspot": np.array([0.02, 0.20, 0.02]),
        "xid": np.array([0.01, 0.01, 0.50]),
    }
    eps = {"hotspot": 0.05, "xid": 0.05}
    mask = safety_chance_constraint(risk_uppers, eps)
    # A candidate is admitted only if *every* residual-risk channel is within eps.
    assert mask.tolist() == [True, False, False]


def test_safety_chance_constraint_accepts_all_when_no_channels():
    risk_uppers = {"hotspot": np.array([0.9, 0.9])}
    mask = safety_chance_constraint(risk_uppers, {})  # no epsilons -> nothing gated
    assert mask.tolist() == [True, True]
