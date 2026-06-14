# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""DE-FIDELITY-CERTIFICATION — the doc-R fidelity + poisoning battery.

Verifies the fidelity-certification orchestrator and the newly added tests:

* :func:`run_fidelity_suite` embeds packs once, runs every certification +
  poisoning test, and returns a populated :class:`FidelityReport`;
* the four previously-missing poisoning tests
  (``paired_branch_action_sensitivity``, ``generator_agreement_trap``,
  ``support_overlap_and_provenance`` with r(x)/ESS/clipped weights,
  ``missingness_semantics`` with CMI + missingness-conditioned C2ST) each *detect*
  their targeted poisoning, not just abstain;
* the counterfactual-fidelity DR test flags an inflated synthetic contrast;
* the TSTR/TRTS harness runs over the doc-L label channels;
* :func:`synthetic_mixing_weight` realises Eq27 ``ω_g(x)=clip(r,0,ω_max)·c_g``
  and zeroes weights when certification fails (the c_g gate); and
* :func:`build_demo_release` ships a *certified* (not ``fidelity=None``) demo
  release whose data card records the certification status + coverage.
"""

from __future__ import annotations

import tempfile

import numpy as np

from gpu_center_world_model.data_engine.release_manifest import (
    build_demo_release,
    certify_demo_packs,
)
from gpu_center_world_model.data_engine.synthetic import generate_pack
from gpu_center_world_model.data_engine.synthetic.fidelity import (
    CERTIFICATION_TESTS,
    POISONING_TESTS,
    FidelityReport,
    branch_feature_dict,
    counterfactual_fidelity_test,
    density_ratio_weights,
    effective_sample_size,
    generator_agreement_trap_test,
    missingness_semantics_test,
    pack_features,
    paired_branch_action_sensitivity_test,
    run_fidelity_suite,
    support_overlap_and_provenance_test,
    synthetic_mixing_weight,
    tstr_trts_over_labels,
)
from gpu_center_world_model.data_engine.synthetic.fidelity import (
    TestResult as FidTestResult,
)


def _packs(scenario: str, seeds=(0, 1)):
    return [generate_pack(seed=s, scenario=scenario) for s in seeds]


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def test_run_fidelity_suite_covers_all_thirteen_tests():
    real = _packs("normal")
    syn = _packs("thermal_stress")
    rep = run_fidelity_suite(real, syn)
    names = {r.name for r in rep.results}
    # every doc-R certification + poisoning test name is present (conformance is
    # only present when an oracle is supplied; it abstains by omission otherwise).
    for name in POISONING_TESTS:
        assert name in names, name
    for name in CERTIFICATION_TESTS:
        if name == "exact_kernel_conformance":
            continue
        assert name in names, name
    assert isinstance(rep.certified, bool)
    cov = rep.coverage
    assert set(cov) == set(CERTIFICATION_TESTS + POISONING_TESTS)


def test_run_fidelity_suite_self_consistency_is_certified():
    # Two disjoint synthetic draws of the SAME family should pass the two-sample
    # certification tests (P_real == P_syn on the embedding); abstaining tests
    # (no real intervention log / single feasible branch) pass too.
    a = _packs("thermal_stress", seeds=(0, 1, 2))
    b = _packs("thermal_stress", seeds=(3, 4, 5))
    rep = run_fidelity_suite(a, b)
    failed = [r.name for r in rep.results if not r.passed]
    assert failed == [], failed
    assert rep.certified


def test_report_gate_matches_certification():
    good = FidelityReport(results=[FidTestResult("x", True, 0.0)])
    bad = FidelityReport(results=[FidTestResult("x", False, 1.0)])
    assert good.gate() == 1.0 and good.certified
    assert bad.gate() == 0.0 and not bad.certified
    # a not-ran report is never certified (the legacy fidelity=None case).
    assert not FidelityReport(results=[], ran=False).certified


# ──────────────────────────────────────────────────────────────────────────────
# Poisoning test: support overlap + provenance (r(x), ESS, clipped weights)
# ──────────────────────────────────────────────────────────────────────────────
def test_support_overlap_detects_disjoint_support():
    rng = np.random.default_rng(0)
    real = rng.normal(0.0, 1.0, size=(200, 4))
    # synthetic shifted far away ⇒ near-zero overlap ⇒ origin AUC ~ 1 ⇒ flagged.
    syn = rng.normal(20.0, 1.0, size=(200, 4))
    res = support_overlap_and_provenance_test(real, syn)
    assert not res.passed
    assert res.detail["origin_auc"] > 0.95


def test_support_overlap_passes_on_overlapping_support():
    rng = np.random.default_rng(1)
    real = rng.normal(0.0, 1.0, size=(200, 4))
    syn = rng.normal(0.0, 1.0, size=(200, 4))
    res = support_overlap_and_provenance_test(real, syn)
    assert res.passed
    assert res.detail["ess_frac"] > 0.1


def test_density_ratio_weights_and_ess():
    rng = np.random.default_rng(2)
    real = rng.normal(0.0, 1.0, size=(150, 3))
    syn = rng.normal(0.0, 1.0, size=(150, 3))
    w, auc = density_ratio_weights(real, syn, omega_max=10.0)
    assert w.shape == (150,)
    assert np.all(w >= 0.0) and np.all(w <= 10.0)
    ess = effective_sample_size(w)
    # overlapping ⇒ ratios near 1 ⇒ ESS close to full sample size.
    assert 0.0 < ess <= 150.0 + 1e-6
    assert 0.3 <= auc <= 0.7


# ──────────────────────────────────────────────────────────────────────────────
# Poisoning test: missingness semantics (CMI + missingness-conditioned C2ST)
# ──────────────────────────────────────────────────────────────────────────────
def test_missingness_semantics_flags_independence():
    rng = np.random.default_rng(3)
    n = 400
    # REAL: missingness is coupled to incidents (MNAR) — missing rises with fault.
    real = []
    for _ in range(n):
        inc = int(rng.random() < 0.3)
        miss = int(rng.random() < (0.7 if inc else 0.05))
        real.append((miss, inc, 0))
    # SYN (poisoned): missingness independent of incident (benign cleaning noise).
    syn = []
    for _ in range(n):
        inc = int(rng.random() < 0.3)
        miss = int(rng.random() < 0.3)
        syn.append((miss, inc, 0))
    res = missingness_semantics_test(real, syn)
    assert not res.passed
    assert res.detail["cmi_real"] > res.detail["cmi_syn"]


def test_missingness_semantics_passes_when_coupling_matches():
    rng = np.random.default_rng(4)
    n = 400

    def draw():
        out = []
        for _ in range(n):
            inc = int(rng.random() < 0.3)
            miss = int(rng.random() < (0.7 if inc else 0.05))
            out.append((miss, inc, 0))
        return out

    res = missingness_semantics_test(draw(), draw())
    assert res.passed


# ──────────────────────────────────────────────────────────────────────────────
# Poisoning test: generator-agreement trap (rank correlation)
# ──────────────────────────────────────────────────────────────────────────────
def test_generator_agreement_trap_flags_misalignment():
    # synthetic-only leaderboard reverses the real-only ranking ⇒ rho < 0.
    syn = {"g0": 1.0, "g1": 2.0, "g2": 3.0, "g3": 4.0}
    real = {"g0": 4.0, "g1": 3.0, "g2": 2.0, "g3": 1.0}
    res = generator_agreement_trap_test(syn, real)
    assert not res.passed
    assert res.statistic < 0.0


def test_generator_agreement_trap_passes_when_aligned():
    syn = {"g0": 1.0, "g1": 2.0, "g2": 3.0, "g3": 4.0}
    real = {"g0": 1.1, "g1": 2.2, "g2": 2.9, "g3": 4.3}
    res = generator_agreement_trap_test(syn, real)
    assert res.passed and res.statistic >= 0.5


def test_generator_agreement_trap_abstains_below_two_variants():
    res = generator_agreement_trap_test({"g0": 1.0}, {"g0": 1.0})
    assert res.passed and res.detail["n_variants"] < 2


# ──────────────────────────────────────────────────────────────────────────────
# Poisoning test: paired-branch action sensitivity
# ──────────────────────────────────────────────────────────────────────────────
def _pack_with_pairs(contrast: float, state_hash: str = "h"):
    """A synthetic pack whose three feasible branches differ in admitted GPU-hours."""
    from gpu_center_world_model.data_engine.synthetic.counterfactual import (
        BranchOutcome,
        CounterfactualPack,
    )
    from gpu_center_world_model.schema.kpis import KPIChannel

    ch = KPIChannel.ADMITTED_GPU_HOURS_DELTA.value
    pack = CounterfactualPack(
        state_hash=state_hash,
        seed_bundle=0,
        horizon_s=300,
        baseline_action_id="a0",
    )
    for i in range(3):
        pack.branches.append(
            BranchOutcome(
                action_id=f"a{i}",
                action=None,  # type: ignore[arg-type]
                feasible=True,
                kpis={ch: contrast * i},
            )
        )
    return pack


def test_paired_branch_flags_collapse():
    # all branches identical ⇒ contrasts collapse toward zero ⇒ flagged.
    packs = [_pack_with_pairs(0.0, state_hash=f"h{i}") for i in range(3)]
    res = paired_branch_action_sensitivity_test(packs)
    assert not res.passed
    assert res.detail["collapsed"] == 1.0


def test_paired_branch_passes_with_action_effect():
    # distinct pre-action hashes (real packs have distinct hashes) ⇒ a stable
    # nonzero action effect is correct, not "deterministic" poisoning.
    packs = [_pack_with_pairs(5.0, state_hash=f"h{i}") for i in range(3)]
    res = paired_branch_action_sensitivity_test(packs)
    assert res.passed
    assert res.detail["mean_abs_contrast"] > 0.0


def test_paired_branch_flags_determinism_across_repeated_hash():
    # the SAME pre-action hash repeated with zero contrast variance ⇒ the
    # generator ignores Û ⇒ implausibly deterministic ⇒ flagged.
    packs = [_pack_with_pairs(5.0, state_hash="same") for _ in range(4)]
    res = paired_branch_action_sensitivity_test(packs)
    assert not res.passed
    assert res.detail["deterministic"] == 1.0


def test_paired_branch_abstains_without_pairs():
    # a single feasible branch per pack ⇒ no comparable pairs ⇒ abstain (pass).
    real = _packs("normal")  # demo packs have one feasible branch (hold)
    res = paired_branch_action_sensitivity_test(real)
    assert res.passed and res.detail["n_pairs"] == 0.0


def test_paired_branch_flags_sign_flip_vs_real():
    # distinct hashes (no determinism signal) so only the sign flip fires.
    packs = [_pack_with_pairs(5.0, state_hash=f"h{i}") for i in range(2)]
    # syn Δ(a0,a2) = -10 (negative); real DR contrast has the OPPOSITE sign.
    from gpu_center_world_model.schema.kpis import KPIChannel

    ch = KPIChannel.ADMITTED_GPU_HOURS_DELTA.value
    real_log = {
        ch: [("a0", 100.0, 0.5), ("a0", 100.0, 0.5), ("a2", 0.0, 0.5), ("a2", 0.0, 0.5)]
    }
    res = paired_branch_action_sensitivity_test(
        packs, real_log, functionals=(ch,)
    )
    assert not res.passed
    assert res.detail["sign_flips"] >= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Certification test: counterfactual fidelity (doubly robust)
# ──────────────────────────────────────────────────────────────────────────────
def test_counterfactual_fidelity_flags_inflated_contrast():
    from gpu_center_world_model.schema.kpis import KPIChannel

    ch = KPIChannel.ADMITTED_GPU_HOURS_DELTA.value
    # synthetic branch contrast Δ(a0,a2) = -10000 (wildly inflated).
    pack = _pack_with_pairs(5000.0)
    # real held-out DR contrast is ~0 with tight SE ⇒ huge standardised error.
    real_log = {
        ch: [("a0", 1.0, 0.5)] * 10 + [("a2", 1.0, 0.5)] * 10,
    }
    res = counterfactual_fidelity_test([pack], real_log, functionals=(ch,))
    assert not res.passed
    assert res.detail["n_compared"] >= 1


def test_counterfactual_fidelity_passes_when_contrast_matches():
    from gpu_center_world_model.schema.kpis import KPIChannel

    ch = KPIChannel.ADMITTED_GPU_HOURS_DELTA.value
    pack = _pack_with_pairs(1.0)  # Δ(a0,a2) = -2
    real_log = {
        ch: [("a0", 0.0, 0.5)] * 20 + [("a2", 2.0, 0.5)] * 20,  # real Δ = -2
    }
    res = counterfactual_fidelity_test([pack], real_log, functionals=(ch,))
    assert res.passed


def test_counterfactual_fidelity_abstains_without_real_log():
    res = counterfactual_fidelity_test(_packs("thermal_stress"), {})
    assert res.passed and res.detail["n_compared"] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# TSTR/TRTS harness over the doc-L label channels
# ──────────────────────────────────────────────────────────────────────────────
def test_tstr_trts_over_labels_runs_on_doc_l_channels():
    real = pack_features(_packs("normal", seeds=(0, 1, 2)))
    syn = pack_features(_packs("thermal_stress", seeds=(0, 1, 2)))
    res = tstr_trts_over_labels(real, syn)
    assert res.name == "tstr_trts"
    assert "worst_delta_tstr" in res.detail


def test_branch_feature_dict_exposes_doc_l_channels():
    pack = generate_pack(seed=0, scenario="fault_stress")
    feat = branch_feature_dict(pack.branches[0])
    assert "feasible" in feat
    assert any(k.startswith("kpi:") for k in feat)
    assert any(k.startswith("haz:") for k in feat)
    assert any(k.startswith("vq:") for k in feat)
    assert "telemetry_missing_rate" in feat


# ──────────────────────────────────────────────────────────────────────────────
# Eq27 mixing weight + certification gate
# ──────────────────────────────────────────────────────────────────────────────
def test_synthetic_mixing_weight_zeroed_when_uncertified():
    rng = np.random.default_rng(5)
    real = rng.normal(0.0, 1.0, size=(50, 3))
    syn = rng.normal(0.0, 1.0, size=(50, 3))
    bad = FidelityReport(results=[FidTestResult("x", False, 1.0)])
    w = synthetic_mixing_weight(real, syn, bad)
    assert np.all(w == 0.0)  # c_g = 0 ⇒ revert to real-only training (T10)
    # None report behaves like an uncertified stratum.
    assert np.all(synthetic_mixing_weight(real, syn, None) == 0.0)


def test_synthetic_mixing_weight_clips_and_gates_when_certified():
    rng = np.random.default_rng(6)
    real = rng.normal(0.0, 1.0, size=(120, 3))
    syn = rng.normal(0.0, 1.0, size=(120, 3))
    good = FidelityReport(results=[FidTestResult("x", True, 0.0)])
    w = synthetic_mixing_weight(real, syn, good, omega_max=10.0)
    assert w.shape == (120,)
    assert np.all(w >= 0.0) and np.all(w <= 10.0)
    assert np.any(w > 0.0)  # certified overlapping stratum keeps positive weight


# ──────────────────────────────────────────────────────────────────────────────
# Demo release is certified (not fidelity=None)
# ──────────────────────────────────────────────────────────────────────────────
def test_build_demo_release_is_certified_not_none():
    scenarios = ("normal", "thermal_stress", "fault_stress", "scheduling_stress")
    with tempfile.TemporaryDirectory() as d:
        res = build_demo_release(d, seeds=(0, 1), scenarios=scenarios)
    card = res["data_card"]
    # the gate is no longer defeated by fidelity=None: the card records a real
    # certification status, coverage, and the abstention ledger.
    assert "fidelity_certified" in card
    assert card["fidelity_ran"] is True
    assert "fidelity_coverage" in card
    assert "fidelity_abstained" in card
    assert card["fidelity_certified"] is True  # demo certifies (with abstentions)
    assert card["fidelity_gate_c_g"] == 1.0


def test_certify_demo_packs_runs_conformance_audit():
    packs = [
        (scen, s, generate_pack(seed=s, scenario=scen))
        for scen in ("normal", "thermal_stress")
        for s in (0, 1)
    ]
    rep = certify_demo_packs(packs)
    names = {r.name for r in rep.results}
    # the deterministic-zero exact-kernel conformance audit actually ran.
    assert "exact_kernel_conformance" in names
    conf = next(r for r in rep.results if r.name == "exact_kernel_conformance")
    assert conf.passed and conf.statistic == 0.0


def test_uncertified_demo_release_marks_card_explicitly():
    scenarios = ("normal", "thermal_stress", "fault_stress", "scheduling_stress")
    with tempfile.TemporaryDirectory() as d:
        res = build_demo_release(d, seeds=(0, 1), scenarios=scenarios, certify=False)
    card = res["data_card"]
    # certify=False ⇒ fidelity=None path ⇒ explicitly uncertified, not omitted.
    assert card["fidelity_certified"] is False
    assert card["fidelity_ran"] is False
