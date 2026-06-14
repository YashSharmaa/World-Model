# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Dependent-data sample-complexity and effective-sample-size calculators (doc O).

Decision windows are temporally dependent and pairwise comparisons inside a
window are clustered, so naive i.i.d. counts overstate statistical power. Doc O
gives the exact corrections; this module implements them as plain closed-form
NumPy/math so a release can *certify* whether it has enough effective samples for
each acceptance claim, and *plan* the held-out test sizes.

Key quantities (doc O ``sample_complexity``):

* **design effect** ``deff = 1 + (m-1) * rho_pair`` and effective pairwise sample
  size ``N_pair_eff = W * m / deff`` (D8-D9), where ``W`` windows each provide
  ``m = B(B-1)/2`` pairwise comparisons among ``B`` branched safe actions;
* **weighted ESS** ``n_ESS = (sum w)^2 / sum w^2`` for real+synthetic mixing
  (D10);
* **mixing ESS** ``n_eff_mix ~= n / tau_mix`` for temporally dependent windows
  (E2);
* **one-sided Bernoulli certification** ``n_eff >= log(1/delta)/(2 gamma^2)``
  (E5; the famous ~600 effective pairwise comparisons for a 0.70 sign-accuracy
  claim with a 0.05 margin);
* **two-sided coverage** ``n_eff >= log(2/delta)/(2 eps^2)`` (E6; ~738 windows);
* **rule of three** ``upper95 ~= 3/N`` for zero-event false-safe (E7; N~=300);
* **Bernstein paired planning-utility** condition (E?: planning-utility row).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────────────────
# Effective sample sizes
# ──────────────────────────────────────────────────────────────────────────────
def pairwise_per_window(n_branches: int) -> int:
    """``m(B) = B(B-1)/2`` pairwise comparisons available per window (D8)."""
    if n_branches < 2:
        return 0
    return n_branches * (n_branches - 1) // 2


def design_effect(m_pairs: int, rho_pair: float) -> float:
    """Cluster design effect ``deff = 1 + (m-1) * rho_pair`` (D9).

    ``rho_pair`` is the intra-window correlation of pairwise outcomes; doc O uses
    rho in [0.2, 0.6] as a realistic band.
    """
    if m_pairs <= 1:
        return 1.0
    return 1.0 + (m_pairs - 1) * rho_pair


def effective_pairwise_samples(
    n_windows: int, n_branches: int, rho_pair: float
) -> float:
    """``N_pair_eff = W * m / deff`` — effective pairwise sample size (D9).

    This is the number that must clear the certification thresholds, *not* the
    raw ``W * m`` count.
    """
    m = pairwise_per_window(n_branches)
    if m == 0:
        return 0.0
    deff = design_effect(m, rho_pair)
    return n_windows * m / deff


def weighted_ess(weights: list[float]) -> float:
    """Kish weighted effective sample size ``(sum w)^2 / sum w^2`` (D10).

    For real+synthetic mixing: equal weights give ``n_ESS = n`` (full power);
    highly skewed weights collapse toward the count of dominant samples.
    """
    s1 = sum(weights)
    s2 = sum(w * w for w in weights)
    if s2 == 0.0:
        return 0.0
    return (s1 * s1) / s2


def mixing_effective_n(n: int, tau_mix: float) -> float:
    """``n_eff_mix ~= n / tau_mix`` — dependence-inflated ESS (E2).

    ``tau_mix = 1 + 2 * sum_k beta(k)`` is the dependence inflation factor; for
    geometric mixing ``beta(k) <= C rho^k`` it is finite and equals
    ``1 + 2 C rho / (1 - rho)``.
    """
    if tau_mix <= 0:
        raise ValueError("tau_mix must be positive")
    return n / tau_mix


def tau_mix_geometric(c: float, rho: float) -> float:
    """Dependence inflation ``1 + 2 C rho/(1-rho)`` under geometric mixing (E2)."""
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    return 1.0 + 2.0 * c * rho / (1.0 - rho)


# ──────────────────────────────────────────────────────────────────────────────
# Required-sample calculators (the inverse direction)
# ──────────────────────────────────────────────────────────────────────────────
def required_n_one_sided(gamma: float, delta: float = 0.05) -> float:
    """One-sided Bernoulli certification: ``n >= log(1/delta) / (2 gamma^2)`` (E5).

    To certify ``p >= p_star`` when the truth is ``p_star + gamma``. With
    ``delta=0.05`` and ``gamma=0.05`` this returns ~600 — the canonical
    "~600 effective pairwise comparisons for a 0.70 sign-accuracy claim".
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return math.log(1.0 / delta) / (2.0 * gamma * gamma)


def required_n_two_sided(eps: float, delta: float = 0.05) -> float:
    """Two-sided coverage within ±eps: ``n >= log(2/delta) / (2 eps^2)`` (E6).

    With ``delta=0.05`` and ``eps=0.05`` this returns ~738 windows per slice.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    return math.log(2.0 / delta) / (2.0 * eps * eps)


def rule_of_three_upper(n_zero_events: int) -> float:
    """Rule of three: 95% upper bound ``~= 3/N`` for zero observed events (E7).

    Supports the false-safe < 1% claim: ``N ~= 300`` zero-event severe-shift
    cases give an upper bound of ~0.01.
    """
    if n_zero_events <= 0:
        return 1.0
    return 3.0 / n_zero_events


def required_n_rule_of_three(target_upper: float) -> float:
    """Inverse rule of three: zero-event cases needed for a target upper bound."""
    if target_upper <= 0:
        raise ValueError("target_upper must be positive")
    return 3.0 / target_upper


def required_n_paired_bernstein(
    gamma_d: float, variance: float, delta: float = 0.05
) -> float:
    """Bernstein paired-difference condition for planning utility (doc O E?).

    ``n >= 2 v log(2/delta)/gamma_D^2 + 2 log(2/delta)/(3 gamma_D)`` where ``v``
    is the paired-difference variance and ``gamma_D`` the absolute target gain.
    Doc O: baseline regret ~0.30, 15% target => threshold 0.045, true gain 0.060
    => gamma_D 0.015, variance 0.04-0.09 => ~1,476-3,116 paired windows.
    """
    if gamma_d <= 0:
        raise ValueError("gamma_d must be positive")
    log_term = math.log(2.0 / delta)
    return 2.0 * variance * log_term / (gamma_d * gamma_d) + (
        2.0 * log_term / (3.0 * gamma_d)
    )


def windows_for_pairwise_target(
    target_pairwise_eff: float, n_branches: int, rho_pair: float
) -> float:
    """Invert ``N_pair_eff`` to windows needed at a given branch count / rho.

    Doc O: ~600 effective pairwise comparisons with B=4 and rho in [0.2, 0.6]
    => ~200-400 windows for the single headline number.
    """
    m = pairwise_per_window(n_branches)
    if m == 0:
        return float("inf")
    deff = design_effect(m, rho_pair)
    return target_pairwise_eff * deff / m


# ──────────────────────────────────────────────────────────────────────────────
# Certification check objects
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class CertificationCheck:
    """Result of checking one acceptance claim against effective sample size."""

    claim: str
    required_effective_n: float
    available_effective_n: float
    sufficient: bool
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "claim": self.claim,
            "required_effective_n": self.required_effective_n,
            "available_effective_n": self.available_effective_n,
            "sufficient": self.sufficient,
            "detail": self.detail,
        }


@dataclass(slots=True)
class SampleComplexityPlan:
    """A full pre-registered sample-complexity plan + per-claim certification.

    Constructed from the doc-O acceptance targets and a description of the test
    set (window count, branches per window, dependence). ``checks`` answers, for
    each headline claim, whether the available effective sample size suffices.
    """

    n_windows: int
    n_branches: int
    rho_pair: float
    tau_mix: float
    n_severe_shift_zero_events: int
    checks: list[CertificationCheck] = field(default_factory=list)

    @property
    def all_sufficient(self) -> bool:
        return all(c.sufficient for c in self.checks)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_windows": self.n_windows,
            "n_branches": self.n_branches,
            "rho_pair": self.rho_pair,
            "tau_mix": self.tau_mix,
            "n_severe_shift_zero_events": self.n_severe_shift_zero_events,
            "all_sufficient": self.all_sufficient,
            "checks": [c.as_dict() for c in self.checks],
        }


def build_certification_plan(
    n_windows: int,
    *,
    n_branches: int = 4,
    rho_pair: float = 0.4,
    tau_mix: float = 2.0,
    n_severe_shift_zero_events: int = 300,
    sign_margin: float = 0.05,
    coverage_eps: float = 0.05,
    delta: float = 0.05,
    planning_gamma_d: float = 0.015,
    planning_variance: float = 0.06,
    n_non_abstained_windows: int | None = None,
) -> SampleComplexityPlan:
    """Assemble the per-claim certification plan for a proposed test set (doc O).

    Defaults mirror doc O's worked example: B=4 (m=6), rho~0.4, and the
    acceptance targets from doc F. Returns a :class:`SampleComplexityPlan` whose
    ``all_sufficient`` flag tells the runner whether the harness can make the
    headline claims at 95% confidence given the available windows.
    """
    pair_eff = effective_pairwise_samples(n_windows, n_branches, rho_pair)
    window_eff = mixing_effective_n(n_windows, tau_mix)
    na_windows = (
        n_non_abstained_windows
        if n_non_abstained_windows is not None
        else round(0.6 * n_windows)  # doc F >= 0.60 coverage floor
    )
    na_pair_eff = effective_pairwise_samples(na_windows, n_branches, rho_pair)

    checks: list[CertificationCheck] = []

    # 1. Overall pairwise sign accuracy (>=0.70), margin 0.05.
    req_sign = required_n_one_sided(sign_margin, delta)
    checks.append(
        CertificationCheck(
            claim="pairwise_sign_accuracy>=0.70",
            required_effective_n=req_sign,
            available_effective_n=pair_eff,
            sufficient=pair_eff >= req_sign,
            detail=(
                f"one-sided Bernoulli (E5): margin={sign_margin}, "
                f"~{req_sign:.0f} effective pairwise comparisons needed"
            ),
        )
    )

    # 2. Non-abstained sign accuracy (>=0.80) at >=0.60 coverage.
    checks.append(
        CertificationCheck(
            claim="non_abstained_sign_accuracy>=0.80@coverage>=0.60",
            required_effective_n=req_sign,
            available_effective_n=na_pair_eff,
            sufficient=na_pair_eff >= req_sign,
            detail=(
                "same one-sided target on the non-abstained subset; "
                f"assuming {na_windows} non-abstained windows"
            ),
        )
    )

    # 3. Calibration / coverage per major slice (>=738 effective windows).
    req_cov = required_n_two_sided(coverage_eps, delta)
    checks.append(
        CertificationCheck(
            claim="interval_coverage_within_band_per_slice",
            required_effective_n=req_cov,
            available_effective_n=window_eff,
            sufficient=window_eff >= req_cov,
            detail=(
                f"two-sided coverage (E6): eps={coverage_eps}, "
                f"~{req_cov:.0f} effective windows PER major slice"
            ),
        )
    )

    # 4. Severe-shift false-safe < 1% (rule of three).
    fs_upper = rule_of_three_upper(n_severe_shift_zero_events)
    checks.append(
        CertificationCheck(
            claim="severe_shift_false_safe<0.01",
            required_effective_n=required_n_rule_of_three(0.01),
            available_effective_n=float(n_severe_shift_zero_events),
            sufficient=fs_upper <= 0.01,
            detail=(
                f"rule of three (E7): {n_severe_shift_zero_events} zero-event "
                f"cases => 95% upper bound {fs_upper:.4f}"
            ),
        )
    )

    # 5. Planning utility (paired Bernstein).
    req_plan = required_n_paired_bernstein(planning_gamma_d, planning_variance, delta)
    checks.append(
        CertificationCheck(
            claim="planning_safe_regret_reduction>=15%",
            required_effective_n=req_plan,
            available_effective_n=window_eff,
            sufficient=window_eff >= req_plan,
            detail=(
                f"paired Bernstein: gamma_D={planning_gamma_d}, "
                f"var={planning_variance}, ~{req_plan:.0f} paired windows needed"
            ),
        )
    )

    return SampleComplexityPlan(
        n_windows=n_windows,
        n_branches=n_branches,
        rho_pair=rho_pair,
        tau_mix=tau_mix,
        n_severe_shift_zero_events=n_severe_shift_zero_events,
        checks=checks,
    )


__all__ = [
    "CertificationCheck",
    "SampleComplexityPlan",
    "build_certification_plan",
    "design_effect",
    "effective_pairwise_samples",
    "mixing_effective_n",
    "pairwise_per_window",
    "required_n_one_sided",
    "required_n_paired_bernstein",
    "required_n_rule_of_three",
    "required_n_two_sided",
    "rule_of_three_upper",
    "tau_mix_geometric",
    "weighted_ess",
    "windows_for_pairwise_target",
]
