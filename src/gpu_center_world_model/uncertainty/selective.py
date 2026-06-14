# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Split-select-certify selective prediction with finite-sample certificates.

This module implements doc L's selective-prediction protocol — the gate that
decides, per candidate, whether the learned model is *trusted* to rank/forecast
or must **abstain** and fall back to conservative ranking. Safety is a gate, not
a weighted metric, so this gate sits on top of the kernel-certified safe set and
only ever *removes* trust.

Protocol (doc L, selective_prediction_rule):

1. **Trust score** (D7):
   ``s(x,a) = -l1 D_epi - l2 N_ood - l3 U_risk + l4 M_margin``
   where ``U_risk`` is the one-sided conformal upper bound on excess safety risk
   and ``M_margin`` is the conformalized benefit margin over the fallback. The
   gate is ``g_tau(x,a) = 1{s(x,a) >= tau}``.
2. **Select** ``tau*`` on a *selection* split: maximise accepted coverage
   subject to the empirical accepted-risk budget.
3. **Certify** on an *independent* split: with ``m`` accepted and ``k`` failures,
   the one-sided exact **Clopper-Pearson** upper bound on accepted risk is
   ``U_R = BetaInv(1 - delta_R; k + 1, m - k)`` and the coverage lower bound is
   ``L_C = BetaInv(delta_C; m, n_cert - m + 1)``. Because ``tau*`` was chosen on
   an independent split, with prob >= ``1 - delta_R - delta_C``:
   ``R(tau*) <= U_R`` and ``C(tau*) >= L_C``.

The **optimism-bias one-sided gate** and the **safety chance-constraint** are
also provided here as the failure-event instantiations doc L lists (P1-P4).

All certificates are finite-sample and exact for the *selected* operating point;
they are marginal / accepted-subset statements, never per-(x,a) guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._betainc import beta_quantile


# ─────────────────────────────────────────────────────────────────────────────
# Clopper-Pearson exact binomial bounds
# ─────────────────────────────────────────────────────────────────────────────
def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """Exact one-sided upper bound on a binomial rate (doc L certificate).

    ``U = BetaInv(1 - delta; k + 1, n - k)`` is the Clopper-Pearson upper
    confidence limit: ``P(p <= U) >= 1 - delta`` for the true rate ``p`` given
    ``k`` successes in ``n`` trials. Edge cases: ``k == n`` -> 1.0; ``n == 0`` ->
    1.0 (no evidence, worst case).
    """
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta_quantile(1.0 - delta, k + 1, n - k))


def clopper_pearson_lower(k: int, n: int, delta: float) -> float:
    """Exact one-sided lower bound on a binomial rate.

    ``L = BetaInv(delta; k, n - k + 1)`` with ``P(p >= L) >= 1 - delta``. Edge
    cases: ``k == 0`` -> 0.0; ``n == 0`` -> 0.0; ``k == n`` -> ``delta^{1/n}``
    (the exact lower confidence limit when *every* trial succeeded — the
    ``Beta(n, 1)`` quantile, which is finite, not the degenerate 0).
    """
    if n <= 0 or k <= 0:
        return 0.0
    if k >= n:
        # Beta(n, 1) has CDF u^n, so its delta-quantile is delta^{1/n}.
        return float(delta ** (1.0 / n))
    return float(beta_quantile(delta, k, n - k + 1))


def _coverage_lower_bound(m: int, n_cert: int, delta_coverage: float) -> float:
    """Guarded coverage lower bound ``L_C`` for ``m`` accepts in ``n_cert`` (P2).

    The coverage ``C = m / n_cert`` is a binomial rate, so its exact one-sided
    Clopper-Pearson lower bound is ``BetaInv(delta; m, n_cert - m + 1)``. The
    previous code used ``Beta(m, n_cert - m + 1)`` directly, which is **degenerate
    at the ``m == n_cert`` edge** (``Beta(m, 1)`` is fine but the call path mixed
    accepted-risk and coverage parameterisations). This guarded helper handles all
    edges: ``m == 0`` -> 0.0 (no evidence of any coverage); ``m == n_cert`` ->
    ``delta^{1/n_cert}`` (the exact all-accept lower limit); otherwise the exact
    Beta quantile.
    """
    if n_cert <= 0 or m <= 0:
        return 0.0
    if m >= n_cert:
        return float(delta_coverage ** (1.0 / n_cert))
    return float(beta_quantile(delta_coverage, m, n_cert - m + 1))


# ─────────────────────────────────────────────────────────────────────────────
# Trust score (doc L, D7)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class TrustScoreWeights:
    """Weights for the selective trust score (doc L, D7)."""

    l1_epistemic: float = 1.0
    l2_ood: float = 1.0
    l3_risk: float = 1.0
    l4_margin: float = 1.0


def trust_score(
    *,
    epistemic: np.ndarray | float,
    ood: np.ndarray | float,
    risk_upper: np.ndarray | float,
    benefit_margin: np.ndarray | float,
    weights: TrustScoreWeights | None = None,
) -> np.ndarray:
    """Compute ``s(x,a) = -l1 D_epi - l2 N_ood - l3 U_risk + l4 M_margin``.

    Higher is more trustworthy. ``risk_upper`` should be the *conformal* upper
    bound on excess safety risk and ``benefit_margin`` the conformalized benefit
    over the conservative fallback (so the gate is statistically honest).
    """
    w = weights or TrustScoreWeights()
    epi = np.asarray(epistemic, float)
    od = np.asarray(ood, float)
    ru = np.asarray(risk_upper, float)
    bm = np.asarray(benefit_margin, float)
    return -w.l1_epistemic * epi - w.l2_ood * od - w.l3_risk * ru + (w.l4_margin * bm)


# ─────────────────────────────────────────────────────────────────────────────
# Selection: choose tau* on the selection split
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class SelectionResult:
    """Outcome of choosing the trust threshold on the selection split."""

    tau: float
    empirical_coverage: float
    empirical_risk: float
    n_accepted: int
    feasible: bool  # True iff a threshold met the empirical risk budget


def select_threshold(
    scores: np.ndarray,
    failures: np.ndarray,
    risk_budget: float,
    min_accept: int = 1,
) -> SelectionResult:
    """Pick ``tau*`` maximising accepted coverage s.t. accepted risk <= budget.

    Doc L selection step. ``failures[i] == 1`` means accepting ``i`` would be a
    "bad" event (false-safe / optimism / sign error, per the policy claim). We
    sweep candidate thresholds (the sorted unique scores) and keep the one with
    the largest acceptance count whose *empirical* accepted-risk does not exceed
    ``risk_budget`` and which accepts at least ``min_accept`` points.

    The certificate is computed on the **independent** certification split, so it
    is valid for this selected ``tau*`` even though selection optimised the
    empirical objective (optimism bias of selection is absorbed by the split).
    """
    s = np.asarray(scores, float)
    f = np.asarray(failures, float)
    if s.size == 0:
        return SelectionResult(float("inf"), 0.0, 0.0, 0, False)
    candidates = np.unique(s)
    best: SelectionResult | None = None
    for tau in candidates:
        accept = s >= tau
        m = int(accept.sum())
        if m < min_accept:
            continue
        risk = float(f[accept].mean())
        cov = m / s.size
        if risk <= risk_budget and (best is None or m > best.n_accepted):
            best = SelectionResult(float(tau), cov, risk, m, True)
    if best is not None:
        return best
    # No threshold meets the budget: return the most conservative (accept none
    # below the max score) and flag infeasible so the caller abstains broadly.
    tau = float(candidates[-1])
    accept = s >= tau
    m = int(accept.sum())
    risk = float(f[accept].mean()) if m else 0.0
    return SelectionResult(tau, m / s.size, risk, m, False)


# ─────────────────────────────────────────────────────────────────────────────
# Certification: certify accepted risk + coverage on an independent split
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class SelectiveCertificate:
    """Finite-sample operating certificate for a selected ``tau*`` (doc L)."""

    tau: float
    n_cert: int
    n_accepted: int
    n_failures: int
    accepted_risk_upper: float  # U_R: P(F=1 | accepted) <= this w.p. 1-delta_R
    coverage_lower: float  # L_C: P(accepted) >= this w.p. 1-delta_C
    delta_risk: float
    delta_coverage: float
    risk_ok: bool = field(default=False)
    coverage_ok: bool = field(default=False)

    @property
    def total_confidence_budget(self) -> float:
        """Union-bound confidence budget spent (doc L: allocate jointly)."""
        return self.delta_risk + self.delta_coverage


def certify(
    scores: np.ndarray,
    failures: np.ndarray,
    tau: float,
    *,
    delta_risk: float = 0.05,
    delta_coverage: float = 0.05,
    risk_target: float | None = None,
    coverage_target: float | None = None,
) -> SelectiveCertificate:
    """Certify accepted-risk and coverage at ``tau`` on the certification split.

    Implements doc L's exact certificate: ``U_R = CP_upper(k, m, delta_R)`` and
    ``L_C = CP_lower-style`` coverage bound via ``BetaInv(delta_C; m, n-m+1)``.
    With probability >= ``1 - delta_R - delta_C`` the true accepted-risk is
    below ``U_R`` and the true coverage is above ``L_C``.

    ``risk_target`` / ``coverage_target`` set the pass/fail flags for the
    operating-point constraints (P1: ``U_R <= 0.05``; P2: ``L_C >= 0.60``).
    """
    s = np.asarray(scores, float)
    f = np.asarray(failures, float)
    n_cert = s.size
    accept = s >= tau
    m = int(accept.sum())
    k = int(f[accept].sum()) if m else 0

    u_r = clopper_pearson_upper(k, m, delta_risk)
    # Coverage lower bound via the guarded helper (handles the m == n_cert edge).
    l_c = _coverage_lower_bound(m, n_cert, delta_coverage)

    cert = SelectiveCertificate(
        tau=float(tau),
        n_cert=n_cert,
        n_accepted=m,
        n_failures=k,
        accepted_risk_upper=u_r,
        coverage_lower=l_c,
        delta_risk=delta_risk,
        delta_coverage=delta_coverage,
    )
    if risk_target is not None:
        cert.risk_ok = u_r <= risk_target
    if coverage_target is not None:
        cert.coverage_ok = l_c >= coverage_target
    return cert


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end split-select-certify
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class SelectiveGate:
    """A fitted split-select-certify selective gate (doc L).

    Fit on a (selection split, certification split) pair; then apply at decision
    time: ``accept(x,a)`` iff ``s(x,a) >= tau*``. The attached
    :class:`SelectiveCertificate` is the auditable finite-sample guarantee that
    travels with every recommendation (planner explanation ledger).
    """

    tau: float = float("inf")
    selection: SelectionResult | None = None
    certificate: SelectiveCertificate | None = None
    weights: TrustScoreWeights = field(default_factory=TrustScoreWeights)

    def accept(self, score: float) -> bool:
        """``True`` iff trusted (do not abstain) at the certified threshold."""
        return float(score) >= self.tau

    def accept_mask(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, float) >= self.tau


def split_select_certify(
    sel_scores: np.ndarray,
    sel_failures: np.ndarray,
    cert_scores: np.ndarray,
    cert_failures: np.ndarray,
    *,
    risk_budget: float = 0.05,
    delta_risk: float = 0.05,
    delta_coverage: float = 0.05,
    coverage_target: float = 0.60,
    min_accept: int = 1,
) -> SelectiveGate:
    """Run the full doc L protocol and return a fitted :class:`SelectiveGate`.

    ``risk_budget`` is the empirical selection budget; ``risk_target`` for the
    certificate is taken equal to it (P1). The selection and certification splits
    must be independent (doc L assumption A6) for the certificate to be valid.
    """
    sel = select_threshold(sel_scores, sel_failures, risk_budget, min_accept=min_accept)
    cert = certify(
        cert_scores,
        cert_failures,
        sel.tau,
        delta_risk=delta_risk,
        delta_coverage=delta_coverage,
        risk_target=risk_budget,
        coverage_target=coverage_target,
    )
    return SelectiveGate(tau=sel.tau, selection=sel, certificate=cert)


# ─────────────────────────────────────────────────────────────────────────────
# Optimism-bias one-sided gate + safety chance constraint (P1-P4)
# ─────────────────────────────────────────────────────────────────────────────
def optimism_bias_gate(excess_risk_upper: np.ndarray | float) -> np.ndarray:
    """Deploy gate ``g(x,a) = 1{U_alpha(x,a) <= 0}`` for optimism control (T6).

    With ``Z = excess interventional risk vs fallback`` and one-sided conformal
    upper bound ``U_alpha`` on ``Z``, deploying only when ``U_alpha <= 0`` gives
    ``P{deploy and Z > 0} <= alpha`` — the clean translation of non-abstained
    optimism-bias control. Returns a boolean accept mask.
    """
    u = np.asarray(excess_risk_upper, float)
    return u <= 0.0


def safety_chance_constraint(
    risk_uppers: dict[str, np.ndarray | float],
    epsilons: dict[str, float],
) -> np.ndarray:
    """Safety gate ``A_gate = {a : U_alpha_e(x,a) <= eps_e for all e}`` (E17).

    Hard feasibility stays symbolic/exact upstream; this filters the residual
    *learned* safety risks by their calibrated one-sided upper bounds. Returns a
    boolean accept mask that is the AND over all risk channels.
    """
    masks = []
    for ch, eps in epsilons.items():
        if ch not in risk_uppers:
            continue
        u = np.asarray(risk_uppers[ch], float)
        masks.append(u <= eps)
    if not masks:
        # No risk channels supplied -> nothing to filter; accept all.
        any_arr = next(iter(risk_uppers.values()), 0.0)
        return np.ones_like(np.asarray(any_arr, float), dtype=bool)
    out = masks[0]
    for m in masks[1:]:
        out = out & m
    return out


# ─────────────────────────────────────────────────────────────────────────────
# P3 severe-shift certificate (distribution-shift robustness of the gate)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class SevereShiftCertificate:
    """Robustness of the accepted-risk certificate under bounded shift (doc L P3).

    The split-select-certify certificate is *marginal* and assumes the
    certification split is exchangeable with deployment. Under a bounded covariate
    shift with total-variation distance ``<= tv_bound`` between the certification
    and deployment distributions, the deployed accepted risk can exceed the
    certified ``U_R`` by at most the worst-case TV inflation (the
    distribution-shift slack of doc L P3):

        R_deploy <= min(1, U_R + tv_bound)   w.p. >= 1 - delta_risk.

    A severe shift is flagged when ``tv_bound`` is large enough that the inflated
    bound no longer clears the operating target, so the gate must abstain more
    broadly rather than trust the nominal certificate.
    """

    base_risk_upper: float
    tv_bound: float
    shifted_risk_upper: float
    risk_target: float
    holds_under_shift: bool


def severe_shift_certificate(
    cert: SelectiveCertificate,
    *,
    tv_bound: float,
    risk_target: float | None = None,
) -> SevereShiftCertificate:
    """Inflate the certified accepted-risk bound for a bounded severe shift (P3).

    ``tv_bound`` is an upper bound on the total-variation distance between the
    certification and deployment input distributions (e.g. estimated from the OOD
    conformal p-value mass). The shifted bound is ``min(1, U_R + tv_bound)``; the
    certificate *holds under shift* iff that still clears ``risk_target`` (default:
    the certificate's own ``delta_risk``-implied accepted-risk target if absent,
    falling back to ``accepted_risk_upper``).
    """
    base = float(cert.accepted_risk_upper)
    tv = float(max(0.0, tv_bound))
    shifted = float(min(1.0, base + tv))
    target = risk_target if risk_target is not None else base
    return SevereShiftCertificate(
        base_risk_upper=base,
        tv_bound=tv,
        shifted_risk_upper=shifted,
        risk_target=float(target),
        holds_under_shift=bool(shifted <= target + 1e-12),
    )


__all__ = [
    "SelectionResult",
    "SelectiveCertificate",
    "SelectiveGate",
    "SevereShiftCertificate",
    "TrustScoreWeights",
    "certify",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "optimism_bias_gate",
    "safety_chance_constraint",
    "select_threshold",
    "severe_shift_certificate",
    "split_select_certify",
    "trust_score",
]
