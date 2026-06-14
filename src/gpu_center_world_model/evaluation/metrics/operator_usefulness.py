# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Operator-usefulness metrics — the deployment-readiness layer (doc F).

Operator usefulness sits *fifth* in doc F's gated hierarchy: it is a
deployment-readiness layer, never a release-blocking gate, and doc F is emphatic
that it must be measured "through behavior, not only through surveys". The
companion ledger row is doc O **L10** ("operator usefulness improves median
decision time by at least 20 percent with no safety loss") — a purely *empirical*
claim that is only as strong as the counterbalanced human study behind it.

This module grades an already-collected interaction-trace study, so it has no
dependency on the learned backend and stays import-clean on CI. The runner feeds
typed records; the suite never simulates operators.

Doc F's ``operator_usefulness`` metric family declares three *offline* metrics and
four *live-demo* metrics:

offline
    * **coverage-versus-win-rate** — among the cases the assistant chose to speak
      on (coverage), how often did it beat the unaided operator (win rate)? A high
      win rate at trivially-low coverage is not usefulness; the curve must be
      published, not collapsed (doc F: "high recommendation acceptance rate does
      not prove usefulness if the system only speaks in obvious cases").
    * **appropriate-reliance proxy** — from the human-AI evaluation literature
      doc F cites: did the operator *follow* the assistant when it was right and
      *override* it when it was wrong? We decompose into the four reliance cells
      (appropriate accept / appropriate reject / over-reliance / under-reliance)
      and report the appropriate-reliance fraction.
    * **explanation-blocker agreement** — on blocked / why-not cases, does the
      operator agree the emitted blocker set explains the block? Set-wise
      agreement against the operator's adjudicated blocker set (doc F
      "explanation-blocker agreement").

live-demo / timed-study
    * **median decision time** with and without the assistant, and the
      doc-F-acceptance **decision-time reduction** (>= 20%).
    * **unsafe acceptance rate**, **correct override rate**, **unnecessary
      override rate** — the override harness (doc F live_demo_metrics).

The doc-F acceptance threshold (``operator_usefulness``) is "at least 20% lower
median decision time in timed scenario tasks with no increase in unsafe
decisions". :func:`evaluate_operator_usefulness` returns ``passed`` per that rule,
but the runner reports it as provisional (never release-blocking), exactly as doc
F instructs ("if such a study is not run ... mark that dimension as provisional").

Pure NumPy; everything operates on plain records / float arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

ArrayLike = np.ndarray | list[float]


def _as_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


# ──────────────────────────────────────────────────────────────────────────────
# The atomic record: one operator's interaction trace on one decision case
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class OperatorCase:
    """One operator × decision-case interaction trace (counterbalanced study).

    The study runs each decision scenario *with* and *without* the assistant in
    counterbalanced order (doc F protocol); one :class:`OperatorCase` records the
    assisted arm for a single (operator, scenario) cell, with the matched unaided
    decision-time carried alongside for the within-subject delta.

    Attributes
    ----------
    case_id:
        Identifier for this (operator, scenario) cell.
    assistant_spoke:
        Did the assistant make a recommendation (vs abstain)? Drives coverage.
    assistant_correct:
        On the cases the assistant spoke on, was its recommendation the safe
        oracle-aligned one? Drives the win rate and reliance cells.
    operator_followed:
        Did the operator accept the assistant's recommendation (vs override)?
    operator_decision_correct:
        Was the operator's *final* decision the safe oracle-aligned one? Used for
        win-rate vs the unaided arm and for the unsafe-acceptance check.
    unaided_decision_correct:
        Was the matched *unaided* decision (same operator, same scenario, no
        assistant) correct? The within-subject win-rate comparator.
    decision_time_s:
        Time-to-decision in the assisted arm (seconds).
    unaided_decision_time_s:
        Matched time-to-decision in the unaided arm (seconds).
    final_decision_unsafe:
        Did the operator's final accepted decision trip a safety threshold? Used
        for the "no increase in unsafe decisions" acceptance clause.
    unaided_decision_unsafe:
        Did the matched unaided decision trip a safety threshold?
    is_blocked_case:
        Is this a blocked / why-not case (drives explanation-blocker agreement)?
    emitted_blockers:
        The blocker labels the assistant emitted (frozenset of strings).
    operator_blockers:
        The blocker labels the operator adjudicated as the true explanation.
    """

    case_id: str
    assistant_spoke: bool = False
    assistant_correct: bool = False
    operator_followed: bool = False
    operator_decision_correct: bool = False
    unaided_decision_correct: bool = False
    decision_time_s: float = float("nan")
    unaided_decision_time_s: float = float("nan")
    final_decision_unsafe: bool = False
    unaided_decision_unsafe: bool = False
    is_blocked_case: bool = False
    emitted_blockers: frozenset[str] = field(default_factory=frozenset)
    operator_blockers: frozenset[str] = field(default_factory=frozenset)


# ──────────────────────────────────────────────────────────────────────────────
# offline metric 1: coverage-versus-win-rate
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class CoverageWinRatePoint:
    """One point of the coverage-vs-win-rate curve (binned by assistant speaking)."""

    coverage: float
    win_rate: float
    n_spoken: int


def coverage_vs_win_rate(cases: Sequence[OperatorCase]) -> CoverageWinRatePoint:
    """Coverage and conditional win rate over the cases the assistant spoke on.

    *Coverage* is the fraction of cases the assistant did not abstain on. The
    *win rate* is, among spoken cases, the fraction where the assisted final
    decision was correct while the matched unaided decision was not (a genuine
    within-subject win), tie-broken so a case where both arms are correct counts
    as half a win (the assistant did not hurt). This rewards speaking only when it
    actually helps — a high win rate at low coverage is visible, not hidden (doc
    F: conditional win-rate-versus-coverage plot).
    """
    n = len(cases)
    if n == 0:
        return CoverageWinRatePoint(float("nan"), float("nan"), 0)
    spoken = [c for c in cases if c.assistant_spoke]
    coverage = len(spoken) / n
    if not spoken:
        return CoverageWinRatePoint(
            coverage=coverage, win_rate=float("nan"), n_spoken=0
        )
    wins = 0.0
    for c in spoken:
        a = c.operator_decision_correct
        u = c.unaided_decision_correct
        if a and not u:
            wins += 1.0
        elif a and u:
            wins += 0.5  # no harm, no help → half credit
        # a==False contributes 0 (assistant did not produce a correct outcome)
    return CoverageWinRatePoint(
        coverage=coverage, win_rate=wins / len(spoken), n_spoken=len(spoken)
    )


# ──────────────────────────────────────────────────────────────────────────────
# offline metric 2: appropriate-reliance proxy
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class RelianceBreakdown:
    """The four reliance cells + the appropriate-reliance fraction (doc F)."""

    n: int
    appropriate_accept: int  # followed a correct assistant
    appropriate_reject: int  # overrode an incorrect assistant
    over_reliance: int  # followed an incorrect assistant
    under_reliance: int  # overrode a correct assistant
    appropriate_reliance_rate: float

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "appropriate_accept": self.appropriate_accept,
            "appropriate_reject": self.appropriate_reject,
            "over_reliance": self.over_reliance,
            "under_reliance": self.under_reliance,
            "appropriate_reliance_rate": self.appropriate_reliance_rate,
        }


def appropriate_reliance(cases: Sequence[OperatorCase]) -> RelianceBreakdown:
    """Decompose follow/override behaviour against assistant correctness (doc F).

    Only cases where the assistant spoke contribute (you cannot rely on silence).
    The appropriate-reliance rate is ``(appropriate_accept + appropriate_reject)
    / n_spoken`` — the fraction of interactions where the operator followed a
    right assistant or overrode a wrong one. Doc F (and the human-AI evaluation
    work it cites) treats this as the right usefulness signal rather than a bare
    acceptance rate, which over-reliance inflates.
    """
    spoken = [c for c in cases if c.assistant_spoke]
    n = len(spoken)
    if n == 0:
        return RelianceBreakdown(0, 0, 0, 0, 0, float("nan"))
    aa = ar = over = under = 0
    for c in spoken:
        if c.operator_followed and c.assistant_correct:
            aa += 1
        elif not c.operator_followed and not c.assistant_correct:
            ar += 1
        elif c.operator_followed and not c.assistant_correct:
            over += 1
        else:  # overrode a correct assistant
            under += 1
    return RelianceBreakdown(
        n=n,
        appropriate_accept=aa,
        appropriate_reject=ar,
        over_reliance=over,
        under_reliance=under,
        appropriate_reliance_rate=(aa + ar) / n,
    )


# ──────────────────────────────────────────────────────────────────────────────
# offline metric 3: explanation-blocker agreement
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ExplanationAgreement:
    """Set-wise agreement of emitted vs operator-adjudicated blocker sets."""

    n_blocked: int
    exact_match_count: int
    exact_match_rate: float
    mean_jaccard: float

    def as_dict(self) -> dict[str, object]:
        return {
            "n_blocked": self.n_blocked,
            "exact_match_count": self.exact_match_count,
            "exact_match_rate": self.exact_match_rate,
            "mean_jaccard": self.mean_jaccard,
        }


def explanation_blocker_agreement(
    cases: Sequence[OperatorCase],
) -> ExplanationAgreement:
    """Agreement between emitted and operator-adjudicated blocker sets (doc F).

    Computed over the blocked / why-not cases only. Reports both the strict
    exact-set-match rate (doc F wants the emitted blocker explanation to *match*,
    not be "mostly right") and the mean Jaccard similarity as a softer companion
    so a near-miss is visible.
    """
    blocked = [c for c in cases if c.is_blocked_case]
    n = len(blocked)
    if n == 0:
        return ExplanationAgreement(0, 0, float("nan"), float("nan"))
    matches = 0
    jaccards: list[float] = []
    for c in blocked:
        emitted = set(c.emitted_blockers)
        adjudicated = set(c.operator_blockers)
        if emitted == adjudicated:
            matches += 1
        union = emitted | adjudicated
        if union:
            jaccards.append(len(emitted & adjudicated) / len(union))
        else:
            # both empty: vacuously a perfect agreement
            jaccards.append(1.0)
    return ExplanationAgreement(
        n_blocked=n,
        exact_match_count=matches,
        exact_match_rate=matches / n,
        mean_jaccard=float(np.mean(jaccards)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# live-demo / timed-study: decision-time + override harness
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class DecisionTimeReport:
    """Assisted vs unaided median decision time and the doc-F reduction."""

    n_paired: int
    median_assisted_s: float
    median_unaided_s: float
    median_reduction: float  # fraction: (unaided - assisted) / unaided
    target_reduction: float
    meets_time_target: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "n_paired": self.n_paired,
            "median_assisted_s": self.median_assisted_s,
            "median_unaided_s": self.median_unaided_s,
            "median_reduction": self.median_reduction,
            "target_reduction": self.target_reduction,
            "meets_time_target": self.meets_time_target,
        }


def decision_time_reduction(
    cases: Sequence[OperatorCase], *, target_reduction: float = 0.20
) -> DecisionTimeReport:
    """Median decision-time reduction from the assistant (doc F live metric).

    Uses only cases with both arms timed. The reduction is computed on the
    *medians* (doc F: "median decision time"), which is robust to the heavy right
    tail of human timing. ``meets_time_target`` is the >= 20% doc-F clause.
    """
    a = _as_array([c.decision_time_s for c in cases])
    u = _as_array([c.unaided_decision_time_s for c in cases])
    valid = ~np.isnan(a) & ~np.isnan(u)
    a, u = a[valid], u[valid]
    n = int(a.size)
    if n == 0:
        return DecisionTimeReport(
            0, float("nan"), float("nan"), float("nan"), target_reduction, False
        )
    med_a = float(np.median(a))
    med_u = float(np.median(u))
    reduction = (med_u - med_a) / med_u if med_u > 0 else float("nan")
    meets = reduction == reduction and reduction >= target_reduction
    return DecisionTimeReport(
        n_paired=n,
        median_assisted_s=med_a,
        median_unaided_s=med_u,
        median_reduction=reduction,
        target_reduction=target_reduction,
        meets_time_target=meets,
    )


@dataclass(slots=True)
class OverrideReport:
    """The override harness: unsafe-acceptance, correct/unnecessary overrides."""

    n_spoken: int
    unsafe_acceptance_rate: float  # followed assistant into an unsafe final decision
    correct_override_rate: float  # overrode an incorrect assistant / all overrides
    unnecessary_override_rate: float  # overrode a correct assistant / all overrides
    # safety-loss clause (doc F acceptance): assisted vs unaided unsafe-decision rate
    assisted_unsafe_rate: float
    unaided_unsafe_rate: float
    no_safety_loss: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "n_spoken": self.n_spoken,
            "unsafe_acceptance_rate": self.unsafe_acceptance_rate,
            "correct_override_rate": self.correct_override_rate,
            "unnecessary_override_rate": self.unnecessary_override_rate,
            "assisted_unsafe_rate": self.assisted_unsafe_rate,
            "unaided_unsafe_rate": self.unaided_unsafe_rate,
            "no_safety_loss": self.no_safety_loss,
        }


def override_harness(cases: Sequence[OperatorCase]) -> OverrideReport:
    """Grade the override behaviour and the no-safety-loss acceptance clause.

    * **unsafe-acceptance rate**: among spoken cases the operator *followed*, the
      fraction whose final decision was unsafe — the cardinal operational failure
      (doc F live metric "unsafe acceptance rate").
    * **correct override rate**: of the cases the operator overrode, the fraction
      where the assistant was actually wrong (a *good* override).
    * **unnecessary override rate**: of the overrides, the fraction where the
      assistant was right (the operator threw away a correct recommendation).
    * **no_safety_loss**: assisted unsafe-decision rate does not exceed the
      matched unaided rate — the doc-F "no increase in unsafe decisions" clause.
    """
    spoken = [c for c in cases if c.assistant_spoke]
    n = len(spoken)
    if n == 0:
        return OverrideReport(
            0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            False,
        )
    followed = [c for c in spoken if c.operator_followed]
    overrode = [c for c in spoken if not c.operator_followed]
    unsafe_accept = (
        sum(1 for c in followed if c.final_decision_unsafe) / len(followed)
        if followed
        else 0.0
    )
    correct_ovr = (
        sum(1 for c in overrode if not c.assistant_correct) / len(overrode)
        if overrode
        else float("nan")
    )
    unnec_ovr = (
        sum(1 for c in overrode if c.assistant_correct) / len(overrode)
        if overrode
        else float("nan")
    )
    # safety-loss clause is computed over *all* cases with both arms recorded
    assisted_unsafe = _as_array(
        [1.0 if c.final_decision_unsafe else 0.0 for c in cases]
    )
    unaided_unsafe = _as_array(
        [1.0 if c.unaided_decision_unsafe else 0.0 for c in cases]
    )
    a_rate = float(np.mean(assisted_unsafe)) if assisted_unsafe.size else float("nan")
    u_rate = float(np.mean(unaided_unsafe)) if unaided_unsafe.size else float("nan")
    no_loss = (a_rate == a_rate) and (u_rate == u_rate) and (a_rate <= u_rate)
    return OverrideReport(
        n_spoken=n,
        unsafe_acceptance_rate=unsafe_accept,
        correct_override_rate=correct_ovr,
        unnecessary_override_rate=unnec_ovr,
        assisted_unsafe_rate=a_rate,
        unaided_unsafe_rate=u_rate,
        no_safety_loss=no_loss,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Aggregated operator-usefulness report (doc O L10 binding object)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class OperatorUsefulnessReport:
    """The full operator-usefulness readout — provisional, never release-blocking.

    ``passed`` encodes the doc-F acceptance threshold (>= 20% median decision-time
    reduction *and* no increase in unsafe decisions). ``provisional`` is always
    True: doc F says operator readiness must not be claimed without a human study,
    and even with one this layer never blocks a release. This object is the
    empirical binding for doc O ledger claim **L10**.
    """

    n_cases: int
    coverage_win_rate: CoverageWinRatePoint
    reliance: RelianceBreakdown
    explanation: ExplanationAgreement
    decision_time: DecisionTimeReport
    overrides: OverrideReport
    passed: bool
    provisional: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "n_cases": self.n_cases,
            "coverage_vs_win_rate": {
                "coverage": self.coverage_win_rate.coverage,
                "win_rate": self.coverage_win_rate.win_rate,
                "n_spoken": self.coverage_win_rate.n_spoken,
            },
            "appropriate_reliance": self.reliance.as_dict(),
            "explanation_blocker_agreement": self.explanation.as_dict(),
            "decision_time": self.decision_time.as_dict(),
            "overrides": self.overrides.as_dict(),
            "passed": self.passed,
            "provisional": self.provisional,
            "acceptance": (
                "doc F operator_usefulness: >= 20% lower median decision time with "
                "no increase in unsafe decisions; reported provisionally and never "
                "release-blocking (doc O L10)."
            ),
        }


def evaluate_operator_usefulness(
    cases: Sequence[OperatorCase], *, target_reduction: float = 0.20
) -> OperatorUsefulnessReport:
    """Compute the full operator-usefulness profile for a timed scenario study.

    Combines all three offline metrics and the timed/override harness, and folds
    the doc-F acceptance rule into ``passed`` — a >= ``target_reduction`` median
    decision-time reduction together with no increase in unsafe decisions. The
    result is always *provisional* (doc F).
    """
    cwr = coverage_vs_win_rate(cases)
    reliance = appropriate_reliance(cases)
    explanation = explanation_blocker_agreement(cases)
    dt = decision_time_reduction(cases, target_reduction=target_reduction)
    ov = override_harness(cases)
    passed = bool(dt.meets_time_target and ov.no_safety_loss)
    return OperatorUsefulnessReport(
        n_cases=len(cases),
        coverage_win_rate=cwr,
        reliance=reliance,
        explanation=explanation,
        decision_time=dt,
        overrides=ov,
        passed=passed,
    )


__all__ = [
    "CoverageWinRatePoint",
    "DecisionTimeReport",
    "ExplanationAgreement",
    "OperatorCase",
    "OperatorUsefulnessReport",
    "OverrideReport",
    "RelianceBreakdown",
    "appropriate_reliance",
    "coverage_vs_win_rate",
    "decision_time_reduction",
    "evaluate_operator_usefulness",
    "explanation_blocker_agreement",
    "override_harness",
]
