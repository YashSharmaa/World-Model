# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Why-not cards — composing exact blockers with predicted risk (doc S, doc M).

When an action is rejected or loses, the operator deserves an auditable,
non-LLM explanation. There are three distinct reasons an action does not get
recommended, and a faithful why-not card must keep them *separate* because they
carry different epistemic weight (doc F: hard safety is a release-blocking exact
property; statistical risk is calibrated probability; ranking is utility):

1. **Hard-infeasible** — the exact kernel returned a minimal semantic
   infeasibility certificate (:class:`~..simulator.contracts.WhyNot`): a
   group-minimal set of blockers (quota, gang, topology, health, …) with exact
   deficits (doc S, D16/A1). This is certain, not probabilistic.
2. **Risk-filtered** — the action was hard-feasible but a modeled hazard's
   conformal upper bound exceeded the operator's chance-constraint budget
   (:mod:`risk_gate`). This is a calibrated probability statement.
3. **Out-ranked** — the action was safe and passed every gate but lost the
   lexicographic comparison to the recommendation; the card explains *on which
   lexicographic field* it lost.

A card composes whichever of these apply. The result is a structured object the
serving layer renders verbatim — no free-text generation, fully reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schema.actions import Action
from ..schema.state import WorldState
from ..simulator.contracts import Blocker, ExactKernel, WhyNot
from .risk_gate import RiskAssessment
from .score import LEX_FIELDS

__all__ = [
    "RankNote",
    "RiskNote",
    "WhyNotCard",
    "blockers_card",
    "compose_why_not",
    "rank_card",
    "risk_card",
]


# ──────────────────────────────────────────────────────────────────────────────
# Card sub-structures
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class RiskNote:
    """One hazard that exceeded its chance-constraint budget."""

    channel: str
    conformal_upper: float
    budget: float
    kind: str = "chance"  # chance | cvar


@dataclass(slots=True)
class RankNote:
    """The lexicographic field on which a safe candidate lost to the winner."""

    field: str
    candidate_value: float
    winner_value: float

    @property
    def margin(self) -> float:
        return self.candidate_value - self.winner_value


@dataclass(slots=True)
class WhyNotCard:
    """A composed, auditable why-not explanation for one rejected/losing action."""

    action_id: str
    feasible: bool = True
    # 1. exact kernel certificate (group-minimal IIS analogue)
    minimal_groups: list[str] = field(default_factory=list)
    blockers: list[Blocker] = field(default_factory=list)
    deficits: dict[str, float] = field(default_factory=dict)
    # 2. statistical-risk rejection
    risk_filtered: bool = False
    risk_notes: list[RiskNote] = field(default_factory=list)
    # 3. lexicographic out-ranking
    out_ranked: bool = False
    rank_note: RankNote | None = None
    headline: str = ""

    def summary(self) -> str:
        """A single-line human summary of the dominant reason."""
        if not self.feasible:
            groups = ", ".join(self.minimal_groups) or "infeasible"
            return f"blocked: {groups}"
        if self.risk_filtered:
            ch = ", ".join(n.channel for n in self.risk_notes)
            return f"risk-filtered: {ch}"
        if self.out_ranked and self.rank_note is not None:
            return f"out-ranked on {self.rank_note.field}"
        return "not recommended"


# ──────────────────────────────────────────────────────────────────────────────
# Card builders (each handles one explanation family)
# ──────────────────────────────────────────────────────────────────────────────
def blockers_card(action_id: str, why: WhyNot) -> WhyNotCard:
    """Build a card from an exact-kernel infeasibility certificate (doc S)."""
    return WhyNotCard(
        action_id=action_id,
        feasible=False,
        minimal_groups=[g.value for g in why.minimal_groups],
        blockers=list(why.blockers),
        deficits=dict(why.deficits),
        headline="hard-infeasible",
    )


def risk_card(action_id: str, assessment: RiskAssessment) -> WhyNotCard:
    """Build a card from a statistical-risk gate rejection (:mod:`risk_gate`)."""
    notes: list[RiskNote] = []
    for ch in assessment.violated_channels:
        if ch in assessment.chance_checks:
            up, bud, _ = assessment.chance_checks[ch]
            notes.append(RiskNote(channel=ch, conformal_upper=up, budget=bud))
        elif ch in assessment.cvar_checks:
            cv, bud, _ = assessment.cvar_checks[ch]
            notes.append(
                RiskNote(channel=ch, conformal_upper=cv, budget=bud, kind="cvar")
            )
    return WhyNotCard(
        action_id=action_id,
        feasible=True,
        risk_filtered=True,
        risk_notes=notes,
        headline="statistical-risk over budget",
    )


def rank_card(
    action_id: str,
    candidate_key: tuple[float, ...],
    winner_key: tuple[float, ...],
) -> WhyNotCard:
    """Build a card explaining the lexicographic field on which an action lost.

    Walks the lexicographic key from highest to lowest priority and reports the
    *first* field where the candidate is strictly worse than the winner — that is
    precisely the field that decided the comparison (doc M, E1).
    """
    note: RankNote | None = None
    for i, fname in enumerate(LEX_FIELDS):
        cv = candidate_key[i] if i < len(candidate_key) else 0.0
        wv = winner_key[i] if i < len(winner_key) else 0.0
        if cv > wv + 1e-12:
            note = RankNote(field=fname, candidate_value=cv, winner_value=wv)
            break
        if cv < wv - 1e-12:
            # Candidate is actually better on this field — it lost on a tie-break
            # of a *lower* field, so keep scanning.
            continue
    return WhyNotCard(
        action_id=action_id,
        feasible=True,
        out_ranked=True,
        rank_note=note,
        headline="out-ranked",
    )


# ──────────────────────────────────────────────────────────────────────────────
# The composition entry point
# ──────────────────────────────────────────────────────────────────────────────
def compose_why_not(
    action_id: str,
    *,
    kernel: ExactKernel | None = None,
    state: WorldState | None = None,
    action: Action | None = None,
    feasible: bool = True,
    risk_assessment: RiskAssessment | None = None,
    candidate_key: tuple[float, ...] | None = None,
    winner_key: tuple[float, ...] | None = None,
) -> WhyNotCard:
    """Compose a complete why-not card from whichever reasons apply.

    Precedence follows epistemic weight (doc F's gated hierarchy): a hard
    infeasibility certificate dominates (it is certain); otherwise a statistical
    risk rejection; otherwise a lexicographic out-ranking note. When ``feasible``
    is False and a kernel+state+action are supplied, the exact minimal
    certificate is fetched from ``kernel.why_not`` so the card carries the precise
    group-minimal blockers and deficits.
    """
    if not feasible:
        if kernel is not None and state is not None and action is not None:
            why = kernel.why_not(state, action)
            return blockers_card(action_id, why)
        return WhyNotCard(
            action_id=action_id, feasible=False, headline="hard-infeasible"
        )

    if risk_assessment is not None and not risk_assessment.passes:
        return risk_card(action_id, risk_assessment)

    if candidate_key is not None and winner_key is not None:
        return rank_card(action_id, candidate_key, winner_key)

    return WhyNotCard(action_id=action_id, headline="not recommended")
