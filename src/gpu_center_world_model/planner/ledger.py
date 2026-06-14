# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The structured (non-LLM) explanation ledger (doc M / F).

Every planner decision must be *auditable*: a reviewer should be able to replay,
field by field, why each candidate was blocked, risk-filtered, ranked, abstained,
or recommended — with no free-text generation in the loop (doc F: "explanation
completeness for blocked and abstained actions"). This module is the append-only
recorder of those structured reasons.

Each :class:`~..planner.contracts.LedgerEntry` has a ``kind`` drawn from a small
fixed vocabulary, an ``action_id``, a short ``detail`` string, and a structured
``evidence`` dict. The ledger is deterministic given the same forecasts and
state, so two runs over the same branch root produce byte-identical ledgers —
which is exactly what the evaluation harness checks for rank-stability and
explanation completeness (doc F live-demo scorecard).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import LedgerEntry
from .risk_gate import RiskAssessment
from .score import LEX_FIELDS
from .why_not import WhyNotCard

__all__ = ["LEDGER_KINDS", "ExplanationLedger"]

#: The fixed, auditable vocabulary of ledger entry kinds (doc M/F).
LEDGER_KINDS: tuple[str, ...] = (
    "enumerated",
    "blocked",
    "risk_filtered",
    "ranked",
    "abstained",
    "recommended",
    "fallback",
)


@dataclass(slots=True)
class ExplanationLedger:
    """Append-only structured reasoning record for one decision window.

    The ledger is constructed empty and the planner appends entries as it walks
    the pipeline (enumerate → score → risk-gate → rank → abstain/recommend). It
    exposes typed ``record_*`` helpers so the planner never hand-builds an entry
    with an off-vocabulary ``kind`` or a malformed evidence dict.
    """

    entries: list[LedgerEntry] = field(default_factory=list)

    # ── generic append ─────────────────────────────────────────────────────
    def _append(
        self, kind: str, action_id: str, detail: str, evidence: dict[str, object]
    ) -> LedgerEntry:
        entry = LedgerEntry(
            kind=kind, action_id=action_id, detail=detail, evidence=evidence
        )
        self.entries.append(entry)
        return entry

    # ── typed recorders ────────────────────────────────────────────────────
    def record_enumerated(self, action_id: str, action_type: str) -> LedgerEntry:
        """Record that a candidate entered the kernel-safe set."""
        return self._append(
            "enumerated",
            action_id,
            f"safe candidate ({action_type})",
            {"action_type": action_type},
        )

    def record_blocked(self, action_id: str, card: WhyNotCard) -> LedgerEntry:
        """Record an exact-kernel hard-infeasibility rejection (doc S)."""
        return self._append(
            "blocked",
            action_id,
            card.summary(),
            {
                "minimal_groups": list(card.minimal_groups),
                "deficits": dict(card.deficits),
                "blockers": [
                    {"group": b.group.value, "detail": b.detail, "deficit": b.deficit}
                    for b in card.blockers
                ],
            },
        )

    def record_risk_filtered(
        self, action_id: str, assessment: RiskAssessment
    ) -> LedgerEntry:
        """Record a statistical-risk chance-constraint rejection (:mod:`risk_gate`)."""
        return self._append(
            "risk_filtered",
            action_id,
            assessment.reason() or "risk over budget",
            {
                "horizon_s": assessment.horizon_s,
                "violated": list(assessment.violated_channels),
                "chance_checks": {
                    k: {"conformal_upper": v[0], "budget": v[1], "exceeds": v[2]}
                    for k, v in assessment.chance_checks.items()
                },
                "cvar_checks": {
                    k: {"cvar": v[0], "budget": v[1], "exceeds": v[2]}
                    for k, v in assessment.cvar_checks.items()
                },
            },
        )

    def record_ranked(
        self, action_id: str, rank: int, lex_key: tuple[float, ...]
    ) -> LedgerEntry:
        """Record a candidate's lexicographic rank and its full key breakdown."""
        keyed = {f: float(v) for f, v in zip(LEX_FIELDS, lex_key, strict=False)}
        return self._append(
            "ranked",
            action_id,
            f"rank {rank}",
            {"rank": rank, "lex_key": keyed},
        )

    def record_abstained(self, action_id: str, reason: str) -> LedgerEntry:
        """Record a window-level or candidate-level abstention."""
        return self._append(
            "abstained", action_id, reason or "abstain", {"reason": reason}
        )

    def record_fallback(self, action_id: str, mode: str, reason: str) -> LedgerEntry:
        """Record selection of a conservative fallback action after abstention."""
        return self._append(
            "fallback",
            action_id,
            f"fallback={mode}: {reason}",
            {"mode": mode, "reason": reason},
        )

    def record_recommended(
        self, action_id: str, lex_key: tuple[float, ...], scalar: float
    ) -> LedgerEntry:
        """Record the final recommended action with its winning key + scalar."""
        keyed = {f: float(v) for f, v in zip(LEX_FIELDS, lex_key, strict=False)}
        return self._append(
            "recommended",
            action_id,
            "recommended (lex-argmin over safe set)",
            {"lex_key": keyed, "scalar_score": float(scalar)},
        )

    # ── views ──────────────────────────────────────────────────────────────
    def by_kind(self, kind: str) -> list[LedgerEntry]:
        """All entries of a given ``kind`` (for the evaluation completeness check)."""
        return [e for e in self.entries if e.kind == kind]

    def to_list(self) -> list[LedgerEntry]:
        """Return the ledger entries as a plain list (the contract surface)."""
        return list(self.entries)
