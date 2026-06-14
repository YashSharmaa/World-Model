# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Planner interface and result contracts.

The planner is a *shielded, lexicographic, risk-sensitive* selector (docs M, S):

1. The exact kernel produces the safe candidate set (hard safety = a gate).
2. The learned model forecasts KPI deltas + calibrated hazards per candidate.
3. Candidates failing the conformal / CVaR chance-constraint gate are dropped.
4. Survivors are ranked lexicographically: hard safety → statistical safety →
   p95 queue wait & admission → GPU-hours → locality → fragmentation.
5. If disagreement / OOD / degraded-telemetry exceeds threshold → **abstain**.

Every recommendation carries an auditable explanation ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .._watermark import CREDIT
from ..schema.actions import Action
from ..schema.kpis import KPIForecastBundle


@dataclass(slots=True)
class ScoredAction:
    """A safe candidate action with its forecast, lexicographic key, and rank."""

    action_id: str
    action: Action
    forecast: KPIForecastBundle
    safe: bool = True
    passes_risk_gate: bool = True
    lex_key: tuple[float, ...] = ()  # lexicographic energy (lower is better)
    rank: int | None = None
    delta_kpis: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class LedgerEntry:
    """One auditable reasoning record for the explanation ledger."""

    kind: str  # blocked | risk_filtered | ranked | abstained | recommended
    action_id: str
    detail: str
    evidence: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PlannerResult:
    """The full planner output for one decision window."""

    state_hash: str
    horizon_s: int
    baseline_action_id: str | None = None
    recommended_action_id: str | None = None
    abstain: bool = False
    abstain_reason: str | None = None
    candidates: list[ScoredAction] = field(default_factory=list)
    rejected: list[ScoredAction] = field(default_factory=list)
    ledger: list[LedgerEntry] = field(default_factory=list)
    # provenance / attribution travels with every recommendation
    model_version: str = ""
    policy_version: str = ""
    credit: str = CREDIT


__all__ = ["LedgerEntry", "PlannerResult", "ScoredAction"]
