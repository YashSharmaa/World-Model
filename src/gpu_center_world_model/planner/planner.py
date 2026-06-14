# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The shielded, lexicographic, risk-sensitive planner (docs M, S, F).

:class:`ShieldedPlanner` is the v1 actor. It is a *selector*, not a generator:
it can only ever emit an action drawn from the exact kernel's safe candidate set,
which gives the planner its **constructive zero-violation property** — every
recommendation is hard-safe by construction (doc M, T1 / doc S, T1).

The decision pipeline for one window (doc M, E2):

    1. enumerate  — ask the exact kernel for the safe candidate set A_K(b).
    2. score      — get a KPIForecastBundle per candidate from the world model
                    and build its lexicographic energy key.
    3. risk-gate  — drop candidates whose conformal hazard upper bound (or CVaR
                    tail) exceeds the operator's chance-constraint budget.
    4. lex-rank   — order survivors by lexicographic energy (lex-argmin).
    5. abstain    — if epistemic disagreement / OOD / degraded telemetry exceeds
                    threshold, fall back to the conservative rule / hold.
    6. recommend  — emit the lex-argmin survivor (or the fallback) and write the
                    full auditable ledger.

The module also implements :func:`safe_regret` for evaluation: the gap between
the planner's chosen action and the oracle's best action over the safe candidate
set, measured with the order-preserving scalar embedding (doc M / F,
``safe_regret_definition_and_bounds``).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model.contracts import WorldModel
from ..schema.actions import Action, ActionType, action_workload_uid
from ..schema.hashing import world_state_hash
from ..schema.kpis import KPIForecastBundle
from ..schema.policy import PolicyConfig
from ..schema.state import WorldState
from ..simulator.contracts import ExactKernel
from .abstain import (
    AbstainDecision,
    aggregate_abstention,
    observation_integrity_bad_prob,
    should_abstain,
)
from .contracts import PlannerResult, ScoredAction
from .enumerate import enumerate_safe_candidates
from .ledger import ExplanationLedger
from .risk_gate import assess_risk
from .score import (
    LexWeights,
    build_lex_key,
    lex_key_components,
    scalar_embedding,
)

__all__ = ["SafeRegret", "ShieldedPlanner", "safe_regret"]


class ShieldedPlanner:
    """Shielded lexicographic risk-sensitive planner over the safe candidate set.

    The kernel (safety shield) and world model (residual forecaster) are injected
    via their frozen Protocols so the planner depends on neither a concrete
    simulator nor MLX internals (doc E/H boundary rule). The planner is stateless
    across windows; all per-decision provenance travels in the returned
    :class:`~.contracts.PlannerResult`.
    """

    def __init__(
        self,
        kernel: ExactKernel,
        model: WorldModel,
        *,
        weights: LexWeights | None = None,
        model_version: str = "",
    ) -> None:
        self.kernel = kernel
        self.model = model
        self.weights = weights or LexWeights()
        self.model_version = model_version

    # ──────────────────────────────────────────────────────────────────────
    # The decision entry point
    # ──────────────────────────────────────────────────────────────────────
    def plan(
        self,
        state: WorldState,
        workload_uid: str,
        horizon_s: int,
        policy: PolicyConfig,
    ) -> PlannerResult:
        """Plan a single decision window for one queued workload.

        Returns a fully-populated :class:`PlannerResult` whose
        ``recommended_action_id`` is *always* a member of the kernel-safe set (or
        the conservative fallback, which is itself safe). The result carries the
        scored candidates, the rejected candidates, and the complete auditable
        ledger.
        """
        ledger = ExplanationLedger()
        state_h = world_state_hash(state)
        result = PlannerResult(
            state_hash=state_h,
            horizon_s=horizon_s,
            model_version=self.model_version,
            policy_version=policy.policy_id,
        )

        # 1. ENUMERATE — kernel-certified safe candidate set (the shield front).
        safe = enumerate_safe_candidates(self.kernel, state, workload_uid, policy)
        for aid, act in safe:
            ledger.record_enumerated(aid, act.type.value)

        # Identify the baseline action (hold for this workload) — KPIs are deltas
        # relative to it (doc kpis: "deltas relative to a declared baseline").
        baseline_id = self._baseline_action_id(safe, workload_uid)
        result.baseline_action_id = baseline_id

        # 2. SCORE — one forecast bundle + lexicographic key per candidate.
        actions = [a for _, a in safe]
        bundles = self._forecast(state, actions, horizon_s)

        scored: dict[str, ScoredAction] = {}
        abstain_decisions: dict[str, AbstainDecision] = {}
        source_bad = observation_integrity_bad_prob(state.source)

        for (aid, act), bundle in zip(safe, bundles, strict=False):
            lex_key = build_lex_key(
                bundle, horizon_s, hard_safety=0.0, risk=policy.risk
            )
            comp = lex_key_components(bundle, horizon_s, risk=policy.risk)
            sa = ScoredAction(
                action_id=aid,
                action=act,
                forecast=bundle,
                safe=True,
                lex_key=lex_key,
                delta_kpis=comp,
            )
            scored[aid] = sa
            abstain_decisions[aid] = should_abstain(
                bundle, policy.abstention, source_bad
            )

        # 3. RISK GATE — statistical chance-constraint / CVaR tail (doc M).
        survivors: list[ScoredAction] = []
        for aid, sa in scored.items():
            assessment = assess_risk(sa.forecast, policy.risk, horizon_s)
            if assessment.passes:
                sa.passes_risk_gate = True
                survivors.append(sa)
            else:
                sa.passes_risk_gate = False
                ledger.record_risk_filtered(aid, assessment)
                result.rejected.append(sa)

        # 4. LEX RANK — lexicographic ordering (lex-argmin) over survivors.
        survivors.sort(key=lambda s: s.lex_key)
        for rank, sa in enumerate(survivors):
            sa.rank = rank
            ledger.record_ranked(sa.action_id, rank, sa.lex_key)
        result.candidates = survivors

        # 5. ABSTAIN — window-level abstention gate.
        surviving_decisions = {
            sa.action_id: abstain_decisions[sa.action_id] for sa in survivors
        }
        window_abstain, window_reason = aggregate_abstention(
            surviving_decisions, state, policy.abstention
        )

        if not survivors:
            # No safe candidate passed the risk gate — abstain / fall back.
            window_abstain = True
            window_reason = window_reason or "no_candidate_passed_risk_gate"

        if window_abstain:
            return self._abstain_or_fallback(
                result,
                ledger,
                survivors,
                scored,
                safe,
                workload_uid,
                horizon_s,
                policy,
                window_reason,
            )

        # The top-ranked survivor may itself be individually flagged for
        # abstention (e.g. high epistemic disagreement on the *best* option even
        # though others are clean). In that case we still abstain conservatively.
        winner = survivors[0]
        winner_decision = abstain_decisions[winner.action_id]
        if winner_decision.abstain:
            return self._abstain_or_fallback(
                result,
                ledger,
                survivors,
                scored,
                safe,
                workload_uid,
                horizon_s,
                policy,
                f"winner_flagged ({winner_decision.reason})",
            )

        # 6. RECOMMEND — emit the lex-argmin survivor (always kernel-safe).
        scalar = scalar_embedding(winner.lex_key, self.weights)
        ledger.record_recommended(winner.action_id, winner.lex_key, scalar)
        result.recommended_action_id = winner.action_id
        result.abstain = False
        result.ledger = ledger.to_list()
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────
    def _forecast(
        self, state: WorldState, actions: list[Action], horizon_s: int
    ) -> list[KPIForecastBundle]:
        """Forecast every safe candidate via the world model's batch hot path.

        Prefers ``score_actions`` (the single compiled hot path the serving layer
        drives); degrades to per-action ``forecast`` if a backend only implements
        the scalar method. Never expands the safe set (doc C/F/M).
        """
        if not actions:
            return []
        try:
            bundles = self.model.score_actions(state, actions, horizon_s)
            if bundles is not None and len(bundles) == len(actions):
                return list(bundles)
        except (NotImplementedError, AttributeError):
            pass
        return [self.model.forecast(state, a, horizon_s) for a in actions]

    @staticmethod
    def _baseline_action_id(
        safe: list[tuple[str, Action]], workload_uid: str
    ) -> str | None:
        """Return the id of the hold action for this workload, if present."""
        for aid, act in safe:
            if act.type is ActionType.HOLD and action_workload_uid(act) == workload_uid:
                return aid
        return None

    def _abstain_or_fallback(
        self,
        result: PlannerResult,
        ledger: ExplanationLedger,
        survivors: list[ScoredAction],
        scored: dict[str, ScoredAction],
        safe: list[tuple[str, Action]],
        workload_uid: str,
        horizon_s: int,
        policy: PolicyConfig,
        reason: str,
    ) -> PlannerResult:
        """Resolve an abstention into a concrete, *still-safe* fallback action.

        Per ``AbstentionConfig.fallback`` (doc L/F):

        * ``hold`` — emit the always-safe hold (the maximally conservative move).
        * ``conservative_rule`` — among risk-gate survivors, pick the one that
          minimises residual statistical-safety pressure first, then prefers hold
          / least-utility-risk; if no survivor exists, fall back to hold.
        * ``none`` — emit no recommendation (pure abstain), leaving the operator
          to decide.

        In every branch the emitted action (if any) is drawn from the kernel-safe
        set, so abstention can never break the hard-safety certificate (doc M G1).
        """
        result.abstain = True
        result.abstain_reason = reason
        mode = policy.abstention.fallback
        baseline_id = result.baseline_action_id

        chosen_id: str | None = None
        if mode == "hold":
            chosen_id = baseline_id
        elif mode == "conservative_rule":
            chosen_id = self._conservative_choice(survivors, baseline_id)
        elif mode == "none":
            chosen_id = None
        else:  # unknown fallback string → safest default is hold
            chosen_id = baseline_id

        # If conservative rule found nothing among survivors, fall back to hold.
        if chosen_id is None and mode != "none":
            chosen_id = baseline_id

        if chosen_id is not None:
            sa = scored.get(chosen_id)
            ledger.record_fallback(chosen_id, mode, reason)
            if sa is not None:
                scalar = scalar_embedding(sa.lex_key, self.weights)
                ledger.record_recommended(chosen_id, sa.lex_key, scalar)
            result.recommended_action_id = chosen_id
        else:
            ledger.record_abstained(workload_uid, reason)

        result.candidates = survivors
        result.ledger = ledger.to_list()
        return result

    @staticmethod
    def _conservative_choice(
        survivors: list[ScoredAction], baseline_id: str | None
    ) -> str | None:
        """Pick the most conservative safe survivor for the fallback path.

        Conservative ordering prefers (a) the lowest residual statistical-safety
        pressure (lex_key[1]), then (b) hold over an active intervention, then (c)
        the lexicographic key. This deliberately *de-prioritises* utility relative
        to the normal ranking, because under abstention we trust the model less
        and lean toward caution (doc L/F conservative fallback).
        """
        if not survivors:
            return None

        def conservative_key(s: ScoredAction) -> tuple[float, int, tuple[float, ...]]:
            stat_safety = s.lex_key[1] if len(s.lex_key) > 1 else 0.0
            is_hold = 0 if s.action.type is ActionType.HOLD else 1
            return (stat_safety, is_hold, s.lex_key)

        best = min(survivors, key=conservative_key)
        # Prefer hold explicitly if it survived and is comparably safe.
        if baseline_id is not None:
            for s in survivors:
                if s.action_id == baseline_id:
                    if s.lex_key[1] <= best.lex_key[1] + 1e-9:
                        return baseline_id
                    break
        return best.action_id


# ──────────────────────────────────────────────────────────────────────────────
# Safe-regret evaluation (doc M / F)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class SafeRegret:
    """The numeric safe-regret of a planner decision against an oracle.

    ``regret = S(â) - S(a*)`` with the order-preserving scalar embedding (doc M,
    D3): non-negative when the oracle is at least as good as the planner. A
    ``regret`` of 0 means the planner chose an oracle-optimal safe action; the
    exact-search decomposition (doc M T1) guarantees ``regret ≤ 2·ε_model`` when
    the whole safe set is scored.
    """

    regret: float
    chosen_action_id: str | None
    oracle_action_id: str | None
    chosen_scalar: float
    oracle_scalar: float
    abstained: bool
    matched_oracle: bool


def safe_regret(
    result: PlannerResult,
    oracle: PlannerResult | dict[str, tuple[float, ...]] | tuple[float, ...],
    *,
    weights: LexWeights | None = None,
) -> SafeRegret:
    """Compute safe-regret of ``result`` against an ``oracle`` (doc M / F).

    The ``oracle`` may be supplied three ways for evaluation convenience:

    * another :class:`PlannerResult` whose recommendation is the oracle's choice
      (e.g. the planner run with the *reference-branch* true KPIs instead of the
      learned forecasts) — the common case;
    * a ``dict[action_id, lex_key]`` of true lexicographic keys over the safe set
      (the planner then takes the true lex-argmin as the oracle); or
    * a bare ``lex_key`` tuple, the oracle's best true key directly.

    Regret uses the order-preserving scalar embedding so it is a single number;
    by construction it preserves the lexicographic order on the bounded candidate
    set, so a positive regret always corresponds to a genuinely worse lex outcome
    (doc M, D2). When the planner abstained, regret is reported against the
    fallback it actually emitted (or against the oracle directly if it emitted
    nothing), and ``abstained`` is flagged so eval can slice on it.
    """
    w = weights or LexWeights()

    # The planner's chosen scalar comes from the recommended action's lex key.
    chosen_id = result.recommended_action_id
    chosen_key = _key_for(result, chosen_id)
    chosen_scalar = (
        scalar_embedding(chosen_key, w) if chosen_key is not None else float("inf")
    )

    # Resolve the oracle's best key + id.
    oracle_id, oracle_key = _oracle_best(oracle, w)
    oracle_scalar = (
        scalar_embedding(oracle_key, w) if oracle_key is not None else float("nan")
    )

    if chosen_key is None or oracle_key is None:
        regret = float("nan")
    else:
        regret = max(0.0, chosen_scalar - oracle_scalar)

    matched = chosen_id is not None and oracle_id is not None and chosen_id == oracle_id
    return SafeRegret(
        regret=regret,
        chosen_action_id=chosen_id,
        oracle_action_id=oracle_id,
        chosen_scalar=chosen_scalar,
        oracle_scalar=oracle_scalar,
        abstained=result.abstain,
        matched_oracle=matched,
    )


def _key_for(result: PlannerResult, action_id: str | None) -> tuple[float, ...] | None:
    """Find the lexicographic key for an action id within a planner result."""
    if action_id is None:
        return None
    for sa in result.candidates:
        if sa.action_id == action_id:
            return sa.lex_key
    for sa in result.rejected:
        if sa.action_id == action_id:
            return sa.lex_key
    return None


def _oracle_best(
    oracle: PlannerResult | dict[str, tuple[float, ...]] | tuple[float, ...],
    weights: LexWeights,
) -> tuple[str | None, tuple[float, ...] | None]:
    """Resolve the oracle's best (id, lex_key) across the supported input forms."""
    if isinstance(oracle, PlannerResult):
        oid = oracle.recommended_action_id
        return oid, _key_for(oracle, oid)
    if isinstance(oracle, dict):
        if not oracle:
            return None, None
        best_id = min(oracle, key=lambda k: oracle[k])
        return best_id, oracle[best_id]
    if isinstance(oracle, tuple):
        return None, oracle
    return None, None
