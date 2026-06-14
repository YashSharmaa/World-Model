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
from ..schema.actions import (
    Action,
    ActionType,
    AdmitAndPlace,
    Hold,
    action_workload_uid,
)
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
from .enumerate import action_id_for, enumerate_safe_candidates
from .ledger import ExplanationLedger
from .risk_gate import assess_risk
from .score import (
    LexWeights,
    build_lex_key,
    lex_key_components,
    scalar_embedding,
)
from .why_not import blockers_card, rank_card, risk_card

__all__ = [
    "SafeRegret",
    "ShieldedPlanner",
    "enumerate_full_candidate_space",
    "safe_regret",
]


# ──────────────────────────────────────────────────────────────────────────────
# Full candidate-space enumeration (the *complete* why-not provenance surface)
# ──────────────────────────────────────────────────────────────────────────────
def enumerate_full_candidate_space(
    state: WorldState,
    workload_uid: str,
    policy: PolicyConfig,
) -> list[tuple[str, Action]]:
    """Enumerate the *full* candidate action space for one queued workload.

    Unlike :func:`enumerate_safe_candidates` (which returns only the kernel-safe
    subset ``A_K(b)``), this returns the complete set of candidate commands the
    operator could have asked about — including the ones the exact kernel will
    later reject — so the ledger can carry a blocked why-not certificate for
    *every* rejected member of the candidate space, not just the survivors (doc
    F: "explanation completeness for blocked and abstained actions"; doc S, D16:
    a why-not certificate per infeasible candidate).

    The space is ``hold`` plus the cross product of admit+place over every
    eligible ``(cluster_queue, resource_flavor)`` pair and every topology domain
    selection: each topology domain individually (``required`` locality, the most
    constrained form, so a single-domain infeasibility surfaces its own minimal
    certificate) plus the unconstrained spread. ``mask_domain`` is excluded — it
    is an exogenous guardrail, not an actor action (doc M, D2). Results are
    deduplicated by the same stable :func:`action_id_for` identity the safe-set
    enumeration uses, so the two sets key on identical ids.
    """
    x = state.symbolic
    allowed = set(policy.allowed_actions)
    out: list[tuple[str, Action]] = []
    seen: set[str] = set()

    def _emit(action: Action) -> None:
        aid = action_id_for(action)
        if aid in seen:
            return
        seen.add(aid)
        out.append((aid, action))

    wl = x.workloads.get(workload_uid)

    # hold is always a candidate the operator can consider (and ask "why not").
    if ActionType.HOLD.value in allowed:
        _emit(Hold(workload_uid=workload_uid))

    if wl is None or ActionType.ADMIT_AND_PLACE.value not in allowed:
        return out

    # Eligible (queue, flavor) pairs: the workload's declared queue/flavor first,
    # then every other configured queue/flavor so impossible pairings (e.g. a
    # flavor a queue cannot borrow) still get an exact certificate.
    queues: list[str] = []
    if wl.cluster_queue_id and wl.cluster_queue_id in x.cluster_queues:
        queues.append(wl.cluster_queue_id)
    queues.extend(q for q in sorted(x.cluster_queues) if q not in queues)
    if not queues:
        queues = [wl.cluster_queue_id or ""]

    flavors: list[str] = []
    if wl.admitted_flavor_id:
        flavors.append(wl.admitted_flavor_id)
    flavors.extend(f for f in sorted(x.flavors) if f not in flavors)
    if not flavors:
        flavors = [wl.admitted_flavor_id or ""]

    # Topology selections mirror the *form* the kernel's safe-set enumeration
    # uses (the workload's declared topology policy, one selection per domain),
    # so the feasible members share an identical stable id with the safe set and
    # the safe set is a strict subset of this space. The per-domain candidates
    # additionally surface the *blocked* placements (e.g. a protected or too-small
    # domain) that get a why-not certificate. The unconstrained spread is added
    # so a "place anywhere" probe is also covered.
    wl_policy = getattr(wl, "topology_policy", None)
    wl_policy_val = getattr(wl_policy, "value", "preferred")
    selections: list[tuple[str, tuple[str, ...]]] = [
        (wl_policy_val, (dom_id,)) for dom_id in x.domains
    ]
    selections.append(("preferred", ()))

    for q in queues:
        for f in flavors:
            for domain_policy, doms in selections:
                _emit(
                    AdmitAndPlace(
                        workload_uid=workload_uid,
                        cluster_queue_id=q,
                        resource_flavor_id=f,
                        domain_policy=domain_policy,
                        domain_id_or_set=doms,
                    )
                )
    return out


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

        # 1b. BLOCKED why-not certificates for the *full* candidate space. Every
        # candidate command the operator could ask about but that the exact
        # kernel rejects gets a group-minimal infeasibility certificate
        # (kernel.why_not) recorded as a `blocked` ledger entry, so explanation
        # completeness covers the whole space, not just the survivors (doc F /
        # doc S, D16). The rejected `blocked` ScoredActions are added to
        # ``result.rejected`` so the serving why-not surface can render them.
        safe_ids = {aid for aid, _ in safe}
        full_space = enumerate_full_candidate_space(state, workload_uid, policy)
        for aid, act in full_space:
            if aid in safe_ids:
                continue
            feas = self.kernel.feasible(state, act)
            if feas.feasible:
                # Hard-feasible but pruned from the safe set (e.g. symmetry /
                # dominance reduction) — not a hard block, so leave it for the
                # risk/rank cards below; do not emit a spurious blocked card.
                continue
            why = self.kernel.why_not(state, act)
            card = blockers_card(aid, why)
            ledger.record_blocked(aid, card)
            ledger.attach_card(card)
            result.rejected.append(
                ScoredAction(
                    action_id=aid,
                    action=act,
                    forecast=KPIForecastBundle(action_id=aid),
                    safe=False,
                    passes_risk_gate=False,
                )
            )

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
        # A risk-filtered reject is *hard-feasible* but a modeled hazard's
        # conformal upper bound exceeded its budget; it carries a risk_card (doc
        # F: a calibrated-probability why-not, kept distinct from a hard block).
        survivors: list[ScoredAction] = []
        for aid, sa in scored.items():
            assessment = assess_risk(sa.forecast, policy.risk, horizon_s)
            if assessment.passes:
                sa.passes_risk_gate = True
                survivors.append(sa)
            else:
                sa.passes_risk_gate = False
                ledger.record_risk_filtered(aid, assessment)
                ledger.attach_card(risk_card(aid, assessment))
                result.rejected.append(sa)

        # 4. LEX RANK — lexicographic ordering (lex-argmin) over survivors.
        survivors.sort(key=lambda s: s.lex_key)
        for rank, sa in enumerate(survivors):
            sa.rank = rank
            ledger.record_ranked(sa.action_id, rank, sa.lex_key)
        result.candidates = survivors

        # 4b. RANK why-not cards for the *out-ranked* survivors. Each survivor
        # that lost to the lex-argmin winner gets a rank_card naming the exact
        # lexicographic field on which it lost (doc M, E1) — completing the
        # three-family why-not provenance (blocked / risk-filtered / out-ranked).
        if survivors:
            winner_key = survivors[0].lex_key
            for sa in survivors[1:]:
                ledger.attach_card(
                    rank_card(sa.action_id, sa.lex_key, winner_key)
                )

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
    # Serving adapter: rank() over a pre-scored safe set
    # ──────────────────────────────────────────────────────────────────────
    def rank(
        self,
        state: WorldState,
        safe_actions: list[tuple[str, Action]],
        forecasts: dict[str, KPIForecastBundle],
        baseline_id: str,
        *,
        policy: PolicyConfig | None = None,
        horizon_s: int | None = None,
    ) -> tuple[list[str], bool, str | None]:
        """Rank a pre-scored kernel-safe set, returning ``(ranking, abstain, reason)``.

        This is the adapter the serving ``compare_actions`` hot path drives so an
        injected :class:`ShieldedPlanner` is exercised *end-to-end*: serving has
        already run the exact kernel (producing ``safe_actions``) and the learned
        model (producing ``forecasts``), so the planner only needs to apply its
        risk gate, lexicographic ordering, and abstention gate over that set — the
        same layers 3-5 of :meth:`plan`, but consuming forecasts serving already
        computed rather than re-forecasting (so there is no double scoring, doc F
        live-demo latency).

        The signature is byte-stable with the reference engine's ``rank`` so it is
        a drop-in: serving calls ``planner.rank(state, safe_actions, forecasts,
        baseline_id)``. ``policy`` / ``horizon_s`` are optional, defaulting to the
        operator policy embedded in the state and the dominant forecast horizon,
        so the existing positional call site keeps working unchanged.
        """
        resolved_policy = policy or self._policy_from_state(state)
        risk = resolved_policy.risk
        hz = horizon_s if horizon_s is not None else _dominant_horizon(forecasts)
        source_bad = observation_integrity_bad_prob(state.source)

        survivors: list[tuple[tuple[float, ...], str]] = []
        survivor_decisions: dict[str, AbstainDecision] = {}
        for aid, _act in safe_actions:
            bundle = forecasts.get(aid)
            if bundle is None:
                continue
            assessment = assess_risk(bundle, risk, hz)
            if not assessment.passes:
                # Statistical-risk reject: drop from the ranking (gate, not score).
                continue
            lex_key = build_lex_key(bundle, hz, hard_safety=0.0, risk=risk)
            survivors.append((lex_key, aid))
            survivor_decisions[aid] = should_abstain(
                bundle, resolved_policy.abstention, source_bad
            )

        survivors.sort(key=lambda t: t[0])
        ranking = [aid for _k, aid in survivors]

        if not ranking:
            # Every safe candidate failed the risk gate → conservative abstain.
            return [], True, "no_candidate_passed_risk_gate"

        decisions = {aid: survivor_decisions[aid] for aid in ranking}
        window_abstain, window_reason = aggregate_abstention(
            decisions, state, resolved_policy.abstention
        )
        if window_abstain:
            return ranking, True, window_reason or "window_abstain"

        winner_decision = survivor_decisions.get(ranking[0])
        if winner_decision is not None and winner_decision.abstain:
            return ranking, True, f"winner_flagged ({winner_decision.reason})"

        return ranking, False, None

    @staticmethod
    def _policy_from_state(state: WorldState) -> PolicyConfig:
        """Build a :class:`PolicyConfig` from the state's operator policy.

        The serving ``rank`` call site does not pass an explicit policy, so the
        thresholds come from the symbolic operator policy embedded in the live
        snapshot (the same source the exact kernel reads its guardrails from).
        Missing budgets fall back to the :class:`PolicyConfig` defaults.
        """
        from ..schema.policy import AbstentionConfig, RiskThresholds

        op = getattr(state.symbolic, "operator_policy", None)
        if op is None:
            return PolicyConfig()
        rt = getattr(op, "risk_thresholds", None) or {}
        risk = RiskThresholds()
        for chan_attr in (
            "max_hotspot_prob",
            "max_headroom_violation_prob",
            "max_pods_ready_timeout_prob",
            "max_requeue_prob",
            "max_xid_fault_prob",
        ):
            if chan_attr in rt:
                setattr(risk, chan_attr, float(rt[chan_attr]))
        abst = AbstentionConfig()
        if "max_epistemic_disagreement" in rt:
            abst.max_epistemic_disagreement = float(rt["max_epistemic_disagreement"])
        if "max_ood_score" in rt:
            abst.max_ood_score = float(rt["max_ood_score"])
        allowed = list(
            getattr(op, "allowed_actions", PolicyConfig().allowed_actions)
        )
        return PolicyConfig(
            policy_id=getattr(op, "policy_id", "default"),
            allowed_actions=allowed,
            protected_domains=list(getattr(op, "protected_domains", [])),
            risk=risk,
            abstention=abst,
        )

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


def _dominant_horizon(forecasts: dict[str, KPIForecastBundle]) -> int:
    """Infer the horizon to gate/rank on from a set of forecast bundles.

    The serving ``rank`` adapter is handed forecasts already produced at one
    horizon (compare snaps to a single nearest horizon). We read it back off the
    bundles rather than re-deriving it, falling back to the canonical 30-minute
    planning horizon when no quantile/hazard carries one.
    """
    for bundle in forecasts.values():
        for q in bundle.quantiles:
            if q.horizon_s:
                return int(q.horizon_s)
        for h in bundle.hazards:
            if h.horizon_s:
                return int(h.horizon_s)
    return 1800


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
