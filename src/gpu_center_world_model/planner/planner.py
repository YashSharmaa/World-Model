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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..fairness.debt import FairnessDebtTracker
from ..fairness.hooks import fairness_rank
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
from ..schema.policy import PolicyConfig, RiskThresholds
from ..schema.state import WorldState
from ..simulator.contracts import ExactKernel
from .abstain import (
    AbstainDecision,
    aggregate_abstention,
    observation_integrity_bad_prob,
    should_abstain,
)
from .contracts import LedgerEntry, PlannerResult, ScoredAction
from .enumerate import (
    CEMSearch,
    MCTSSearch,
    MPPISearch,
    SearchResult,
    action_id_for,
    aggregate_sequence_lex_key,
    enumerate_safe_candidates,
    guarded_proposal,
)
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
    "DROPEResult",
    "SafeRegret",
    "SafeRegretOracle",
    "ShieldedPlanner",
    "build_safe_regret_oracle",
    "doubly_robust_ope",
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
        debt_tracker: FairnessDebtTracker | None = None,
        amortization: object | None = None,
    ) -> None:
        self.kernel = kernel
        self.model = model
        self.weights = weights or LexWeights()
        self.model_version = model_version
        # Optional amortized proposal/value/abstention heads (doc M D1). When
        # supplied the planner uses a *guarded* proposal (top-k recall) to avoid
        # scoring the full safe set, falling back to exhaustive scoring whenever
        # u_ω flags OOD/disagreement or the candidate count is below threshold
        # (doc M G2). ``None`` keeps the planner in exact, fully-enumerated mode,
        # so the v1 default path — and the golden — are unchanged.
        self.amortization = amortization
        # Slow-core fairness-debt memory (doc V D31). ``None`` in v1 / fairness
        # off; when supplied (fairness on) it carries per-tenant decayed debt
        # across windows so the Eq11 worst-served-first re-rank and the debt
        # budget have real history. Never consulted while ``fairness_mode`` is
        # off, so the default path is unchanged.
        self.debt_tracker = debt_tracker

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

        # 1c. GUARDED AMORTIZED PROPOSAL (doc M D3/G2). When amortization heads are
        # injected, prune the safe set to the proposal head's top-k *before*
        # scoring — but only outside the fallback region. The guard falls back to
        # the full safe set whenever u_ω flags OOD/disagreement or the candidate
        # count is below the enumeration threshold, so utility-equivalence with
        # the exact planner holds there (doc M G2). With no heads this is a no-op,
        # so the default path scores the full safe set exactly (golden unchanged).
        scoring_set = safe
        if self.amortization is not None and safe:
            gp = guarded_proposal(
                self.amortization, state, safe, horizon_s
            )
            scoring_set = gp.actions
            ledger.entries.append(
                LedgerEntry(
                    kind="ranked",
                    action_id="__amortization__",
                    detail=(
                        f"guarded_proposal reason={gp.reason}; "
                        f"used_fallback={gp.used_fallback}; "
                        f"n_proposed={len(gp.actions)}/{len(safe)}"
                    ),
                    evidence={
                        "amortization_reason": gp.reason,
                        "used_fallback": gp.used_fallback,
                        "n_proposed": len(gp.actions),
                        "n_safe": len(safe),
                        "abstain_scores": dict(gp.abstain_scores),
                    },
                )
            )

        # 2. SCORE — one forecast bundle + lexicographic key per candidate.
        safe = scoring_set
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

        # 3b/4-fairness. FAIRNESS RE-RANK — Eq11 budgeted-lexicographic rule.
        # Runs *after* the risk gate (so fairness only ever reorders hard-safe,
        # risk-passing actions) and *before* the final rank assignment, but only
        # when the operator turned fairness on (``fairness_mode != off``). When
        # off this is a strict no-op and the lex order above is final, so the v1
        # golden is byte-stable (doc V H10 / Eq11).
        if policy.fairness_mode != "off" and survivors:
            survivors = self._fairness_rerank(
                survivors, state, policy, horizon_s, ledger
            )

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

    # ──────────────────────────────────────────────────────────────────────
    # Bounded multi-step search (doc M enumeration_vs_search D3/D4; E_lex)
    # ──────────────────────────────────────────────────────────────────────
    def plan_sequence(
        self,
        state: WorldState,
        workload_uid: str,
        horizon_s: int,
        policy: PolicyConfig,
        *,
        backend: str = "cem",
        sequence_horizon: int = 2,
        rng_seed: int = 0,
        **backend_kwargs: object,
    ) -> SearchResult:
        """Plan a multi-step safe action *sequence* via bounded search (doc M D3).

        Where :meth:`plan` selects a single action by *exhaustively* scoring the
        kernel-safe set (decision-theoretically optimal while that set is finite
        and small, doc M enumeration_vs_search D1), this drives an admissible
        bounded-search backend over H-step safe action sequences ``a_0:H-1`` and
        scores each by the lexicographic sequence energy
        ``E_lex(b_0, a_0:H-1)`` (doc M E1). It is the *search-later* path the doc
        prescribes for when the action ontology / lookahead grows past what local
        exhaustive evaluation can support (doc M D3/D4).

        Backends (``"cem"`` | ``"mppi"`` | ``"mcts"``) share one interface:

        * the per-step candidate sampler is the exact kernel's safe set (the
          shield front — search never leaves ``A_K(b)``, so the constructive
          zero-violation property is preserved, doc M G1);
        * the sequence scorer rolls the world model forward over the whole
          sequence and aggregates the per-step lexicographic energy keys (doc M
          D4 ``H_hard`` / ``R_stat`` / ``ψ`` summed across the horizon);
        * the injected amortization **value head** warm-starts the first step and
          (for CEM/MPPI) the sampling distribution (doc M D3).

        The emitted action is the first action of the best sequence found
        (receding-horizon control); the full sequence + its energy travel in the
        returned :class:`SearchResult`.
        """
        rng = np.random.default_rng(rng_seed)

        def sampler(s: WorldState, step: int) -> list[Action]:
            safe = enumerate_safe_candidates(self.kernel, s, workload_uid, policy)
            return [a for _, a in safe]

        def scorer(s: WorldState, seq: list[Action]) -> tuple[float, ...]:
            step_keys: list[tuple[float, ...]] = []
            for a in seq:
                bundles = self._forecast(s, [a], horizon_s)
                bundle = bundles[0] if bundles else None
                if bundle is None:
                    step_keys.append((float("inf"),))
                    continue
                # hard_safety stays 0 (sequence drawn from the kernel-safe set);
                # the risk budgets give the doc-M positive-excess R_stat term.
                step_keys.append(
                    build_lex_key(bundle, horizon_s, hard_safety=0.0, risk=policy.risk)
                )
            return aggregate_sequence_lex_key(step_keys)

        backends = {"cem": CEMSearch, "mppi": MPPISearch, "mcts": MCTSSearch}
        cls = backends.get(backend)
        if cls is None:
            raise ValueError(
                f"unknown search backend {backend!r}; expected one of "
                f"{sorted(backends)}"
            )
        engine = cls(
            sampler,
            scorer,
            horizon=sequence_horizon,
            value_head=self.amortization,
            rng=rng,
            **backend_kwargs,  # type: ignore[arg-type]
        )
        return engine.search(state, horizon_s)

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

    def _fairness_rerank(
        self,
        survivors: list[ScoredAction],
        state: WorldState,
        policy: PolicyConfig,
        horizon_s: int,
        ledger: ExplanationLedger,
    ) -> list[ScoredAction]:
        """Apply the Eq11 budgeted-lexicographic fairness re-rank (doc V H10).

        Wires three pieces of evidence into :func:`fairness_rank`:

        * **realized-service history** ``U_t`` per tenant, so the under-served gap
          is the *true* ``(B_t - U_t)_+`` (doc V D31) — taken from the slow-core
          debt tracker's decayed realised service when present, else from the
          Kueue LocalQueue historical-usage observables retained on the state;
        * the **action-conditioned drift delta** ``δ_FD(a|s)`` per candidate
          (doc V D23), derived from each candidate's forecast cross-rack / fairness
          pressure relative to the hold baseline so the drift budget ``δ_F`` gates
          on a real per-action quantity rather than a constant;
        * the operator **debt budget** ``δ_D`` and **drift budget** ``δ_F`` read
          off the :class:`PolicyConfig`.

        The returned list is the fairness-reordered survivor set; every candidate
        also gets an H8 ledger entry (serving tenant, benchmark service, gap,
        debt, drift, both budget verdicts) attached to ``ledger``.
        """
        realized = self._realized_service(state)
        drift_delta = self._action_drift_deltas(survivors, horizon_s)
        res = fairness_rank(
            survivors,
            state,
            policy,
            horizon_s=float(horizon_s),
            debt_tracker=self.debt_tracker,
            realized_service=realized or None,
            action_drift_delta=drift_delta or None,
        )
        # Record the fairness evidence into the explanation ledger (H8). Each
        # entry is a `ranked` record annotated with the Eq11 fairness fields and
        # is appended to the same auditable ledger the rest of plan() writes to.
        if res.applied:
            ledger.entries.extend(res.ledger)
        return res.ranked if res.applied else survivors

    @staticmethod
    def _realized_service(state: WorldState) -> dict[str, float]:
        """Per-tenant realised cumulative service ``U_t`` from retained history.

        Maps the Kueue LocalQueue ``historical_usage`` observable (decayed
        consumed resources, doc V H2/D22) up to the tenant that owns each
        LocalQueue, so the under-served gap ``B_t - U_t`` uses the platform's own
        decayed history rather than a self-reported number. Tenants with no
        recorded history contribute 0 (maximally under-served), which is the
        conservative worst-served-first default (doc V D31).
        """
        sym = state.symbolic
        out: dict[str, float] = {}
        local_queues = getattr(sym, "local_queues", {}) or {}
        # Resolve LocalQueue -> tenant via the queue's own tenant_id when present,
        # else via any workload pointing at that LocalQueue's cluster queue.
        for lq in local_queues.values():
            tenant = getattr(lq, "tenant_id", None)
            usage = float(getattr(lq, "historical_usage", 0.0))
            if tenant:
                out[tenant] = out.get(tenant, 0.0) + usage
        return out

    def _action_drift_deltas(
        self, survivors: list[ScoredAction], horizon_s: int
    ) -> dict[str, float]:
        """Action-conditioned fairness-drift delta ``δ_FD(a|s)`` per candidate.

        Doc V D23 defines ``δ_FD(a|s) = E[FD_H|s,a] - E[FD_H|s,hold]``. The
        planner does not re-simulate the cohort weighted-share trajectory per
        candidate, so it uses the forecast's own fairness-relevant pressure as a
        calibrated proxy: an admit/place that concentrates load on already-loaded
        domains (higher cross-rack / locality cost and higher admitted GPU-hours
        on a contended pool) *worsens* weighted-share spread, while a hold leaves
        drift unchanged (its delta is exactly 0 by definition). The proxy is
        signed and bounded, monotone in the locality cost, and reduces to 0 for
        hold — exactly the qualitative shape Eq11's drift budget needs to gate
        on. The hold survivor (if present) anchors the baseline at 0.
        """
        deltas: dict[str, float] = {}
        for sa in survivors:
            if sa.action.type is ActionType.HOLD:
                deltas[sa.action_id] = 0.0
                continue
            comp = sa.delta_kpis or {}
            # locality cost (cross-rack rate, [0,1]) is the borrowing-layer drift
            # proxy: cross-rack placement spreads a tenant's share across the
            # cohort, widening the weighted-share gap (doc V D21/Eq12).
            locality = float(comp.get("locality", 0.0))
            fragmentation = float(comp.get("fragmentation", 0.0))
            deltas[sa.action_id] = locality + 0.5 * fragmentation
        return deltas

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


# ──────────────────────────────────────────────────────────────────────────────
# Safe-regret oracle from the full enumerated safe set (doc M safe_regret D1)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class SafeRegretOracle:
    """The lex-argmin oracle over a window's safe set under *true* KPIs (doc M D1).

    Built from the full enumerated kernel-safe set scored with the reference-
    branch true lexicographic KPIs (not the learned forecasts), this is the
    comparator the planner's chosen action is regretted against
    ``r_t^safe = S_t(â_t) - S_t(a_t^⋆)`` (doc M safe_regret D1/D3). Carrying the
    full ``true_keys`` map lets :func:`safe_regret` difference *any* chosen action
    against the same oracle on the same scalar embedding.
    """

    oracle_action_id: str | None
    oracle_key: tuple[float, ...] | None
    true_keys: dict[str, tuple[float, ...]]  # action_id -> true lex key
    safe_action_ids: tuple[str, ...]

    def as_oracle_dict(self) -> dict[str, tuple[float, ...]]:
        """The ``{action_id: true_key}`` form :func:`safe_regret` accepts."""
        return dict(self.true_keys)


def build_safe_regret_oracle(
    safe_actions: Sequence[tuple[str, Action]],
    true_forecasts: Mapping[str, KPIForecastBundle],
    horizon_s: int,
    *,
    risk: RiskThresholds | None = None,
) -> SafeRegretOracle:
    """Build the safe-regret oracle by scoring the *full* enumerated safe set.

    The oracle is the lex-argmin over the kernel-certified safe candidate set
    ``C_t^safe`` evaluated with the **reference-branch true KPIs** (doc M D1):
    each safe action's ``true_forecasts`` bundle (produced by replaying the
    reference branch, not the learned model) is mapped to its true lexicographic
    key via the same :func:`build_lex_key` the planner ranks on, and the
    lex-smallest is the oracle ``a_t^⋆``. Actions with no supplied true forecast
    are skipped (they cannot be scored truthfully and must not silently win).

    This is the honest comparator doc M/F mandate for planning utility: regret is
    measured against the best *safe* action under the branch environment, never
    inferred from model loss. Pair the returned oracle with :func:`safe_regret`.
    """
    true_keys: dict[str, tuple[float, ...]] = {}
    for aid, _act in safe_actions:
        bundle = true_forecasts.get(aid)
        if bundle is None:
            continue
        true_keys[aid] = build_lex_key(
            bundle, horizon_s, hard_safety=0.0, risk=risk
        )
    oracle_id: str | None = None
    oracle_key: tuple[float, ...] | None = None
    if true_keys:
        oracle_id = min(true_keys, key=lambda k: true_keys[k])
        oracle_key = true_keys[oracle_id]
    return SafeRegretOracle(
        oracle_action_id=oracle_id,
        oracle_key=oracle_key,
        true_keys=true_keys,
        safe_action_ids=tuple(aid for aid, _ in safe_actions),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Doubly-robust OPE with a logging-support / coverage guard (doc M E1 / A1 / E2)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class DROPEResult:
    """A sequential doubly-robust off-policy value estimate (doc M safe_regret E1).

    Attributes
    ----------
    value:
        The DR value estimate ``V̂_0^DR`` (doc M E1). ``nan`` when the coverage
        guard refused to estimate (out-of-support target).
    direct_value:
        The pure direct-method (model ``V̂``) baseline, for comparison.
    supported:
        ``True`` iff every step's target action lay in the logger's support with
        positive propensity (doc M A1 overlap / positivity). When ``False`` the
        estimate is labelled **unsupported** and ``value`` is ``nan`` — DR is
        never used to justify out-of-coverage actions (doc M E2 / INFERENCE).
    min_overlap:
        The smallest logging propensity encountered on the target path — the
        binding overlap diagnostic (doc M A1).
    max_weight:
        The largest cumulative importance ratio ``ρ_{0:t}`` — large values flag a
        fragile estimate even when nominally supported (doc M E2 truncation).
    n_steps:
        Horizon length scored.
    note:
        Human-readable provenance / refusal reason.
    """

    value: float
    direct_value: float
    supported: bool
    min_overlap: float
    max_weight: float
    n_steps: int
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "direct_value": self.direct_value,
            "supported": self.supported,
            "min_overlap": self.min_overlap,
            "max_weight": self.max_weight,
            "n_steps": self.n_steps,
            "note": self.note,
        }


def doubly_robust_ope(
    rewards: Sequence[float],
    target_propensities: Sequence[float],
    behavior_propensities: Sequence[float],
    v_hat: Sequence[float],
    q_hat: Sequence[float],
    *,
    gamma: float = 1.0,
    overlap_floor: float = 1e-3,
    weight_cap: float | None = None,
) -> DROPEResult:
    """Sequential doubly-robust OPE with a logging-support / coverage guard.

    Implements the standard finite-horizon sequential DR estimator (doc M
    safe_regret E1):

        ``V̂_0^DR = V̂(s_0) + Σ_t ρ_{0:t} [ r_t + γ V̂(s_{t+1}) - Q̂(s_t, a_t) ]``,
        ``ρ_{0:t} = Π_{u≤t} π(a_u|s_u) / μ(a_u|s_u)``.

    **Coverage guard (doc M A1 / E2).** OPE is only valid under overlap /
    positivity: the target action must have support under the logger on the
    relevant state distribution. If *any* step's behavior propensity falls below
    ``overlap_floor`` while the target puts mass there, the estimate is declared
    **unsupported** and ``value`` is ``nan`` — the guard exists precisely so OPE
    is *never* used to justify out-of-coverage actions (placement / mask choices
    the logger never attempted, doc M E2 INFERENCE). On supported paths an
    optional ``weight_cap`` truncates the cumulative ratio (the truncated-DR
    remedy under weak overlap, doc M E2), and ``max_weight`` is reported so a
    fragile-but-supported estimate is still flagged.

    Inputs are aligned per-step sequences of length ``H``: realized ``rewards``
    ``r_t``; target / behavior action propensities ``π(a_t|s_t)`` and
    ``μ(a_t|s_t)``; the model's state values ``v_hat`` = ``[V̂(s_0..s_H)]``
    (length ``H+1``); and ``q_hat`` = ``[Q̂(s_t,a_t)]`` (length ``H``).
    """
    H = len(rewards)
    if H == 0 or len(target_propensities) != H or len(behavior_propensities) != H:
        return DROPEResult(
            value=float("nan"),
            direct_value=float("nan"),
            supported=False,
            min_overlap=float("nan"),
            max_weight=float("nan"),
            n_steps=H,
            note="empty or mis-aligned inputs",
        )
    if len(v_hat) < H + 1 or len(q_hat) < H:
        return DROPEResult(
            value=float("nan"),
            direct_value=float(v_hat[0]) if v_hat else float("nan"),
            supported=False,
            min_overlap=float("nan"),
            max_weight=float("nan"),
            n_steps=H,
            note="v_hat must have length H+1 and q_hat length H",
        )

    direct = float(v_hat[0])

    # Coverage guard: any target-supported step the logger barely covers ⇒ refuse.
    min_overlap = float("inf")
    for t in range(H):
        pi_t = float(target_propensities[t])
        mu_t = float(behavior_propensities[t])
        if pi_t > 0.0:
            min_overlap = min(min_overlap, mu_t)
    if not np.isfinite(min_overlap):
        min_overlap = 1.0  # target put no mass anywhere ⇒ vacuously covered
    if min_overlap < overlap_floor:
        return DROPEResult(
            value=float("nan"),
            direct_value=direct,
            supported=False,
            min_overlap=min_overlap,
            max_weight=float("nan"),
            n_steps=H,
            note=(
                f"out-of-coverage: min logging propensity {min_overlap:.3g} < "
                f"overlap_floor {overlap_floor:.3g}; DR unsupported (doc M A1/E2)."
            ),
        )

    value = direct
    rho = 1.0
    max_weight = 0.0
    for t in range(H):
        pi_t = float(target_propensities[t])
        mu_t = max(float(behavior_propensities[t]), overlap_floor)
        rho *= pi_t / mu_t
        if weight_cap is not None:
            rho = min(rho, weight_cap)
        max_weight = max(max_weight, rho)
        td = (
            float(rewards[t])
            + gamma * float(v_hat[t + 1])
            - float(q_hat[t])
        )
        value += rho * td

    return DROPEResult(
        value=float(value),
        direct_value=direct,
        supported=True,
        min_overlap=min_overlap,
        max_weight=max_weight,
        n_steps=H,
        note="supported DR estimate (doc M E1).",
    )
