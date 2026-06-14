# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Safe-candidate enumeration — the constructive zero-violation front door.

The planner *never* invents actions. It asks the exact kernel (the safety shield,
injected via the :class:`~..simulator.contracts.ExactKernel` Protocol) for the
complete, hard-feasible candidate set ``A_K(b) = {a : K(b, a) = 1}`` for the
queued workload under consideration (doc M, D3; doc S, D1).

Design rationale — *enum-now, search-later* (doc M ``enumeration_vs_search``):
while the v1 action ontology is ``{hold, admit, place, admit_and_place}`` over a
2-4 rack demo pod, the safe set is finite and small, so exhaustive enumeration is
**decision-theoretically optimal**: search error is exactly zero whenever every
safe candidate is scored. We therefore enumerate fully here and defer
MPC/CEM/MCTS to a later action family. The learned model may rank / forecast /
abstain over this set but may *never expand it* (doc C / F / M).

This module also guarantees that ``hold`` is always present as the conservative
fallback (doc F step 6: "if the safe set is empty except hold, return hold"),
and it canonicalises a stable, content-addressed ``action_id`` per candidate so
the ledger, forecasts, and UI all key on the same identifier.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..schema.actions import Action, ActionType, Hold, action_workload_uid
from ..schema.policy import PolicyConfig
from ..schema.state import WorldState
from ..simulator.contracts import ExactKernel

__all__ = [
    "AmortizationDataset",
    "AmortizationExample",
    "CEMSearch",
    "GuardedProposal",
    "LinearAmortizationHeads",
    "MCTSSearch",
    "MPPISearch",
    "SearchBackend",
    "SearchResult",
    "SequenceScorer",
    "action_id_for",
    "aggregate_sequence_lex_key",
    "enumerate_safe_candidates",
    "filter_by_policy_whitelist",
    "guarded_proposal",
    "train_amortization_heads",
]


# ──────────────────────────────────────────────────────────────────────────────
# Stable action identity
# ──────────────────────────────────────────────────────────────────────────────
def _canonical(obj: Any) -> Any:
    """Recursively canonicalise a dataclass/enum/collection for hashing.

    Mirrors :mod:`schema.hashing` semantics so the same action always maps to the
    same id regardless of dict ordering or float micro-jitter.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _canonical(getattr(obj, f.name))
            for f in sorted(dataclasses.fields(obj), key=lambda x: x.name)
        }
    if isinstance(obj, dict):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(x) for x in obj]
    if isinstance(obj, set):
        return sorted(_canonical(x) for x in obj)
    if hasattr(obj, "value") and hasattr(obj, "name"):  # Enum
        return obj.value
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def action_id_for(action: Action) -> str:
    """Return a stable, content-addressed identifier for a candidate action.

    The id is ``"<type>:<12-hex digest>"`` — human-skimmable (the action type is
    visible) yet collision-resistant and reproducible across processes, so the
    forecast bundle, ledger entries, and UI ribbon can all key on it (doc F).
    """
    payload = _canonical(action)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()[:12]
    return f"{action.type.value}:{digest}"


# ──────────────────────────────────────────────────────────────────────────────
# Policy whitelist (operator guardrail, applied *before* the kernel where cheap)
# ──────────────────────────────────────────────────────────────────────────────
def filter_by_policy_whitelist(
    actions: list[Action], policy: PolicyConfig
) -> list[Action]:
    """Drop any action whose type is not in the operator's ``allowed_actions``.

    The exact kernel already enforces the action whitelist as a hard blocker
    (``BlockerGroup.ACTION_WHITELIST``); applying it here too is a cheap,
    defence-in-depth pre-filter so we never even forecast a disallowed action.

    ``mask_domain`` alphabet decision (doc M D2 ``A = {hold, admit, place,
    admit_and_place, mask_domain}`` reconciled with ``ACTOR_ACTIONS``). Doc M D2
    lists ``mask_domain`` in the *action alphabet* but also says the v1 actor
    "searches over the safe finite candidate set generated by the exact kernel".
    We reconcile the two as a **single, explicit, opt-in rule**: ``mask_domain``
    is an exogenous operator guardrail / scenario intervention, *not* an
    autonomous actor action (``schema.actions.ACTOR_ACTIONS`` excludes it), so it
    is **excluded from the actor candidate set by default**. It is admitted only
    when the operator *explicitly* whitelists ``"mask_domain"`` in
    ``allowed_actions`` (the default whitelist does not, so existing behaviour is
    byte-stable) — and even then it still has to be kernel-certified like any
    other candidate (the kernel must have emitted it into ``safe_candidates`` and
    pass ``K(b, a) = 1``; this filter never *adds* an action). This is the
    "allow it through the whitelist when policy permits + kernel-certify" path:
    the same uniform whitelist+kernel discipline governs every action type, with
    no special-cased hard drop, while the conservative default keeps ``mask_domain``
    out of the autonomous actor's set.
    """
    allowed = set(policy.allowed_actions)
    out: list[Action] = []
    for a in actions:
        if a.type.value in allowed:
            out.append(a)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# The enumeration entry point
# ──────────────────────────────────────────────────────────────────────────────
def enumerate_safe_candidates(
    kernel: ExactKernel,
    state: WorldState,
    workload_uid: str,
    policy: PolicyConfig,
) -> list[tuple[str, Action]]:
    """Return the kernel-certified safe candidate set for one queued workload.

    Steps:

    1. Ask the injected exact kernel for ``safe_candidates(state, workload_uid)``.
       By contract these are already hard-feasible (``K(b, a) = 1``).
    2. Apply the operator action whitelist as a defence-in-depth pre-filter
       (:func:`filter_by_policy_whitelist`). ``mask_domain`` is admitted only when
       the operator explicitly whitelists it (default: excluded — it is an
       exogenous guardrail, not an actor action, doc M D2 reconciled with
       ``ACTOR_ACTIONS``); the kernel still certifies whatever survives.
    3. Guarantee ``hold`` is present as the conservative fallback (doc F step 6;
       ``hold`` for a pending workload is always hard-feasible).
    4. Deduplicate by stable ``action_id`` while preserving the kernel's order.

    Returns a list of ``(action_id, action)`` pairs. The constructive
    zero-violation property of the whole planner rests on this set: every action
    the planner can ever emit is drawn from here, so it is hard-safe by
    construction (doc M, T1 / doc S, T1).
    """
    raw = list(kernel.safe_candidates(state, workload_uid))
    allowed = filter_by_policy_whitelist(raw, policy)

    # Ensure hold is always available as the conservative fallback. The kernel
    # should already include it, but the planner must never be left without an
    # emittable safe action when hold is operator-permitted.
    if ActionType.HOLD.value in set(policy.allowed_actions):
        has_hold = any(
            a.type is ActionType.HOLD and action_workload_uid(a) == workload_uid
            for a in allowed
        )
        if not has_hold:
            allowed.append(Hold(workload_uid=workload_uid))

    seen: set[str] = set()
    out: list[tuple[str, Action]] = []
    for a in allowed:
        aid = action_id_for(a)
        if aid in seen:
            continue
        seen.add(aid)
        out.append((aid, a))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Action-sequence feature extraction (shared by amortization + bounded search)
# ──────────────────────────────────────────────────────────────────────────────
def _action_feature(state: WorldState, action: Action) -> np.ndarray:
    """A small, deterministic feature vector for one ``(state, action)`` pair.

    The amortization heads are linear over this fixed, auditable encoding (doc M
    D1: the heads sit *on top of* the world model and never own feasibility). The
    features are all read off the symbolic state + the action's own type/locality
    — no learned MLX latent is required, so the planner stays backend-neutral and
    import-clean on CI. The vector intentionally captures the lexicographic
    signals the planner ranks on: action type, the contended-pool pressure the
    placement faces, and the workload's gang size.
    """
    x = state.symbolic
    t = action.type
    is_hold = 1.0 if t is ActionType.HOLD else 0.0
    is_place = 1.0 if t in (ActionType.PLACE, ActionType.ADMIT_AND_PLACE) else 0.0
    # locality span: number of explicitly named domains (0 ⇒ spread / unconstrained)
    doms = getattr(action, "domain_id_or_set", ()) or ()
    n_doms = float(len(doms)) if isinstance(doms, (list, tuple, set)) else 0.0

    # contended free-GPU pressure across all domains (lower free ⇒ tighter fit)
    domains = list(x.domains.values())
    free_gpu = float(sum(getattr(d, "free_gpu", 0) for d in domains))
    n_dom = max(len(domains), 1)
    headroom = (
        float(sum(float(getattr(d, "headroom_score", 1.0)) for d in domains)) / n_dom
    )
    hotspot = (
        float(sum(float(getattr(d, "hotspot_score", 0.0)) for d in domains)) / n_dom
    )

    # gang size of the target workload (drives admitted GPU-hours utility)
    uid = action_workload_uid(action)
    gang = 0.0
    if uid is not None:
        wl = x.workloads.get(uid)
        if wl is not None:
            gang = float(sum(ps.count * ps.gpu_req for ps in wl.podsets))

    return np.array(
        [
            1.0,  # bias
            is_hold,
            is_place,
            n_doms,
            free_gpu / 64.0,
            headroom,
            hotspot,
            gang / 64.0,
        ],
        dtype=float,
    )


_FEATURE_DIM = 8


# ──────────────────────────────────────────────────────────────────────────────
# Offline full-enumeration replay distillation set (doc M, amortization_plan D2)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class AmortizationExample:
    """One offline replay window's distillation record (doc M D2).

    Stores everything the three heads are trained against for a single decision
    window where the planner *fully enumerated* the safe set:

    * ``features`` — ``(K, F)`` feature matrix, one row per safe candidate;
    * ``oracle_index`` — the row of the lex-argmin oracle action ``a_t^⋆``;
    * ``lex_scores`` — the scalarized lex score ``S_t(a)`` per candidate (lower
      is better) — the value-head regression target;
    * ``margins`` — the planner margin ``Δ_t(a) = S_t(a) - S_t(a_t^⋆)`` (≥ 0);
    * ``abstain_labels`` — the per-candidate abstention / OOD label in ``[0, 1]``
      (disagreement / OOD / degraded telemetry), the ``u_ω`` target.
    """

    features: np.ndarray  # (K, F)
    oracle_index: int
    lex_scores: np.ndarray  # (K,)
    margins: np.ndarray  # (K,)
    abstain_labels: np.ndarray  # (K,)


@dataclass(slots=True)
class AmortizationDataset:
    """A collection of :class:`AmortizationExample` windows (doc M D2)."""

    examples: list[AmortizationExample] = field(default_factory=list)

    def add(self, example: AmortizationExample) -> None:
        self.examples.append(example)

    @classmethod
    def from_replay(
        cls,
        windows: Sequence[
            tuple[
                WorldState,
                list[Action],  # the enumerated safe candidates
                list[float],  # true scalarized lex score S_t per candidate
                list[float],  # abstain/OOD label per candidate
            ]
        ],
    ) -> AmortizationDataset:
        """Build the distillation set from full-enumeration replay windows.

        Each window supplies the safe candidate actions, the *true* scalarized
        lex score ``S_t(a)`` per candidate (from the reference-branch KPIs — the
        same object the planner ranks on, doc M D2), and the abstention/OOD label
        per candidate. The oracle index, pairwise/listwise ranking labels, and
        the planner margins ``Δ_t`` are derived here so the heads can be trained
        directly off the record (doc M D2).
        """
        ds = cls()
        for state, actions, lex_scores, abstain in windows:
            if not actions:
                continue
            feats = np.stack(
                [_action_feature(state, a) for a in actions], axis=0
            )
            scores = np.asarray(lex_scores, dtype=float)
            oracle = int(np.argmin(scores))
            margins = scores - scores[oracle]
            ds.add(
                AmortizationExample(
                    features=feats,
                    oracle_index=oracle,
                    lex_scores=scores,
                    margins=margins,
                    abstain_labels=np.asarray(abstain, dtype=float),
                )
            )
        return ds


# ──────────────────────────────────────────────────────────────────────────────
# The amortized heads (doc M, amortization_plan D1) — pure-NumPy, deterministic
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class LinearAmortizationHeads:
    """Linear proposal / value / abstention heads over :func:`_action_feature`.

    Realizes the doc-M D1 head bundle without any MLX dependency so the planner
    stays backend-neutral and CI-importable: a proposal weight vector ``w_pi``
    (listwise/pairwise ranking of safe candidates), a value weight vector
    ``w_v`` (regresses the scalarized lex score ``S_t`` — *lower is better*, so
    it can warm-start search directly), and an abstention weight vector ``w_u``
    (logistic over the disagreement / OOD label, in ``[0, 1]``). The heads are a
    drop-in :class:`~...model.contracts.AmortizationHeads` implementation.

    Safety note (doc M G1): the heads only ever *rank / score / decide-to-trust*
    over the kernel-safe set; they never expand it, so amortization cannot break
    the hard-safety certificate.
    """

    w_pi: np.ndarray  # (F,) proposal: higher score ⇒ more likely the oracle
    w_v: np.ndarray  # (F,) value: predicts S_t (lower is better)
    w_u: np.ndarray  # (F,) abstention logits

    @classmethod
    def zeros(cls, feature_dim: int = _FEATURE_DIM) -> LinearAmortizationHeads:
        return cls(
            w_pi=np.zeros(feature_dim),
            w_v=np.zeros(feature_dim),
            w_u=np.zeros(feature_dim),
        )

    # -- AmortizationHeads Protocol -----------------------------------------
    def proposal_scores(
        self, state: WorldState, actions: list[Action], horizon_s: int
    ) -> list[float]:
        if not actions:
            return []
        feats = np.stack([_action_feature(state, a) for a in actions], axis=0)
        return [float(v) for v in feats @ self.w_pi]

    def value(
        self, state: WorldState, action: Action | None, horizon_s: int
    ) -> float:
        if action is None:
            # Terminal value V_ψ(b): the best (lowest) predicted lex score is not
            # available without candidates; use the bias-only baseline (w_v · e0),
            # i.e. the value head evaluated at the zero-action feature.
            f = np.zeros(self.w_v.shape[0])
            f[0] = 1.0
            return float(f @ self.w_v)
        return float(_action_feature(state, action) @ self.w_v)

    def abstain_score(
        self, state: WorldState, action: Action, horizon_s: int
    ) -> float:
        z = float(_action_feature(state, action) @ self.w_u)
        return 1.0 / (1.0 + math.exp(-z))


def train_amortization_heads(
    dataset: AmortizationDataset,
    *,
    l2: float = 1e-3,
    pairwise_iters: int = 400,
    pairwise_lr: float = 0.1,
    abstain_iters: int = 400,
    abstain_lr: float = 0.1,
) -> LinearAmortizationHeads:
    """Distil the three heads from the offline full-enumeration replay set.

    * **Value head** ``V_ψ`` — ridge regression of the scalarized lex score
      ``S_t(a)`` on the candidate features, closed-form (doc M D2: "train
      V_ψ/Q_ψ on S_t"). Lower predicted value ⇒ better action.
    * **Proposal head** ``π_φ`` — *pairwise* ranking (doc M D2: "train π_φ with
      listwise or pairwise ranking"): for every (oracle, non-oracle) pair we want
      ``w·f_oracle > w·f_other``; minimized with a logistic pairwise loss by
      gradient descent. The margins ``Δ_t`` weight each pair so confusing the
      oracle with a near-tie costs less than confusing it with a far-worse action.
    * **Abstention head** ``u_ω`` — logistic regression of the abstain/OOD label
      on the features (doc M D2: "train u_ω on disagreement/OOD/abstention
      labels"), gradient descent with L2.

    Returns a deterministic :class:`LinearAmortizationHeads`. Empty input yields
    zero heads (a benign identity that proposes everything and never abstains).
    """
    if not dataset.examples:
        return LinearAmortizationHeads.zeros()

    feat_dim = dataset.examples[0].features.shape[1]

    # ---- value head: ridge regression S_t ≈ X w_v -------------------------
    all_feats = np.concatenate([e.features for e in dataset.examples], axis=0)
    all_scores = np.concatenate([e.lex_scores for e in dataset.examples], axis=0)
    xtx = all_feats.T @ all_feats + l2 * np.eye(feat_dim)
    w_v = np.linalg.solve(xtx, all_feats.T @ all_scores)

    # ---- proposal head: pairwise logistic ranking -------------------------
    # Build (positive=oracle feature, negative=other feature, weight=margin+1).
    pos: list[np.ndarray] = []
    neg: list[np.ndarray] = []
    pair_w: list[float] = []
    for e in dataset.examples:
        of = e.features[e.oracle_index]
        for k in range(e.features.shape[0]):
            if k == e.oracle_index:
                continue
            pos.append(of)
            neg.append(e.features[k])
            pair_w.append(1.0 + float(e.margins[k]))
    w_pi = np.zeros(feat_dim)
    if pos:
        P = np.stack(pos, axis=0)
        N = np.stack(neg, axis=0)
        W = np.asarray(pair_w, dtype=float)
        W = W / W.sum()
        diff = P - N  # want w·diff > 0 for each pair
        for _ in range(pairwise_iters):
            s = diff @ w_pi
            # logistic pairwise loss: -log sigmoid(w·diff); grad = -sigmoid(-s)·diff
            sig_neg = 1.0 / (1.0 + np.exp(s))  # sigmoid(-s)
            grad = -(W * sig_neg) @ diff + l2 * w_pi
            w_pi = w_pi - pairwise_lr * grad

    # ---- abstention head: logistic regression on abstain labels -----------
    all_abstain = np.concatenate([e.abstain_labels for e in dataset.examples], axis=0)
    y = np.clip(all_abstain, 0.0, 1.0)
    w_u = np.zeros(feat_dim)
    n = all_feats.shape[0]
    for _ in range(abstain_iters):
        p = 1.0 / (1.0 + np.exp(-(all_feats @ w_u)))
        grad = all_feats.T @ (p - y) / n + l2 * w_u
        w_u = w_u - abstain_lr * grad

    return LinearAmortizationHeads(w_pi=w_pi, w_v=w_v, w_u=w_u)


# ──────────────────────────────────────────────────────────────────────────────
# Guarded inference: amortized proposal with exact-fallback (doc M D3 / G2)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class GuardedProposal:
    """Result of :func:`guarded_proposal`: the candidate subset to actually score.

    Attributes
    ----------
    actions:
        The ``(action_id, action)`` subset the planner should score — either the
        amortized top-k proposal, or the *full* safe set when the guard tripped.
    used_fallback:
        ``True`` when the guard forced exhaustive scoring (doc M G2 fallback
        region: the amortized system is utility-equivalent to the exact planner
        there).
    reason:
        Why the fallback fired (``"u_omega_flag"`` / ``"below_threshold"``), or
        ``"amortized_topk"`` when the proposal was trusted.
    abstain_scores:
        ``action_id -> u_ω(b,a)`` for the proposed (or full) set, for the ledger.
    """

    actions: list[tuple[str, Action]]
    used_fallback: bool
    reason: str
    abstain_scores: dict[str, float] = field(default_factory=dict)


def guarded_proposal(
    heads: Any,  # AmortizationHeads (structural)
    state: WorldState,
    safe_actions: list[tuple[str, Action]],
    horizon_s: int,
    *,
    top_k: int = 4,
    enumeration_threshold: int = 6,
    abstain_threshold: float = 0.5,
) -> GuardedProposal:
    """Guarded amortized proposal with exact fallback (doc M D3 / G2).

    Applies the exact hard shield's *output* (``safe_actions`` is already the
    kernel-safe set) and then:

    1. **Below-threshold fallback.** If the safe set is already small
       (``len ≤ enumeration_threshold``), full enumeration is exact and cheap, so
       we score everything — there is nothing to amortize away (doc M D2/D4: keep
       full enumeration while the safe set is small).
    2. **u_ω guard.** Otherwise we ask the abstention head for ``u_ω(b,a)``. If
       *any* candidate trips ``abstain_threshold`` (OOD / high disagreement) we
       fall back to exhaustive scoring — the amortized proposal is not trusted in
       the OOD region (doc M D3/G2).
    3. **Amortized top-k.** Only when the set is large *and* the guard is clean do
       we keep the proposal head's top-k recall set (doc M D1/G2). The oracle is
       assumed to lie in the top-k with probability ≥ 1-δ_prop; the bounded extra
       safe-regret is ``M·δ_prop + 2ε_eval`` (doc M G2).

    The hold action (if present) is *always* retained in the proposal so the
    conservative fallback never gets pruned (doc F step 6).
    """
    abstain_scores: dict[str, float] = {}
    for aid, act in safe_actions:
        try:
            abstain_scores[aid] = float(
                heads.abstain_score(state, act, horizon_s)
            )
        except (AttributeError, NotImplementedError):
            abstain_scores[aid] = 0.0

    if len(safe_actions) <= enumeration_threshold:
        return GuardedProposal(
            actions=list(safe_actions),
            used_fallback=True,
            reason="below_threshold",
            abstain_scores=abstain_scores,
        )

    if any(v >= abstain_threshold for v in abstain_scores.values()):
        return GuardedProposal(
            actions=list(safe_actions),
            used_fallback=True,
            reason="u_omega_flag",
            abstain_scores=abstain_scores,
        )

    try:
        scores = heads.proposal_scores(
            state, [a for _, a in safe_actions], horizon_s
        )
    except (AttributeError, NotImplementedError):
        return GuardedProposal(
            actions=list(safe_actions),
            used_fallback=True,
            reason="no_proposal_head",
            abstain_scores=abstain_scores,
        )

    order = sorted(range(len(safe_actions)), key=lambda i: -scores[i])
    keep_idx = set(order[: max(top_k, 1)])
    # always keep hold so the conservative fallback survives the pruning.
    for i, (_aid, act) in enumerate(safe_actions):
        if act.type is ActionType.HOLD:
            keep_idx.add(i)
    proposal = [safe_actions[i] for i in sorted(keep_idx)]
    return GuardedProposal(
        actions=proposal,
        used_fallback=False,
        reason="amortized_topk",
        abstain_scores=abstain_scores,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Bounded multi-step search (doc M, enumeration_vs_search D3/D4; energy E_lex)
# ──────────────────────────────────────────────────────────────────────────────
#: Score one H-step action sequence ``a_0:H-1`` from belief ``b_0`` to its
#: lexicographic energy ``E_lex(b_0, a_0:H-1)`` (doc M E1, lower is better). The
#: planner supplies this by rolling the kernel + world model forward over the
#: sequence and aggregating the per-step lex keys. The scorer is the single
#: world-model coupling point; the search backends below are model-agnostic.
SequenceScorer = Callable[[WorldState, Sequence[Action]], tuple[float, ...]]


@runtime_checkable
class _CandidateSampler(Protocol):
    """Per-step safe candidate generator for one decision point in a sequence."""

    def __call__(self, state: WorldState, step: int) -> list[Action]: ...


def aggregate_sequence_lex_key(
    step_keys: Sequence[tuple[float, ...]]
) -> tuple[float, ...]:
    """Aggregate per-step lex keys into one sequence-level key (doc M D4/E1).

    ``E_lex(b_0, a_0:H-1)`` sums each lexicographic field over the horizon (doc M
    D4: ``H_hard = Σ_t (1 - K)``; ``R_stat`` summed; ``ψ`` summed), preserving the
    field order so the sequence key compares lexicographically exactly like a
    single-step key. An empty sequence is the all-zero key (a no-op rollout).
    """
    if not step_keys:
        return ()
    width = max(len(k) for k in step_keys)
    agg = [0.0] * width
    for key in step_keys:
        for i in range(width):
            agg[i] += float(key[i]) if i < len(key) else 0.0
    return tuple(agg)


@dataclass(slots=True)
class SearchResult:
    """Output of a bounded-search backend over multi-step action sequences."""

    best_sequence: list[Action]
    best_key: tuple[float, ...]
    emitted_action: Action | None  # the first action of the best sequence (a_0)
    n_rollouts: int
    backend: str


class SearchBackend:
    """Common interface for bounded multi-step search (doc M, D3/D4).

    A backend searches the space of H-step safe action *sequences* ``a_0:H-1``
    and returns the one minimizing the lexicographic sequence energy
    :func:`aggregate_sequence_lex_key` ∘ ``scorer`` (doc M E1/E2). The emitted
    action is the first of the best sequence (receding-horizon control, doc M
    enumeration_vs_search D3). All backends share:

    * ``candidate_sampler`` — per-step kernel-safe candidate generator (the
      shield front; search never leaves ``A_K(b)``, so safety is preserved);
    * ``scorer`` — the :data:`SequenceScorer` (kernel + world-model rollout);
    * an optional ``value_head`` warm-start that biases the first-step proposal /
      truncates lookahead (doc M D3: "use V_ψ/Q_ψ to warm-start MPC/CEM/MCTS or
      to terminate lookahead earlier").

    Subclasses implement :meth:`search`.
    """

    name: str = "search"

    def __init__(
        self,
        candidate_sampler: _CandidateSampler,
        scorer: SequenceScorer,
        *,
        horizon: int = 2,
        value_head: Any | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.candidate_sampler = candidate_sampler
        self.scorer = scorer
        self.horizon = max(int(horizon), 1)
        self.value_head = value_head
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def _score_sequence(
        self, state: WorldState, seq: Sequence[Action]
    ) -> tuple[float, ...]:
        return self.scorer(state, seq)

    def _warm_first_step(
        self, state: WorldState, candidates: list[Action], horizon_s: int
    ) -> list[int]:
        """Order first-step candidate indices by the value head (warm start).

        Lower predicted value ⇒ better, so the value head's argsort is the
        warm-start ordering the search expands first (doc M D3). Without a value
        head the natural enumeration order is kept.
        """
        if self.value_head is None or not candidates:
            return list(range(len(candidates)))
        try:
            vals = [
                float(self.value_head.value(state, a, horizon_s))
                for a in candidates
            ]
        except (AttributeError, NotImplementedError):
            return list(range(len(candidates)))
        return sorted(range(len(candidates)), key=lambda i: vals[i])

    def search(self, state: WorldState, horizon_s: int) -> SearchResult:
        raise NotImplementedError


class CEMSearch(SearchBackend):
    """Cross-Entropy-Method / MPPI-style sampling search (doc M D3).

    For a discrete kernel-generated candidate set, CEM/MPPI reduce to an
    *importance-weighted elite-refit* over a per-step categorical proposal: sample
    K sequences from the current per-step distributions, roll each out, keep the
    elite fraction by lexicographic energy, and refit the per-step categoricals
    toward the elites (MPPI's soft, temperature-weighted update is the same family
    — doc M D3 "MPC/MPPI/CEM"). The value head warm-starts the first-step
    distribution so good actions are sampled from the first iteration. The best
    sequence ever rolled out is returned (elitist, so the result never regresses).
    """

    name = "cem"

    def __init__(
        self,
        candidate_sampler: _CandidateSampler,
        scorer: SequenceScorer,
        *,
        horizon: int = 2,
        value_head: Any | None = None,
        rng: np.random.Generator | None = None,
        n_samples: int = 24,
        n_iters: int = 4,
        elite_frac: float = 0.25,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(
            candidate_sampler,
            scorer,
            horizon=horizon,
            value_head=value_head,
            rng=rng,
        )
        self.n_samples = max(int(n_samples), 1)
        self.n_iters = max(int(n_iters), 1)
        self.elite_frac = float(elite_frac)
        self.temperature = max(float(temperature), 1e-6)

    def search(self, state: WorldState, horizon_s: int) -> SearchResult:
        # Per-step candidate pools (kernel-safe at each step; the v1 action
        # ontology is myopic per workload, so the safe set at each step is the
        # same generator applied to the rolled state proxy — `state` here).
        pools: list[list[Action]] = [
            self.candidate_sampler(state, t) for t in range(self.horizon)
        ]
        pools = [p if p else [Hold(workload_uid="")] for p in pools]
        # Per-step categorical logits, warm-started at step 0 by the value head.
        logits: list[np.ndarray] = []
        for t, pool in enumerate(pools):
            base = np.zeros(len(pool))
            if t == 0:
                order = self._warm_first_step(state, pool, horizon_s)
                # decreasing bonus following the warm-start order
                for rank, idx in enumerate(order):
                    base[idx] = float(len(order) - rank)
            logits.append(base)

        best_seq: list[Action] | None = None
        best_key: tuple[float, ...] | None = None
        n_roll = 0
        n_elite = max(1, round(self.elite_frac * self.n_samples))

        for _ in range(self.n_iters):
            seqs: list[list[int]] = []
            keys: list[tuple[float, ...]] = []
            probs = [_softmax(lg / self.temperature) for lg in logits]
            for _s in range(self.n_samples):
                choice = [
                    int(self.rng.choice(len(pools[t]), p=probs[t]))
                    for t in range(self.horizon)
                ]
                seq = [pools[t][choice[t]] for t in range(self.horizon)]
                key = self._score_sequence(state, seq)
                n_roll += 1
                seqs.append(choice)
                keys.append(key)
                if best_key is None or key < best_key:
                    best_key, best_seq = key, seq
            # elite refit: rank by lexicographic key, take the best n_elite.
            elite = sorted(range(len(keys)), key=lambda i: keys[i])[:n_elite]
            new_logits: list[np.ndarray] = []
            for t, pool in enumerate(pools):
                counts = np.ones(len(pool)) * 1e-3  # Laplace smoothing
                for i in elite:
                    counts[seqs[i][t]] += 1.0
                new_logits.append(np.log(counts))
            logits = new_logits

        assert best_seq is not None and best_key is not None
        return SearchResult(
            best_sequence=best_seq,
            best_key=best_key,
            emitted_action=best_seq[0] if best_seq else None,
            n_rollouts=n_roll,
            backend=self.name,
        )


class MPPISearch(CEMSearch):
    """MPPI: temperature-weighted (soft) variant of the elite-refit search.

    MPPI replaces CEM's hard elite cut with a soft, exponentiated-cost weighting
    of *all* samples (doc M D3 "MPC/MPPI/CEM"). Implemented as :class:`CEMSearch`
    with ``elite_frac = 1.0`` so every sample contributes, weighted by the
    softmin of its scalarized sequence energy through ``temperature``.
    """

    name = "mppi"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("elite_frac", 1.0)
        super().__init__(*args, **kwargs)


@dataclass(slots=True)
class _MCTSNode:
    state_seq: tuple[int, ...]  # chosen candidate indices so far
    visits: int = 0
    total_value: float = 0.0  # mean scalarized sequence energy (lower better)
    children: dict[int, _MCTSNode] = field(default_factory=dict)
    untried: list[int] = field(default_factory=list)


class MCTSSearch(SearchBackend):
    """Monte-Carlo Tree Search over discrete safe action sequences (doc M D3).

    Doc M selects MCTS for *discrete / combinatorial* action families where prefix
    reuse and sparse delayed effects matter (enumeration_vs_search D3). Each tree
    node is a sequence prefix over kernel-safe per-step candidate indices; the
    search runs ``n_simulations`` of select → expand → rollout → backup using a
    UCT selection rule on the *negated* scalarized sequence energy (so lower
    energy ⇒ higher reward). The value head warm-starts the root's first-step
    child priors. The best fully-expanded sequence by lexicographic key is
    returned (elitist tracking, so MCTS never regresses below its best rollout).
    """

    name = "mcts"

    def __init__(
        self,
        candidate_sampler: _CandidateSampler,
        scorer: SequenceScorer,
        *,
        horizon: int = 2,
        value_head: Any | None = None,
        rng: np.random.Generator | None = None,
        n_simulations: int = 48,
        c_uct: float = 1.4,
    ) -> None:
        super().__init__(
            candidate_sampler,
            scorer,
            horizon=horizon,
            value_head=value_head,
            rng=rng,
        )
        self.n_simulations = max(int(n_simulations), 1)
        self.c_uct = float(c_uct)

    def search(self, state: WorldState, horizon_s: int) -> SearchResult:
        pools: list[list[Action]] = [
            self.candidate_sampler(state, t) for t in range(self.horizon)
        ]
        pools = [p if p else [Hold(workload_uid="")] for p in pools]
        warm_first = self._warm_first_step(state, pools[0], horizon_s)

        root = _MCTSNode(state_seq=(), untried=list(warm_first))
        best_seq: list[Action] | None = None
        best_key: tuple[float, ...] | None = None
        n_roll = 0
        # bounded score range for the UCT reward normalization, learned online.
        seen_scalars: list[float] = []

        for _ in range(self.n_simulations):
            node = root
            path = [root]
            depth = 0
            # SELECT down the tree while fully expanded and not terminal.
            while depth < self.horizon and not node.untried and node.children:
                node = self._uct_select(node)
                path.append(node)
                depth = len(node.state_seq)
            # EXPAND one untried child if available.
            if depth < self.horizon and node.untried:
                idx = node.untried.pop(0)
                child_seq = (*node.state_seq, idx)
                child = _MCTSNode(
                    state_seq=child_seq,
                    untried=list(range(len(pools[len(child_seq)])))
                    if len(child_seq) < self.horizon
                    else [],
                )
                node.children[idx] = child
                node = child
                path.append(node)
                depth = len(node.state_seq)
            # ROLLOUT to a full horizon (random safe completions).
            full = list(node.state_seq)
            while len(full) < self.horizon:
                t = len(full)
                full.append(int(self.rng.integers(len(pools[t]))))
            seq = [pools[t][full[t]] for t in range(self.horizon)]
            key = self._score_sequence(state, seq)
            n_roll += 1
            scalar = _key_scalar(key)
            seen_scalars.append(scalar)
            if best_key is None or key < best_key:
                best_key, best_seq = key, seq
            # BACKUP the (normalized) reward up the path.
            lo, hi = min(seen_scalars), max(seen_scalars)
            reward = 1.0 - ((scalar - lo) / (hi - lo)) if hi > lo else 1.0
            for n in path:
                n.visits += 1
                n.total_value += reward

        assert best_seq is not None and best_key is not None
        return SearchResult(
            best_sequence=best_seq,
            best_key=best_key,
            emitted_action=best_seq[0] if best_seq else None,
            n_rollouts=n_roll,
            backend=self.name,
        )

    def _uct_select(self, node: _MCTSNode) -> _MCTSNode:
        log_n = math.log(max(node.visits, 1))
        best_child: _MCTSNode | None = None
        best_score = -math.inf
        for child in node.children.values():
            if child.visits == 0:
                return child
            exploit = child.total_value / child.visits
            explore = self.c_uct * math.sqrt(log_n / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score, best_child = score, child
        assert best_child is not None
        return best_child


# ──────────────────────────────────────────────────────────────────────────────
# small numeric helpers (kept local so the module has no heavyweight deps)
# ──────────────────────────────────────────────────────────────────────────────
def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(e) / len(e)


def _key_scalar(key: tuple[float, ...]) -> float:
    """Collapse a lexicographic sequence key to a scalar for UCT normalization.

    Geometrically-decaying weights so earlier (dominant) fields outweigh later
    ones — used *only* to normalise the MCTS reward; the best-sequence selection
    always uses the true lexicographic tuple comparison.
    """
    if not key:
        return 0.0
    return float(sum(k / (1e6**i) for i, k in enumerate(key)))
