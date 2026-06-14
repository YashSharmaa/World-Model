# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Direct policy ranker baseline (doc F baseline #5).

A **contextual bandit / logged-action ranker trained from logs without a learned
world model** (doc F: "Contextual bandit or logged-action ranker without a
learned world model, evaluated with off-policy estimators only where support
exists"). This is the foil that asks the brutal question the brief keeps
returning to: *do we even need a world model?* — a policy that simply learns to
rank the logged actions by their realized reward, with no dynamics, no rollout,
and no calibrated uncertainty.

Design (the honest contextual-bandit recipe):

* **Reward model (direct method, ``Q̂``).** A ridge-regularised linear value head
  ``Q̂(x, a) = w_a · φ(x)`` is fit per action type on the logged
  ``(context, action, reward)`` rows. This is the *direct method* baseline value
  the DR estimator corrects (doc F off-policy estimators).
* **Softmax target policy ``π``.** The ranker turns the fitted action-values into
  a tempered softmax over the *feasible* action types, ``π(a|x) ∝ exp(Q̂/T)`` —
  a contextual bandit policy with an explicit temperature so it has support on
  every feasible action (positivity, needed for OPE).
* **Doubly-robust OPE sanity check.** Where logging support exists (the logged
  behaviour propensities ``μ(a|x)`` overlap the target ``π``), the ranker's value
  is sanity-checked with the **sequential doubly-robust estimator the planner
  lane already shipped** (``planner.doubly_robust_ope``, doc M E1/A1/E2) rather
  than a second copy of the math. The coverage guard there refuses to estimate
  out-of-support actions, exactly as doc F demands ("only where support exists").

It exposes the same ``score_actions(state, actions, horizon_s)`` surface as the
other baselines so the runner can build baseline ``PlanningWindow``s from it. The
forecast bundle it emits is *reward-shaped*: a single learned utility per action
mapped into the planner's ``admitted_gpu_hours`` / ``queue_wait`` channels with a
degenerate band (the ranker carries no calibrated uncertainty — that absence is
the whole point of the foil).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...schema.actions import Action, ActionType, action_workload_uid
from ...schema.kpis import (
    HORIZONS_S,
    HazardChannel,
    HazardForecast,
    KPIChannel,
    KPIForecastBundle,
    QuantileForecast,
)
from ...schema.state import WorldState
from .features import flatten_state

# The ranker scores its learned utility into these planner channels so the
# downstream lexicographic key (admitted GPU-hours up, queue-wait down) reflects
# its ranking. It models no other channel — a deliberately myopic foil.
_UTILITY_CHANNELS: tuple[KPIChannel, ...] = (
    KPIChannel.ADMITTED_GPU_HOURS_DELTA,
    KPIChannel.QUEUE_WAIT_DELTA_S,
)


@dataclass(slots=True)
class LoggedDecision:
    """One logged decision row for the contextual bandit (doc F off-policy logs).

    Attributes
    ----------
    state:
        The pre-action context ``x`` the logger observed.
    action:
        The action ``a`` the logging policy actually took.
    reward:
        The realized scalar reward ``r`` (higher is better) — e.g. admitted
        GPU-hours minus a queue-wait penalty from the reference branch.
    behavior_propensity:
        The logging policy's probability ``μ(a|x)`` of taking ``a``. Needed for
        the importance ratio in OPE; when unknown a uniform-over-feasible prior
        is assumed at fit time (the standard logged-bandit fallback).
    feasible_action_types:
        The feasible action set the logger chose from (for the uniform prior and
        for the target policy's support). Defaults to all action types.
    """

    state: WorldState
    action: Action
    reward: float
    behavior_propensity: float | None = None
    feasible_action_types: tuple[ActionType, ...] = ()


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge regression ``w = (XᵀX + λI)⁻¹ Xᵀy`` with a bias column."""
    n, d = X.shape
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
    A = Xb.T @ Xb + lam * np.eye(d + 1)
    w = np.linalg.solve(A, Xb.T @ y)
    return w


@dataclass(slots=True)
class DirectPolicyRankerBaseline:
    """Contextual-bandit / logged-action ranker with doubly-robust OPE (doc F #5).

    Fit with :meth:`fit` over :class:`LoggedDecision` rows; query with
    :meth:`score_actions`. The fitted per-action ridge value head is the direct
    method ``Q̂``; :meth:`target_propensity` exposes the softmax target policy
    ``π``; :meth:`doubly_robust_value` runs the planner lane's DR estimator over a
    logged path to sanity-check the ranker *only where logging support exists*.

    Parameters
    ----------
    temperature:
        Softmax temperature ``T`` for the target policy (smaller ⇒ greedier).
    ridge_lambda:
        Ridge regularisation strength for the per-action value heads.
    """

    temperature: float = 1.0
    ridge_lambda: float = 1.0
    #: action_type.value -> fitted ridge weight vector (incl. bias term)
    value_heads: dict[str, np.ndarray] = field(default_factory=dict)
    #: reward standardisation so the softmax temperature is scale-stable
    _reward_mean: float = 0.0
    _reward_scale: float = 1.0
    _fitted: bool = False

    # ------------------------------------------------------------------ #
    # Feature / value plumbing
    # ------------------------------------------------------------------ #
    def _featurize(self, state: WorldState) -> np.ndarray:
        """Context features ``φ(x)`` — the flat global state (no action signal).

        The action enters through the *per-action* value head (one ridge model
        keyed by action type), so the context vector itself is action-free; this
        is the contextual-bandit factorisation ``Q̂(x, a) = w_a · φ(x)``.
        """
        return flatten_state(state).astype(np.float64)

    def _value(self, state: WorldState, action: Action) -> float:
        """Predicted (standardised-space) reward ``Q̂(x, a)`` for one action."""
        w = self.value_heads.get(action.type.value)
        if w is None:
            return 0.0
        phi = self._featurize(state)
        xb = np.concatenate([phi, [1.0]])
        return float(xb @ w)

    def _value_raw(self, state: WorldState, action: Action) -> float:
        """Predicted reward in the original (un-standardised) reward units."""
        return self._value(state, action) * self._reward_scale + self._reward_mean

    # ------------------------------------------------------------------ #
    # Fit (the logged-bandit training step)
    # ------------------------------------------------------------------ #
    def fit(self, log: list[LoggedDecision]) -> DirectPolicyRankerBaseline:
        """Fit the per-action ridge value heads from logged decisions (doc F #5).

        Rewards are standardised (zero mean / unit scale) so the softmax
        temperature is comparable across packs, then one ridge value head is fit
        per observed action type on the rows that took that action. Action types
        never logged simply have no head (their value defaults to the pooled
        mean, i.e. the contextual-bandit prior).
        """
        if not log:
            self._fitted = True
            return self
        rewards = np.array([d.reward for d in log], dtype=np.float64)
        self._reward_mean = float(np.mean(rewards))
        scale = float(np.std(rewards))
        self._reward_scale = scale if scale > 1e-9 else 1.0
        # Bucket rows by action type and fit one ridge head per type.
        by_type: dict[str, list[int]] = {}
        for i, d in enumerate(log):
            by_type.setdefault(d.action.type.value, []).append(i)
        for atype, idxs in by_type.items():
            X = np.stack([self._featurize(log[i].state) for i in idxs])
            y = (rewards[idxs] - self._reward_mean) / self._reward_scale
            self.value_heads[atype] = _ridge_fit(X, y, self.ridge_lambda)
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Target policy π(a|x): tempered softmax over feasible action values
    # ------------------------------------------------------------------ #
    def target_propensity(
        self, state: WorldState, actions: list[Action]
    ) -> dict[str, float]:
        """The softmax target policy ``π(a|x)`` over the supplied (feasible) set.

        Returns ``action_id -> π``. The tempered softmax over the fitted
        action-values gives every feasible action positive mass (positivity),
        which is exactly the overlap the doubly-robust estimator needs.
        """
        if not actions:
            return {}
        vals = np.array([self._value(state, a) for a in actions], dtype=np.float64)
        t = self.temperature if self.temperature > 1e-9 else 1e-9
        z = (vals - vals.max()) / t
        ex = np.exp(z)
        denom = float(ex.sum()) or 1.0
        probs = ex / denom
        return {
            self._action_id(a): float(p)
            for a, p in zip(actions, probs, strict=False)
        }

    def best_action(self, state: WorldState, actions: list[Action]) -> Action | None:
        """Greedy ``argmax_a Q̂(x, a)`` over the feasible candidate set."""
        if not actions:
            return None
        return max(actions, key=lambda a: self._value(state, a))

    # ------------------------------------------------------------------ #
    # Doubly-robust OPE sanity check (reuses the planner lane's estimator)
    # ------------------------------------------------------------------ #
    def doubly_robust_value(
        self,
        path: list[LoggedDecision],
        *,
        gamma: float = 1.0,
        overlap_floor: float = 1e-3,
        weight_cap: float | None = None,
    ):
        """DR-OPE value of this ranker over a logged path (doc F "where support").

        Reuses :func:`planner.doubly_robust_ope` — the sequential doubly-robust
        estimator the planner lane shipped — rather than re-deriving the math. For
        each logged step we supply: the realized reward ``r_t``; the *target*
        propensity ``π(a_t|s_t)`` from this ranker's softmax policy; the logged
        *behaviour* propensity ``μ(a_t|s_t)`` (uniform-over-feasible when the log
        did not record one); the direct-method state value ``V̂(s_t) =
        max_a Q̂(s_t, a)`` over the feasible set; and the action value
        ``Q̂(s_t, a_t)``. The estimator's own coverage guard refuses to estimate
        out-of-support paths (``supported=False``, ``value=nan``), which is the
        doc-F "only where support exists" clause enforced for free.

        Returns the planner lane's :class:`DROPEResult`.
        """
        # Imported lazily so the baselines package stays import-clean on a build
        # that excludes the planner (the OPE check is opt-in, doc F).
        from ...planner import doubly_robust_ope

        rewards: list[float] = []
        target_props: list[float] = []
        behavior_props: list[float] = []
        v_hat: list[float] = []
        q_hat: list[float] = []
        for d in path:
            feasible = self._feasible_actions(d)
            aid = self._action_id(d.action)
            pi = self.target_propensity(d.state, feasible)
            # Direct-method state value: best feasible action value (greedy V̂).
            best_v = max(
                (self._value_raw(d.state, a) for a in feasible),
                default=self._value_raw(d.state, d.action),
            )
            rewards.append(float(d.reward))
            target_props.append(float(pi.get(aid, 0.0)))
            behavior_props.append(self._behavior_propensity(d, feasible))
            v_hat.append(float(best_v))
            q_hat.append(self._value_raw(d.state, d.action))
        # ``v_hat`` must be length H+1: append a terminal bootstrap of 0 (the
        # logged path ends at the decision, no continuation value, doc M E1).
        v_hat.append(0.0)
        return doubly_robust_ope(
            rewards,
            target_props,
            behavior_props,
            v_hat,
            q_hat,
            gamma=gamma,
            overlap_floor=overlap_floor,
            weight_cap=weight_cap,
        )

    def _feasible_actions(self, d: LoggedDecision) -> list[Action]:
        """Reconstruct the feasible candidate set for a logged decision.

        We only have the realized action and the feasible *types*; we represent
        each feasible type by the logged action's workload so the softmax has the
        right support cardinality. The realized action is always included.
        """
        wl = action_workload_uid(d.action)
        types = d.feasible_action_types or (d.action.type,)
        out: list[Action] = []
        seen: set[str] = set()
        for at in types:
            a = _proxy_action(at, wl)
            aid = self._action_id(a)
            if aid not in seen:
                out.append(a)
                seen.add(aid)
        if self._action_id(d.action) not in seen:
            out.append(d.action)
        return out

    def _behavior_propensity(self, d: LoggedDecision, feasible: list[Action]) -> float:
        """Logged ``μ(a|x)``; uniform-over-feasible prior when unrecorded."""
        if d.behavior_propensity is not None:
            return max(0.0, float(d.behavior_propensity))
        n = max(len(feasible), 1)
        return 1.0 / n

    # ------------------------------------------------------------------ #
    # score_actions surface (shared with every other baseline)
    # ------------------------------------------------------------------ #
    def _action_id(self, action: Action) -> str:
        return f"{action_workload_uid(action) or '*'}::{action.type.value}"

    def score_actions(
        self, state: WorldState, actions: list[Action], horizon_s: int = 1800
    ) -> list[KPIForecastBundle]:
        """Map each action's learned utility into a reward-shaped forecast bundle.

        The ranker has no dynamics and no calibrated uncertainty, so the band is
        degenerate (point mass at the learned utility). The single learned
        utility ``Q̂(x, a)`` drives the admitted-GPU-hours channel (more utility ⇒
        more admitted work) and, negated, the queue-wait channel (more utility ⇒
        less waiting), so the planner's lexicographic key ranks exactly on the
        ranker's preference order.
        """
        out: list[KPIForecastBundle] = []
        for a in actions:
            u = self._value_raw(state, a)
            quantiles: list[QuantileForecast] = []
            for ch in _UTILITY_CHANNELS:
                # admitted-GPU-hours is "more is better"; queue-wait is "less is
                # better" — so queue-wait carries the negated utility.
                v = u if ch is KPIChannel.ADMITTED_GPU_HOURS_DELTA else -u
                for h in HORIZONS_S:
                    quantiles.append(
                        QuantileForecast(
                            channel=ch,
                            horizon_s=h,
                            quantiles={0.05: v, 0.5: v, 0.95: v},
                        )
                    )
            # A myopic constant base-rate hazard (the ranker models no events).
            hazards = [
                HazardForecast(
                    channel=HazardChannel.REQUEUE,
                    horizon_s=h,
                    probability=0.05,
                    conformal_upper=0.05,
                )
                for h in HORIZONS_S
            ]
            out.append(
                KPIForecastBundle(
                    action_id=self._action_id(a),
                    quantiles=quantiles,
                    hazards=hazards,
                )
            )
        return out


def _proxy_action(atype: ActionType, workload_uid: str | None) -> Action:
    """Build a minimal action of ``atype`` for the same workload (support proxy)."""
    from ...schema.actions import Admit, AdmitAndPlace, Hold, MaskDomain, Place

    wl = workload_uid or ""
    if atype == ActionType.HOLD:
        return Hold(workload_uid=wl)
    if atype == ActionType.ADMIT:
        return Admit(
            workload_uid=wl,
            cluster_queue_id="cq0",
            resource_flavor_id="f0",
        )
    if atype == ActionType.PLACE:
        return Place(workload_uid=wl)
    if atype == ActionType.ADMIT_AND_PLACE:
        return AdmitAndPlace(
            workload_uid=wl,
            cluster_queue_id="cq0",
            resource_flavor_id="f0",
        )
    return MaskDomain(domain_id_or_set=("d0",))


__all__ = [
    "DirectPolicyRankerBaseline",
    "LoggedDecision",
]
