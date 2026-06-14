# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Deterministic counterfactual branch replay (the :class:`BranchRunner`).

A *branch* is a single counterfactual rollout from a frozen pre-action state.
Paired branches must differ **only** by the clamped action — never by drifting
exogenous conditions — or action effects get contaminated by changed context,
the single biggest source of false optimism in learned world models (docs D, I,
R). Two mechanisms enforce this:

* :meth:`root` freezes the pre-action ``state_hash`` (``world_state_hash`` over
  ``X_t`` + ``C_t``) and a :class:`SeedBundle`. Every branch off the same root
  re-derives the *same* per-stream seeds, so all branches see common random
  numbers (doc seeding module / doc D/R/O).
* :meth:`rollout` applies the **exact symbolic step** ``X_{t+1} = K(X_t, a)``
  first (the safety shield), then draws stochastic exogenous events under those
  shared seeds, then — if a residual model is injected — asks it for the
  learned transition. Without a residual model the rollout is symbolic-only and
  fully deterministic.

The residual transition is *injected*, never imported, so the simulator does not
depend on MLX or the learned backend (doc E/H boundary rule). The injected
callable has the shape::

    residual(state: WorldState, action: Action, horizon_s: int, seed: int)
        -> WorldState

i.e. it returns the next state with the residual belief / stochastic events
applied; if it instead conforms to the model ``WorldModel.forecast`` Protocol we
adapt it (see :class:`ResidualTransition`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Protocol, runtime_checkable

from ..schema.actions import Action
from ..schema.hashing import world_state_hash
from ..schema.state import WorldState
from ..utils.seeding import SeedBundle
from .contracts import BranchRoot
from .exact_kernel.kernel import ExactSymbolicKernel
from .rare_events import apply_event

#: A residual transition: maps (state, action, horizon, seed) -> next state.
ResidualFn = Callable[[WorldState, Action, int, int], WorldState]


@runtime_checkable
class ResidualTransition(Protocol):
    """Optional injected stochastic/learned transition (kept backend-neutral)."""

    def __call__(
        self, state: WorldState, action: Action, horizon_s: int, seed: int
    ) -> WorldState: ...


class BranchRunner:
    """Deterministic branch replay from a fixed pre-action hash + seed bundle.

    Implements the :class:`simulator.contracts.BranchRunner` Protocol. The
    residual model and the rare-event schedule are optional; the bare runner does
    a correct symbolic-only rollout that is byte-stable across processes.
    """

    def __init__(
        self,
        kernel: ExactSymbolicKernel | None = None,
        residual: ResidualFn | None = None,
        *,
        step_s: int = 300,
        rare_event_schedule: Sequence[tuple[int, str]] | None = None,
        rare_event_kwargs: dict[str, dict[str, object]] | None = None,
    ) -> None:
        self.kernel = kernel or ExactSymbolicKernel()
        self.residual = residual
        #: Meso control step in seconds (doc K, A1: 1-5 min control rate).
        self.step_s = step_s
        #: Optional ``[(step_index, event_name), ...]`` schedule applied under CRN.
        self.rare_event_schedule = list(rare_event_schedule or ())
        self.rare_event_kwargs = rare_event_kwargs or {}

    # ── BranchRunner.root ──────────────────────────────────────────────────
    def root(self, state: WorldState, horizon_s: int, seed: int) -> BranchRoot:
        """Freeze the pre-action state hash + seed bundle for a counterfactual.

        The returned :class:`BranchRoot` is the shared anchor for a *family* of
        branches: every action's rollout off this root re-derives the identical
        per-stream seeds, guaranteeing common random numbers across the family.
        """
        return BranchRoot(
            state_hash=world_state_hash(state),
            seed_bundle=seed,
            horizon_s=horizon_s,
        )

    # ── BranchRunner.rollout ───────────────────────────────────────────────
    def rollout(
        self, root: BranchRoot, state: WorldState, action: Action
    ) -> list[WorldState]:
        """Roll a single action branch forward under common random numbers.

        Procedure per meso step (doc K transition + doc S shield):

        1. **Verify alignment.** The provided ``state`` must hash to
           ``root.state_hash`` at step 0 — otherwise the branch is contaminated
           by drift and we refuse silently-wrong deltas (raises ``ValueError``).
        2. **Exact symbolic step.** Apply the safety shield's deterministic
           transition ``X_{t+1} = K(X_t, a)``. Only the first step carries the
           clamped intervention ``action``; subsequent steps carry ``hold`` so the
           branch isolates the one-shot effect of the action (doc I do-semantics).
        3. **Exogenous events under CRN.** Apply any scheduled rare events using
           seeds derived from the *shared* :class:`SeedBundle`, so paired branches
           see identical exogenous draws.
        4. **Residual transition.** If a residual model is injected, advance the
           learned belief / stochastic dynamics with a CRN-derived seed.

        Returns the trajectory ``[X_0, X_1, ..., X_H]`` including the root state.
        """
        # 1. Alignment check (the whole point of the frozen hash).
        if world_state_hash(state) != root.state_hash:
            raise ValueError(
                "branch state does not match frozen root hash; counterfactual "
                "branches must start from the identical pre-action state"
            )

        bundle = SeedBundle(master=root.seed_bundle)
        n_steps = max(1, root.horizon_s // self.step_s)
        traj: list[WorldState] = [state]
        cur = state

        from ..schema.actions import Hold

        for step_i in range(n_steps):
            # 2. Exact symbolic transition (intervention only on the first step).
            step_action: Action = (
                action
                if step_i == 0
                else Hold(workload_uid=getattr(action, "workload_uid", "") or "")
            )
            cur = self.kernel.step(cur, step_action)
            cur = replace(cur, ts_ns=cur.ts_ns + self.step_s * 1_000_000_000)

            # 3. Exogenous rare events under common random numbers.
            cur = self._apply_scheduled_events(cur, bundle, step_i)

            # 4. Optional residual / learned stochastic transition.
            if self.residual is not None:
                resid_seed = bundle.stream("phase") ^ (step_i * 0x9E3779B1)
                cur = self.residual(cur, step_action, self.step_s, resid_seed)

            traj.append(cur)
        return traj

    # ── internals ──────────────────────────────────────────────────────────
    def _apply_scheduled_events(
        self, state: WorldState, bundle: SeedBundle, step_i: int
    ) -> WorldState:
        """Apply every rare event scheduled for ``step_i`` under shared seeds.

        The seed for an event is derived from the event's natural noise stream so
        a fault burst draws from ``health``, a queue burst from ``arrivals``, etc.
        — keeping each exogenous channel independent yet reproducible across
        branches (doc seeding STREAMS).
        """
        stream_of = {
            "gpu_fault_burst": "health",
            "pods_ready_timeout": "phase",
            "cooling_derate": "cooling",
            "power_derate": "cooling",
            "queue_burst": "arrivals",
            "maintenance_mask": "operator",
            "telemetry_degradation": "telemetry",
        }
        cur = state
        for sched_step, name in self.rare_event_schedule:
            if sched_step != step_i:
                continue
            base = bundle.stream(stream_of.get(name, "regime"))
            seed = base ^ (step_i * 0x85EBCA6B)
            kwargs = self.rare_event_kwargs.get(name, {})
            cur = apply_event(name, cur, seed, **kwargs)
        return cur


__all__ = ["BranchRunner", "ResidualFn", "ResidualTransition"]
