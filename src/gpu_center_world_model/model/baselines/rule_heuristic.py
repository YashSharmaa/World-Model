# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Exact-rule heuristic baseline (doc F baseline #1).

Fair-share queue ordering plus exact feasibility masks, topology-aware best-fit /
same-domain preference, and a simple headroom filter. This baseline uses the
*exact kernel* for hard safety (so it can never recommend an infeasible action)
and a deterministic, hand-written headroom rule for ranking — it carries **no
learned dynamics**. It is the conservative fallback the abstention gate defers to
(doc J R6 / doc L selective-prediction).

Because the exact kernel lives in the simulator module (outside this package),
this baseline accepts any object satisfying the :class:`ExactKernel` Protocol;
if none is supplied it degrades to a self-contained feasibility check over the
symbolic state (free-GPU + protected-domain rules) so it still runs in CI.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...schema.actions import Action, ActionType, action_workload_uid
from ...schema.entities import WorkloadStatus
from ...schema.kpis import (
    HORIZONS_S,
    HazardChannel,
    HazardForecast,
    KPIChannel,
    KPIForecastBundle,
    QuantileForecast,
)
from ...schema.state import WorldState


def _self_feasible(state: WorldState, action: Action) -> bool:
    """Self-contained feasibility check (used when no exact kernel is provided).

    Conservative: a non-hold actor action is feasible only if the targeted
    workload is pending/reserving, the action is in the operator whitelist, and
    (for placement) some non-protected domain has enough free GPUs.
    """
    pol = state.symbolic.operator_policy
    if action.type.value not in pol.allowed_actions and action.type != ActionType.HOLD:
        return False
    if action.type == ActionType.HOLD:
        return True
    wl = action_workload_uid(action)
    w = state.symbolic.workloads.get(wl) if wl else None
    if w is None or w.status not in (WorkloadStatus.PENDING, WorkloadStatus.RESERVING):
        return False
    demand = sum(ps.count * ps.gpu_req for ps in w.podsets)
    protected = set(pol.protected_domains)
    for d in state.symbolic.domains.values():
        if d.domain_id in protected:
            continue
        if d.free_gpu >= demand:
            return True
    return demand == 0


@dataclass(slots=True)
class RuleHeuristicBaseline:
    """Exact-kernel + headroom-filter ranker (no learned dynamics).

    Parameters
    ----------
    kernel:
        Optional object implementing ``feasible(state, action)`` (the
        :class:`ExactKernel` Protocol). When ``None`` the self-contained check is
        used so the baseline runs standalone in CI.
    headroom_floor:
        Minimum domain headroom score required to *prefer* a placement; below
        this the action is feasible but ranked worse (the "simple headroom
        filter" of doc F).
    """

    kernel: object | None = None
    headroom_floor: float = 0.2

    def _feasible(self, state: WorldState, action: Action) -> bool:
        if self.kernel is not None and hasattr(self.kernel, "feasible"):
            return bool(self.kernel.feasible(state, action).feasible)
        return _self_feasible(state, action)

    def _headroom_score(self, state: WorldState, action: Action) -> float:
        """Best non-protected domain headroom for the action's workload demand."""
        wl = action_workload_uid(action)
        w = state.symbolic.workloads.get(wl) if wl else None
        if w is None:
            return 1.0
        demand = sum(ps.count * ps.gpu_req for ps in w.podsets)
        protected = set(state.symbolic.operator_policy.protected_domains)
        best = 0.0
        for d in state.symbolic.domains.values():
            if d.domain_id in protected or d.free_gpu < demand:
                continue
            best = max(best, d.headroom_score * (1.0 - d.hotspot_score))
        return best

    def score_actions(
        self, state: WorldState, actions: list[Action], horizon_s: int = 1800
    ) -> list[KPIForecastBundle]:
        """Return a forecast-shaped bundle per action (deterministic, no UQ).

        Continuous KPI deltas are filled with a degenerate (point) quantile band
        derived from the headroom rule; hazards are a fixed, conservative base
        rate scaled by the headroom deficit so unsafe placements rank worse.
        """
        out: list[KPIForecastBundle] = []
        for a in actions:
            feasible = self._feasible(state, a)
            head = self._headroom_score(state, a) if feasible else 0.0
            deficit = max(0.0, self.headroom_floor - head)
            wl = action_workload_uid(a) or "*"
            quantiles = self._point_quantiles(a, head)
            hazards = self._hazards(deficit)
            out.append(
                KPIForecastBundle(
                    action_id=f"{wl}::{a.type.value}",
                    quantiles=quantiles,
                    hazards=hazards,
                    epistemic_score=0.0,
                    ood_score=0.0,
                    abstain=not feasible,
                    abstain_reason=None if feasible else "infeasible_rule",
                )
            )
        return out

    def _point_quantiles(self, action: Action, head: float) -> list[QuantileForecast]:
        # A held action has zero KPI delta by definition (it is the baseline).
        base = 0.0 if action.type == ActionType.HOLD else -(1.0 - head)
        out: list[QuantileForecast] = []
        for ch in (
            KPIChannel.QUEUE_WAIT_DELTA_S,
            KPIChannel.ADMITTED_GPU_HOURS_DELTA,
            KPIChannel.RACK_THERMAL_HEADROOM_C,
        ):
            for h in HORIZONS_S:
                v = base if ch != KPIChannel.RACK_THERMAL_HEADROOM_C else head * 5.0
                out.append(
                    QuantileForecast(
                        channel=ch,
                        horizon_s=h,
                        quantiles={0.05: v, 0.5: v, 0.95: v},
                    )
                )
        return out

    def _hazards(self, deficit: float) -> list[HazardForecast]:
        out: list[HazardForecast] = []
        for ch in (HazardChannel.HOTSPOT, HazardChannel.HEADROOM_VIOLATION):
            for h in HORIZONS_S:
                p = min(1.0, 0.02 + deficit)
                out.append(
                    HazardForecast(
                        channel=ch, horizon_s=h, probability=p, conformal_upper=p
                    )
                )
        return out


__all__ = ["RuleHeuristicBaseline"]
