# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Exact-kernel + empirical-tables baseline (doc F baseline #2).

Keeps the exact rule kernel for hard safety, then attaches **empirical
quantile / median tables** for the genuinely uncertain quantities — PodsReady
latency, remaining runtime, headroom residuals, requeue risk — conditioned on a
coarse context key ``(workload class, load bucket)``. No relational learning: it
is a lookup of marginal-conditional empirical distributions.

This is a deliberately strong-but-dumb baseline (doc F: "a skeptical engineer
will not accept better-than-a-weak-neural-baseline as evidence"). It uses the
empirical CDF on a calibration table, so its intervals are honest within the
table's support but cannot react to relational structure (topology, gang
coupling) the way the residual world model can.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ...schema.actions import Action, action_workload_uid
from ...schema.entities import CommClass
from ...schema.kpis import (
    HORIZONS_S,
    QUANTILES,
    HazardChannel,
    HazardForecast,
    KPIChannel,
    KPIForecastBundle,
    QuantileForecast,
)
from ...schema.state import WorldState


def _load_bucket(state: WorldState) -> str:
    """Coarse load bucket from pod GPU occupancy (low / mid / high)."""
    gpus = state.symbolic.gpus
    if not gpus:
        return "low"
    busy = sum(1 for g in gpus.values() if g.allocated_to) / max(len(gpus), 1)
    return "low" if busy < 0.33 else ("mid" if busy < 0.66 else "high")


def _workload_class(state: WorldState, workload_uid: str | None) -> str:
    """Coarse workload class from its comm intensity (drives contention tails)."""
    if workload_uid is None:
        return "none"
    w = state.symbolic.workloads.get(workload_uid)
    if w is None:
        return "none"
    return "comm_heavy" if w.comm_class == CommClass.ALLREDUCE_HEAVY else "comm_light"


@dataclass(slots=True)
class EmpiricalTablesBaseline:
    """Empirical-quantile lookup tables keyed by ``(class, load, channel, horizon)``.

    Fit on a labelled calibration table with :meth:`fit`; query with
    :meth:`score_actions`. Unseen keys fall back to the pooled marginal
    distribution (graceful degrade), and missing channels emit a flat zero band.
    """

    # key -> list of observed KPI-delta values (per channel/horizon)
    kpi_samples: dict[tuple, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # key -> list of observed binary hazard outcomes
    hazard_samples: dict[tuple, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def observe_kpi(
        self,
        wl_class: str,
        load: str,
        channel: KPIChannel,
        horizon_s: int,
        value: float,
    ) -> None:
        self.kpi_samples[(wl_class, load, channel.value, horizon_s)].append(
            float(value)
        )

    def observe_hazard(
        self,
        wl_class: str,
        load: str,
        channel: HazardChannel,
        horizon_s: int,
        occurred: bool,
    ) -> None:
        self.hazard_samples[(wl_class, load, channel.value, horizon_s)].append(
            1.0 if occurred else 0.0
        )

    def _kpi_quantiles(self, key: tuple) -> dict[float, float]:
        samples = self.kpi_samples.get(key)
        if not samples:
            # pooled fall-back over the same channel/horizon across keys
            pooled = [
                v for k, vs in self.kpi_samples.items() if k[2:] == key[2:] for v in vs
            ]
            samples = pooled
        if not samples:
            return {q: 0.0 for q in QUANTILES}
        arr = np.asarray(samples)
        return {q: float(np.quantile(arr, q)) for q in QUANTILES}

    def _hazard_prob(self, key: tuple) -> float:
        samples = self.hazard_samples.get(key)
        if not samples:
            pooled = [
                v
                for k, vs in self.hazard_samples.items()
                if k[2:] == key[2:]
                for v in vs
            ]
            samples = pooled
        if not samples:
            return 0.05
        return float(np.mean(samples))

    def score_actions(
        self, state: WorldState, actions: list[Action], horizon_s: int = 1800
    ) -> list[KPIForecastBundle]:
        load = _load_bucket(state)
        out: list[KPIForecastBundle] = []
        for a in actions:
            wl = action_workload_uid(a)
            cls = _workload_class(state, wl)
            quantiles: list[QuantileForecast] = []
            for ch in (
                KPIChannel.QUEUE_WAIT_DELTA_S,
                KPIChannel.ADMITTED_GPU_HOURS_DELTA,
                KPIChannel.COMPLETION_TIME_DELTA_S,
                KPIChannel.RACK_THERMAL_HEADROOM_C,
            ):
                for h in HORIZONS_S:
                    qd = self._kpi_quantiles((cls, load, ch.value, h))
                    quantiles.append(
                        QuantileForecast(channel=ch, horizon_s=h, quantiles=qd)
                    )
            hazards: list[HazardForecast] = []
            for ch in (
                HazardChannel.PODS_READY_TIMEOUT,
                HazardChannel.REQUEUE,
                HazardChannel.HOTSPOT,
            ):
                for h in HORIZONS_S:
                    p = self._hazard_prob((cls, load, ch.value, h))
                    hazards.append(
                        HazardForecast(
                            channel=ch, horizon_s=h, probability=p, conformal_upper=p
                        )
                    )
            out.append(
                KPIForecastBundle(
                    action_id=f"{wl or '*'}::{a.type.value}",
                    quantiles=quantiles,
                    hazards=hazards,
                )
            )
        return out


__all__ = ["EmpiricalTablesBaseline"]
