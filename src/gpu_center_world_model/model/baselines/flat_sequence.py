# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Flat single-timescale sequence baseline (doc F baseline #4, ablation #1).

A single GRU over concatenated flat state tensors with a matched parameter
budget but **no temporal hierarchy and no typed object relations** (doc F: "no
hierarchy and no explicit object relations"). This is the foil for the
"remove temporal hierarchy" ablation: if long-horizon counterfactual ranking
degrades relative to the relational world model, the fast/slow split earns its
keep.

It reuses the backend shim and the same :class:`architecture.GRUCell` so the
comparison is apples-to-apples on the numeric backend; the difference is purely
architectural — one flat recurrence over a global feature vector instead of
typed cores + fixed-incidence message passing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...schema.actions import Action, action_workload_uid
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
from ..architecture import Dense, GRUCell
from ..backend import Backend, get_backend
from .features import FLAT_STATE_DIM, action_onehot, flatten_state

_TARGET_CHANNELS: tuple[KPIChannel, ...] = (
    KPIChannel.QUEUE_WAIT_DELTA_S,
    KPIChannel.ADMITTED_GPU_HOURS_DELTA,
    KPIChannel.COMPLETION_TIME_DELTA_S,
)
_HAZARD_CHANNELS: tuple[HazardChannel, ...] = (
    HazardChannel.PODS_READY_TIMEOUT,
    HazardChannel.REQUEUE,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


@dataclass(slots=True)
class FlatSequenceBaseline:
    """Single-timescale GRU over flat features (no hierarchy, no relations)."""

    be: Backend
    hidden: int
    encoder: Dense
    core: GRUCell
    kpi_out: Dense  # -> (n_channels * n_horizons * n_quantiles)
    hz_out: Dense  # -> (n_hazards * n_horizons) logits

    @classmethod
    def build(
        cls, hidden: int = 48, seed: int = 7, backend: Backend | None = None
    ) -> FlatSequenceBaseline:
        be = backend or get_backend()
        rng = np.random.default_rng(seed)

        def dense(out_dim: int, in_dim: int) -> Dense:
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            W = rng.uniform(-limit, limit, (out_dim, in_dim)).astype(np.float32)
            b = np.zeros((out_dim,), dtype=np.float32)
            return Dense(be.from_numpy(W), be.from_numpy(b))

        in_dim = FLAT_STATE_DIM + 5  # flat state + 5 action one-hot slots
        encoder = dense(hidden, in_dim)
        n = hidden + hidden
        limit = np.sqrt(6.0 / (n + hidden))
        core = GRUCell(
            be.from_numpy(rng.uniform(-limit, limit, (hidden, n)).astype(np.float32)),
            be.from_numpy(rng.uniform(-limit, limit, (hidden, n)).astype(np.float32)),
            be.from_numpy(rng.uniform(-limit, limit, (hidden, n)).astype(np.float32)),
        )
        n_kpi = len(_TARGET_CHANNELS) * len(HORIZONS_S) * len(QUANTILES)
        n_hz = len(_HAZARD_CHANNELS) * len(HORIZONS_S)
        kpi_out = dense(n_kpi, hidden)
        hz_out = dense(n_hz, hidden)
        return cls(be, hidden, encoder, core, kpi_out, hz_out)

    def _encode(self, state: WorldState, action: Action):
        be, xp = self.be, self.be.xp
        x = np.concatenate([flatten_state(state), action_onehot(action)]).astype(
            np.float32
        )
        xv = be.from_numpy(x)
        h0 = xp.zeros((self.hidden,), dtype=be.float_dtype)
        enc = xp.tanh(self.encoder(xv, be))
        return self.core(h0, enc, be)

    def score_actions(
        self, state: WorldState, actions: list[Action], horizon_s: int = 1800
    ) -> list[KPIForecastBundle]:
        be = self.be
        out: list[KPIForecastBundle] = []
        nQ, nH = len(QUANTILES), len(HORIZONS_S)
        for a in actions:
            h = self._encode(state, a)
            kpi = np.asarray(be.to_numpy(self.kpi_out(h, be))).reshape(
                len(_TARGET_CHANNELS), nH, nQ
            )
            hz = _sigmoid(
                np.asarray(be.to_numpy(self.hz_out(h, be))).reshape(
                    len(_HAZARD_CHANNELS), nH
                )
            )
            quantiles: list[QuantileForecast] = []
            for ci, ch in enumerate(_TARGET_CHANNELS):
                for hi, hz_s in enumerate(HORIZONS_S):
                    row = np.sort(kpi[ci, hi])  # enforce monotone quantiles
                    quantiles.append(
                        QuantileForecast(
                            channel=ch,
                            horizon_s=hz_s,
                            quantiles={QUANTILES[j]: float(row[j]) for j in range(nQ)},
                        )
                    )
            hazards: list[HazardForecast] = []
            for ci, ch in enumerate(_HAZARD_CHANNELS):
                survive = 1.0
                for hi, hz_s in enumerate(HORIZONS_S):
                    survive *= 1.0 - float(hz[ci, hi])
                    p = min(max(1.0 - survive, 0.0), 1.0)
                    hazards.append(
                        HazardForecast(
                            channel=ch, horizon_s=hz_s, probability=p, conformal_upper=p
                        )
                    )
            out.append(
                KPIForecastBundle(
                    action_id=f"{action_workload_uid(a) or '*'}::{a.type.value}",
                    quantiles=quantiles,
                    hazards=hazards,
                )
            )
        return out


__all__ = ["FlatSequenceBaseline"]
