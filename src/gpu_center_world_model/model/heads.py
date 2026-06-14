# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Query-conditioned prediction heads producing a :class:`KPIForecastBundle`.

Three head families (docs L, J, F):

* **Multi-horizon quantile KPI-delta heads** (doc L head H1 / doc J H1). For each
  KPI channel and horizon they emit the :data:`schema.QUANTILES` grid trained
  with the pinball loss (doc J Eq. 11–12) and a non-crossing penalty (Eq. 13 /
  doc L Eq. 8). KPIs are *deltas vs. the baseline action* (typically ``hold``),
  never bare point estimates (doc kpis rationale).
* **Hazard heads** (doc L head H2 / doc J H2). Mutually-exclusive terminal events
  use a discrete-time *competing-risks* softmax over interval-hazards; co-
  occurring incidents use *parallel binary* hazards (doc kpis: HazardChannel
  semantics). Each hazard carries a one-sided conformal upper bound slot the
  calibrator later fills (doc L Eq. 13 / one-sided).
* **Abstention / OOD head** (doc L selective-prediction / doc J H3). Emits an
  epistemic-disagreement score and an OOD score; the abstention *gate* itself is
  applied later (in :mod:`world_model`) so the head can never override exact-
  kernel infeasibility (doc J R6).

All heads read the per-entity latents ``Z`` produced by
:class:`architecture.RelationalWorldModel` and are *query-conditioned*: a KPI
about a workload pulls that workload's latent; a rack-power KPI pulls a rack
latent; hazards about GPUs pull GPU latents (doc L H2: per-entity heads). The
maths is written against the backend shim so it runs identically on MLX/NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schema.kpis import (
    HORIZONS_S,
    QUANTILES,
    HazardChannel,
    HazardForecast,
    KPIChannel,
    KPIForecastBundle,
    QuantileForecast,
)
from .architecture import Dense, RelationalWorldModel
from .backend import ArrayLike, Backend, get_backend

# KPI channels and which entity-type latent each one queries (doc L H1 mapping).
KPI_QUERY_ENTITY: dict[KPIChannel, str] = {
    KPIChannel.QUEUE_WAIT_DELTA_S: "queue",
    KPIChannel.ADMITTED_GPU_HOURS_DELTA: "global",
    KPIChannel.TOPOLOGY_FRAGMENTATION_DELTA: "topology_domain",
    KPIChannel.RACK_POWER_KW: "rack",
    KPIChannel.RACK_THERMAL_HEADROOM_C: "rack",
    KPIChannel.COMPLETION_TIME_DELTA_S: "workload",
    KPIChannel.SLOWDOWN: "workload",
    KPIChannel.CROSS_RACK_RATE: "switch",
    # Fairness share-debt delta (doc V lambda_unfair*share_debt) is a
    # cluster-queue / cohort quantity → read it from the queue latent block.
    KPIChannel.FAIRNESS_SHARE_DEBT_DELTA: "queue",
}

# Hazard channels: which entity, and whether the event is a competing-risk
# terminal outcome or a parallel binary incident (doc kpis / doc L H2).
HAZARD_QUERY_ENTITY: dict[HazardChannel, str] = {
    HazardChannel.HOTSPOT: "rack",
    HazardChannel.PODS_READY_TIMEOUT: "workload",
    HazardChannel.REQUEUE: "workload",
    HazardChannel.XID_FAULT: "gpu",
    HazardChannel.HEADROOM_VIOLATION: "rack",
    HazardChannel.NODE_FAILURE: "node",
}

#: Mutually-exclusive terminal events modelled as competing risks (doc L H2).
COMPETING_RISK_CHANNELS: tuple[HazardChannel, ...] = (
    HazardChannel.PODS_READY_TIMEOUT,
    HazardChannel.REQUEUE,
)
#: Co-occurring incidents modelled as parallel binary hazards.
PARALLEL_BINARY_CHANNELS: tuple[HazardChannel, ...] = (
    HazardChannel.HOTSPOT,
    HazardChannel.XID_FAULT,
    HazardChannel.HEADROOM_VIOLATION,
    HazardChannel.NODE_FAILURE,
)

KPI_CHANNELS: tuple[KPIChannel, ...] = tuple(KPIChannel)
HAZARD_CHANNELS: tuple[HazardChannel, ...] = tuple(HazardChannel)


def _init_dense(rng: np.random.Generator, out_dim: int, in_dim: int) -> tuple:
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    W = rng.uniform(-limit, limit, size=(out_dim, in_dim)).astype(np.float32)
    b = np.zeros((out_dim,), dtype=np.float32)
    return W, b


@dataclass(slots=True)
class HeadConfig:
    """Static head configuration (horizons, quantile grid, hidden width)."""

    horizons_s: tuple[int, ...] = HORIZONS_S
    quantiles: tuple[float, ...] = QUANTILES
    hidden: int = 32
    seed: int = 1


@dataclass(slots=True)
class ForecastHeads:
    """The query-conditioned head bundle (quantile + hazard + abstention).

    The KPI head is parameterised as a *median + monotone non-negative offsets*
    so quantiles are non-crossing by construction (doc J Eq. 13 / doc L Eq. 8):
    ``q_τ = median + sign(τ-0.5) · cumulative_softplus(offset_τ)``. This is a
    stronger structural guarantee than a soft penalty alone.
    """

    cfg: HeadConfig
    be: Backend
    kpi_trunk: dict[KPIChannel, Dense]
    kpi_out: dict[KPIChannel, Dense]  # -> (n_horizons * n_quantiles)
    hz_competing: Dense  # softmax over {survive, timeout, requeue} per horizon
    hz_binary: dict[HazardChannel, Dense]  # -> n_horizons logits
    abstain_trunk: Dense
    abstain_out: Dense  # -> [epistemic_logit, ood_logit]

    @classmethod
    def build(
        cls,
        latent_dims: dict[str, int],
        cfg: HeadConfig | None = None,
        backend: Backend | None = None,
    ) -> ForecastHeads:
        cfg = cfg or HeadConfig()
        be = backend or get_backend()
        rng = np.random.default_rng(cfg.seed)
        H, Q = len(cfg.horizons_s), len(cfg.quantiles)

        def dense(out_dim: int, in_dim: int) -> Dense:
            W, b = _init_dense(rng, out_dim, in_dim)
            return Dense(be.from_numpy(W), be.from_numpy(b))

        kpi_trunk: dict[KPIChannel, Dense] = {}
        kpi_out: dict[KPIChannel, Dense] = {}
        for ch in KPI_CHANNELS:
            ent = KPI_QUERY_ENTITY[ch]
            d = latent_dims[ent]
            kpi_trunk[ch] = dense(cfg.hidden, d)
            kpi_out[ch] = dense(H * Q, cfg.hidden)

        # competing-risks head: per horizon, classes = {survive, timeout, requeue}
        n_cr = len(COMPETING_RISK_CHANNELS) + 1
        hz_competing = dense(H * n_cr, latent_dims["workload"])

        hz_binary: dict[HazardChannel, Dense] = {}
        for ch in PARALLEL_BINARY_CHANNELS:
            ent = HAZARD_QUERY_ENTITY[ch]
            hz_binary[ch] = dense(H, latent_dims[ent])

        abstain_trunk = dense(cfg.hidden, latent_dims["global"])
        abstain_out = dense(2, cfg.hidden)

        return cls(
            cfg=cfg,
            be=be,
            kpi_trunk=kpi_trunk,
            kpi_out=kpi_out,
            hz_competing=hz_competing,
            hz_binary=hz_binary,
            abstain_trunk=abstain_trunk,
            abstain_out=abstain_out,
        )

    # ------------------------------------------------------------------ #
    # Query pooling: reduce an entity-type latent block to one query vector.
    # ------------------------------------------------------------------ #
    def _query_vec(
        self,
        Z: dict[str, ArrayLike],
        masks: dict[str, ArrayLike],
        entity: str,
        row: int | None = None,
    ) -> ArrayLike:
        """Masked-mean pool an entity latent into one query vector (doc Q V4).

        If ``row`` is given, returns that specific entity row (e.g. the queried
        workload); otherwise returns the masked mean over valid rows — a
        permutation-invariant readout (doc Q Th. 4).
        """
        xp = self.be.xp
        h = Z[entity]
        if h.ndim == 1:
            return h
        m = masks.get(entity)
        if row is not None and 0 <= row < h.shape[0]:
            return h[row]
        if m is None:
            return xp.mean(h, axis=0)
        denom = xp.maximum(xp.sum(m), 1.0)
        return xp.sum(h * m[:, None], axis=0) / denom

    # ------------------------------------------------------------------ #
    # Quantile KPI-delta head (non-crossing by construction).
    # ------------------------------------------------------------------ #
    def _softplus(self, x: ArrayLike) -> ArrayLike:
        xp = self.be.xp
        return xp.log1p(xp.exp(-xp.abs(x))) + xp.maximum(x, 0.0)

    def quantile_forecasts(
        self,
        Z: dict[str, ArrayLike],
        masks: dict[str, ArrayLike],
        query_row: dict[str, int] | None = None,
    ) -> list[QuantileForecast]:
        """Emit non-crossing quantile forecasts for every KPI channel/horizon."""
        be, xp = self.be, self.be.xp
        Hh, Qq = self.cfg.horizons_s, self.cfg.quantiles
        try:
            mid = Qq.index(0.5)
        except ValueError:
            mid = len(Qq) // 2
        out: list[QuantileForecast] = []
        for ch in KPI_CHANNELS:
            ent = KPI_QUERY_ENTITY[ch]
            row = (query_row or {}).get(ent)
            q = self._query_vec(Z, masks, ent, row)
            trunk = xp.tanh(self.kpi_trunk[ch](q, be))
            raw = self.kpi_out[ch](trunk, be)  # (H*Q,)
            raw = np.asarray(be.to_numpy(raw)).reshape(len(Hh), len(Qq))
            for hi, hz in enumerate(Hh):
                vals = self._noncrossing_row(raw[hi], mid, Qq)
                out.append(
                    QuantileForecast(
                        channel=ch,
                        horizon_s=hz,
                        quantiles={Qq[j]: float(vals[j]) for j in range(len(Qq))},
                    )
                )
        return out

    def _noncrossing_row(
        self, row: np.ndarray, mid: int, quantiles: tuple[float, ...]
    ) -> np.ndarray:
        """Build a monotone-in-τ row from a median + non-negative offsets.

        ``q_mid`` is the median; lower quantiles subtract a cumulative softplus
        of the offsets and upper quantiles add one, guaranteeing
        ``q_{τ} ≤ q_{τ'}`` for ``τ < τ'`` (doc J Eq. 13 made structural).
        """
        median = float(row[mid])
        out = np.empty(len(quantiles), dtype=np.float32)
        out[mid] = median
        # upper side
        acc = 0.0
        for j in range(mid + 1, len(quantiles)):
            acc += float(np.log1p(np.exp(-abs(row[j]))) + max(row[j], 0.0))
            out[j] = median + acc
        # lower side
        acc = 0.0
        for j in range(mid - 1, -1, -1):
            acc += float(np.log1p(np.exp(-abs(row[j]))) + max(row[j], 0.0))
            out[j] = median - acc
        return out

    # ------------------------------------------------------------------ #
    # Hazard heads (competing risks + parallel binaries).
    # ------------------------------------------------------------------ #
    def hazard_forecasts(
        self,
        Z: dict[str, ArrayLike],
        masks: dict[str, ArrayLike],
        query_row: dict[str, int] | None = None,
    ) -> list[HazardForecast]:
        """Emit per-horizon hazard probabilities for every hazard channel.

        Competing-risk channels share a discrete-time survival softmax over
        ``{survive, timeout, requeue}`` and report the *cumulative incidence*
        ``F_k(t) = Σ_{u≤t} S(u-1)·π_{k,u}`` (doc L E5–E6). Parallel-binary
        channels report ``1 - Π_{u≤t}(1 - h_u)`` cumulative event probability.
        """
        be, _xp = self.be, self.be.xp
        Hh = self.cfg.horizons_s
        out: list[HazardForecast] = []

        # ---- competing risks (workload-level) ---------------------------- #
        wrow = (query_row or {}).get("workload")
        qw = self._query_vec(Z, masks, "workload", wrow)
        cr_logits = np.asarray(be.to_numpy(self.hz_competing(qw, be)))
        n_cr = len(COMPETING_RISK_CHANNELS) + 1
        cr_logits = cr_logits.reshape(len(Hh), n_cr)
        # per-interval class probs; class 0 = survive (doc L E5)
        surv = 1.0
        cum_inc = {ch: 0.0 for ch in COMPETING_RISK_CHANNELS}
        for hi, hz in enumerate(Hh):
            p = _np_softmax(cr_logits[hi])
            for k, ch in enumerate(COMPETING_RISK_CHANNELS):
                cum_inc[ch] += surv * float(p[k + 1])
            surv *= float(p[0])  # survive into next interval
            for ch in COMPETING_RISK_CHANNELS:
                out.append(
                    HazardForecast(
                        channel=ch,
                        horizon_s=hz,
                        probability=min(max(cum_inc[ch], 0.0), 1.0),
                    )
                )

        # ---- parallel binary incidents ---------------------------------- #
        for ch in PARALLEL_BINARY_CHANNELS:
            ent = HAZARD_QUERY_ENTITY[ch]
            row = (query_row or {}).get(ent)
            q = self._query_vec(Z, masks, ent, row)
            logits = np.asarray(be.to_numpy(self.hz_binary[ch](q, be)))
            haz = _np_sigmoid(logits)  # per-interval hazard
            survive = 1.0
            for hi, hz in enumerate(Hh):
                survive *= 1.0 - float(haz[hi])
                out.append(
                    HazardForecast(
                        channel=ch,
                        horizon_s=hz,
                        probability=min(max(1.0 - survive, 0.0), 1.0),
                    )
                )
        return out

    # ------------------------------------------------------------------ #
    # Abstention / OOD head.
    # ------------------------------------------------------------------ #
    def abstention_scores(
        self,
        Z: dict[str, ArrayLike],
        masks: dict[str, ArrayLike],
        epistemic_disagreement: float = 0.0,
    ) -> tuple[float, float]:
        """Return ``(epistemic_score, ood_score)`` in ``[0, 1]`` (doc J H3).

        The head reads the global latent (the planner-query context summary). The
        externally-supplied ``epistemic_disagreement`` (ensemble variance from
        :mod:`world_model`) is fused in so the score reflects both the learned
        OOD detector and live member disagreement (doc L epistemic_method).
        """
        be, xp = self.be, self.be.xp
        g = self._query_vec(Z, masks, "global")
        trunk = xp.tanh(self.abstain_trunk(g, be))
        logits = np.asarray(be.to_numpy(self.abstain_out(trunk, be)))
        epi = float(_np_sigmoid(np.array([logits[0]]))[0])
        ood = float(_np_sigmoid(np.array([logits[1]]))[0])
        # fuse external ensemble disagreement (clipped) into the epistemic score
        epi = float(min(1.0, max(epi, epistemic_disagreement)))
        return epi, ood

    # ------------------------------------------------------------------ #
    # The full bundle.
    # ------------------------------------------------------------------ #
    def forecast_bundle(
        self,
        action_id: str,
        Z: dict[str, ArrayLike],
        masks: dict[str, ArrayLike],
        query_row: dict[str, int] | None = None,
        epistemic_disagreement: float = 0.0,
    ) -> KPIForecastBundle:
        """Assemble the planner-facing :class:`KPIForecastBundle` for one action."""
        quantiles = self.quantile_forecasts(Z, masks, query_row)
        hazards = self.hazard_forecasts(Z, masks, query_row)
        epi, ood = self.abstention_scores(Z, masks, epistemic_disagreement)
        return KPIForecastBundle(
            action_id=action_id,
            quantiles=quantiles,
            hazards=hazards,
            epistemic_score=epi,
            ood_score=ood,
        )

    def to_backend(self, be: Backend) -> ForecastHeads:
        if be.name == self.be.name:
            return self
        conv = self.be.to_numpy

        def cd(d: Dense) -> Dense:
            return Dense(be.from_numpy(conv(d.W)), be.from_numpy(conv(d.b)))

        return ForecastHeads(
            cfg=self.cfg,
            be=be,
            kpi_trunk={k: cd(v) for k, v in self.kpi_trunk.items()},
            kpi_out={k: cd(v) for k, v in self.kpi_out.items()},
            hz_competing=cd(self.hz_competing),
            hz_binary={k: cd(v) for k, v in self.hz_binary.items()},
            abstain_trunk=cd(self.abstain_trunk),
            abstain_out=cd(self.abstain_out),
        )


# small standalone numpy helpers (heads post-process on the CPU side) ---------- #
def _np_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.clip(e.sum(), 1e-12, None)


def _np_sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def latent_dims_of(model: RelationalWorldModel) -> dict[str, int]:
    """Return the per-entity latent widths a model exposes (for head sizing)."""
    from .architecture import latent_widths

    return latent_widths(model.cfg.latent)


__all__ = [
    "COMPETING_RISK_CHANNELS",
    "HAZARD_CHANNELS",
    "HAZARD_QUERY_ENTITY",
    "KPI_CHANNELS",
    "KPI_QUERY_ENTITY",
    "PARALLEL_BINARY_CHANNELS",
    "ForecastHeads",
    "HeadConfig",
    "latent_dims_of",
]
