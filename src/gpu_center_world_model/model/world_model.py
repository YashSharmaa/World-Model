# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""``ResidualWorldModel`` — the learned residual model behind the WorldModel
contract.

This class wires the typed relational encoder (:mod:`architecture`), the
query-conditioned heads (:mod:`heads`), an epistemic-disagreement ensemble (doc
L BatchEnsemble-style, here realized as a small bag of seed-perturbed encoders),
and the conformal calibrator (:mod:`calibration`) into one object implementing
:class:`model.contracts.WorldModel`:

* :meth:`encode` returns the per-entity latent belief ``Z_t``.
* :meth:`forecast` rolls one action forward and returns a calibrated
  :class:`KPIForecastBundle`.
* :meth:`score_actions` is the **single compiled hot path** the serving layer
  drives: it encodes once per (state, action), runs the heads, fuses ensemble
  disagreement into the epistemic score, calibrates, and applies the abstention
  gate — the learned model may only rank / forecast / abstain over the
  kernel-certified safe set, never expand it (the project spine, doc F/M).

The whole path is backend-neutral (MLX on Apple silicon, NumPy on Linux CI). The
exact kernel stays outside this object entirely (doc E boundary rule); the model
consumes only the candidate :class:`Action` list the kernel already certified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..schema.actions import Action, action_workload_uid
from ..schema.kpis import KPIForecastBundle
from ..schema.policy import AbstentionConfig
from ..schema.state import WorldState
from ..schema.tensors import ShapeCaps
from ..version import SCHEMA_VERSION
from .architecture import (
    ArchConfig,
    FeatureBuilder,
    RelationalWorldModel,
    TensorState,
    kernel_state_summary,
)
from .backend import Backend, get_backend
from .calibration.conformal import ConformalBundleCalibrator
from .heads import ForecastHeads, HeadConfig, latent_dims_of


def _action_name(action: Action) -> str:
    """Return the canonical action-type string for a candidate action."""
    return action.type.value


# ─────────────────────────────────────────────────────────────────────────────
# Rank-1 BatchEnsemble fast weights (doc L epistemic_method, BatchEnsemble)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class BatchEnsembleMember:
    """Per-member rank-1 fast weights over the shared relational backbone.

    BatchEnsemble (Wen et al. 2020, doc L epistemic_method) replaces a deep
    ensemble of ``M`` full copies with **one** shared "slow" backbone plus, per
    member, a *rank-1* multiplicative perturbation ``W_m = W ∘ (s_m r_mᵀ)``. Here
    we realise the rank-1 fast weight over each per-entity latent block as a pair
    of vectors ``(r_t, s_t)`` per latent type ``t``: the shared encoded latent
    ``z_t`` is modulated to ``z_t^m = s_t ∘ tanh((r_t ∘ z_t))`` — a genuine rank-1
    fast-weight transform of the encoder/temporal-core readout, applied *after*
    the shared backbone forward pass so no slow weights are duplicated. The members
    differ only by the ``O(M · Σ_t d_t)`` rank-1 parameters, which is the
    memory-budget win doc E reserves for the ensemble.

    Member 0 is the identity (the shared backbone itself), so ``encode`` of the
    base model and member 0 agree exactly.
    """

    r: dict[str, np.ndarray]
    s: dict[str, np.ndarray]
    seed: int

    @classmethod
    def build(cls, latent_dims: dict[str, int], seed: int) -> BatchEnsembleMember:
        rng = np.random.default_rng(seed)
        r: dict[str, np.ndarray] = {}
        s: dict[str, np.ndarray] = {}
        for t, d in latent_dims.items():
            # rank-1 vectors centred at 1 (identity-near) so the perturbation is a
            # small per-feature gain — the BatchEnsemble initialisation.
            r[t] = (1.0 + 0.1 * rng.normal(size=d)).astype(np.float64)
            s[t] = (1.0 + 0.1 * rng.normal(size=d)).astype(np.float64)
        return cls(r=r, s=s, seed=seed)

    def apply(self, z: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply the rank-1 fast weight to the shared latent dict (per type)."""
        out: dict[str, np.ndarray] = {}
        for t, v in z.items():
            rv = self.r.get(t)
            sv = self.s.get(t)
            if rv is None or sv is None:
                out[t] = v
                continue
            # broadcast the rank-1 (r, s) over the leading entity axis.
            modulated = sv * np.tanh(rv * v)
            out[t] = modulated
        return out


@dataclass(slots=True)
class ResidualWorldModel:
    """The learned residual, action- and query-conditioned world model.

    Build with :meth:`build` (deterministic), move backends with
    :meth:`to_backend`, and serve with :meth:`score_actions`. ``ensemble`` holds
    ``n_members`` seed-perturbed encoders whose latent disagreement is the
    epistemic signal feeding the abstention gate (doc L epistemic_method).
    """

    backbone: RelationalWorldModel
    heads: ForecastHeads
    feature_builder: FeatureBuilder
    calibrator: ConformalBundleCalibrator
    abstention: AbstentionConfig
    #: The *deep* ensemble of independent backbones is retained ONLY as an
    #: explicitly-labelled offline disagreement oracle (it is no longer the
    #: serving epistemic path). It defaults empty; serving uses the rank-1
    #: BatchEnsemble below (doc L epistemic_method / doc E memory budget).
    ensemble: list[RelationalWorldModel] = field(default_factory=list)
    #: Rank-1 BatchEnsemble fast-weight members over the shared backbone — the
    #: production epistemic path (shared slow weights + per-member rank-1 r/s).
    be_members: list[BatchEnsembleMember] = field(default_factory=list)
    #: Held-out disagreement scale used to map the raw latent-distance to a
    #: calibrated epistemic score in [0, 1] (replaces the ad-hoc 1 - exp(-d)).
    epistemic_scale: float = 0.0
    schema_hash: str = SCHEMA_VERSION
    model_version: str = "rwm-v1"

    @property
    def be(self) -> Backend:
        return self.backbone.be

    # ------------------------------------------------------------------ #
    # Construction.
    # ------------------------------------------------------------------ #
    @classmethod
    def build(
        cls,
        arch: ArchConfig | None = None,
        head_cfg: HeadConfig | None = None,
        abstention: AbstentionConfig | None = None,
        n_ensemble: int = 3,
        backend: Backend | None = None,
        n_oracle: int = -1,
    ) -> ResidualWorldModel:
        """Build the residual model with a rank-1 BatchEnsemble epistemic path.

        ``n_ensemble`` now sizes the **rank-1 BatchEnsemble** (shared backbone +
        per-member rank-1 fast weights) — the production epistemic path that
        replaces the deep ensemble of independent copies (doc L BatchEnsemble; doc
        E memory budget). ``n_oracle`` (default 0) optionally builds the deep
        ensemble of independent backbones, kept ONLY as a labelled offline
        disagreement oracle for evaluation — it is never on the serving path.
        """
        be = backend or get_backend()
        arch = arch or ArchConfig()
        backbone = RelationalWorldModel.build(arch, be)
        heads = ForecastHeads.build(latent_dims_of(backbone), head_cfg, be)
        latent_dims = latent_dims_of(backbone)
        # rank-1 BatchEnsemble members (shared slow weights, per-member r/s) — the
        # *serving* epistemic path. Sized by ``n_ensemble`` and reconstructed
        # deterministically from ``arch.seed`` on every build (no serialization
        # needed: they are fixed fast-weight inits, not trained backbone arrays).
        be_members: list[BatchEnsembleMember] = []
        for m in range(1, max(1, n_ensemble)):
            be_members.append(
                BatchEnsembleMember.build(latent_dims, seed=arch.seed + 7919 * m)
            )
        # The deep ensemble of independent backbones is retained ONLY as the
        # labelled offline disagreement oracle (doc L). ``n_oracle == -1`` (the
        # default) means "match ``n_ensemble``" so the existing checkpoint manifest
        # (which sizes the ensemble by ``len(model.ensemble)``) round-trips
        # byte-stably; ``n_oracle == 0`` skips it entirely on the memory-lean
        # serving build while still getting the rank-1 epistemic members.
        n_oracle_eff = max(0, n_ensemble - 1) if n_oracle < 0 else n_oracle
        ensemble: list[RelationalWorldModel] = []
        for m in range(1, n_oracle_eff + 1):
            member_cfg = ArchConfig(
                latent=arch.latent,
                message_dim=arch.message_dim,
                n_rounds=arch.n_rounds,
                attn_dim=arch.attn_dim,
                action_embed_dim=arch.action_embed_dim,
                seed=arch.seed + 1000 * m,
            )
            ensemble.append(RelationalWorldModel.build(member_cfg, be))
        return cls(
            backbone=backbone,
            heads=heads,
            feature_builder=FeatureBuilder(be),
            calibrator=ConformalBundleCalibrator(),
            abstention=abstention or AbstentionConfig(),
            ensemble=ensemble,
            be_members=be_members,
        )

    # ------------------------------------------------------------------ #
    # WorldModel.encode
    # ------------------------------------------------------------------ #
    def encode(self, state: WorldState, caps: ShapeCaps | None = None) -> object:
        """Encode ``state`` into the latent belief ``Z_t`` under a ``hold`` baseline.

        Returns the per-entity latent dict from the backbone. The optional
        ``caps`` pins a compile bucket; otherwise the smallest fitting bucket is
        chosen (doc E shape management).
        """
        ts = self.feature_builder.build(state, caps)
        return self.backbone.encode(ts, "hold")

    def _tensor_state(self, state: WorldState, caps: ShapeCaps | None) -> TensorState:
        return self.feature_builder.build(state, caps)

    # ------------------------------------------------------------------ #
    # Epistemic disagreement from the rank-1 BatchEnsemble (doc L E9–E10).
    # ------------------------------------------------------------------ #
    def _member_latents(
        self, ts: TensorState, action_name: str, z_main: dict
    ) -> list[dict[str, np.ndarray]]:
        """Per-member latents from the rank-1 BatchEnsemble (shared backbone).

        The shared backbone is encoded ONCE (``z_main``); each member is a cheap
        rank-1 fast-weight modulation of that latent (no second backbone forward
        pass). If a deep-ensemble oracle is present *and no rank-1 members are
        configured*, fall back to the oracle's independent encodes so the offline
        labelled-oracle path keeps working.
        """
        be = self.be
        ref = {k: np.asarray(be.to_numpy(v)) for k, v in z_main.items()}
        if self.be_members:
            return [m.apply(ref) for m in self.be_members]
        if self.ensemble:
            out: list[dict[str, np.ndarray]] = []
            for member in self.ensemble:
                zm = member.encode(ts, action_name)
                out.append({k: np.asarray(be.to_numpy(v)) for k, v in zm.items()})
            return out
        return []

    def _raw_disagreement(
        self, ts: TensorState, members: list[dict], z_main: dict
    ) -> float:
        """Mean masked per-type latent disagreement (the raw E10 distance)."""
        be = self.be
        ref = {k: np.asarray(be.to_numpy(v)) for k, v in z_main.items()}
        if not members:
            return 0.0
        # ensemble mean over the modulated members (BatchEnsemble q_bar).
        diffs: list[float] = []
        for zm in members:
            acc, n = 0.0, 0
            for t, a in zm.items():
                if t not in ref:
                    continue
                a = np.asarray(a)
                err = np.abs(a - ref[t])
                if t in ts.masks:
                    m = np.asarray(be.to_numpy(ts.masks[t]))[..., None]
                    err = err * m
                    denom = max(float(m.sum()) * a.shape[-1], 1.0)
                    acc += float(err.sum()) / denom
                else:
                    acc += float(err.mean())
                n += 1
            diffs.append(acc / max(n, 1))
        return float(np.mean(diffs))

    def _epistemic(self, ts: TensorState, action_name: str, z_main: dict) -> float:
        """Calibrated epistemic disagreement score in [0, 1] (doc L E10).

        The raw across-member latent distance ``d`` is mapped to a score by a
        **held-out disagreement scale** ``epistemic_scale`` (fitted by
        :meth:`calibrate_epistemic_scale` on an ID corpus as a robust high
        quantile of the raw distance): ``score = clip(d / scale, 0, 1)``. This is a
        genuine calibration to the model's own ID disagreement spread, replacing
        the ad-hoc ``1 - exp(-d)`` squash (which has no operational meaning). With
        no fitted scale yet it degrades to the bounded ``1 - exp(-d)`` so the gate
        still functions before calibration.
        """
        members = self._member_latents(ts, action_name, z_main)
        if not members:
            return 0.0
        d = self._raw_disagreement(ts, members, z_main)
        if self.epistemic_scale > 0.0:
            return float(np.clip(d / self.epistemic_scale, 0.0, 1.0))
        return float(1.0 - np.exp(-d))

    def calibrate_epistemic_scale(
        self,
        states: list[WorldState],
        actions: list[Action] | None = None,
        *,
        quantile: float = 0.95,
        caps: ShapeCaps | None = None,
    ) -> float:
        """Fit ``epistemic_scale`` to a held-out ID disagreement quantile (doc L).

        Computes the raw across-member latent disagreement on an in-distribution
        corpus and sets the scale to its robust high quantile, so an ID window
        scores ~``quantile`` and only genuinely-higher-than-ID disagreement
        saturates toward 1. This calibrates the disagreement->score mapping to the
        model's own ID spread instead of the arbitrary ``1 - exp(-d)`` curve.
        """
        ds: list[float] = []
        for i, s in enumerate(states):
            a_name = (
                actions[i].type.value
                if actions is not None and i < len(actions)
                else "hold"
            )
            ts = self._tensor_state(s, caps)
            z = self.backbone.encode(ts, a_name)
            members = self._member_latents(ts, a_name, z)
            ds.append(self._raw_disagreement(ts, members, z))
        if not ds:
            return self.epistemic_scale
        scale = float(np.quantile(np.asarray(ds, float), quantile))
        self.epistemic_scale = scale if scale > 1e-9 else self.epistemic_scale
        return self.epistemic_scale

    # ------------------------------------------------------------------ #
    # WorldModel.forecast
    # ------------------------------------------------------------------ #
    def forecast(
        self,
        state: WorldState,
        action: Action,
        horizon_s: int,
        caps: ShapeCaps | None = None,
    ) -> KPIForecastBundle:
        """Roll one action forward and return a calibrated forecast bundle.

        ``horizon_s`` is accepted for contract compatibility; the heads already
        emit every horizon in :data:`schema.HORIZONS_S`, so the bundle is
        horizon-complete and the planner reads the slice it needs.
        """
        ts = self._tensor_state(state, caps)
        return self._forecast_from_tensor(ts, action, state=state, horizon_s=horizon_s)

    def _forecast_from_tensor(
        self,
        ts: TensorState,
        action: Action,
        state: WorldState | None = None,
        horizon_s: int | None = None,
    ) -> KPIForecastBundle:
        a_name = _action_name(action)
        action_id = self._action_id(action)
        z = self.backbone.encode(ts, a_name)
        # Free-running predictor rollout (doc J D6, doc E compiled outer rollout
        # step): advance the encoded belief through ``P_θ`` for the requested
        # horizon, conditioned on the action and the stop-grad exact kernel-state
        # summary, then forecast from the *rolled* latent so the bundle reflects
        # the deployed recursive core — not a single-step encoding. The latent
        # key set is unchanged, so the head contract and KPIForecastBundle fields
        # stay byte-stable.
        z_head = self._rollout_latent(z, a_name, ts, state, horizon_s)
        epi = self._epistemic(ts, a_name, z)
        query_row = self._query_row(ts, action)
        bundle = self.heads.forecast_bundle(
            action_id, z_head, ts.masks, query_row=query_row, epistemic_disagreement=epi
        )
        bundle = self.calibrator.calibrate(bundle)
        return self._apply_gate(bundle)

    def _rollout_latent(
        self,
        z0: object,
        action_name: str,
        ts: TensorState,
        state: WorldState | None,
        horizon_s: int | None,
    ) -> dict:
        """Advance ``z0`` through the predictor for ``horizon_s`` (doc J D6).

        Returns the final rolled latent ``ẑ_H`` the heads forecast from. With no
        predictor or no state context the encoded ``z0`` is returned unchanged
        (the heads already emit every horizon), so the path degrades gracefully.
        """
        bb = self.backbone
        if bb.predictor is None or state is None:
            return z0  # type: ignore[return-value]
        # Map the wall-clock horizon to a small number of slow predictor steps,
        # capped by the configured H_slow (doc J B3). One slow step ≈ the planner
        # decision cadence; HORIZONS_S[-1] (the longest band) maps to H_slow.
        from ..schema.kpis import HORIZONS_S

        hmax = min(getattr(bb.cfg, "t_slow", 12), 12)
        if horizon_s is None or not HORIZONS_S:
            steps = 1
        else:
            longest = float(max(HORIZONS_S))
            steps = max(1, round(hmax * float(horizon_s) / longest))
        psi = kernel_state_summary(state, bb.be)
        traj = bb.rollout(
            z0,  # type: ignore[arg-type]
            action_name,
            ts.masks,
            kernel_states=[psi] * steps,
            horizon=steps,
            detach_every=0,  # inference: no BPTT truncation needed
        )
        return traj[-1] if traj else z0  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # WorldModel.score_actions — the single compiled hot path.
    # ------------------------------------------------------------------ #
    def score_actions(
        self,
        state: WorldState,
        actions: list[Action],
        horizon_s: int,
        caps: ShapeCaps | None = None,
    ) -> list[KPIForecastBundle]:
        """Forecast a batch of *kernel-certified safe* candidate actions.

        Tensorizes the state **once** (the expensive symbolic→array bridge) and
        reuses it for every candidate, so the per-action cost is just the encoder
        + heads forward pass (doc F live-demo latency target). Returns one
        :class:`KPIForecastBundle` per action, in input order.
        """
        ts = self._tensor_state(state, caps)
        return [
            self._forecast_from_tensor(ts, a, state=state, horizon_s=horizon_s)
            for a in actions
        ]

    # ------------------------------------------------------------------ #
    # Abstention gate (doc L selective-prediction / doc J R6).
    # ------------------------------------------------------------------ #
    def _apply_gate(self, bundle: KPIForecastBundle) -> KPIForecastBundle:
        """Set the abstention flag from epistemic / OOD scores (never expands set).

        The gate can only cause the learned model to *abstain* (defer to the
        conservative rule fallback); it can never make an unsafe action look
        safe, and it never overrides exact-kernel infeasibility (doc J R6).
        """
        cfg = self.abstention
        reason = None
        if bundle.epistemic_score > cfg.max_epistemic_disagreement:
            reason = "epistemic_disagreement"
        elif bundle.ood_score > cfg.max_ood_score:
            reason = "ood"
        bundle.abstain = reason is not None
        bundle.abstain_reason = reason
        return bundle

    # ------------------------------------------------------------------ #
    # Query plumbing.
    # ------------------------------------------------------------------ #
    def _action_id(self, action: Action) -> str:
        wl = action_workload_uid(action) or "*"
        return f"{wl}::{action.type.value}"

    def _query_row(self, ts: TensorState, action: Action) -> dict[str, int]:
        """Map the targeted workload to its row index for query-conditioned heads."""
        wl = action_workload_uid(action)
        if wl is None:
            return {}
        order = ts.order.get("workload", [])
        if wl in order:
            return {"workload": order.index(wl)}
        return {}

    # ------------------------------------------------------------------ #
    # Backend movement.
    # ------------------------------------------------------------------ #
    def to_backend(self, be: Backend) -> ResidualWorldModel:
        if be.name == self.be.name:
            return self
        return ResidualWorldModel(
            backbone=self.backbone.to_backend(be),
            heads=self.heads.to_backend(be),
            feature_builder=FeatureBuilder(be),
            calibrator=self.calibrator,
            abstention=self.abstention,
            ensemble=[m.to_backend(be) for m in self.ensemble],
            be_members=self.be_members,  # rank-1 r/s are backend-neutral numpy
            epistemic_scale=self.epistemic_scale,
            schema_hash=self.schema_hash,
            model_version=self.model_version,
        )


__all__ = ["BatchEnsembleMember", "ResidualWorldModel"]
