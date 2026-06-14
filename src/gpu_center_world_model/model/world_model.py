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
from .architecture import ArchConfig, FeatureBuilder, RelationalWorldModel, TensorState
from .backend import Backend, get_backend
from .calibration.conformal import ConformalBundleCalibrator
from .heads import ForecastHeads, HeadConfig, latent_dims_of


def _action_name(action: Action) -> str:
    """Return the canonical action-type string for a candidate action."""
    return action.type.value


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
    ensemble: list[RelationalWorldModel] = field(default_factory=list)
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
    ) -> ResidualWorldModel:
        be = backend or get_backend()
        arch = arch or ArchConfig()
        backbone = RelationalWorldModel.build(arch, be)
        heads = ForecastHeads.build(latent_dims_of(backbone), head_cfg, be)
        # ensemble: seed-perturbed copies of the backbone (BatchEnsemble proxy).
        ensemble: list[RelationalWorldModel] = []
        for m in range(1, max(1, n_ensemble)):
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
    # Epistemic disagreement from the ensemble (doc L E9–E10).
    # ------------------------------------------------------------------ #
    def _epistemic(self, ts: TensorState, action_name: str, z_main: dict) -> float:
        """Mean per-type latent disagreement across ensemble members (doc L E10)."""
        if not self.ensemble:
            return 0.0
        be = self.be
        ref = {k: np.asarray(be.to_numpy(v)) for k, v in z_main.items()}
        diffs: list[float] = []
        for member in self.ensemble:
            zm = member.encode(ts, action_name)
            acc, n = 0.0, 0
            for t, v in zm.items():
                if t not in ref:
                    continue
                a = np.asarray(be.to_numpy(v))
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
        # squash the (unbounded) mean latent-distance into a [0, 1] score.
        d = float(np.mean(diffs))
        return float(1.0 - np.exp(-d))

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
        return self._forecast_from_tensor(ts, action)

    def _forecast_from_tensor(
        self, ts: TensorState, action: Action
    ) -> KPIForecastBundle:
        a_name = _action_name(action)
        action_id = self._action_id(action)
        z = self.backbone.encode(ts, a_name)
        epi = self._epistemic(ts, a_name, z)
        query_row = self._query_row(ts, action)
        bundle = self.heads.forecast_bundle(
            action_id, z, ts.masks, query_row=query_row, epistemic_disagreement=epi
        )
        bundle = self.calibrator.calibrate(bundle)
        return self._apply_gate(bundle)

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
        return [self._forecast_from_tensor(ts, a) for a in actions]

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
            schema_hash=self.schema_hash,
            model_version=self.model_version,
        )


__all__ = ["ResidualWorldModel"]
