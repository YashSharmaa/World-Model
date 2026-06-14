# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Checkpoint bundle: model weights + feature_stats + schema_hash + calibration.

Doc E recommends *multiple files per checkpoint* rather than one monolith:
``model.safetensors`` (weights), ``training_state.json`` (step/schedule),
``feature_stats_and_schema.json`` (normalization, schema hash, entity caps), and
a separate ``calibration_bundle.json`` (versioned independently). This module
realizes that layout for the :class:`ResidualWorldModel`.

Weights are stored as ``safetensors`` when the optional ``safetensors`` wheel is
present (doc E preferred format) and otherwise fall back to NumPy ``.npz`` so the
package still saves/loads on plain CI (HARD RULE 3). The schema hash is recorded
so a checkpoint cannot be silently loaded against an incompatible schema (doc E
reproducibility requirement).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..version import SCHEMA_VERSION
from .architecture import ArchConfig, Dense, GRUCell, RelationalWorldModel
from .backend import Backend, get_backend
from .calibration.conformal import ConformalBundleCalibrator
from .heads import ForecastHeads, HeadConfig, latent_dims_of
from .world_model import ResidualWorldModel

try:  # optional, preferred per doc E
    from safetensors.numpy import load_file as _st_load
    from safetensors.numpy import save_file as _st_save

    HAS_SAFETENSORS = True
except ImportError:  # pragma: no cover - NumPy fallback path
    _st_load = _st_save = None  # type: ignore
    HAS_SAFETENSORS = False


# --------------------------------------------------------------------------- #
# Flatten / unflatten a model's parameters to a flat name -> ndarray dict.
# --------------------------------------------------------------------------- #
def _flatten_backbone(m: RelationalWorldModel, prefix: str) -> dict[str, np.ndarray]:
    be = m.be
    out: dict[str, np.ndarray] = {}

    def dense(name: str, d: Dense) -> None:
        out[f"{name}.W"] = np.asarray(be.to_numpy(d.W))
        out[f"{name}.b"] = np.asarray(be.to_numpy(d.b))

    def gru(name: str, g: GRUCell) -> None:
        out[f"{name}.Wz"] = np.asarray(be.to_numpy(g.Wz))
        out[f"{name}.Wr"] = np.asarray(be.to_numpy(g.Wr))
        out[f"{name}.Wh"] = np.asarray(be.to_numpy(g.Wh))

    for k, v in m.encoders.items():
        dense(f"{prefix}.enc.{k}", v)
    gru(f"{prefix}.fast_core", m.fast_core)
    gru(f"{prefix}.slow_core", m.slow_core)
    for k, v in m.relation_update.items():
        gru(f"{prefix}.relupd.{k}", v)
    for k, v in m.pool_proj.items():
        dense(f"{prefix}.pool.{k}", v)
    for k, v in m.broadcast_update.items():
        gru(f"{prefix}.bcast.{k}", v)
    # Wave-1 typed per-relation message blocks (doc Q D2-D4).
    for k, v in m.rel_src.items():
        dense(f"{prefix}.relsrc.{k}", v)
    for k, v in m.rel_dst.items():
        dense(f"{prefix}.reldst.{k}", v)
    for k, v in m.rel_phi.items():
        dense(f"{prefix}.relphi.{k}", v)
    for k, v in m.rel_gate.items():
        dense(f"{prefix}.relgate.{k}", v)
    out[f"{prefix}.action_embed"] = np.asarray(be.to_numpy(m.action_embed))
    dense(f"{prefix}.attn_q", m.attn_q)
    dense(f"{prefix}.attn_k", m.attn_k)
    dense(f"{prefix}.attn_v", m.attn_v)
    dense(f"{prefix}.global_readout", m.global_readout)
    dense(f"{prefix}.film_gen", m.film_gen)
    # World-transition predictor P_θ (doc J D6).
    if m.predictor is not None:
        for k, v in m.predictor.cells.items():
            gru(f"{prefix}.pred.cells.{k}", v)
        for k, v in m.predictor.psi_proj.items():
            dense(f"{prefix}.pred.psi.{k}", v)
    # Exact-field decoders D_f for the consistency loss (doc J D11).
    if m.field_decoders is not None:
        for k, v in m.field_decoders.decoders.items():
            dense(f"{prefix}.fdec.{k}", v)
    return out


def _unflatten_backbone(
    m: RelationalWorldModel, prefix: str, params: dict[str, np.ndarray]
) -> None:
    be = m.be

    def dense(name: str) -> Dense:
        return Dense(
            be.from_numpy(params[f"{name}.W"]),
            be.from_numpy(params[f"{name}.b"]),
        )

    def gru(name: str) -> GRUCell:
        return GRUCell(
            be.from_numpy(params[f"{name}.Wz"]),
            be.from_numpy(params[f"{name}.Wr"]),
            be.from_numpy(params[f"{name}.Wh"]),
        )

    for k in m.encoders:
        m.encoders[k] = dense(f"{prefix}.enc.{k}")
    m.fast_core = gru(f"{prefix}.fast_core")
    m.slow_core = gru(f"{prefix}.slow_core")
    for k in m.relation_update:
        m.relation_update[k] = gru(f"{prefix}.relupd.{k}")
    for k in m.pool_proj:
        m.pool_proj[k] = dense(f"{prefix}.pool.{k}")
    for k in m.broadcast_update:
        m.broadcast_update[k] = gru(f"{prefix}.bcast.{k}")
    for k in m.rel_src:
        m.rel_src[k] = dense(f"{prefix}.relsrc.{k}")
    for k in m.rel_dst:
        m.rel_dst[k] = dense(f"{prefix}.reldst.{k}")
    for k in m.rel_phi:
        m.rel_phi[k] = dense(f"{prefix}.relphi.{k}")
    for k in m.rel_gate:
        m.rel_gate[k] = dense(f"{prefix}.relgate.{k}")
    m.action_embed = be.from_numpy(params[f"{prefix}.action_embed"])
    m.attn_q = dense(f"{prefix}.attn_q")
    m.attn_k = dense(f"{prefix}.attn_k")
    m.attn_v = dense(f"{prefix}.attn_v")
    m.global_readout = dense(f"{prefix}.global_readout")
    m.film_gen = dense(f"{prefix}.film_gen")
    if m.predictor is not None:
        for k in m.predictor.cells:
            m.predictor.cells[k] = gru(f"{prefix}.pred.cells.{k}")
        for k in m.predictor.psi_proj:
            m.predictor.psi_proj[k] = dense(f"{prefix}.pred.psi.{k}")
    if m.field_decoders is not None:
        for k in m.field_decoders.decoders:
            m.field_decoders.decoders[k] = dense(f"{prefix}.fdec.{k}")


def _flatten_heads(h: ForecastHeads) -> dict[str, np.ndarray]:
    be = h.be
    out: dict[str, np.ndarray] = {}

    def dense(name: str, d: Dense) -> None:
        out[f"{name}.W"] = np.asarray(be.to_numpy(d.W))
        out[f"{name}.b"] = np.asarray(be.to_numpy(d.b))

    for ch, d in h.kpi_trunk.items():
        dense(f"heads.kpi_trunk.{ch.value}", d)
    for ch, d in h.kpi_out.items():
        dense(f"heads.kpi_out.{ch.value}", d)
    dense("heads.hz_competing", h.hz_competing)
    for ch, d in h.hz_binary.items():
        dense(f"heads.hz_binary.{ch.value}", d)
    dense("heads.abstain_trunk", h.abstain_trunk)
    dense("heads.abstain_out", h.abstain_out)
    return out


def _unflatten_heads(h: ForecastHeads, params: dict[str, np.ndarray]) -> None:
    be = h.be

    def dense(name: str) -> Dense:
        return Dense(
            be.from_numpy(params[f"{name}.W"]),
            be.from_numpy(params[f"{name}.b"]),
        )

    for ch in h.kpi_trunk:
        h.kpi_trunk[ch] = dense(f"heads.kpi_trunk.{ch.value}")
    for ch in h.kpi_out:
        h.kpi_out[ch] = dense(f"heads.kpi_out.{ch.value}")
    h.hz_competing = dense("heads.hz_competing")
    for ch in h.hz_binary:
        h.hz_binary[ch] = dense(f"heads.hz_binary.{ch.value}")
    h.abstain_trunk = dense("heads.abstain_trunk")
    h.abstain_out = dense("heads.abstain_out")


# --------------------------------------------------------------------------- #
# Weight I/O (safetensors preferred, npz fallback).
# --------------------------------------------------------------------------- #
def _save_weights(path: Path, params: dict[str, np.ndarray]) -> str:
    arrs = {k: np.ascontiguousarray(v.astype(np.float32)) for k, v in params.items()}
    if HAS_SAFETENSORS:
        _st_save(arrs, str(path.with_suffix(".safetensors")))
        return str(path.with_suffix(".safetensors"))
    np.savez(path.with_suffix(".npz"), **arrs)
    return str(path.with_suffix(".npz"))


def _load_weights(path: Path) -> dict[str, np.ndarray]:
    st = path.with_suffix(".safetensors")
    npz = path.with_suffix(".npz")
    if st.exists() and HAS_SAFETENSORS:
        return dict(_st_load(str(st)))
    if npz.exists():
        with np.load(npz) as data:
            return {k: data[k] for k in data.files}
    raise FileNotFoundError(f"No weights at {st} or {npz}")


@dataclass(slots=True)
class CheckpointManifest:
    """The human-readable checkpoint metadata block (doc E reproducibility)."""

    schema_hash: str
    model_version: str
    arch: dict[str, Any]
    head: dict[str, Any]
    n_ensemble: int
    weights_file: str
    author: str = "Yash Sharma"
    author_url: str = "https://www.linkedin.com/in/yashsharmaa/"


# --------------------------------------------------------------------------- #
# Public save / load.
# --------------------------------------------------------------------------- #
def save_checkpoint(
    model: ResidualWorldModel,
    directory: str | Path,
    feature_stats: dict[str, Any] | None = None,
) -> CheckpointManifest:
    """Write the multi-file checkpoint bundle (doc E layout).

    Produces ``model.{safetensors|npz}``, ``manifest.json`` (schema hash +
    architecture config), ``calibration.json`` (conformal thresholds), and
    ``feature_stats.json`` (normalization). The schema hash gates reloading.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)

    params: dict[str, np.ndarray] = {}
    params.update(_flatten_backbone(model.backbone, "backbone"))
    for i, member in enumerate(model.ensemble):
        params.update(_flatten_backbone(member, f"ensemble.{i}"))
    params.update(_flatten_heads(model.heads))
    weights_file = _save_weights(d / "model", params)

    arch = model.backbone.cfg
    head = model.heads.cfg
    manifest = CheckpointManifest(
        schema_hash=model.schema_hash,
        model_version=model.model_version,
        arch={
            "message_dim": arch.message_dim,
            "n_rounds": arch.n_rounds,
            "attn_dim": arch.attn_dim,
            "action_embed_dim": arch.action_embed_dim,
            "seed": arch.seed,
            "latent": arch.latent.as_dict(),
        },
        head={
            "horizons_s": list(head.horizons_s),
            "quantiles": list(head.quantiles),
            "hidden": head.hidden,
            "seed": head.seed,
        },
        n_ensemble=len(model.ensemble) + 1,
        weights_file=Path(weights_file).name,
    )
    (d / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    (d / "calibration.json").write_text(
        json.dumps(model.calibrator.state_dict(), indent=2), encoding="utf-8"
    )
    (d / "feature_stats.json").write_text(
        json.dumps(
            {"schema_hash": SCHEMA_VERSION, "stats": feature_stats or {}}, indent=2
        ),
        encoding="utf-8",
    )
    return manifest


def load_checkpoint(
    directory: str | Path, backend: Backend | None = None
) -> ResidualWorldModel:
    """Load a checkpoint bundle into a fresh :class:`ResidualWorldModel`.

    Raises if the on-disk schema hash differs from the running
    :data:`SCHEMA_VERSION` (doc E: a checkpoint must never silently load against
    an incompatible schema).
    """
    d = Path(directory)
    be = backend or get_backend()
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_hash"] != SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint schema {manifest['schema_hash']} != running "
            f"{SCHEMA_VERSION}; refusing to load (doc E reproducibility)."
        )
    a = manifest["arch"]
    from ..schema.tensors import LatentDims

    lat = LatentDims(
        **{k: v for k, v in a["latent"].items() if k != "global"},
        global_=a["latent"]["global"],
    )
    arch_cfg = ArchConfig(
        latent=lat,
        message_dim=a["message_dim"],
        n_rounds=a["n_rounds"],
        attn_dim=a["attn_dim"],
        action_embed_dim=a["action_embed_dim"],
        seed=a["seed"],
    )
    h = manifest["head"]
    head_cfg = HeadConfig(
        horizons_s=tuple(h["horizons_s"]),
        quantiles=tuple(h["quantiles"]),
        hidden=h["hidden"],
        seed=h["seed"],
    )
    model = ResidualWorldModel.build(
        arch=arch_cfg,
        head_cfg=head_cfg,
        n_ensemble=manifest["n_ensemble"],
        backend=be,
    )

    params = _load_weights(d / "model")
    _unflatten_backbone(model.backbone, "backbone", params)
    for i, member in enumerate(model.ensemble):
        _unflatten_backbone(member, f"ensemble.{i}", params)
    _unflatten_heads(model.heads, params)

    cal_path = d / "calibration.json"
    if cal_path.exists():
        model.calibrator = ConformalBundleCalibrator.from_state_dict(
            json.loads(cal_path.read_text(encoding="utf-8"))
        )
    model.schema_hash = manifest["schema_hash"]
    model.model_version = manifest["model_version"]
    return model


def feature_stats_from_latents(model: ResidualWorldModel) -> dict[str, Any]:
    """Return the per-entity latent widths as a compact feature-stats block."""
    return {"latent_dims": latent_dims_of(model.backbone)}


__all__ = [
    "HAS_SAFETENSORS",
    "CheckpointManifest",
    "feature_stats_from_latents",
    "load_checkpoint",
    "save_checkpoint",
]
