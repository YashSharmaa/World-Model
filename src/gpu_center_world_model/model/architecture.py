# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The typed relational world model (doc Q) realized over the backend shim.

This module implements the v1 architecture recommended by doc Q:

* **Per-entity encoders** (small MLPs) map raw, masked, padded entity features
  into the per-type latent widths fixed by :data:`schema.DEFAULT_LATENT_DIMS`
  (doc P latent-dimension budget).
* **Fast / slow GRU temporal cores** (doc E/J) compress the multi-rate
  observation lanes — a fast core for gpu/node/rack/switch transients and a slow
  core for workload/queue/scheduler context — into per-step summaries.
* **Typed message passing as FIXED-INCIDENCE batched matmuls** ``U_r = S_r·H_σ``
  and ``Σ_r = T_rᵀ·M_r`` (doc Q Eqs. 1–3, D1–D4). No sparse graph runtime: the
  incidence matrices are one-hot gather/scatter operators, so the scheme is the
  *implementation* of a typed MPNN, not a different expressive class (doc Q T1).
* **Hierarchical PNA-style pooling** gpu→node→rack→domain (doc Q P1–P7) with the
  sum/mean/std/max/count reducer, followed by **one downward broadcast** sweep
  (doc Q P10–P11) to recover long-range context cheaply.
* **Tiny candidate-set attention** for workload↔domain scoring over the small,
  exact-kernel-masked feasible set (doc Q Eq. 6, V5).

The whole forward path is written against :class:`backend.Backend` so the exact
same code runs on MLX (Apple silicon) and the NumPy reference path used by CI.
Weights are deterministic NumPy arrays seeded once and pushed through the
backend's ``from_numpy``; this keeps the architecture testable with NumPy alone
while remaining a drop-in for an ``mlx.nn``-trained checkpoint.

Strict **per-type permutation equivariance** (doc Q G1–G2, Th. 1–3): every
update is row-shared over the entity axis and every reducer is a masked
sum/mean/max, so within-type relabelling commutes with the forward map. Padding
positions are zeroed by the validity masks and never carry slot-specific
parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..schema.entities import (
    CommClass,
    DeviceHealth,
    NodeHealth,
    Workload,
    WorkloadStatus,
)
from ..schema.state import WorldState
from ..schema.tensors import (
    DEFAULT_LATENT_DIMS,
    MESSAGE_PASSING_ROUNDS,
    LatentDims,
    ShapeCaps,
    pick_bucket,
)
from ..schema.topology import HIERARCHY_UP
from .backend import ArrayLike, Backend, get_backend

# Entity types that live on the fast telemetry clock vs. the slow scheduler clock
# (doc E temporal cores / doc J B4 multi-timescale design).
FAST_TYPES: tuple[str, ...] = ("gpu", "node", "rack", "switch")
SLOW_TYPES: tuple[str, ...] = ("workload", "podset", "queue", "global")

# Raw input feature widths per entity type (hand-curated from the schema's
# dynamic fields; deliberately small and auditable, not learned column counts).
RAW_FEATURE_DIMS: dict[str, int] = {
    "workload": 14,
    "podset": 8,
    "queue": 8,
    "topology_domain": 6,
    "node": 10,
    "gpu": 12,
    "rack": 8,
    "switch": 4,
    "global": 8,
}

# The four-action alphabet embedding index (doc C / P action-as-factor).
ACTION_INDEX: dict[str, int] = {
    "hold": 0,
    "admit": 1,
    "place": 2,
    "admit_and_place": 3,
    "mask_domain": 4,
}
N_ACTIONS = 5


# --------------------------------------------------------------------------- #
# Deterministic dense-layer parameters (backend-neutral container).
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Dense:
    """A row-shared affine layer ``y = x·Wᵀ + b`` (kept type/relation shared)."""

    W: ArrayLike  # shape (out, in)
    b: ArrayLike  # shape (out,)

    def __call__(self, x: ArrayLike, be: Backend) -> ArrayLike:
        return be.xp.matmul(x, self.W.swapaxes(-1, -2)) + self.b


@dataclass(slots=True)
class GRUCell:
    """A minimal GRU cell (update/reset/candidate gates), row-shared over slots.

    Implements the gated residual update of doc Q Eq. 1 / doc J B5 (positive
    update-gate bias for stability). Operates on the trailing feature axis so it
    is permutation-equivariant over leading entity / batch axes.
    """

    Wz: ArrayLike
    Wr: ArrayLike
    Wh: ArrayLike  # each (hidden, input + hidden)

    def __call__(self, h: ArrayLike, x: ArrayLike, be: Backend) -> ArrayLike:
        xp = be.xp
        xh = xp.concatenate([x, h], axis=-1)
        z = be.sigmoid(xp.matmul(xh, self.Wz.swapaxes(-1, -2)))
        r = be.sigmoid(xp.matmul(xh, self.Wr.swapaxes(-1, -2)))
        xrh = xp.concatenate([x, r * h], axis=-1)
        hcand = xp.tanh(xp.matmul(xrh, self.Wh.swapaxes(-1, -2)))
        return (1.0 - z) * h + z * hcand


def _init_dense(rng: np.random.Generator, out_dim: int, in_dim: int) -> tuple:
    # Glorot/Xavier-uniform; deterministic given rng (doc J B5 stability).
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    W = rng.uniform(-limit, limit, size=(out_dim, in_dim)).astype(np.float32)
    b = np.zeros((out_dim,), dtype=np.float32)
    return W, b


def _init_gru(rng: np.random.Generator, hidden: int, in_dim: int) -> tuple:
    n = in_dim + hidden
    limit = np.sqrt(6.0 / (n + hidden))
    Wz = rng.uniform(-limit, limit, size=(hidden, n)).astype(np.float32)
    Wr = rng.uniform(-limit, limit, size=(hidden, n)).astype(np.float32)
    Wh = rng.uniform(-limit, limit, size=(hidden, n)).astype(np.float32)
    return Wz, Wr, Wh


# --------------------------------------------------------------------------- #
# Architecture configuration.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ArchConfig:
    """Static architecture hyper-parameters (doc E memory budget / doc Q V1)."""

    latent: LatentDims = field(default_factory=lambda: DEFAULT_LATENT_DIMS)
    message_dim: int = 32  # typed-edge message width (doc Q D2)
    n_rounds: int = MESSAGE_PASSING_ROUNDS  # upward sweeps (doc Q V2)
    attn_dim: int = 32  # candidate-set attention key/query width (doc Q Eq. 6)
    action_embed_dim: int = 8  # action-as-factor embedding (doc P R4)
    seed: int = 0


# PNA reducer fan-out: sum/mean/std/max are each child-width, plus a single
# scalar log1p(count) (doc Q Agt5). So a pool over child-width ``cd`` yields
# ``4*cd + 1`` features.
_PNA_VEC = 4  # sum, mean, std, max


def _pna_width(cd: int) -> int:
    return _PNA_VEC * cd + 1


def latent_widths(latent: LatentDims) -> dict[str, int]:
    """Resolve per-latent-type widths, mapping ``topology_domain`` → rack width.

    The frozen :class:`schema.LatentDims` (doc P budget) does not carry a
    dedicated ``topology_domain`` key, so we reuse the rack width (doc P assigns
    the domain a band comparable to the rack: shared power/cooling inertia and
    medium-rank hotspot dynamics). This keeps the latent budget faithful to the
    frozen contract without modifying it.
    """
    w = latent.as_dict()
    w["topology_domain"] = w["rack"]
    return w


# --------------------------------------------------------------------------- #
# The model parameter bundle.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class RelationalWorldModel:
    """The typed relational world model (doc Q recommended v1 architecture).

    Holds all learnable parameters as backend arrays and exposes a pure
    ``encode``/``forward`` path. Construct via :meth:`build` (deterministic) and
    move it to a backend with :meth:`to_backend`. The same object serves the
    NumPy reference path (tests / Linux CI) and an MLX-loaded checkpoint.
    """

    cfg: ArchConfig
    be: Backend
    # per-entity encoders + temporal cores + relation/pool/broadcast params
    encoders: dict[str, Dense]
    fast_core: GRUCell
    slow_core: GRUCell
    relation_msg: dict[str, Dense]  # type -> incoming-edge message MLP
    relation_update: dict[str, GRUCell]  # type -> gated update
    pool_proj: dict[str, Dense]  # parent-type -> PNA pool projection
    broadcast_update: dict[str, GRUCell]  # child-type -> downward-broadcast update
    action_embed: ArrayLike  # (N_ACTIONS, action_embed_dim)
    attn_q: Dense  # workload query for candidate attention
    attn_k: Dense  # domain key for candidate attention
    attn_v: Dense  # domain value for candidate attention

    # ---- construction ---------------------------------------------------- #
    @classmethod
    def build(
        cls, cfg: ArchConfig | None = None, backend: Backend | None = None
    ) -> RelationalWorldModel:
        cfg = cfg or ArchConfig()
        be = backend or get_backend()
        rng = np.random.default_rng(cfg.seed)
        lat = latent_widths(cfg.latent)

        def dense(out_dim: int, in_dim: int) -> Dense:
            W, b = _init_dense(rng, out_dim, in_dim)
            return Dense(be.from_numpy(W), be.from_numpy(b))

        def gru(hidden: int, in_dim: int) -> GRUCell:
            Wz, Wr, Wh = _init_gru(rng, hidden, in_dim)
            return GRUCell(be.from_numpy(Wz), be.from_numpy(Wr), be.from_numpy(Wh))

        encoders = {
            t: dense(lat[t], RAW_FEATURE_DIMS[t] + cfg.action_embed_dim)
            for t in ENTITY_TYPES_LAT
        }
        # Fast / slow cores operate on a *shared* working width = max latent so a
        # single cell serves every type on that clock (row-shared, equivariant).
        fast_w = max(lat[t] for t in FAST_TYPES)
        slow_w = max(lat[t] for t in SLOW_TYPES)
        fast_core = gru(fast_w, fast_w)
        slow_core = gru(slow_w, slow_w)

        # Typed message MLP: input = [W_src·U, W_dst·V, action, global] summary.
        relation_msg: dict[str, Dense] = {}
        relation_update: dict[str, GRUCell] = {}
        for t in ENTITY_TYPES_LAT:
            d = lat[t]
            # message takes concat(self, summed-incoming, action_embed, global)
            msg_in = d + cfg.message_dim + cfg.action_embed_dim + lat["global"]
            relation_msg[t] = dense(cfg.message_dim, d + cfg.message_dim)
            relation_update[t] = gru(d, msg_in)

        # Hierarchical PNA pools (gpu->node->rack->domain), child width -> parent.
        pool_proj: dict[str, Dense] = {}
        broadcast_update: dict[str, GRUCell] = {}
        for child, parent in _UP_PAIRS:
            cd, pd = lat[child], lat[parent]
            pool_proj[parent] = dense(pd, pd + _pna_width(cd))
            broadcast_update[child] = gru(cd, cd + pd)

        action_embed = be.from_numpy(
            rng.normal(0, 0.1, size=(N_ACTIONS, cfg.action_embed_dim)).astype(
                np.float32
            )
        )
        attn_q = dense(cfg.attn_dim, lat["workload"])
        attn_k = dense(cfg.attn_dim, lat["topology_domain"])
        attn_v = dense(lat["workload"], lat["topology_domain"])

        return cls(
            cfg=cfg,
            be=be,
            encoders=encoders,
            fast_core=fast_core,
            slow_core=slow_core,
            relation_msg=relation_msg,
            relation_update=relation_update,
            pool_proj=pool_proj,
            broadcast_update=broadcast_update,
            action_embed=action_embed,
            attn_q=attn_q,
            attn_k=attn_k,
            attn_v=attn_v,
        )

    # ---- helpers --------------------------------------------------------- #
    def _pna_reduce(
        self, child: ArrayLike, assign: ArrayLike, n_parents: int
    ) -> ArrayLike:
        """PNA reducer over a child→parent assignment (doc Q Agt5, P1–P7).

        ``assign`` is the one-hot parent incidence ``P_{c→p}`` of shape
        (n_parents, n_children). Returns concat(sum, mean, std, max, log1p(n))
        over the child axis for each parent — the count- and shape-aware reducer
        doc Q insists on (mean-only is insufficient: Agt2–Agt4).
        """
        xp = self.be.xp
        # sum & count
        psum = xp.matmul(assign, child)  # (P, d)
        cnt = xp.sum(assign, axis=-1, keepdims=True)  # (P, 1)
        denom = xp.maximum(cnt, 1.0)
        pmean = psum / denom
        # E[x^2] for std
        psq = xp.matmul(assign, child * child) / denom
        pvar = xp.maximum(psq - pmean * pmean, 0.0)
        pstd = xp.sqrt(pvar + 1e-6)
        # masked max: push padded children to -inf before reducing
        # assign: (P, C); child: (C, d) -> (P, C, d)
        big = child[None, :, :] * assign[:, :, None]
        neg = (1.0 - assign)[:, :, None] * (-1e9)
        pmax = xp.max(big + neg, axis=1)
        plog = xp.log1p(cnt)
        return xp.concatenate([psum, pmean, pstd, pmax, plog], axis=-1)

    # ---- the public forward path ----------------------------------------- #
    def encode(
        self, batch: TensorState, action_name: str = "hold"
    ) -> dict[str, ArrayLike]:
        """Encode a tensorized state into per-entity latents ``Z`` (doc Q Eq. 8).

        TemporalEncode → RelFuse (two upward sweeps + one downward broadcast).
        ``action_name`` selects the action embedding broadcast through every
        update via FiLM-style concatenation (doc Q A4: row-shared, never
        slot-specific). Returns a dict ``{entity_type: latent[n, d]}``.
        """
        be, xp, cfg = self.be, self.be.xp, self.cfg
        a_idx = ACTION_INDEX.get(action_name, 0)
        a_emb = self.action_embed[a_idx]  # (action_embed_dim,)

        # 1) Per-entity encode (raw features + action) -> latent ------------- #
        H: dict[str, ArrayLike] = {}
        for t in ENTITY_TYPES_LAT:
            feats = batch.features[t]  # (n, raw)
            n = feats.shape[0]
            a_row = xp.broadcast_to(a_emb[None, :], (n, a_emb.shape[0]))
            enc_in = xp.concatenate([feats, a_row], axis=-1)
            H[t] = xp.tanh(self.encoders[t](enc_in, be)) * batch.masks[t][:, None]

        # 2) Temporal cores: blend the encoded latent with its telemetry-summary
        # GRU step. With one snapshot per state we run a single recurrent step
        # from a zero hidden state (the cores still gate/normalize the signal).
        for t in FAST_TYPES:
            H[t] = self._temporal(H[t], self.fast_core, batch.masks[t])
        for t in SLOW_TYPES:
            if t == "global":
                continue
            H[t] = self._temporal(H[t], self.slow_core, batch.masks[t])

        # 3) Relational fusion: upward typed message passing -------------- #
        for _ in range(cfg.n_rounds):
            H = self._message_round(H, batch, a_emb)

        # 4) Hierarchical PNA pooling gpu->node->rack->domain ------------- #
        H = self._upward_pool(H, batch)

        # 5) Global read-out + one downward broadcast --------------------- #
        H["global"] = self._global_readout(H, batch)[None, :]  # keep (1, d)
        H = self._downward_broadcast(H, batch)
        return H

    def _temporal(self, latent: ArrayLike, core: GRUCell, mask: ArrayLike) -> ArrayLike:
        """Run one recurrent step, padding the latent to the core working width."""
        be, xp = self.be, self.be.xp
        d = latent.shape[-1]
        w = core.Wz.shape[0]
        if d < w:
            pad = xp.zeros((latent.shape[0], w - d), dtype=be.float_dtype)
            x = xp.concatenate([latent, pad], axis=-1)
        else:
            x = latent[:, :w]
        h0 = xp.zeros((latent.shape[0], w), dtype=be.float_dtype)
        out = core(h0, x, be)[:, :d]
        return out * mask[:, None]

    def _message_round(
        self, H: dict[str, ArrayLike], batch: TensorState, a_emb: ArrayLike
    ) -> dict[str, ArrayLike]:
        """One typed message-passing round (doc Q D1–D4, Eq. 1).

        For each typed relation r: σ→τ, gather sources via ``S_r`` (here folded
        into the dense parent/child assignment), build a row-shared message, and
        scatter-add into targets via ``T_rᵀ`` — exactly the typed MPNN of Eq. 3.
        """
        be, xp, cfg = self.be, self.be.xp, self.cfg
        g = H.get("global")
        if g is None:
            g = xp.zeros((cfg.latent.global_,), dtype=be.float_dtype)
        elif g.ndim == 2:  # global is stored as a single padded row (1, d)
            g = g[0]
        msg_dim = cfg.message_dim
        # Accumulate incoming messages per target type.
        incoming: dict[str, ArrayLike] = {
            t: xp.zeros((H[t].shape[0], msg_dim), dtype=be.float_dtype)
            for t in ENTITY_TYPES_LAT
        }
        counts: dict[str, ArrayLike] = {
            t: xp.zeros((H[t].shape[0], 1), dtype=be.float_dtype)
            for t in ENTITY_TYPES_LAT
        }
        for (src_t, dst_t), assign in batch.incidence.items():
            # assign: (n_dst, n_src) one-hot scatter for relation src->dst.
            src = H[src_t]
            # message MLP wants (self_dst_proxy, projected_src). Use a fixed
            # projection of the source latent into message space, then scatter.
            src_msg = self._project_to_msg(src, dst_t)
            scattered = xp.matmul(assign, src_msg)  # T_rᵀ · M_r  (n_dst, msg)
            incoming[dst_t] = incoming[dst_t] + scattered
            counts[dst_t] = counts[dst_t] + xp.sum(assign, axis=-1, keepdims=True)
        # Mean-normalize messages (doc Q D3) and run the gated update.
        for t in ENTITY_TYPES_LAT:
            mu = incoming[t] / xp.maximum(counts[t], 1.0)
            n = H[t].shape[0]
            a_row = xp.broadcast_to(a_emb[None, :], (n, a_emb.shape[0]))
            g_row = xp.broadcast_to(g[None, :], (n, g.shape[0]))
            upd_in = xp.concatenate([H[t], mu, a_row, g_row], axis=-1)
            H[t] = self.relation_update[t](H[t], upd_in, be) * batch.masks[t][:, None]
        return H

    def _project_to_msg(self, src: ArrayLike, dst_t: str) -> ArrayLike:
        """Project a source latent into the shared message space (doc Q D2)."""
        be, xp, _cfg = self.be, self.be.xp, self.cfg
        d = src.shape[-1]
        # pad/truncate to (d_dst + msg_dim) so the dst message MLP applies.
        target_in = self.relation_msg[dst_t].W.shape[1]
        if d < target_in:
            pad = xp.zeros((src.shape[0], target_in - d), dtype=be.float_dtype)
            x = xp.concatenate([src, pad], axis=-1)
        else:
            x = src[:, :target_in]
        return xp.tanh(self.relation_msg[dst_t](x, be))

    def _upward_pool(
        self, H: dict[str, ArrayLike], batch: TensorState
    ) -> dict[str, ArrayLike]:
        """gpu→node→rack→domain PNA pooling (doc Q Eq. 2–4, P1–P7)."""
        be, xp = self.be, self.be.xp
        for child, parent in _UP_PAIRS:
            assign = batch.hierarchy.get((child, parent))
            if assign is None or H[parent].shape[0] == 0:
                continue
            reduced = self._pna_reduce(H[child], assign, H[parent].shape[0])
            pooled_in = xp.concatenate([H[parent], reduced], axis=-1)
            H[parent] = xp.tanh(self.pool_proj[parent](pooled_in, be))
            H[parent] = H[parent] * batch.masks[parent][:, None]
        return H

    def _global_readout(self, H: dict[str, ArrayLike], batch: TensorState) -> ArrayLike:
        """Invariant global summary (doc Q V4) — masked mean over domains+racks."""
        be, xp = self.be, self.be.xp
        gd = self.cfg.latent.global_
        parts = []
        for t in ("topology_domain", "rack", "workload"):
            h, m = H[t], batch.masks[t]
            denom = xp.maximum(xp.sum(m), 1.0)
            mean = xp.sum(h * m[:, None], axis=0) / denom
            parts.append(mean)
        cat = xp.concatenate(parts, axis=-1)
        # fold into the global width by truncation/pad (cheap, deterministic).
        if cat.shape[0] >= gd:
            return cat[:gd]
        pad = xp.zeros((gd - cat.shape[0],), dtype=be.float_dtype)
        return xp.concatenate([cat, pad], axis=-1)

    def _downward_broadcast(
        self, H: dict[str, ArrayLike], batch: TensorState
    ) -> dict[str, ArrayLike]:
        """One downward broadcast sweep (doc Q Eq. 5, P10–P11)."""
        be, xp = self.be, self.be.xp
        for child, parent in reversed(_UP_PAIRS):
            assign = batch.hierarchy.get((child, parent))
            if assign is None or H[child].shape[0] == 0:
                continue
            # B_{p->c} = P_{c->p}ᵀ · H_p  (each child gets its parent's latent)
            broadcast = xp.matmul(assign.swapaxes(-1, -2), H[parent])  # (n_c, d_p)
            upd_in = xp.concatenate([H[child], broadcast], axis=-1)
            H[child] = self.broadcast_update[child](H[child], upd_in, be)
            H[child] = H[child] * batch.masks[child][:, None]
        return H

    def candidate_scores(
        self, H: dict[str, ArrayLike], batch: TensorState
    ) -> ArrayLike:
        """Tiny workload↔domain candidate attention (doc Q Eq. 6–7, V5).

        Returns a (J, D) score matrix masked by the exact-kernel feasible
        candidate mask ``F_WD`` so the learned block can only re-weight inside
        the kernel-certified safe set — never expand it (the project spine).
        """
        be, xp = self.be, self.be.xp
        wl, dom = H["workload"], H["topology_domain"]
        if wl.shape[0] == 0 or dom.shape[0] == 0:
            return xp.zeros((wl.shape[0], dom.shape[0]), dtype=be.float_dtype)
        q = self.attn_q(wl, be)  # (J, attn)
        k = self.attn_k(dom, be)  # (D, attn)
        scale = 1.0 / np.sqrt(float(self.cfg.attn_dim))
        logits = xp.matmul(q, k.swapaxes(-1, -2)) * scale  # (J, D)
        fmask = batch.candidate_mask  # (J, D) in {0,1}
        logits = logits + (fmask - 1.0) * 1e9  # mask infeasible domains
        return be.softmax(logits, axis=-1) * fmask

    def to_backend(self, be: Backend) -> RelationalWorldModel:
        """Return a copy of this model with all params moved to ``be``."""
        if be.name == self.be.name:
            return self
        conv = self.be.to_numpy

        def cd(d: Dense) -> Dense:
            return Dense(be.from_numpy(conv(d.W)), be.from_numpy(conv(d.b)))

        def cg(g: GRUCell) -> GRUCell:
            return GRUCell(
                be.from_numpy(conv(g.Wz)),
                be.from_numpy(conv(g.Wr)),
                be.from_numpy(conv(g.Wh)),
            )

        return RelationalWorldModel(
            cfg=self.cfg,
            be=be,
            encoders={k: cd(v) for k, v in self.encoders.items()},
            fast_core=cg(self.fast_core),
            slow_core=cg(self.slow_core),
            relation_msg={k: cd(v) for k, v in self.relation_msg.items()},
            relation_update={k: cg(v) for k, v in self.relation_update.items()},
            pool_proj={k: cd(v) for k, v in self.pool_proj.items()},
            broadcast_update={k: cg(v) for k, v in self.broadcast_update.items()},
            action_embed=be.from_numpy(conv(self.action_embed)),
            attn_q=cd(self.attn_q),
            attn_k=cd(self.attn_k),
            attn_v=cd(self.attn_v),
        )


# Latent-bearing entity types (subset of ENTITY_TYPES that get learned latents).
# resource_flavor / power_domain / cooling_zone / operator_policy / local_queue /
# cluster_queue collapse into the entities we keep latents for; we keep one
# "queue" latent type and one "topology_domain" type matching DEFAULT_LATENT_DIMS.
ENTITY_TYPES_LAT: tuple[str, ...] = (
    "workload",
    "podset",
    "queue",
    "topology_domain",
    "node",
    "gpu",
    "rack",
    "switch",
    "global",
)

# Pull the schema-named child->parent hierarchy pairs into latent-type names.
_UP_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (c, ("topology_domain" if p == "topology_domain" else p)) for c, p in HIERARCHY_UP
)


# --------------------------------------------------------------------------- #
# State -> tensor feature extraction (the bridge from WorldState to arrays).
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class TensorState:
    """Padded per-entity features + masks + incidence for one :class:`WorldState`.

    This is the model-side realization of :class:`schema.TensorBatch`: it holds
    the actual backend arrays. ``incidence`` keys are (src_type, dst_type) and
    values are dense one-hot scatter matrices (n_dst, n_src); ``hierarchy`` are
    the natural child→parent assignments for PNA pooling; ``candidate_mask`` is
    the exact-kernel feasible (workload, domain) mask.
    """

    caps: ShapeCaps
    features: dict[str, ArrayLike]
    masks: dict[str, ArrayLike]
    incidence: dict[tuple[str, str], ArrayLike]
    hierarchy: dict[tuple[str, str], ArrayLike]
    candidate_mask: ArrayLike
    order: dict[str, list[str]]  # entity_type -> ordered uid list (for read-out)


def _wl_features(w: Workload) -> list[float]:
    gpu_req = sum(ps.count * ps.gpu_req for ps in w.podsets)
    status_run = 1.0 if w.status == WorkloadStatus.RUNNING else 0.0
    status_pend = 1.0 if w.status == WorkloadStatus.PENDING else 0.0
    comm_heavy = 1.0 if w.comm_class == CommClass.ALLREDUCE_HEAVY else 0.0
    return [
        float(w.priority),
        float(gpu_req),
        float(len(w.podsets)),
        status_run,
        status_pend,
        comm_heavy,
        float(w.age_s) / 3600.0,
        float(w.elapsed_run_s) / 3600.0,
        float(w.est_remaining_s) / 3600.0,
        float(w.ready_fraction),
        float(w.requeue_count),
        float(w.max_exec_s) / 3600.0,
        1.0 if w.admitted_domain_id else 0.0,
        1.0 if w.admitted_flavor_id else 0.0,
    ]


def _node_features(n: Any) -> list[float]:
    return [
        float(n.gpu_cap),
        float(n.cpu_cap) / 128.0,
        float(n.mem_cap_gb) / 1024.0,
        float(n.cpu_alloc) / max(n.cpu_cap, 1.0),
        float(n.mem_alloc_gb) / max(n.mem_cap_gb, 1.0),
        float(n.running_pods),
        1.0 if n.ready else 0.0,
        1.0 if n.schedulable else 0.0,
        1.0 if n.node_health == NodeHealth.READY else 0.0,
        1.0 if n.node_health == NodeHealth.DRAINING else 0.0,
    ]


def _gpu_features(g: Any) -> list[float]:
    return [
        float(g.util),
        float(g.mem_util),
        float(g.temp_c) / 100.0,
        float(g.memory_temp_c) / 100.0,
        float(g.power_w) / 1000.0,
        1.0 if g.allocated_to else 0.0,
        float(g.xid_code > 0),
        1.0 if g.health_state == DeviceHealth.HEALTHY else 0.0,
        1.0 if g.health_state == DeviceHealth.DEGRADED else 0.0,
        1.0 if g.health_state == DeviceHealth.INTERMITTENTLY_FAULTY else 0.0,
        1.0 if g.health_state == DeviceHealth.MASKED else 0.0,
        float(g.memory_total_mb) / 100000.0,
    ]


class FeatureBuilder:
    """Builds a :class:`TensorState` from a :class:`WorldState` deterministically.

    Ordering of entities is the sorted uid order so the forward map is stable
    and the masks zero out padding positions (doc Q A1). The builder is exact /
    auditable: every feature traces back to a named schema field (doc C).
    """

    def __init__(self, backend: Backend | None = None) -> None:
        self.be = backend or get_backend()

    def build(self, state: WorldState, caps: ShapeCaps | None = None) -> TensorState:
        be, _xp = self.be, self.be.xp
        X = state.symbolic
        caps = caps or pick_bucket(len(X.workloads), len(X.nodes), len(X.gpus))
        order: dict[str, list[str]] = {}

        def pack(uids: list[str], rows: list[list[float]], cap: int, dim: int):
            n = min(len(uids), cap)
            arr = np.zeros((cap, dim), dtype=np.float32)
            mask = np.zeros((cap,), dtype=np.float32)
            for i in range(n):
                arr[i, : len(rows[i])] = rows[i][:dim]
                mask[i] = 1.0
            return be.from_numpy(arr), be.from_numpy(mask), uids[:n]

        features: dict[str, ArrayLike] = {}
        masks: dict[str, ArrayLike] = {}

        wl_uids = sorted(X.workloads)
        f, m, ordered = pack(
            wl_uids,
            [_wl_features(X.workloads[u]) for u in wl_uids],
            caps.J_max,
            RAW_FEATURE_DIMS["workload"],
        )
        features["workload"], masks["workload"], order["workload"] = f, m, ordered

        ps_uids = sorted(X.podsets)
        ps_rows = [
            [
                float(p.count),
                float(p.gpu_req),
                float(p.ready_count),
                float(p.running_count),
                float(p.pending_count),
                float(p.cpu_req) / 128.0,
                float(p.mem_req_gb) / 1024.0,
                1.0 if p.podset_group else 0.0,
            ]
            for p in (X.podsets[u] for u in ps_uids)
        ]
        f, m, ordered = pack(
            ps_uids, ps_rows, caps.P_max * caps.J_max, RAW_FEATURE_DIMS["podset"]
        )
        features["podset"], masks["podset"], order["podset"] = f, m, ordered

        cq_uids = sorted(X.cluster_queues)
        cq_rows = [
            [
                float(c.fairshare_weight),
                float(c.reserved_gpu),
                float(c.borrowed_gpu),
                float(c.pending_gpu),
                float(c.weighted_share),
                float(c.active_workloads),
                1.0 if c.preemption_policy != "never" else 0.0,
                float(len(c.flavor_order)),
            ]
            for c in (X.cluster_queues[u] for u in cq_uids)
        ]
        f, m, ordered = pack(cq_uids, cq_rows, caps.Q_max, RAW_FEATURE_DIMS["queue"])
        features["queue"], masks["queue"], order["queue"] = f, m, ordered

        dm_uids = sorted(X.domains)
        dm_rows = [
            [
                float(d.free_gpu),
                float(d.free_cpu) / 128.0,
                float(d.free_mem_gb) / 1024.0,
                float(d.headroom_score),
                float(d.hotspot_score),
                1.0 if d.parent_domain_id else 0.0,
            ]
            for d in (X.domains[u] for u in dm_uids)
        ]
        f, m, ordered = pack(
            dm_uids, dm_rows, caps.D_max, RAW_FEATURE_DIMS["topology_domain"]
        )
        features["topology_domain"] = f
        masks["topology_domain"] = m
        order["topology_domain"] = ordered

        nd_uids = sorted(X.nodes)
        f, m, ordered = pack(
            nd_uids,
            [_node_features(X.nodes[u]) for u in nd_uids],
            caps.N_max,
            RAW_FEATURE_DIMS["node"],
        )
        features["node"], masks["node"], order["node"] = f, m, ordered

        gpu_uids = sorted(X.gpus)
        f, m, ordered = pack(
            gpu_uids,
            [_gpu_features(X.gpus[u]) for u in gpu_uids],
            caps.G_max,
            RAW_FEATURE_DIMS["gpu"],
        )
        features["gpu"], masks["gpu"], order["gpu"] = f, m, ordered

        rk_uids = sorted(X.racks)
        rk_rows = [
            [
                float(r.designed_gpu_cap),
                float(r.rack_power_kw) / 100.0,
                float(r.rack_headroom_kw) / 100.0,
                float(r.inferred_thermal_margin_c) / 20.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            for r in (X.racks[u] for u in rk_uids)
        ]
        f, m, ordered = pack(rk_uids, rk_rows, caps.R_max, RAW_FEATURE_DIMS["rack"])
        features["rack"], masks["rack"], order["rack"] = f, m, ordered

        sw_uids = sorted(X.switches)
        sw_rows = [
            [
                float(s.port_util_p95),
                float(s.egress_hotspot_score),
                1.0 if s.tier == "tor" else 0.0,
                0.0,
            ]
            for s in (X.switches[u] for u in sw_uids)
        ]
        f, m, ordered = pack(sw_uids, sw_rows, caps.S_max, RAW_FEATURE_DIMS["switch"])
        features["switch"], masks["switch"], order["switch"] = f, m, ordered

        # global gets a single padded row of context features.
        c = state.context
        gfeat = np.zeros((1, RAW_FEATURE_DIMS["global"]), dtype=np.float32)
        gfeat[0, 0] = float(c.ambient_temp_c) / 30.0
        gfeat[0, 1] = 1.0 if c.regime == "normal" else 0.0
        gfeat[0, 2] = 1.0 if c.regime == "conservative" else 0.0
        gfeat[0, 3] = 1.0 if c.regime == "maintenance" else 0.0
        gfeat[0, 4] = 1.0 if c.regime == "incident" else 0.0
        gfeat[0, 5] = float(len(c.manual_drains))
        gfeat[0, 6] = float(len(X.workloads)) / 64.0
        gfeat[0, 7] = float(len(X.gpus)) / 256.0
        features["global"] = be.from_numpy(gfeat)
        masks["global"] = be.from_numpy(np.ones((1,), dtype=np.float32))
        order["global"] = ["global"]

        hierarchy = self._build_hierarchy(state, order, caps)
        incidence = self._build_incidence(state, order, hierarchy, caps)
        candidate_mask = self._build_candidate_mask(state, order, caps)

        return TensorState(
            caps=caps,
            features=features,
            masks=masks,
            incidence=incidence,
            hierarchy=hierarchy,
            candidate_mask=candidate_mask,
            order=order,
        )

    def _index(self, order: dict[str, list[str]], t: str) -> dict[str, int]:
        return {u: i for i, u in enumerate(order[t])}

    def _build_hierarchy(self, state, order, caps) -> dict:
        be = self.be
        X = state.symbolic
        gpu_idx = self._index(order, "gpu")
        node_idx = self._index(order, "node")
        rack_idx = self._index(order, "rack")
        dom_idx = self._index(order, "topology_domain")
        H: dict[tuple[str, str], Any] = {}

        # gpu -> node
        A = np.zeros((caps.N_max, caps.G_max), dtype=np.float32)
        for u, g in X.gpus.items():
            if u in gpu_idx and g.node_uid in node_idx:
                A[node_idx[g.node_uid], gpu_idx[u]] = 1.0
        H[("gpu", "node")] = be.from_numpy(A)

        # node -> rack
        A = np.zeros((caps.R_max, caps.N_max), dtype=np.float32)
        for u, n in X.nodes.items():
            if u in node_idx and n.rack_id in rack_idx:
                A[rack_idx[n.rack_id], node_idx[u]] = 1.0
        H[("node", "rack")] = be.from_numpy(A)

        # rack -> topology_domain (rack's block maps to a domain id)
        A = np.zeros((caps.D_max, caps.R_max), dtype=np.float32)
        for u, r in X.racks.items():
            if u in rack_idx and r.block_id in dom_idx:
                A[dom_idx[r.block_id], rack_idx[u]] = 1.0
        H[("rack", "topology_domain")] = be.from_numpy(A)
        return H

    def _build_incidence(self, state, order, hierarchy, caps) -> dict:
        """Typed relation scatter matrices (n_dst, n_src) at padded caps.

        Incidence matrices are built at the *padded* bucket dimensions (doc Q A2)
        so that ``T_rᵀ·M_r`` aligns with the padded latent tensors; padding rows
        are all-zero and contribute nothing under the masked update.
        """
        be = self.be
        X = state.symbolic
        inc: dict[tuple[str, str], Any] = dict(hierarchy)  # reuse hierarchy edges
        wl_idx = self._index(order, "workload")
        node_idx = self._index(order, "node")
        # workload -> node (RUNS_ON): connect a running workload to the nodes
        # whose GPUs it currently owns (an exact, auditable assignment).
        A = np.zeros((caps.N_max, caps.J_max), dtype=np.float32)
        for g in X.gpus.values():
            w = g.allocated_to
            if w in wl_idx and g.node_uid in node_idx:
                A[node_idx[g.node_uid], wl_idx[w]] = 1.0
        inc[("workload", "node")] = be.from_numpy(A)
        return inc

    def _build_candidate_mask(self, state, order, caps) -> ArrayLike:
        """Feasible (workload, domain) mask ``F_WD`` (doc Q Eq. 6).

        Without the exact kernel in this module we conservatively allow a
        pending workload to consider any non-protected domain with free GPUs ≥
        its demand; the planner replaces this with the kernel-certified set. The
        learned block may only re-weight inside this set, never expand it.
        """
        be = self.be
        X = state.symbolic
        protected = set(X.operator_policy.protected_domains)
        mask = np.zeros((caps.J_max, caps.D_max), dtype=np.float32)
        dom_idx = self._index(order, "topology_domain")
        for wi, wu in enumerate(order["workload"]):
            w = X.workloads[wu]
            if w.status not in (WorkloadStatus.PENDING, WorkloadStatus.RESERVING):
                continue
            demand = sum(ps.count * ps.gpu_req for ps in w.podsets)
            for du, di in dom_idx.items():
                if du in protected:
                    continue
                if X.domains[du].free_gpu >= demand:
                    mask[wi, di] = 1.0
        return be.from_numpy(mask)


__all__ = [
    "ACTION_INDEX",
    "ENTITY_TYPES_LAT",
    "FAST_TYPES",
    "N_ACTIONS",
    "RAW_FEATURE_DIMS",
    "SLOW_TYPES",
    "ArchConfig",
    "Dense",
    "FeatureBuilder",
    "GRUCell",
    "RelationalWorldModel",
    "TensorState",
]
