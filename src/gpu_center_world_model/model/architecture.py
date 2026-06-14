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

from collections.abc import Sequence
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
    # --- multi-rate temporal cores (doc E temporal cores / doc J B3–B4) ------ #
    # The fast core compresses ``T_fast`` telemetry inputs per slow step; the
    # slow core carries the planning-horizon memory over ``T_slow`` steps. Used
    # by the sequence temporal encoder and the free-running predictor rollout.
    t_fast: int = 24  # fast-telemetry inputs folded into each slow step (doc J B3)
    t_slow: int = 12  # slow/meso steps that form the outer BPTT clock (doc J B3)
    # --- predictor / world-transition (doc J D6, doc E WorldTransition) ------ #
    # ``u_dim`` is the width of the exogenous context vector u_h (missingness /
    # staleness indicators) fed to the predictor at every rollout step (doc J D6).
    u_dim: int = 4


# PNA reducer fan-out: sum/mean/std/max are each child-width, plus a single
# scalar log1p(count) (doc Q Agt5). So a pool over child-width ``cd`` yields
# ``4*cd + 1`` features.
_PNA_VEC = 4  # sum, mean, std, max


def _pna_width(cd: int) -> int:
    return _PNA_VEC * cd + 1


# --------------------------------------------------------------------------- #
# Typed relation set (doc Q D1–D4) and exact-semantic shortcut relations (R5).
# --------------------------------------------------------------------------- #
# Each typed relation r is the directed pair (src_type, dst_type). Message
# passing keys *per relation* (doc Q D2: W_r^src, W_r^dst, φ_r, η_r) rather than
# per destination type, so a switch receiving from a node and a rack receiving
# from a node learn distinct edge functions. The base typed relations come from
# the hierarchy edges plus the running-workload assignment; the R5 set adds the
# exact semantic shortcuts that let the switch latent and power/cooling coupling
# participate without opaque learned rewiring (doc Q R5/V6).
# Each relation is (name, src_latent_type, dst_latent_type). The *name* keys the
# learned per-relation edge functions (so node→switch and node→rack are distinct
# even when both gather a node source); the latent types name the actual latent
# blocks gathered. Shortcut relations whose schema destination is not a
# latent-bearing type (power_domain / cooling_zone / candidate_domain) are routed
# to the latent block that carries that semantics: power/cooling coupling becomes
# a rack↔rack sibling edge; the candidate edge connects a domain source to the
# querying workload (doc Q R5/V6: exact semantic shortcuts, no learned rewiring).
RELATIONS: tuple[tuple[str, str, str], ...] = (
    # --- hierarchy / assignment relations (upward, doc Q D1) ----------------- #
    ("gpu->node", "gpu", "node"),
    ("node->rack", "node", "rack"),
    ("rack->topology_domain", "rack", "topology_domain"),
    ("workload->node", "workload", "node"),
    # --- R5 exact-semantic shortcut relations (doc Q R5) -------------------- #
    ("node->switch", "node", "switch"),  # CONNECTED_TO: switch latent participates
    ("switch->node", "switch", "node"),  # reverse: egress hotspots reach nodes
    ("rack->power_domain", "rack", "rack"),  # POWERED_BY: rack↔rack shared power
    ("rack->cooling_zone", "rack", "rack"),  # COOLED_BY: rack↔rack shared cooling
    ("candidate_domain->workload", "topology_domain", "workload"),  # feasible cand.
)


def _relation_key(name: str) -> str:
    """Stable string key for a typed relation (for the per-relation param dicts)."""
    return name


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
# The world-transition predictor P_θ (doc J D6, doc E WorldTransition).
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class WorldTransition:
    """The free-running residual transition predictor ``P_θ`` (doc J D6).

    Realizes the deployed recursive core

        ``ẑ_{h+1} = P_θ(ẑ_h, e_a(a_h), ψ(s^K_{h+1}), u_h)``    (doc J D6)

    as a **per-type gated GRU update**: for every entity type ``τ`` a row-shared
    GRU consumes the current latent ``ẑ_{h,τ}`` as the hidden state and an input
    assembled from the action embedding ``e_a(a_h)``, a stop-gradient encoding of
    the exact post-action symbolic state ``ψ(s^K_{h+1})`` (the kernel-certified
    next state, broadcast per type), and the exogenous context ``u_h``
    (missingness / staleness indicators). The kernel term enters as a constant in
    the gradient (doc J Eqs. 20–22, T4: ``∂K/∂θ = 0``), which is what makes the
    supervised world-model gradient exact rather than approximate.

    The update is a residual gate, so a zero action / zero kernel-delta leaves
    the latent close to identity (doc Q R15 strong identity shortcut), while a
    distinct action perturbs every type's latent — the action-sensitivity the
    branch-difference loss supervises (doc J D5/Eq. 9).
    """

    cfg: ArchConfig
    be: Backend
    cells: dict[str, GRUCell]  # per-type transition GRU
    psi_proj: dict[str, Dense]  # ψ(s^K) -> per-type kernel-state summary
    psi_dim: int

    @classmethod
    def build(
        cls,
        cfg: ArchConfig,
        lat: dict[str, int],
        rng: np.random.Generator,
        be: Backend,
    ) -> WorldTransition:
        psi_dim = cfg.message_dim  # width of the kernel-state summary ψ(s^K)

        def dense(out_dim: int, in_dim: int) -> Dense:
            W, b = _init_dense(rng, out_dim, in_dim)
            return Dense(be.from_numpy(W), be.from_numpy(b))

        def gru(hidden: int, in_dim: int) -> GRUCell:
            Wz, Wr, Wh = _init_gru(rng, hidden, in_dim)
            return GRUCell(be.from_numpy(Wz), be.from_numpy(Wr), be.from_numpy(Wh))

        cells: dict[str, GRUCell] = {}
        psi_proj: dict[str, Dense] = {}
        for t in ENTITY_TYPES_LAT:
            d = lat[t]
            # input = [e_a(a_h), ψ_τ(s^K), u_h]
            in_dim = cfg.action_embed_dim + psi_dim + cfg.u_dim
            cells[t] = gru(d, in_dim)
            # ψ summary projects the (stop-grad) kernel-state feature vector to a
            # per-type width; its raw input is a fixed exact summary (see step).
            psi_proj[t] = dense(psi_dim, _PSI_RAW_DIM)
        return cls(cfg=cfg, be=be, cells=cells, psi_proj=psi_proj, psi_dim=psi_dim)

    def step(
        self,
        z: dict[str, ArrayLike],
        action_emb: ArrayLike,
        psi: ArrayLike,
        u: ArrayLike,
        masks: dict[str, ArrayLike],
    ) -> dict[str, ArrayLike]:
        """One free-running predictor step ``ẑ_h -> ẑ_{h+1}`` (doc J D6).

        ``action_emb`` is ``e_a(a_h)`` (action_embed_dim,), ``psi`` is the raw
        exact kernel-state summary ``ψ(s^K_{h+1})`` (already stop-gradient'd by
        the caller, ``_PSI_RAW_DIM``,), and ``u`` is the exogenous context
        ``u_h`` (u_dim,). Returns the next-step per-type latent dict; types not
        present in ``z`` are passed through unchanged.
        """
        be, xp = self.be, self.be.xp
        out: dict[str, ArrayLike] = {}
        for t, h in z.items():
            cell = self.cells.get(t)
            if cell is None:
                out[t] = h
                continue
            n = h.shape[0]
            psi_t = xp.tanh(self.psi_proj[t](psi[None, :], be))[0]  # (psi_dim,)
            a_row = xp.broadcast_to(action_emb[None, :], (n, action_emb.shape[0]))
            psi_row = xp.broadcast_to(psi_t[None, :], (n, psi_t.shape[0]))
            u_row = xp.broadcast_to(u[None, :], (n, u.shape[0]))
            x = xp.concatenate([a_row, psi_row, u_row], axis=-1)
            nxt = cell(h, x, be)  # GRU: hidden=ẑ_h, input=[e_a, ψ, u]
            m = masks.get(t)
            out[t] = nxt * m[:, None] if m is not None else nxt
        return out

    def to_backend(self, be: Backend) -> WorldTransition:
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

        return WorldTransition(
            cfg=self.cfg,
            be=be,
            cells={k: cg(v) for k, v in self.cells.items()},
            psi_proj={k: cd(v) for k, v in self.psi_proj.items()},
            psi_dim=self.psi_dim,
        )


# Width of the raw exact kernel-state summary ψ(s^K) fed to the predictor. The
# summary is a small fixed, auditable vector of exact post-action symbolic counts
# (free-GPU pressure, pending demand, protected-domain count, hotspot/headroom
# aggregates) — see :func:`kernel_state_summary` (doc J D6 ``ψ`` / T4 constants).
_PSI_RAW_DIM = 8


def kernel_state_summary(
    state: WorldState, be: Backend
) -> ArrayLike:
    """Exact post-action symbolic-state summary ``ψ(s^K)`` (doc J D6, T4).

    A small, auditable vector of exact counts the kernel emits — these are the
    constants the predictor conditions on (``∂K/∂θ = 0``, doc J Eqs. 20–22). The
    caller stop-gradients this before feeding it to :meth:`WorldTransition.step`.
    """
    X = state.symbolic
    doms = list(X.domains.values())
    n_dom = max(len(doms), 1)
    free_gpu = sum(d.free_gpu for d in doms)
    headroom = sum(float(d.headroom_score) for d in doms) / n_dom
    hotspot = sum(float(d.hotspot_score) for d in doms) / n_dom
    pending = sum(
        sum(ps.count * ps.gpu_req for ps in w.podsets)
        for w in X.workloads.values()
        if w.status in (WorkloadStatus.PENDING, WorkloadStatus.RESERVING)
    )
    protected = len(set(X.operator_policy.protected_domains))
    n_pending = sum(
        1
        for w in X.workloads.values()
        if w.status in (WorkloadStatus.PENDING, WorkloadStatus.RESERVING)
    )
    vec = np.array(
        [
            float(free_gpu) / 64.0,
            float(pending) / 64.0,
            headroom,
            hotspot,
            float(protected),
            float(n_pending) / 16.0,
            float(len(X.gpus)) / 256.0,
            float(n_dom) / 16.0,
        ],
        dtype=np.float32,
    )
    return be.from_numpy(vec)


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
    # ``relation_msg`` is retained as a per-destination-type message Dense for
    # checkpoint compatibility (serialization flattens it); the *active* forward
    # path uses the typed per-relation blocks below (doc Q D2). Kept as a stable
    # field so existing serialized checkpoints round-trip unchanged.
    relation_msg: dict[str, Dense]
    # --- typed per-relation message blocks (doc Q D2: W_r^src/W_r^dst/φ_r/η_r) #
    rel_src: dict[str, Dense]  # relation key -> W_r^src  (gathered source U_r)
    rel_dst: dict[str, Dense]  # relation key -> W_r^dst  (gathered target V_r)
    rel_phi: dict[str, Dense]  # relation key -> φ_r  (LN'd concat -> message_dim)
    rel_gate: dict[str, Dense]  # relation key -> η_r edge-gate logit (scalar)
    relation_update: dict[str, GRUCell]  # type -> gated node update (doc Q D4)
    pool_proj: dict[str, Dense]  # parent-type -> PNA pool projection
    broadcast_update: dict[str, GRUCell]  # child-type -> downward-broadcast update
    action_embed: ArrayLike  # (N_ACTIONS, action_embed_dim)
    attn_q: Dense  # workload query for candidate attention
    attn_k: Dense  # domain key for candidate attention
    attn_v: Dense  # domain value for candidate attention
    # --- global PNA readout + FiLM downward rebroadcast (doc Q V4/R6) -------- #
    global_readout: Dense  # PNA-pooled all-type summary -> global latent
    film_gen: Dense  # global latent -> per-child (γ, β) FiLM modulation (R6/V4)
    # --- world-transition predictor P_θ (doc J D6, doc E WorldTransition) ---- #
    predictor: WorldTransition | None = None

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

        # --- Typed per-relation message blocks (doc Q D2) ------------------- #
        # For each relation r = (σ, τ): U_r = S_r·H_σ projected by W_r^src, V_r =
        # T_r·H_τ projected by W_r^dst, φ_r maps the LN'd concat
        # [W_r^src U_r, W_r^dst V_r, W_r^edge X_r, A_r(a), G_r(g)] -> message_dim,
        # and η_r is a per-edge scalar gate (sigmoid). N_r = T_r^⊤ η_r normalizes.
        rel_src: dict[str, Dense] = {}
        rel_dst: dict[str, Dense] = {}
        rel_phi: dict[str, Dense] = {}
        rel_gate: dict[str, Dense] = {}
        proj_w = cfg.message_dim
        for name, src_lat, dst_lat in RELATIONS:
            key = _relation_key(name)
            sd = lat[src_lat]
            dd = lat[dst_lat]
            rel_src[key] = dense(proj_w, sd)  # W_r^src
            rel_dst[key] = dense(proj_w, dd)  # W_r^dst
            # φ_r input: [W_src U, W_dst V, edge_scalar, A_r(a), G_r(g)]; the
            # edge feature here is the exact scalar X_r (1 col) and A_r/G_r are
            # the row-shared action/global FiLM summaries (doc Q D2/A4).
            phi_in = 2 * proj_w + 1 + cfg.action_embed_dim + lat["global"]
            rel_phi[key] = dense(cfg.message_dim, phi_in)
            rel_gate[key] = dense(1, phi_in)  # η_r edge-gate logit

        # Legacy per-destination-type message Dense, retained for checkpoint
        # compatibility (serialization flattens ``relation_msg``). Not used by the
        # active forward path — the typed per-relation blocks above supersede it.
        relation_msg: dict[str, Dense] = {}
        for t in ENTITY_TYPES_LAT:
            d = lat[t]
            relation_msg[t] = dense(cfg.message_dim, d + cfg.message_dim)

        # Per-destination-type gated node update (doc Q D4): consumes the type
        # summary C_τ = [Ψ_τ(H_τ), Σ_r, μ_r, log(1+N_r), A_τ(a), G_τ(g)] folded
        # into a fixed per-type width (mean message + count + action + global).
        relation_update: dict[str, GRUCell] = {}
        for t in ENTITY_TYPES_LAT:
            d = lat[t]
            upd_in = d + cfg.message_dim + 1 + cfg.action_embed_dim + lat["global"]
            relation_update[t] = gru(d, upd_in)

        # Hierarchical PNA pools (gpu->node->rack->domain), child width -> parent.
        pool_proj: dict[str, Dense] = {}
        broadcast_update: dict[str, GRUCell] = {}
        for child, parent in _UP_PAIRS:
            cd, pd = lat[child], lat[parent]
            pool_proj[parent] = dense(pd, pd + _pna_width(cd))
            # downward broadcast input: child latent + parent latent + FiLM (2*cd)
            broadcast_update[child] = gru(cd, cd + pd + 2 * cd)

        action_embed = be.from_numpy(
            rng.normal(0, 0.1, size=(N_ACTIONS, cfg.action_embed_dim)).astype(
                np.float32
            )
        )
        attn_q = dense(cfg.attn_dim, lat["workload"])
        attn_k = dense(cfg.attn_dim, lat["topology_domain"])
        attn_v = dense(lat["workload"], lat["topology_domain"])

        # --- Global PNA readout (doc Q V4/Agt5) + FiLM generator (R6) -------- #
        # The global latent is built from a PNA-style sum/mean/std/max/count
        # reducer over *every* entity type (each masked-pooled to its own width)
        # concatenated, then a learned Dense to the global width. The FiLM
        # generator emits a (γ, β) pair per child type for the downward
        # rebroadcast; we size it to the max child width and slice per type.
        readout_in = sum(_pna_width(lat[t]) for t in ENTITY_TYPES_LAT if t != "global")
        global_readout = dense(lat["global"], readout_in)
        max_child = max(lat[t] for t in ENTITY_TYPES_LAT)
        film_gen = dense(2 * max_child, lat["global"])

        predictor = WorldTransition.build(cfg, lat, rng, be)

        return cls(
            cfg=cfg,
            be=be,
            encoders=encoders,
            fast_core=fast_core,
            slow_core=slow_core,
            relation_msg=relation_msg,
            rel_src=rel_src,
            rel_dst=rel_dst,
            rel_phi=rel_phi,
            rel_gate=rel_gate,
            relation_update=relation_update,
            pool_proj=pool_proj,
            broadcast_update=broadcast_update,
            action_embed=action_embed,
            attn_q=attn_q,
            attn_k=attn_k,
            attn_v=attn_v,
            global_readout=global_readout,
            film_gen=film_gen,
            predictor=predictor,
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

        TemporalEncode → RelFuse (two upward sweeps + one downward broadcast) →
        candidate-attention fusion. ``action_name`` selects the action embedding
        broadcast through every update via FiLM-style concatenation (doc Q A4:
        row-shared, never slot-specific). Returns a dict
        ``{entity_type: latent[n, d]}`` whose per-type keys are byte-stable for
        the serving contract; the candidate-attention context is fused *into* the
        workload / domain latent blocks the heads read (not added as a new key).

        Pass ``batch.windows`` (multi-rate observation sequences) to drive the
        fast/slow GRUs as genuine recurrences; with a single snapshot the cores
        run one recurrent step from a zero hidden state (still gating/normalising).
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

        # 2) Temporal cores: drive the fast/slow GRUs over the multi-rate
        # observation windows (doc J B3/B4). When windows are present the fast
        # core consumes T_fast telemetry steps, compressing them into each of the
        # T_slow slow-core steps; otherwise a single recurrent step is run.
        for t in FAST_TYPES:
            H[t] = self._temporal_seq(t, H[t], self.fast_core, batch, fast=True)
        for t in SLOW_TYPES:
            if t == "global":
                continue
            H[t] = self._temporal_seq(t, H[t], self.slow_core, batch, fast=False)

        # 3) Relational fusion: upward typed message passing -------------- #
        for _ in range(cfg.n_rounds):
            H = self._message_round(H, batch, a_emb)

        # 4) Hierarchical PNA pooling gpu->node->rack->domain ------------- #
        H = self._upward_pool(H, batch)

        # 5) Global PNA read-out + one FiLM downward broadcast ------------ #
        H["global"] = self._global_readout(H, batch)[None, :]  # keep (1, d)
        H = self._downward_broadcast(H, batch)

        # 6) Candidate attention: fuse P_domain / S_workload into the domain /
        # workload latents that the placement/action-value heads read (doc Q
        # Eq. 6–8, V5). The attention value path (attn_v) supplies the context,
        # not just the softmax weights, so the heads see a candidate-aware latent.
        H = self._fuse_candidate_context(H, batch, a_emb)
        return H

    # ---- the free-running predictor rollout (doc J D6/Eqs. 8–10, B1–B4) --- #
    def rollout(
        self,
        z0: dict[str, ArrayLike],
        action_name: str,
        masks: dict[str, ArrayLike],
        kernel_states: Sequence[ArrayLike] | None = None,
        exogenous: Sequence[ArrayLike] | None = None,
        horizon: int = 2,
        detach_every: int = 4,
    ) -> list[dict[str, ArrayLike]]:
        """Free-running ``H``-step latent rollout ``ẑ_1..ẑ_H`` (doc J D6, Eq. 8).

        Recurses the predictor ``P_θ`` from the encoded ``z0`` *without teacher
        forcing* (doc J B1): each step's output feeds the next step's input, so
        the returned latents ``ẑ_1..ẑ_H`` are genuinely distinct (this is the
        deployed recursive core, not the degenerate replication it replaces).

        Conditioning per step (doc J D6):

        * ``e_a(a_h)`` — the action embedding for ``action_name``;
        * ``ψ(s^K_{h+1})`` — the stop-gradient exact post-action kernel-state
          summary (``kernel_states[h]``; falls back to a zero vector when the
          caller does not supply the kernel rollout);
        * ``u_h`` — exogenous context (``exogenous[h]``; zero when absent).

        **Truncated BPTT** (doc J B3): every ``detach_every`` slow steps the
        carried latent is ``stop_gradient``'d, so the gradient window is bounded
        (``H_slow = 12`` with ``K = 4`` detaches in the v1 setting) while the
        rollout itself stays full-length and free-running.
        """
        be, xp = self.be, self.be.xp
        if self.predictor is None:
            # No predictor configured: degrade to repeating z0 (kept for safety;
            # the build path always constructs a predictor).
            return [dict(z0) for _ in range(max(horizon, 0))]
        a_idx = ACTION_INDEX.get(action_name, 0)
        a_emb = self.action_embed[a_idx]
        zero_psi = xp.zeros((_PSI_RAW_DIM,), dtype=be.float_dtype)
        zero_u = xp.zeros((self.cfg.u_dim,), dtype=be.float_dtype)
        traj: list[dict[str, ArrayLike]] = []
        z = dict(z0)
        for h in range(max(horizon, 0)):
            psi = zero_psi
            if kernel_states is not None and h < len(kernel_states):
                psi = be.stop_gradient(kernel_states[h])  # ∂K/∂θ = 0 (doc J T4)
            u = zero_u
            if exogenous is not None and h < len(exogenous):
                u = exogenous[h]
            z = self.predictor.step(z, a_emb, psi, u, masks)
            # truncated BPTT: detach the carried latent every K slow steps so the
            # gradient window is bounded (doc J B3); the rollout stays full-length.
            if detach_every > 0 and ((h + 1) % detach_every == 0):
                z = {t: be.stop_gradient(v) for t, v in z.items()}
            traj.append(dict(z))
        return traj

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

    def _temporal_seq(
        self,
        t: str,
        latent: ArrayLike,
        core: GRUCell,
        batch: TensorState,
        fast: bool,
    ) -> ArrayLike:
        """Run the temporal core over the multi-rate observation window (doc J B4).

        ``batch.windows[t]`` (when present) is a ``(T, n, raw)`` sequence of the
        same per-entity features at successive telemetry/scheduler ticks. The
        fast core (T_fast inputs) carries its hidden state across the fast lane
        and compresses each fast burst into a slow summary; the slow core
        (T_slow inputs) carries the planning-horizon memory. The recurrence is a
        genuine carried-hidden-state GRU, not a single degenerate step. With no
        window the single-snapshot one-step fallback (:meth:`_temporal`) is used.
        """
        be, xp = self.be, self.be.xp
        windows = getattr(batch, "windows", None)
        seq = None if windows is None else windows.get(t)
        if seq is None:
            return self._temporal(latent, core, batch.masks[t])
        d = latent.shape[-1]
        w = core.Wz.shape[0]
        mask = batch.masks[t]
        n = latent.shape[0]
        h = xp.zeros((n, w), dtype=be.float_dtype)
        steps = int(seq.shape[0])
        n_inner = self.cfg.t_fast if fast else self.cfg.t_slow
        # Encode each window step into latent space, then carry the GRU hidden
        # state across at most ``n_inner`` steps (doc J B3 T_fast/T_slow caps).
        for s in range(min(steps, n_inner)):
            xs = seq[s]  # (n, raw)
            a_row = xp.zeros((n, self.cfg.action_embed_dim), dtype=be.float_dtype)
            enc_in = xp.concatenate([xs, a_row], axis=-1)
            z = xp.tanh(self.encoders[t](enc_in, be))  # (n, d)
            if d < w:
                pad = xp.zeros((n, w - d), dtype=be.float_dtype)
                x = xp.concatenate([z, pad], axis=-1)
            else:
                x = z[:, :w]
            h = core(h, x, be)
        # Fuse the final encoded snapshot with the recurrent summary so the
        # carried temporal context augments (not replaces) the snapshot latent.
        out = 0.5 * (latent + h[:, :d])
        return out * mask[:, None]

    def _message_round(
        self, H: dict[str, ArrayLike], batch: TensorState, a_emb: ArrayLike
    ) -> dict[str, ArrayLike]:
        """One typed per-relation message-passing round (doc Q D1–D4, Eq. 1–3).

        For each typed relation ``r: σ→τ`` keyed by the *(src_type, dst_type)*
        pair (so a node→switch edge and a node→rack edge learn distinct edge
        functions), this computes:

        * **D1 gather:** ``U_r = S_r·H_σ`` (gather sources to edges via the dense
          assignment) and ``V_r = T_r·H_τ`` (gather destinations to edges).
        * **D2 typed edge message:** ``M_r = η_r ⊙ φ_r(LN([W_r^src U_r,
          W_r^dst V_r, W_r^edge X_r, A_r(a), G_r(g)]))`` with a row-shared edge
          MLP ``φ_r``, a sigmoid edge gate ``η_r``, the action FiLM ``A_r(a)``
          and global FiLM ``G_r(g)`` (doc Q A4).
        * **D3 aggregation:** ``Σ_r = T_r^⊤ M_r``, ``N_r = T_r^⊤ η_r``,
          ``μ_r = Σ_r / max(N_r, 1)`` — gate-weighted, not raw-count mean (R16:
          never mean-only without a count feature).
        * **D4 update:** the per-type GRU consumes the type summary
          ``[H_τ, μ_r, log(1+N_r), A_τ(a), G_τ(g)]``.

        Because every edge function is shared by relation type and every reducer
        is a masked scatter-add, the round is strictly per-type equivariant
        (doc Q G1–G2, Th. 1–3).
        """
        be, xp, cfg = self.be, self.be.xp, self.cfg
        g = H.get("global")
        if g is None:
            g = xp.zeros((cfg.latent.global_,), dtype=be.float_dtype)
        elif g.ndim == 2:  # global is stored as a single padded row (1, d)
            g = g[0]
        msg_dim = cfg.message_dim
        agg: dict[str, ArrayLike] = {
            t: xp.zeros((H[t].shape[0], msg_dim), dtype=be.float_dtype)
            for t in ENTITY_TYPES_LAT
        }
        gate_n: dict[str, ArrayLike] = {
            t: xp.zeros((H[t].shape[0], 1), dtype=be.float_dtype)
            for t in ENTITY_TYPES_LAT
        }
        relations = self._relations(batch)
        for name, src_t, dst_t in RELATIONS:
            if name not in relations:
                continue
            assign, edge_x = relations[name]
            key = _relation_key(name)
            if key not in self.rel_phi or dst_t not in agg:
                continue
            src, dst = H[src_t], H[dst_t]
            # D1: U_r/V_r gathered to the (n_dst, n_src) dense edge tensor. We
            # realise the edge axis as the destination rows (one aggregated edge
            # bundle per dst), gathering the source via the assignment matmul.
            n_dst = dst.shape[0]
            # gather mean source latent per destination row (S_r·H_σ scattered)
            cnt = xp.sum(assign, axis=-1, keepdims=True)  # (n_dst, 1) raw degree
            src_gather = xp.matmul(assign, src) / xp.maximum(cnt, 1.0)  # (n_dst, d_s)
            u_r = self.rel_src[key](src_gather, be)  # W_r^src U_r  (n_dst, msg)
            v_r = self.rel_dst[key](dst, be)  # W_r^dst V_r  (n_dst, msg)
            a_row = xp.broadcast_to(a_emb[None, :], (n_dst, a_emb.shape[0]))
            g_row = xp.broadcast_to(g[None, :], (n_dst, g.shape[0]))
            concat = xp.concatenate([u_r, v_r, edge_x, a_row, g_row], axis=-1)
            concat = self._layer_norm(concat)
            m_r = xp.tanh(self.rel_phi[key](concat, be))  # φ_r  (n_dst, msg)
            eta = be.sigmoid(self.rel_gate[key](concat, be))  # η_r  (n_dst, 1)
            has_edge = (cnt > 0.0).astype(be.float_dtype)  # zero-out empty dst
            eta = eta * has_edge
            agg[dst_t] = agg[dst_t] + eta * m_r  # Σ_r = T_r^⊤ M_r
            gate_n[dst_t] = gate_n[dst_t] + eta  # N_r = T_r^⊤ η_r
        # D3 normalize + D4 gated per-type update.
        for t in ENTITY_TYPES_LAT:
            mu = agg[t] / xp.maximum(gate_n[t], 1.0)  # μ_r
            log_n = xp.log1p(gate_n[t])  # log(1+N_r)
            n = H[t].shape[0]
            a_row = xp.broadcast_to(a_emb[None, :], (n, a_emb.shape[0]))
            g_row = xp.broadcast_to(g[None, :], (n, g.shape[0]))
            upd_in = xp.concatenate([H[t], mu, log_n, a_row, g_row], axis=-1)
            H[t] = self.relation_update[t](H[t], upd_in, be) * batch.masks[t][:, None]
        return H

    def _relations(
        self, batch: TensorState
    ) -> dict[str, tuple[ArrayLike, ArrayLike]]:
        """Resolve every typed relation to its (assign, edge_scalar) pair.

        Returns a dict keyed by the relation *name* mapping to a dense scatter
        ``assign`` (n_dst, n_src) and the per-destination exact edge scalar
        ``X_r`` (n_dst, 1) = the in-degree along that relation, an exact,
        auditable edge feature (doc Q D2 ``W_r^edge X_r``). The incidence is keyed
        by relation name in :attr:`TensorState.relations`; relations absent from
        the batch are skipped.
        """
        be, xp = self.be, self.be.xp
        out: dict[str, tuple[ArrayLike, ArrayLike]] = {}
        rel_inc = getattr(batch, "relations", None) or {}
        for name, _src_lat, _dst_lat in RELATIONS:
            assign = rel_inc.get(name)
            if assign is None:
                continue
            deg = xp.sum(assign, axis=-1, keepdims=True)  # (n_dst, 1) exact edge X_r
            out[name] = (assign, deg.astype(be.float_dtype))
        return out

    def _layer_norm(self, x: ArrayLike, eps: float = 1e-5) -> ArrayLike:
        """Row-wise LayerNorm over the trailing feature axis (doc Q D2 ``LN``)."""
        xp = self.be.xp
        mu = xp.mean(x, axis=-1, keepdims=True)
        var = xp.mean((x - mu) * (x - mu), axis=-1, keepdims=True)
        return (x - mu) / xp.sqrt(var + eps)

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
        """Invariant global summary via a PNA reducer over every type (doc Q V4).

        For each entity type the masked sum/mean/std/max/count reducer
        (doc Q Agt5) produces a ``4·d+1`` summary; the concatenation across all
        types is mapped by a learned Dense to the global width. This replaces the
        mean-only truncated read-out (R16: avoid repeated mean-only aggregation
        without count or variance features) and stays permutation-invariant
        because every reducer is a masked sum / mean / max over the entity axis
        (doc Q Th. 4).
        """
        be, xp = self.be, self.be.xp
        parts = []
        for t in ENTITY_TYPES_LAT:
            if t == "global":
                continue
            h, m = H[t], batch.masks[t]
            mcol = m[:, None]
            psum = xp.sum(h * mcol, axis=0)  # (d,)
            cnt = xp.maximum(xp.sum(m), 1.0)
            pmean = psum / cnt
            psq = xp.sum(h * h * mcol, axis=0) / cnt
            pstd = xp.sqrt(xp.maximum(psq - pmean * pmean, 0.0) + 1e-6)
            neg = (1.0 - m)[:, None] * (-1e9)
            pmax = xp.max(h * mcol + neg, axis=0)
            plog = xp.reshape(xp.log1p(xp.sum(m)), (1,))  # log1p(count) (Agt5)
            parts.append(xp.concatenate([psum, pmean, pstd, pmax, plog], axis=-1))
        cat = xp.concatenate(parts, axis=-1)  # (Σ_t 4·d_t+1,)
        return xp.tanh(self.global_readout(cat, be))

    def _downward_broadcast(
        self, H: dict[str, ArrayLike], batch: TensorState
    ) -> dict[str, ArrayLike]:
        """One FiLM downward broadcast sweep (doc Q Eq. 5, R6/V4, P10–P11).

        The global latent generates a row-shared FiLM modulation ``(γ, β)`` that
        rebroadcasts long-range context to every entity (doc Q R6); along the
        hierarchy each child additionally receives its parent latent
        ``B_{p→c} = P_{c→p}^⊤ H_p``. The per-child GRU consumes
        ``[H_c, B_{p→c}, γ_c, β_c]`` so the downward sweep is a gated residual.
        """
        be, xp = self.be, self.be.xp
        g = H["global"][0] if H["global"].ndim == 2 else H["global"]
        film = self.film_gen(g, be)  # (2·max_child,)
        half = film.shape[0] // 2
        gamma_all, beta_all = film[:half], film[half:]
        for child, parent in reversed(_UP_PAIRS):
            assign = batch.hierarchy.get((child, parent))
            if assign is None or H[child].shape[0] == 0:
                continue
            d = H[child].shape[-1]
            gamma = gamma_all[:d]
            beta = beta_all[:d]
            n = H[child].shape[0]
            gamma_row = xp.broadcast_to(gamma[None, :], (n, d))
            beta_row = xp.broadcast_to(beta[None, :], (n, d))
            # B_{p->c} = P_{c->p}ᵀ · H_p  (each child gets its parent's latent)
            broadcast = xp.matmul(assign.swapaxes(-1, -2), H[parent])  # (n_c, d_p)
            upd_in = xp.concatenate([H[child], broadcast, gamma_row, beta_row], axis=-1)
            H[child] = self.broadcast_update[child](H[child], upd_in, be)
            H[child] = H[child] * batch.masks[child][:, None]
        return H

    def candidate_scores(
        self,
        H: dict[str, ArrayLike],
        batch: TensorState,
        f_wd: ArrayLike | None = None,
    ) -> ArrayLike:
        """Tiny workload↔domain candidate attention (doc Q Eq. 6–7, V5).

        Returns a (J, D) score matrix masked by the exact-kernel feasible
        candidate mask ``F_WD`` so the learned block can only re-weight inside
        the kernel-certified safe set — never expand it (the project spine).
        ``f_wd`` overrides the builder's free-GPU heuristic with the
        kernel-certified mask when the planner supplies it (doc Q Eq. 6, V6).
        """
        be, xp = self.be, self.be.xp
        wl, dom = H["workload"], H["topology_domain"]
        if wl.shape[0] == 0 or dom.shape[0] == 0:
            return xp.zeros((wl.shape[0], dom.shape[0]), dtype=be.float_dtype)
        q = self.attn_q(wl, be)  # (J, attn)
        k = self.attn_k(dom, be)  # (D, attn)
        scale = 1.0 / np.sqrt(float(self.cfg.attn_dim))
        logits = xp.matmul(q, k.swapaxes(-1, -2)) * scale  # (J, D)
        fmask = batch.candidate_mask if f_wd is None else f_wd  # (J, D) in {0,1}
        logits = logits + (fmask - 1.0) * 1e9  # mask infeasible domains
        return be.softmax(logits, axis=-1) * fmask

    def _fuse_candidate_context(
        self,
        H: dict[str, ArrayLike],
        batch: TensorState,
        a_emb: ArrayLike,
        f_wd: ArrayLike | None = None,
    ) -> dict[str, ArrayLike]:
        """Fuse the candidate attention context into the workload/domain latents.

        Computes the attention weights ``Q_WD = F_WD ⊙ softmax(q·kᵀ)`` (Eq. 6),
        then the *value-weighted* candidate context (doc Q Eq. 7):

        * ``S_workload = Σ_D Q_WD · attn_v(domain)`` — each workload's context is
          the value-weighted mix of its feasible domains (uses ``attn_v`` to
          produce the context, not just the softmax weights);
        * ``P_domain = Agg_workload(Q_WD)`` — masked per-domain candidate
          pressure, projected back through ``attn_v`` so the domain latent carries
          its demand pressure.

        ``S_workload`` is added (residually) into ``H['workload']`` and the domain
        pressure into ``H['topology_domain']`` so the placement / action-value
        heads — which read those latent blocks — become candidate-aware without
        any change to the head signatures or the per-type key set.
        """
        be, xp = self.be, self.be.xp
        wl, dom = H["workload"], H["topology_domain"]
        if wl.shape[0] == 0 or dom.shape[0] == 0:
            return H
        q_wd = self.candidate_scores(H, batch, f_wd)  # (J, D) weights, Eq. 6
        val = self.attn_v(dom, be)  # (D, d_workload) value path, Eq. 7
        # S_workload: value-weighted candidate context per workload (Eq. 7).
        s_workload = xp.matmul(q_wd, val)  # (J, d_workload)
        H["workload"] = (H["workload"] + s_workload) * batch.masks["workload"][:, None]
        # P_domain: per-domain candidate pressure = column-sum of weights, then
        # mapped into domain width via a transpose of the value path (Eq. 7).
        pressure = xp.sum(q_wd, axis=0, keepdims=True)  # (1, D)
        # broadcast scalar pressure onto each domain's value vector, pool to dom
        dom_ctx = pressure.swapaxes(-1, -2) * val  # (D, d_workload)
        d_dom = dom.shape[-1]
        if dom_ctx.shape[-1] >= d_dom:
            dom_ctx = dom_ctx[:, :d_dom]
        else:
            gap = d_dom - dom_ctx.shape[-1]
            pad = xp.zeros((dom_ctx.shape[0], gap), be.float_dtype)
            dom_ctx = xp.concatenate([dom_ctx, pad], axis=-1)
        dmask = batch.masks["topology_domain"][:, None]
        H["topology_domain"] = (H["topology_domain"] + dom_ctx) * dmask
        return H

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
            rel_src={k: cd(v) for k, v in self.rel_src.items()},
            rel_dst={k: cd(v) for k, v in self.rel_dst.items()},
            rel_phi={k: cd(v) for k, v in self.rel_phi.items()},
            rel_gate={k: cd(v) for k, v in self.rel_gate.items()},
            relation_update={k: cg(v) for k, v in self.relation_update.items()},
            pool_proj={k: cd(v) for k, v in self.pool_proj.items()},
            broadcast_update={k: cg(v) for k, v in self.broadcast_update.items()},
            action_embed=be.from_numpy(conv(self.action_embed)),
            attn_q=cd(self.attn_q),
            attn_k=cd(self.attn_k),
            attn_v=cd(self.attn_v),
            global_readout=cd(self.global_readout),
            film_gen=cd(self.film_gen),
            predictor=(self.predictor.to_backend(be) if self.predictor else None),
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
    # Typed per-relation scatter matrices keyed by the relation *name* in
    # :data:`RELATIONS` (n_dst, n_src), built by the feature builder including the
    # R5 shortcut relations (doc Q R5). Defaults to empty for back-compat with any
    # externally-constructed TensorState.
    relations: dict[str, ArrayLike] = field(default_factory=dict)
    # Optional multi-rate observation windows per entity type: ``(T, n, raw)``
    # sequences fed to the fast / slow temporal cores (doc J B3/B4). When absent
    # the cores run a single recurrent step on the snapshot features.
    windows: dict[str, ArrayLike] | None = None


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
        relations = self._build_relations(state, order, hierarchy, caps, candidate_mask)

        return TensorState(
            caps=caps,
            features=features,
            masks=masks,
            incidence=incidence,
            hierarchy=hierarchy,
            candidate_mask=candidate_mask,
            order=order,
            relations=relations,
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

    def _build_relations(
        self, state, order, hierarchy, caps, candidate_mask
    ) -> dict[str, ArrayLike]:
        """Name-keyed typed relation scatter matrices, incl. R5 shortcuts (doc Q).

        Builds one ``(n_dst, n_src)`` one-hot scatter per entry of
        :data:`RELATIONS`. The hierarchy/assignment relations reuse the already-
        built incidence; the R5 shortcuts (node↔switch, rack↔power_domain,
        rack↔cooling_zone, workload↔candidate_domain) are constructed here so the
        switch latent and the power/cooling coupling participate in message
        passing (doc Q R5/V6). All matrices are at padded caps so the scatter
        aligns with the padded latent tensors; padding rows are all-zero.
        """
        be = self.be
        X = state.symbolic
        node_idx = self._index(order, "node")
        rack_idx = self._index(order, "rack")
        sw_idx = self._index(order, "switch")
        wl_idx = self._index(order, "workload")
        rel: dict[str, ArrayLike] = {}

        # --- hierarchy / assignment relations (reuse pre-built incidence) ---- #
        rel["gpu->node"] = hierarchy[("gpu", "node")]
        rel["node->rack"] = hierarchy[("node", "rack")]
        rel["rack->topology_domain"] = hierarchy[("rack", "topology_domain")]
        # workload->node was added by _build_incidence; rebuild deterministically.
        A = np.zeros((caps.N_max, caps.J_max), dtype=np.float32)
        for g in X.gpus.values():
            w = g.allocated_to
            if w in wl_idx and g.node_uid in node_idx:
                A[node_idx[g.node_uid], wl_idx[w]] = 1.0
        rel["workload->node"] = be.from_numpy(A)

        # --- R5: node -> switch (CONNECTED_TO) and reverse ------------------- #
        # node -> switch: dst=switch, src=node (n_switch, n_node).
        A_ns = np.zeros((caps.S_max, caps.N_max), dtype=np.float32)
        for u, n in X.nodes.items():
            if u in node_idx and getattr(n, "switch_id", None) in sw_idx:
                A_ns[sw_idx[n.switch_id], node_idx[u]] = 1.0
        rel["node->switch"] = be.from_numpy(A_ns)
        # switch -> node: dst=node, src=switch (n_node, n_switch) = transpose.
        rel["switch->node"] = be.from_numpy(A_ns.T.copy())

        # --- R5: rack <-> power_domain / cooling_zone as rack↔rack siblings -- #
        # Racks sharing a power domain (resp. cooling zone) are connected so the
        # coupling is visible without a separate latent type. Self-edges excluded
        # so the message carries *other* racks' state (doc Q R5 semantics).
        def _sibling(attr: str) -> np.ndarray:
            groups: dict[str, list[int]] = {}
            for u, r in X.racks.items():
                if u not in rack_idx:
                    continue
                key = getattr(r, attr, None)
                if key is None:
                    continue
                groups.setdefault(key, []).append(rack_idx[u])
            M = np.zeros((caps.R_max, caps.R_max), dtype=np.float32)
            for members in groups.values():
                for a in members:
                    for b in members:
                        if a != b:
                            M[a, b] = 1.0
            return M
        rel["rack->power_domain"] = be.from_numpy(_sibling("power_domain_id"))
        rel["rack->cooling_zone"] = be.from_numpy(_sibling("cooling_zone_id"))

        # --- R5: candidate_domain -> workload (feasible candidate edges) ----- #
        # dst=workload, src=topology_domain (n_workload, n_domain): the exact
        # feasible candidate mask F_WD is the incidence (doc Q Eq. 6, V6).
        cm = np.asarray(be.to_numpy(candidate_mask))  # (J, D) in {0,1}
        rel["candidate_domain->workload"] = be.from_numpy(cm.astype(np.float32).copy())
        return rel

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
    "RELATIONS",
    "SLOW_TYPES",
    "ArchConfig",
    "Dense",
    "FeatureBuilder",
    "GRUCell",
    "RelationalWorldModel",
    "TensorState",
    "WorldTransition",
    "kernel_state_summary",
]
