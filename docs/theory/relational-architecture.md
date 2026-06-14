<!--
  GPU-Center World Model — theory: relational architecture & expressivity.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Relational architecture & expressivity

> Theory summary by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.
> Source: research design doc Q. Realised in `model/architecture.py`,
> `schema/topology.py` (incidence), `schema/tensors.py` (buckets).

The world-transition backbone is a **typed message-passing network realised as
fixed-incidence batched matmuls**, with hierarchical pooling and tiny
candidate-set attention. This page is why that specific design.

## Typed message passing as fixed incidence

For each entity type $\tau$ the padded latent is $H_\tau \in \mathbb
R^{B\times n_\tau\times d_\tau}$ with a validity mask. For each directed relation
$r:\sigma\to\tau$, one-hot incidence matrices $S_r$ (edge × source) and $T_r$ (edge
× target) gather and scatter:

$$ U_r = S_r H_\sigma,\qquad V_r = T_r H_\tau,\qquad
\Sigma_r = T_r^\top M_r,\qquad \mu_r = \Sigma_r / \max(N_r, 1) $$

Typed messages $M_r = \eta_r \odot \phi_r(\text{LN}[\,W_r^\text{src}U_r,
W_r^\text{dst}V_r, W_r^\text{edge}X_r, A_r(a_t), G_r(g_t)\,])$ are aggregated to the
target type and fused with the local state through a **GRU-style gated update**
(Eq. 1). The action and global summary are broadcast via **row-shared FiLM /
concat — never slot-specific parameters**.

!!! note "Implementation-equivalence theorem (T1)"
    If $S_r, T_r$ encode the same typed edges a sparse edge-list runtime would, and
    parameters are shared by type/relation (not by slot), then the fixed-incidence
    scheme realises the **same function class** as the corresponding typed MPNN.
    The implementation does **not** change WL power — it is a typed MPNN, just
    written for MLX.

## Hierarchical PNA pooling + one downward broadcast

The real system already has an explicit hierarchy, so coarsening happens **early
and deliberately** (rather than via many flat 1-hop layers that smooth/squash):

- **gpu→node→rack→domain** upward pools using PNA-style **sum/mean/std/max/count**
  reducers, each fused with **exact arithmetic** from the kernel (free GPU, MIG
  histogram, XID stats; rack headroom; power slack; cooling margin).
- **one downward broadcast** ($B_{p\to c} = P_{c\to p}^\top H_p$) recovers most
  long-range context lost by upward compression without increasing flat depth.

Why PNA: GPU→node pooling must expose **count and distribution**, not only average
utilization — one node with 8 free GPUs is operationally very different from 8 nodes
with 1 free GPU each.

## Tiny candidate-set attention

Workload↔candidate-domain scoring uses a dense **masked** $J\times D$ block gated
by the exact-kernel feasible mask $F_\text{WD}(a_t)$. Because $D_\text{max}$ is
small and feasibility is exact, a dense masked block is more faithful and
compile-stable than a dynamic sparse candidate graph. Attention is used **only**
here — not as the transition backbone.

## Expressivity & equivariance

- Typed relation-wise MPNN with injective-enough per-relation MLPs is upper-bounded
  by **typed 1-WL** — sufficient because the exact kernel already exposes the
  arithmetic/legality the learned block would otherwise have to rediscover; higher
  order / subgraph GNNs are rejected until a residual target provably depends on
  motifs the kernel + typed-1-WL backbone cannot expose.
- **Per-type permutation equivariance** holds by construction (symmetry group
  $\mathcal G = \prod_\tau S_{n_\tau}$): row-shared updates + masked reducers, so
  relabelling within a type commutes with the forward map; padding positions are
  zeroed and never carry slot-specific parameters.

## Architecture comparison (why MPNN wins for v1)

| Option | Expressivity | MLX fit |
|---|---|---|
| **typed MPNN (fixed incidence)** | typed 1-WL | **best** — matmuls/broadcasts/reductions, compile-stable |
| graph attention | local++ | feasible on small sets, less compile-clean |
| full graph transformer | strongest global | weakest v1 fit — all-pairs cost vs. unified-memory rollouts |

## Compile stability (T2)

With $n_\tau$ and $e_r$ fixed by bucket and shape-dependent logic kept **outside**
compiled code, the relational block compiles **once per bucket** and reuses. MLX
recompiles on shape change and shapeless compilation is unsafe for shape-dependent
graph logic — hence the 3–6 padded buckets in `schema/tensors.py: DEFAULT_BUCKETS`.

## Recommended v1 architecture

Per-type temporal encoders first (fast GRUs for gpu/node/rack/switch, slow GRUs for
workload/queue/scheduler) → one relational-fusion block per imagined step (two
upward sweeps + one downward broadcast) → query heads. Exact-kernel shortcut
relations and arithmetic are **always visible** to the learned block; no learned
rewiring in v1. See [ADR-0003](../adr/0003-mlx-and-typed-mpnn.md).

## Why this matters

This is the concrete mechanism behind "filter $X_t$ exactly, learn $Z_t$": the
exact kernel feeds legality and arithmetic into the graph, and the typed MPNN
models only the residual relational dynamics — compile-stable, equivariant, and
auditable.
