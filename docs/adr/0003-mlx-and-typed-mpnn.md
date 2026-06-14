<!--
  GPU-Center World Model — ADR-0003.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# ADR-0003 — MLX backend + typed MPNN over fixed incidence

- **Status:** Accepted
- **Date:** 2026-06-14
- **Author:** [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)
- **Deciders:** Yash Sharma

## Context

The learned residual world model is a **typed relational** model over a
heterogeneous graph (workloads, podsets, queues, nodes, GPUs, racks, switches,
domains). It must train and serve on an **Apple-silicon MacBook** for the public
demo, and it must also run on plain Linux CI (where MLX is unavailable).

## Decision

Two coupled choices:

1. **Backend: Apple MLX, behind a small `WorldModel` contract.** MLX gives
   unified memory, lazy evaluation, `compile()`, and a custom-Metal escape hatch.
   The learned backend sits behind `model/contracts.py` so the kernel, planner,
   and evaluation run anywhere; a **NumPy reference path** provides the same
   forward pass on CI and non-Apple machines.

2. **Architecture: a typed MPNN realised as fixed-incidence batched matmuls**
   (doc Q). Typed message passing is implemented with one-hot incidence matrices
   $S_r$ (edge × source) and $T_r$ (edge × target):

   $$ U_r = S_r H_\sigma,\qquad \Sigma_r = T_r^\top M_r $$

   followed by **hierarchical PNA pooling** (gpu→node→rack→domain with
   sum/mean/std/max/count) and **one downward broadcast** sweep, plus **tiny
   attention** only on the small workload↔candidate-domain set.

The exact symbolic kernel is kept **outside** `mx.compile`, and shapes are
**bucketed** into 3–6 padded pod sizes so `compile()` recompiles only once per
bucket.

## Consequences

**Positive**

- **Compile stability:** the relational block is matmuls/broadcasts/reductions,
  so it compiles once per bucket and reuses (doc Q, T2). Stable padded shapes plus
  masks keep the hot path compilable — `compile()` recompiles on shape change, and
  shapeless compilation is unsafe for shape-dependent graph logic.
- **No graph-runtime dependency:** the fixed-incidence scheme is the
  *implementation* of a typed MPNN, not a different expressive class (doc Q, T1);
  it removes the need for a sparse-graph library and improves auditability.
- **Portability:** identical numerics through the NumPy fallback on CI.
- Per-type **permutation equivariance** holds by construction (row-shared updates,
  masked reducers; doc Q G1–G3).

**Negative / costs**

- Some arithmetic is wasted vs. a sparse runtime (padded edges), acceptable at
  demo-pod scale.
- The latent widths and bucket caps are fixed contracts (`schema/tensors.py`) that
  must be respected across modules.

## Alternatives rejected

- **Full hetero graph-transformer backbone** (doc Q): all-pairs attention,
  heterogeneous masking, and structural-encoding cost are misaligned with MLX
  compile stability and unified-memory rollouts.
- **PyG-style sparse runtimes / dynamic rewiring:** conflict with compile-stability
  constraints and weaken auditability.
- **Pure attention over all resources:** the exact kernel already defines the legal
  candidate set; only small-set ranking benefits from attention.

## References

- [Relational architecture](../theory/relational-architecture.md),
  [Representation & information](../theory/representation-information.md)
- [Learning objective](../theory/learning-objective.md)
- [Schema → State & hashing](../schema/state.md) (latent widths, bucket caps)
