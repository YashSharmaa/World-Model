<!--
  GPU-Center World Model — schema state & hashing.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# State & hashing

> Project by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.

`schema/state.py` defines the hybrid `WorldState`, and `schema/hashing.py` defines
the deterministic pre-action `state_hash` that anchors every counterfactual
branch. Together they encode two of the project's load-bearing commitments: the
**four-block factorisation** and **branch alignment**.

## The hybrid state $S_t = (X_t, Z_t, C_t, M_t)$

A `WorldState` carries an event time (`ts_ns`), a discrete index (`tick`), and four
blocks (with `.X`, `.Z`, `.C`, `.M` accessors matching the doc-I notation):

| Block | Dataclass | What it holds | Filtered how |
|---|---|---|---|
| $X_t$ | `SymbolicState` | exact, authoritative symbolic state (all entity tables + typed relations) | **exact** (the kernel) |
| $Z_t$ | `ResidualBelief` | learned residual latent (named numeric blocks, summaries, epistemic) | **learned** |
| $C_t$ | `ExogenousContext` | regime, maintenance windows, manual drains, ambient, optional carbon | partly exogenous |
| $M_t$ | `SourceState` | collector health, channel missing-prob, age, transport delay | learned filter |

### $X_t$ — `SymbolicState`

The source of truth for the exact kernel: `workloads`, `podsets`, `local_queues`,
`cluster_queues`, `flavors`, `domains`, `nodes`, `gpus`, `switches`, `racks`,
`power_domains`, `cooling_zones`, `operator_policy`, and `relations`. **It must
never be learned** — quota, gang, topology, health, and policy semantics are read
directly from these typed tables.

### $Z_t$ — `ResidualBelief`

A **backend-neutral** container of named numeric blocks produced by the learned
backend: per-entity `latents`, planner-facing `summaries` (mixture weights /
quantiles / hazards), and an `epistemic` dictionary. The kernel and planner never
depend on MLX internals — they read summaries, not arrays.

### $M_t$ — `SourceState`

Telemetry source-health, missingness, and staleness modelled as **state, never
preprocessing noise**. Real GPU failures often surface first as degraded
telemetry, so this block is informative.

## The planner belief

$$ B_t(x,z,c,m) = \delta_{x_t^{\text{obs}}}(x)\, q_\phi(z,c,m \mid \mathcal H_t) $$

Filtering is **exact over the symbolic block** ($\delta$ at the observed $X_t$) and
**learned over the residual block** — Rao-Blackwellization (doc T). See
[Bayesian filtering](../theory/bayesian-filtering.md).

## Deterministic pre-action hashing

A **state hash** is taken immediately before every operator/scheduler action.
Every real or simulated rollout points back to that same pre-action hash so
counterfactual branches differ **only** by the clamped action — never by drifting
exogenous conditions. Without this, action effects get contaminated by changed
context, the single biggest source of false optimism in learned world models
(docs D, I, R).

```python
from gpu_center_world_model.schema.hashing import state_hash, world_state_hash

sh  = state_hash(state.symbolic, state.context)   # hashes X_t (+ C_t)
sh2 = world_state_hash(state)                     # convenience over a WorldState
```

Properties of the digest:

- It hashes the **symbolic block $X_t$ plus exogenous context $C_t$** — and
  **excludes the learned belief $Z_t$**, which is an estimate, not ground truth,
  and would make the hash non-reproducible across model versions.
- It is **canonical**: deterministic regardless of dict insertion order, stable
  under float jitter below `1e-6` (floats are quantized to 6 decimals), and
  identical across Python processes (`sha256` over canonical JSON).

The serving layer uses the hash to align `compare` / `why-not` requests to a
frozen pre-action state; an unknown hash returns HTTP 409 so branches never drift.
See [Serving API → REST](../api/rest.md) and
[Architecture → Data & control flow](../architecture/data-flow.md).
