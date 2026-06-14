<!--
  GPU-Center World Model — architecture overview.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Architecture overview

> Designed and built by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.

This is the master map. Deep derivations live in the
[theory foundations](../theory/causal-foundations.md); the contract everything
shares lives in the [schema](../schema/overview.md).

## The thesis in one factorisation

The GPU center is modelled as a **typed dynamical graph**, not video or pixels.
The full hidden state factorises (doc I, Definition 1) as

$$ S_t = (X_t,\; Z_t,\; C_t,\; M_t) $$

| Block | Name | Owner | Learned? |
|---|---|---|---|
| $X_t$ | exact symbolic state (entities + typed relations) | exact kernel | **no** |
| $Z_t$ | residual latent (phase, runtime, hazards, headroom) | learned model | yes |
| $C_t$ | exogenous context (regime, maintenance, drains, ambient) | data engine | partly |
| $M_t$ | observation-source state (missingness, staleness, health) | filter | yes |

The planner belief collapses to

$$ B_t(x,z,c,m) = \delta_{x_t^{\text{obs}}}(x)\, q_\phi(z,c,m \mid \mathcal H_t) $$

— **exact filtering over the symbolic block, learned filtering over the residual
block** (Rao-Blackwellization; doc T). $X_t$ is reconstructed directly from
scheduler / topology / inventory surfaces; uncertainty is concentrated in $Z_t$.
See [Bayesian filtering](../theory/bayesian-filtering.md).

## The spine

```
ingest ─▶ data_engine ─▶ {decision windows, tensors, synthetic packs}
                              │
        ┌─────────────────────┴───────────────────────┐
        ▼                                              ▼
 simulator (exact kernel)  ◀── safe set ──▶  model (learned residual + UQ)
        │                                              │
        └──────────────▶ planner (shield + lex rank) ◀┘
                              │
                  serving (FastAPI + SSE) ─▶ apps/web (three.js Decision Studio)
                              │
                     evaluation (safety-gated benchmark)
```

Three commitments make this a *world model* rather than a dashboard:

1. **Hybrid by design.** Hard feasibility is symbolic and exact; uncertain
   dynamics are learned with calibrated uncertainty. See
   [Exact kernel vs learned residual](exact-kernel-vs-learned.md).
2. **Safety is a gate, not a metric.** The learned model may only rank / forecast
   / abstain over the kernel-certified safe set; it can never expand it. See
   [ADR-0002](../adr/0002-safety-as-a-gate.md).
3. **Intervention-faithful, not visually plausible.** Every counterfactual branch
   is rooted at a frozen **pre-action state hash** under common random numbers, so
   branch deltas estimate intervention effects, not intervention-plus-noise. See
   [Causal foundations](../theory/causal-foundations.md).

## Action alphabet

The v1 controllable actor alphabet is intentionally small, because in GPU-center
operations actions are already explicit, logged, and unambiguous — there is no
need to *learn latent actions*:

`hold` · `admit` · `place` · `admit_and_place`

`mask_domain` exists too, but as an **operator guardrail / scenario
intervention** (a change to policy/context), not a normal actor action. See
[Schema → Actions](../schema/actions.md).

## Scope and honesty

- v1 is **offline replay / shadow mode** only. See [ADR-0004](../adr/0004-shadow-mode-only-v1.md).
- The *zero hard-violation* claim is narrow and constructive: it covers
  **modelled current-state hard feasibility**, under correct kernel semantics and
  atomic re-validation before any (future) actuation.
- All counterfactual claims are relative to the **branch reference environment**.
- Each release ships a [provable-vs-empirical ledger](../benchmark/acceptance.md).

## Apple-silicon stack rationale

Python + MLX for the learned model (unified memory, lazy eval, `compile()`, custom
Metal escape hatch), a pure-Python symbolic kernel kept **outside** `mx.compile`,
bucketed padded shapes to avoid compile churn, FastAPI serving, and TypeScript +
three.js for the UI. The learned backend sits behind a small `WorldModel`
contract so the kernel/planner/eval run anywhere (Linux CI included). See
[ADR-0003](../adr/0003-mlx-and-typed-mpnn.md) and the
[relational architecture](../theory/relational-architecture.md) page.
