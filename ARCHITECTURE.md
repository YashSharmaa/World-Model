<!--
 GPU-Center World Model — architecture overview
 Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
 License: Apache-2.0 (docs: CC-BY-4.0)
-->

# Architecture

> Designed and built by **Yash Sharma** — https://www.linkedin.com/in/yashsharmaa/

This document is the master map. Deep derivations live in [`docs/theory/`](docs/theory).

## 1. The thesis in one diagram

The GPU center is modelled as a **typed dynamical graph**, and the full hidden
state factorises as `S_t = (X_t, Z_t, C_t, M_t)`:

| Block | Name | Owner | Learned? |
|---|---|---|---|
| `X_t` | exact symbolic state (entities + relations) | exact kernel | **no** |
| `Z_t` | residual latent (phase, runtime, hazards, headroom) | learned model | yes |
| `C_t` | exogenous context (regime, maintenance, drains, ambient) | data engine | partly |
| `M_t` | observation-source state (missingness, staleness, health) | filter | yes |

The planner belief collapses to `B_t = δ_{x_t}(x) · q(z, c, m)`: **exact filtering
over the symbolic block, learned filtering over the residual block**
(Rao-Blackwellization).

## 2. Data & control flow

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

## 3. Module boundaries (and the rule that separates them)

> **Boundary rule:** *If a rule can be read directly from scheduler, topology,
> device-allocation, or policy state, it belongs in the exact kernel — never in
> the learned model.*

| Module | Owns | Must not own |
|---|---|---|
| `schema` | the typed world-state contract | any ML / IO logic |
| `ingest` | source adapters → canonical state | feasibility policy |
| `data_engine` | windows, tensorization, synthetic SCM, provenance | feasibility, scoring, geometry |
| `simulator` | hard feasibility, blockers, deterministic branch replay | runtime/hazard estimation |
| `model` | residual dynamics, KPI/hazard forecasts, calibration, UQ | rule enforcement, API |
| `planner` | safe enumeration, lex ranking, abstention, why-not ledger | ingestion, rendering |
| `serving` | REST/SSE DTOs, compiled-graph & replay cache | physics / scheduling logic |
| `apps/web` | semantic 3D read model + branch playback | feasibility, KPI computation |
| `evaluation` | the benchmark (source of truth for claims) | production serving |

## 4. The theory → code map

Every research foundation is realised in code:

| Theory doc | Realised in |
|---|---|
| Causal / probabilistic foundations (SCM, g-formula, do-calculus) | `data_engine/synthetic`, `evaluation/metrics` |
| Learning objective (JEPA latent + counterfactual rollout loss) | `model/mlx_backend` |
| Queueing & control dynamics (MMPP arrivals, Lyapunov virtual queues) | `simulator`, `planner` |
| Uncertainty & conformal guarantees | `uncertainty/`, `model/calibration` |
| Constrained risk-sensitive planning (shielded CMDP, CVaR, safe regret) | `planner/`, `simulator/exact_kernel` |
| Physical surrogate & system ID (RC thermal, power, network) | `data_engine/synthetic` |
| Generalization & sim2real theory | `evaluation/`, provable-vs-empirical ledger |
| Representation / information theory (info bottleneck, action sensitivity) | `model/mlx_backend` |
| Relational architecture & expressivity (typed MPNN, PNA pooling) | `model/mlx_backend` |
| Synthetic generative process & fidelity certification | `data_engine/synthetic` |
| Combinatorial exact kernel & shield (ILP, why-not IIS) | `simulator/exact_kernel` |
| Bayesian filtering & state estimation (EKF/UKF/RBPF/EnKF) | `uncertainty/`, `model` |
| Reliability & changepoint detection (competing risks, CUSUM/SR/BOCPD) | `reliability/` |
| Fairness & mechanism design (DRF/PF/NBS — v2) | `fairness/` |

## 5. Apple-silicon stack rationale

Python + MLX for the learned model (unified memory, lazy eval, `compile()`,
custom Metal escape hatch), a pure-Python symbolic kernel kept **outside**
`mx.compile`, bucketed padded shapes to avoid compile churn, FastAPI serving, and
TypeScript + three.js for the UI. The learned backend is behind a small
`WorldModel` contract so the kernel/planner/eval run anywhere (Linux CI included).

## 6. Safety, scope, and honesty

* `v1` is **offline replay / shadow mode** only.
* The *zero hard-violation* claim is narrow and constructive: it covers
  **modelled current-state hard feasibility**, under correct kernel semantics and
  atomic re-validation before any (future) actuation.
* All counterfactual claims are relative to the **branch reference environment**.
* Each release ships a provable-vs-empirical ledger.
