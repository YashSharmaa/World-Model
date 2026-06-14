<!--
  GPU-Center World Model — theory: causal foundations.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Causal & probabilistic foundations

> Theory summary by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.
> Source: research design doc I. Realised in `schema/state.py`,
> `schema/hashing.py`, `data_engine/synthetic`, `evaluation`.

This page fixes the precise meaning of *intervention-faithful*: what the world
model claims to get right, and the probabilistic object it is.

## The hybrid latent state (Def. 1–4)

At decision epoch $t$ the state factorises as $S_t = (X_t, Z_t, C_t, M_t)$:

- $X_t$ — **exact symbolic state** over the authoritative entity schema (all
  thirteen entity kinds + typed relations $E_t^{\text{rel}}$);
- $Z_t$ — **residual latent** $= (\Phi_t, \Omega_t, \Lambda_t, \Theta_t, \nu_t,
  \xi_t)$: per-workload phase, remaining-time/slowdown, podset readiness,
  rack/domain/power/cooling residuals, switch congestion, per-GPU failure hazards;
- $C_t$ — **exogenous context** (maintenance, drains, ambient, optional carbon,
  arrival/mix regimes);
- $M_t$ — **observation-source state**: missingness is **endogenous state**, not a
  preprocessing artifact.

## The belief and the Rao-Blackwell split (Def. 5)

$$ B_t(s) = \delta_{x_t^{\text{obs}}}(x)\, q_\phi(z, c, m \mid \mathcal H_t) $$

The exact symbolic block is reconstructed directly from scheduler/topology/
inventory surfaces whenever available; **all uncertainty is concentrated in the
residual block**. This is the formal basis for "filter $X_t$ exactly, learn $Z_t$".

## Command-level intervention semantics (Def. 6)

$\mathrm{do}(a)$ acts on the **command presented to the exact kernel**, not on a
physically impossible post-action state. Therefore:

!!! warning "The honest definition of an intervention here"
    If a command is infeasible, the interventional semantics are **reject / no-op
    with a blocker ledger** returned by the kernel — *not* teleportation into an
    infeasible allocation.

This is what stops counterfactual reasoning from inventing impossible futures.

## Intervention correctness (Def. 7)

For horizon $H$ and KPI family $\mathcal K$, a hybrid world model $\hat M$ is
**$H$-step intervention-correct** on reachable beliefs $\mathcal B$ if, for every
reachable belief $b_t$ and every command sequence $\bar a$ in the kernel-defined
feasible support,

$$ \mathrm{Law}_{\hat M}\!\big(\mathrm{KPI}_{t:t+H} \mid \mathrm{do}(\bar a),
B_t = b_t\big) \;=\; \mathrm{Law}_{\text{env}}\!\big(\cdots\big) $$

and the kernel-implied feasibility / blocker outputs match exactly almost surely.
This is the mathematically precise form of the project's scientific claim — and
note it is a claim about **interventional distributions of KPIs**, not about pixel
fidelity.

## The environment is a lexicographic constrained POMDP (Def. 8–10)

The environment is a finite-horizon **lexicographic constrained POMDP**
$M = (S, A, O, T, \Omega, G, B_0, H)$. The admissible command set at belief $b_t$
is $A_{\text{feas}}(x_t) = \{a : \textsf{Feasible}_K(x_t,a)=1\}$ (Def. 9);
commands outside it are valid *what-if* inputs for explanation but resolve to
deterministic rejection. The planner minimizes a **lexicographically ordered**
risk/performance vector — feasibility and policy protection sit *above* predicted
throughput or wait-time gains (Def. 10), the seed of
[safety-as-a-gate](../adr/0002-safety-as-a-gate.md).

## How the code realises it

- The four-block state is `WorldState = (SymbolicState, ResidualBelief,
  ExogenousContext, SourceState)` (`schema/state.py`), with `.X/.Z/.C/.M`
  accessors matching this notation.
- Command-level semantics are enforced by the exact kernel: infeasible commands
  return a `FeasibilityResult(feasible=False, blockers=[...])`, never a mutated
  state.
- Intervention correctness is *measured*, not assumed: the
  [counterfactual benchmark track](../benchmark/tracks.md) scores same-state action
  ranking and delta error, and the [ledger](../benchmark/acceptance.md) states that
  counterfactual correctness is only relative to the branch reference environment.

## Why this matters

The 2025–2026 world-model literature (V-JEPA 2 / 2.1 and action-conditioned
variants) reframed world modelling as **latent, action-conditioned, intervention
prediction** rather than generation over observations. This page is the formal
adoption of that stance for GPU-center operations: the model is judged on whether
its *interventional KPI distributions* are right, under exact kernel feasibility.
