<!--
  GPU-Center World Model — theory: physical surrogate & system ID.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Physical surrogate & system identification

> Theory summary by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.
> Source: research design doc N. Realised in `data_engine/synthetic/thermal.py`,
> `/power.py`, `/network.py`.

The thermal, power, and network surrogates are the physics half of the data
engine. The design rule throughout: **structure-preserving physics first, bounded
learned residual second** — so the surrogate stays energy-consistent and
intervention-faithful under counterfactual actions.

## Thermal: an anchored RC graph

Compartments (GPU package, node chassis, rack plenum, cooling-zone return) are
lumped temperatures on a thermal graph. With capacitance $C \succ 0$, thermal
Laplacian $L_\text{th} = H^\top G_e H$, and boundary losses $G_\text{env}$:

$$ C\dot\theta = -(L_\text{th} + G_\text{env})\theta + B_p p + B_w w $$

written as a deviation $\theta = T - T_\text{ref}$ from a measured inlet/ambient
reference. Stability is **structural**, not bolted on:

- **T1** — if each component has an anchored boundary leak so $L_\text{th} +
  G_\text{env} \succ 0$, then $A_c$ is Hurwitz;
- **T2** — exact zero-order-hold discretization $A_d = \exp(A_c\Delta)$ has spectral
  radius $< 1$ for every $\Delta > 0$;
- **T3** — since $\mathbf 1^\top L_\text{th} = 0$, internal transfer **conserves
  stored heat**; only sources and boundary losses change net energy.

Prefer matrix-exponential discretization (or implicit Euler for stiff modes);
explicit Euler is rejected (its stability depends on $\Delta$ and can break
positivity). Reduced-order models keep only dominant rack/zone modes by Hankel
singular values and **planner-KPI sufficiency**, not one-step MSE.

## Power: switching semi-Markov source model

Workload **phase** $z_j$ is the latent parent of util mix, network demand, power,
and progress; phases are **semi-Markov** (compute/comm/checkpoint/tail have
different dwell distributions). Per-GPU power saturates:

$$ p_{g,k+1} = \text{sat}_{[0,P_\text{max}]}\big(a_{z_k} p_{g,k} + b_{z_k}^\top
\phi_{g,k} + c_{z_k}\big),\quad 0 \le a_{z_k} < 1 $$

Aggregate rack power adds a shot-noise term over **synchronized transitions**
(gang-scheduled jobs move many GPUs through phase boundaries together), with an
explicit ramp constraint. A single util→power regression is rejected (no
phase-specific saturation or synchronized ramps); independent per-GPU sampling is
rejected (violates the latent-parent rule). The **same phase sequence** feeds power
and network so synchronized comm phases affect both consistently.

## Network: two-tier routine + stress

- **Routine tier** — weighted **max-min fairness** (progressive filling) on the
  exact path graph, with transfer time $\tau_k = b_k/\max(r_k,\epsilon) +
  \sum_e \ell_e + d_q(u_e)$. Topology, paths, and capacities stay **exact symbolic
  objects**.
- **Stress tier** — a packet-level teacher (ECMP imbalance, PFC/DCQCN, synchronized
  collectives) **distilled** offline into a lightweight correction over the fluid
  base.

Pure online packet simulation is rejected (the planner compares many action
sequences); pure black-box GNN prediction is rejected (it obscures exact capacity
semantics and extrapolates poorly under new placements/masks).

## Hybrid physics + residual coupling

The residual modulates **physical parameters**, not the state directly:

$$ g_{\text{eff},k} = g_0\, e^{\delta_g(\phi_k)},\quad B_{\text{eff},k} = B_0 +
\Delta_B(\phi_k) $$

keeping the model inside the RC family. Conservation constraints: internal residual
transfer must be **Laplacian** ($\mathbf 1^\top L_\phi = 0$); unaccounted losses
enter only through positive diagonal dissipation $G_\phi$; source corrections are
bounded and attributable to measured context. A physics-informed loss adds rollout
fit, a physics-residual term, an **energy-consistency** penalty, a **Lyapunov-margin**
penalty, and an uncertainty-aware likelihood. Unconstrained additive MLP residuals
are rejected (they can inject artificial energy and break passivity).

## Stability constraints (S1–S6)

Positive thermal storage; nonnegative anchored conductances; energy conservation
of internal transfer; discrete-time stability via matrix-exponential or a
stable-by-construction map; **passivity** (port-Hamiltonian form $\dot\theta =
-(J+R)Q\theta + Gu$, $J = -J^\top$, $R \succeq 0$); and "the residual cannot create
energy" — learned as conductance/dissipation modulation, never an unconstrained
additive state update.

## Why this matters

Power and cooling are **risk signals**, not control knobs ([ADR-0004](../adr/0004-shadow-mode-only-v1.md)) — but to be *useful* risk signals their
forecasts must be physically consistent under counterfactual placements. Anchored
RC + bounded residual is what makes a "what if I admit this gang into the hot rack"
thermal forecast trustworthy rather than a free-form extrapolation. Feeds the
`rack_power_kw`, `rack_thermal_headroom_c`, and `headroom_violation` KPI/hazard
channels.
