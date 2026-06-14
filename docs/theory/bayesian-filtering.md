<!--
  GPU-Center World Model — theory: Bayesian filtering & state estimation.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Bayesian filtering & state estimation

> Theory summary by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.
> Source: research design doc T. Realised in `uncertainty/filters.py`,
> `uncertainty/missingness.py`, and the `model` belief interface.

How the belief over the residual block is maintained online, with the exact kernel
handling everything it can analytically.

## Rao-Blackwellization by the exact kernel (always on)

The full latent is $s_t = (x_t^K, x_t^C, m_t^D, h_t^R, \theta_t, r_t^\text{src})$:
exact symbolic, continuous physical, discrete multimodal, learned relational
residual, parameters, source-health. The major tractability move:

$$ x_t^K = \Phi_K(x_{t-1}^K, a_{t-1}, e_t) \text{ exactly} \;\Rightarrow\;
p(x_t^K, \xi_t \mid \cdot) = \delta[x_t^K = \Phi_K(\cdot)]\, p(\xi_t \mid x_t^K,
\cdot) $$

All exact scheduler/topology semantics are handled **analytically**; only the
residual uncertain state is filtered approximately. This reduces variance,
preserves exact semantics, and gives the planner a clean causal interface. Learning
quota/topology semantics inside a generic recurrent latent is rejected — it
violates the intervention-faithful split.

## A filter family matched to each substate

| Substate | Filter | Why |
|---|---|---|
| continuous thermal/power/headroom (rack/zone/domain aggregates) | **square-root EKF / UKF** | low-dimensional, near-Gaussian after aggregation; PF would waste variance |
| multimodal regimes (phase, fault/no-fault, reset/recover, source-health incidents) | **Rao-Blackwellized particle filter** | the planner cares about posterior **mode probabilities**, not just mean shifts |
| high-dimensional relational residual latent | **Ensemble Kalman filter** (topology-localized) | continuous, graph-coupled, too high-dim for EKF/UKF; PF degenerates |

Selection is by substate character: EKF when Jacobians are available and speed
matters, square-root UKF when more nonlinear but unimodal, RBPF for rare-event
bursts (`gpu_fault_burst`, `telemetry_degradation`), EnKF for many continuous entity
embeddings.

## GRU as an amortized filter

At serving time the residual belief is maintained by a **GRU inference network**
$h_t^\text{GRU} = \text{GRU}_\phi(h_{t-1}^\text{GRU}, \psi_\phi(o_t, a_{t-1},
x_t^K))$, where $\psi_\phi$ **must** include numeric telemetry, typed event
embeddings, **and explicit missingness/staleness indicators** — i.e. the GRU is an
amortized approximation to the principled filters above, not a black box that
ignores observation integrity.

## Informative missingness (MNAR)

Each channel carries availability $\delta_{t,j}$, staleness age $\tau_{t,j}$, and
censoring metadata. Missingness is **MNAR at incident times**: $p(\delta_t \mid
s_t)$ depends on hidden state. Zero-filling inside the filter is forbidden — the
missingness *itself* (and its age) is the signal (see
[reliability & changepoint](reliability-changepoint.md), where degraded telemetry
co-occurs with worsening health).

## Observability classes

The hidden state partitions by observability: **O1** exact symbolic states surfaced
by control-plane APIs (no filtering needed); **O2** continuous physical states that
admit local linearization (EKF/UKF); **O3** nonlinear/relational hidden states with
discrete modes, unmeasured inputs, or graph coupling (RBPF/EnKF/GRU). The deeper a
state sits, the more its belief — and uncertainty — drives **abstention**.

## Coupling to kernel and planner

The planner consumes a **hybrid belief** $b_t = \delta_{x_t^K} \otimes
b_t^\text{residual}$ — *not* a single point estimate. Deterministic state (queue
membership, quota accounting, observed counts) needs no filtering and is read
exactly; only the residual carries posterior uncertainty into the chance/CVaR gates
and the abstention logic.

## Filter consistency diagnostics

NIS / NEES (and pseudo-variants), innovation-whiteness/lag tests, an observability
monitor, particle/ensemble degeneration checks, and **planner-facing calibration
diagnostics** — so a mis-tuned filter is caught before it produces overconfident
forecasts.

## Why this matters

This page is the runtime realisation of the
[Rao-Blackwell belief split](causal-foundations.md): the exact kernel collapses the
symbolic block to a delta, and the right approximate filter handles each piece of
the residual — with missingness treated as informative state so that *quiet
failures* raise uncertainty instead of being silently imputed away.
