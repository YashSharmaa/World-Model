<!--
  GPU-Center World Model — theory: queueing & control dynamics.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Queueing & control dynamics

> Theory summary by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.
> Source: research design doc K. Realised in `simulator`, `planner`,
> `data_engine/synthetic/arrivals.py`.

The queueing/control layer is what lets the analytic spine and the learned
residual share the same coordinates — and what gives the planner its stability
constraints.

## The analytic / learned partition

The dynamics are split so each half does what it is good at (doc K):

**Analytic priors** (read from exact state + queueing math): the exact feasible
pattern set and blocker ledger; effective-load and shadow prices; a fluid
mean-forecast of pending queues, occupancies, fairness debt, and safety virtual
queues over 30–60 min; phase-type transient moments; deterministic rack-power /
hotspot forecasts; KPI transforms.

**Learned residuals** (genuinely data-dependent): residual service-time multiplier
& remaining-time hazard; startup fork-join tail & pods-ready timeout hazard; rack
headroom & switch-hotspot residuals; failure / telemetry-collapse hazard;
domain-quality residual (throughput loss from communication/contention).

**Coupling.** The learned head predicts the **innovation**: e.g.
$\mathbb E[S_{r,p}\mid x] = m_{r,p}^\text{analytic}(x) + \delta_{r,p}^\text{svc}(x)$,
with $\delta$ shrunk toward 0 when evidence is weak. Because residual targets are
$y_\text{realized} - y_\text{analytic}$, **the analytic spine is a control
variate** — it reduces variance in training and planning (EqP6). Learning residuals
in raw observation space was rejected: actions act on bottleneck coordinates and
feasible patterns, so the residual must be conditioned on those sufficient
statistics.

## State-space collapse to a few bottlenecks

Kueue/Kubernetes semantics expose only a small set of practically active
bottlenecks (flavor quotas, shared cohorts, topology domains, a few switches).
Markov-modulated JSQ and multiscale Jackson results justify **state-space collapse
to a low-dimensional latent** — the formal reason the per-entity latent widths can
stay modest (see [Representation & information](representation-information.md)).

## Lyapunov / virtual-queue planner constraints

The bridge from control theory to the planner is a set of **virtual-queue
constraints** the planner enforces over the horizon (doc K, C1–C10):

- **C1 Hard kernel feasibility** — $a \in A_\text{exact}(X_t)$ (a hard constraint,
  never a learned penalty).
- **C2 Capacity-reserve chance constraint** — $\mathbb P[U_\ell(h) > C_\ell(h) -
  r_\ell] \le \alpha_\ell$.
- **C3 Rack-headroom virtual queue** — $Z_{\text{head}}$ stays bounded / has
  negative drift above threshold.
- **C4 Hotspot virtual queue** — same negative-drift logic on $Z_\text{hot}$.
- **C5 Fair-share debt** — $D_c$ stays stable (the v2 fairness hook).
- **C6 Startup/requeue chance constraint** — encodes all-or-nothing startup.
- **C7 Fragmentation reserve** — preserves the largest same-domain feasible bundle.
- **C8 Horizon-wise Lyapunov drift test** — reject any action sequence whose
  cumulative predicted drift fails the bound.
- **C9 OOD / disagreement abstention** — fall back to hold / conservative
  placements when residual-head disagreement is high.
- **C10 Starvation guard** — bound waiting-age virtual queues so conservatism does
  not create unbounded tenant starvation.

These are realised as the planner's chance / CVaR gates and the abstention logic
(see [Constrained planning](constrained-planning.md)).

## Fragmentation as online optimization

Kueue's `required` / `preferred` / `unconstrained` topology policies *are* an
online fragmentation-vs-locality tradeoff: `unconstrained` is explicitly a way to
fill small gaps and reduce fragmentation. The planner's fragmentation term (C7,
and the `topology_fragmentation_delta` KPI) captures this — same-domain placement
that helps the current gang can destroy a future contiguous block.

## Why this matters

This page is why the system is *more* than a learned predictor: it inherits
decades of queueing-stability theory as **hard planner constraints**, while
letting the learned residual model exactly the parts that theory cannot pin down.
The arrival/event side of the synthetic engine uses a Markov-modulated Hawkes-Cox
process so bursts and regimes are realistic (see
[Synthetic fidelity](synthetic-fidelity.md)).
