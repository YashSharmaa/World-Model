<!--
  GPU-Center World Model — theory: constrained risk-sensitive planning.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Constrained, risk-sensitive planning

> Theory summary by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.
> Source: research design doc M. Realised in `planner/` (`planner.py`,
> `risk_gate.py`, `score.py`, `abstain.py`, `enumerate.py`, `why_not.py`).

The planner is a **shielded, lexicographic, risk-sensitive selector**. This page
gives the formal object it solves and why it is neither a plain POMDP nor a plain
CMDP.

## The shielded constrained belief-MDP

The correct formal object is a **shielded constrained belief-MDP** $M = (B, A, O,
\tau, c_0, \{c_\ell\}, \gamma, K)$:

- $K$ is the exact-kernel feasibility predicate;
- the **shielded policy class** is $\Pi_K = \{\pi : \pi(a\mid b)=0 \text{ whenever }
  K(b,a)=0\}$;
- $c_0$ is the primary (lexicographic) cost; $c_\ell$ are risk costs.

Deterministic feasibility is enforced **outside** the learning objective; uncertain
hazards are handled **inside** a risk-constrained layer. That is the precise reason
safety is a gate and not a weight (doc M, inference).

## Shielded Bellman optimality

$$ V^\*(b) = \min_{a \in A_K(b)} \mathbb E\big[c_0(b,a,B') + \gamma V^\*(B') \mid
b,a\big],\quad A_K(b) = \{a : K(b,a)=1\} $$

The minimization ranges only over the kernel-safe set.

## Two-layer risk objective

- **Layer 1 — lexicographic hard safety** with exact symbolic gating.
- **Layer 2 — statistical tail-risk control** on modelled hazards via **CVaR** and,
  where operator policy demands, explicit **chance constraints**:

$$ \mathbb P_\pi(Z_\ell^\pi > 0) \le \delta_\ell, \qquad
\mathrm{CVaR}_{\alpha_\ell}(Z_\ell^\pi) \le \beta_\ell, $$

using the Rockafellar–Uryasev form $\mathrm{CVaR}_\alpha(Z) = \inf_\eta\{\eta +
\frac{1}{1-\alpha}\mathbb E[(Z-\eta)_+]\}$. Mean-variance is **rejected** (it
penalizes benign upside and does not isolate catastrophic tails);
expectation-only ignores tails entirely.

## The hard/soft separation theorem (T1)

> If $K$ is sound for all modelled hard constraints, restricting to $\Pi_K$ yields
> **zero modelled hard-constraint violations by construction**; all remaining
> safety guarantees are statistical and must be expressed through chance / CVaR
> constraints.

A policy in $\Pi_1$ (hard-safe **and** satisfying the statistical budgets) is
hard-safe by exact certification and statistically safe for the modelled hazard
families and declared confidence — guarantee scope is exactly those hazards and
the calibration regime (T2).

## Lexicographic optimum

$$ \pi^\* \in \text{lex-argmin}_{\pi \in \Pi_1}\;\big(W_{95}(\pi),\; -P_\text{adm}
(\pi),\; -\text{GPUHours}_\text{adm}(\pi),\; \text{CrossRack}(\pi),\;
\text{Fragmentation}(\pi)\big) $$

p95 queue wait & admission success dominate; then admitted GPU-hours, then
locality, then fragmentation. For numeric **safe regret** the lexicographic vector
is mapped to an order-preserving scalarization (doc M, D2–D3) with dominating
weights $M_1 \gg M_2 \gg \dots$ on the bounded candidate set.

## Enumeration is optimal at this scale

Because the exact kernel generates the **complete** safe set and the v1 action
family is tiny, **exhaustive enumeration** is decision-theoretically optimal —
$\text{argmin}_{a \in C^\text{safe}_t} \hat S_t(a)$ with **zero search error**. No
learned policy search is needed in v1; amortization is a later optimization.

## Safe regret

For window $t$, $C^\text{safe}_t = \{a : K(b_t,a)=1 \text{ and all conformal/CVaR/
chance gates pass}\}$; the oracle is $a^\star_t \in \text{lex-argmin}_{a \in
C^\text{safe}_t} \psi_t(a)$ where $\psi_t$ is the true lexicographic KPI vector.
Safe regret is the realized gap vs. that oracle — the
[Track C](../benchmark/tracks.md) metric.

## Value-equivalence: train the queries, not the pixels

The relevant function class is the **planner queries that drive decisions** —
queue-wait deltas, admission success, GPU-hours, locality, fragmentation,
headroom/hotspot trajectories, hazards under shielded actions — not generic
observation reconstruction. Approximate value-equivalence is why preserving
action-relevant query answers is *sufficient* for planning (connects to
[Representation & information](representation-information.md) and the
[learning objective](learning-objective.md)).

## How the code realises it

`planner/enumerate.py` builds $C^\text{safe}_t$ via the kernel;
`planner/risk_gate.py` applies the conformal/CVaR/chance gates; `planner/score.py`
computes the lexicographic key; `planner/abstain.py` triggers fallback on
disagreement/OOD; `planner/why_not.py` emits the minimal certificate. Every
recommendation carries an auditable `PlannerResult` ledger.
