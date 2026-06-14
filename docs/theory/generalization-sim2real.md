<!--
  GPU-Center World Model — theory: generalization & sim2real.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Generalization & sim2real guarantees

> Theory summary by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.
> Source: research design doc O. Realised in `evaluation/`,
> `evaluation/sample_complexity.py`, the provable-vs-empirical ledger
> (`evaluation/reports/provable_vs_empirical.py`).

This page is the project's honesty layer: what generalizes, how much data it takes,
and exactly which claims are proved vs. empirical.

## The statistical unit is the decision window

A decision window is $Z_t = (X_t, A_t, Y_t, E_t)$: pre-action typed belief, a safe
action (post-kernel), horizon-indexed KPI/risk + **pairwise action-delta** targets,
and a shared exogenous seed bundle. The learned part is residual and latent;
**hard feasibility is the exact kernel, not learned** (assumption A3). Contiguous
temporal splits prevent post-action leakage (A4). Treating windows as i.i.d. rows
with a flat VC bound is rejected — logs are dependent and the class is
typed-relational.

## Four generalization bounds

- **GB1 — dependent decision-window bound.** Via blocking, $L(\theta) \le \hat
  L_n(\theta) + O(\Xi_G/\sqrt{n_\text{eff,mix}}) + O(\sqrt{\log(1/\delta)/
  n_\text{eff,mix}})$, where $\Xi_G$ is graph sensitivity and $n_\text{eff,mix}
  \approx n/\tau_\text{mix}$ discounts for temporal dependence.
- **GB2 — adaptive-sequence martingale** (Freedman/Bernstein) for certifying future
  contiguous blocks; a sequential-information form shows strong priors and
  conservative fine-tuning shrink algorithm-dependent complexity.
- **GB3 — dependent-data PAC-Bayes** for the exact-kernel + residual split:
  $L(Q) \le \hat L_n(Q) + c_3\Xi_G/\sqrt{n_\text{eff,mix}} + c_4\sqrt{(\text{KL}(Q\|
  P) + \log(1/\delta) + \Psi_\text{dep})/n_\text{eff,mix}}$. Because $K$ is exact and
  fixed, the model **does not spend sample complexity learning hard rules** (E3) —
  the formal reason to prefer simulator/self-supervised initialization, spectral
  control, and symbolic hard rules (T6).
- **GB4 — margin bound for counterfactual ranking.** With action margin $\gamma_0$,
  if pairwise error $< \gamma_0/2$ on every pair, the full window ranking is
  correct (T7); under sub-Gaussian pairwise error, sign-error $\le
  \exp(-n_\text{eff,mix}\gamma_0^2/(8\sigma_\Delta^2))$ (E4). **Sample complexity is
  controlled by the action margin, not raw observation fidelity** — the formal case
  against certifying one-step latent MSE.

## Sim2real, compounding error, identifiability

- **Sim2real transfer:** target risk $\le$ sim risk + a query-conditioned
  divergence term; **calibrated** domain randomization (fit to offline real data)
  shrinks misspecification while widening beyond calibrated uncertainty *increases*
  the shift term (claim L4).
- **Compounding error:** with $L_F$-Lipschitz dynamics under shared noise,
  multi-step error grows but is bounded and mitigated by hierarchy + uncertainty
  propagation (L6).
- **Identifiability (negative result):** from single-view partial telemetry, the
  latent is identifiable **only up to** transformations preserving the observation
  law and action-sufficiency (L7). Hence the focus on *action-sufficient* structure.

## Sample-complexity numbers

- One-sided Bernoulli certification: $n_\text{eff} \ge \log(1/\delta)/(2\gamma^2)
  \approx 600$ for $\delta=0.05, \gamma=0.05$.
- Two-sided 90% coverage within $\pm5$ pts at 95% confidence: $\approx 738$ windows.
- **Rule of three:** zero false-safe in $N$ severe-shift cases ⇒ 95% upper bound
  $\approx 3/N$; for false-safe $< 1\%$, $N \approx 300$.
- Optimism-bias $< 5\%$ on the non-abstained subset: $\approx 1.7$k non-abstained
  cases (⇒ $\approx 2.8$k windows at 0.60 coverage).
- **Recommended release:** ~20k–50k real windows; ~14k–35k train, 3k–7k val/seen,
  3k–8k held-out across slices; **4** safe branches/window (6 pairwise comparisons).

Below ~10k real windows the model can still be useful, but the strongest claims
lean on simulator assumptions — and the ledger must say so.

## The provable-vs-empirical-assumed ledger

Every release ships a ledger classifying each claim:

| Claim | Status |
|---|---|
| L1 no exact-kernel-unsafe recommendation | provable under implementation assumptions |
| L2 generalization gap shrinks at $n_\text{eff}^{-1/2}$ | provable conditional (constants empirical) |
| L3 counterfactual ranking is the right target | provable formulation; empirical performance |
| L4 calibrated DR beats blind DR | provable conditional |
| L5 counterfactual correctness relative to branch env only | assumed & explicitly limited |
| L6 rollout error bounded/mitigated | provable conditional (constants empirical) |
| L7 latent not uniquely identifiable from telemetry | provable negative |
| L8 synthetic data can raise effective sample size | provable conditional |
| L9–L10 acceptance / operator usefulness | empirical |
| L11 query-conditioned intervention-faithful design | theory-motivated, strong empirical support |

See [Benchmark → Acceptance](../benchmark/acceptance.md) for how these are reported.

## Why this matters

It is the antidote to optimism bias at the *project* level: no number in the repo
is allowed to masquerade as a proof, and every "provable" line is pinned to its
assumptions (kernel applied first, faithful, bug-free; exchangeable calibration; no
leakage).
