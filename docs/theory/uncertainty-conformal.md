<!--
  GPU-Center World Model — theory: uncertainty & conformal guarantees.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Uncertainty & conformal guarantees

> Theory summary by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.
> Source: research design doc L. Realised in `uncertainty/` (`conformal.py`,
> `selective.py`, `ood.py`, `decomposition.py`, `_betainc.py`) and `model`
> calibration.

Calibrated uncertainty is what turns the learned forecasts into a **gate** rather
than decoration. This page is the project's recipe for finite-sample guarantees.

## A three-layer UQ stack

1. **Aleatoric** via quantile / hazard heads,
2. **Epistemic** via ensembles (BatchEnsemble / GRU-BatchEnsemble — matching deep
   ensembles at lower cost, MLX-friendly),
3. **Post-hoc conformal** calibration for finite-sample guarantees.

The components are estimated **separately and calibrated jointly**, never collapsed
into one confidence scalar.

## Predictive heads

- **Multi-horizon quantile KPI-delta head** on a $\tau$-grid
  $\{0.05, 0.10, 0.50, 0.90, 0.95\}$ for each KPI × horizon, with a non-crossing
  penalty.
- **Discrete-time survival / competing-risks head** for mutually exclusive terminal
  events (finish vs requeue vs timeout) and parallel binary hazards for
  co-occurring incidents.
- **Bundle-average residual-risk head** $\bar r_e(x,a) = \frac1B \sum_b
  L_{e,b}(x,a)$ over the exogenous seed bundle, conformalized one-sided upward.

## Conformal methods used

| Method | Used for |
|---|---|
| **Split conformal** | marginal coverage on KPI intervals (the classic exchangeability theorem) |
| **CQR** (conformalized quantile regression) | sharper, quantile-adaptive intervals |
| **Mondrian group-conditional** | coverage on prespecified slices $G = g(X)$ (action family × topology × regime) — full conditional coverage is impossible in general, group-conditional is feasible |
| **One-sided upper conformal** | risk targets that only need an *upper* bound |
| **Beta small-sample correction** | optional stronger statement: coverage $\ge 90\%$ with confidence $1-\delta$ over the calibration draw |

## Safety gate as a chance constraint

For each $a \in A_\text{exact}(x)$ and hazard $e$, the gate keeps $a$ only if the
**conformal upper bound** $U_\alpha(x,a) \le$ the operator's risk budget — never
the nominal point estimate. This is the formal hinge between conformal UQ and
[safety-as-a-gate](../adr/0002-safety-as-a-gate.md): conformal risk control extends
to bounded, non-monotonic, OCE, and CVaR-like tail risks.

## Selective prediction (abstention) with a certificate

A **split-select-certify** protocol: on a selection split, pick the trust
threshold $\tau^\*$ maximizing coverage subject to risk budgets; on an
*independent* certification split, certify the accepted risk with an exact one-sided
**Clopper–Pearson** upper bound $U_R(\tau^\*;\delta) = \text{BetaInv}(1-\delta_R;
k_\tau+1, m_\tau-k_\tau)$ and a coverage lower bound. Operating point targets:
overall optimism budget $\le 0.05$; coverage floor $\ge 0.60$ for ranking claims;
on the severe-shift subgroup, accepted risk $\le 0.01$ and coverage upper-bound
$\le 0.20$ (equivalently abstain/fall-back on $\ge 80\%$ of flagged cases). The
incomplete-beta inverse is implemented in `uncertainty/_betainc.py`.

## One-sided optimism-bias bound

Optimism bias is a **one-sided underestimation** problem: define $Z(x,a) =
R^\*(x,a) - R^\*(x,a_\text{fb})$ (true excess interventional risk relative to a
fallback) and bound it one-sided upward via conformal recalibration. This certifies
the < 5% non-abstained optimism-bias [acceptance target](../benchmark/acceptance.md).

## OOD detection → abstain, never act riskier

Full-conformal novelty detection gives distribution-free FDR control with a
shift-aware extension. Because conformal novelty detectors can be adversarially
perturbed, OOD signals **trigger abstention**, not riskier actions (doc K, C9).

## Shift handling

- **Covariate shift** — weighted exchangeability within a fixed action bucket.
- **Temporal drift** — ACI / BC-ACI give long-run / time-uniform coverage (not
  classical one-shot finite-sample per point — stated honestly in the ledger).
- **Action-induced shift** — the interventional target is handled by the
  branch-conditioned calibration buckets.

## Provable vs empirical

- **Provable:** zero exact-kernel violations (kernel-first); marginal 90% coverage
  per exchangeable bucket + group-conditional on Mondrian slices; one-sided upper
  coverage for excess-risk targets.
- **Weakened/conditional:** the 85–95% empirical-coverage check; ACI drift
  guarantees are asymptotic/time-uniform; the $\ge 80\%$ severe-shift abstention.
- **Empirical:** pairwise sign accuracy, safe-regret improvement, interval
  sharpness, OOD recall, operator usefulness.

See [Benchmark → Acceptance](../benchmark/acceptance.md) and
[Generalization & sim2real](generalization-sim2real.md).
