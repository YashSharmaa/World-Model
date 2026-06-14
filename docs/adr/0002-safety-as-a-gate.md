<!--
  GPU-Center World Model — ADR-0002.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# ADR-0002 — Safety is a gate, not a weighted metric

- **Status:** Accepted
- **Date:** 2026-06-14
- **Author:** [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)
- **Deciders:** Yash Sharma

## Context

It is tempting to fold safety into a single scalar objective — e.g.
`reward = utility − λ·risk`. This is the standard reinforcement-learning framing,
and it is **wrong for this project**: a large enough utility can always buy back a
hard violation, and the resulting system can recommend an infeasible or unsafe
action with high confidence.

## Decision

Treat hard safety as a **gate** applied *before* any learned scoring, and treat
statistical safety as **explicit chance / CVaR constraints**, never as a weight.

1. The exact kernel computes the safe candidate set
   $A_{\text{exact}}(X_t) = \{a : K(X_t,a)=1\}$.
2. The learned model forecasts KPI deltas and **calibrated** hazards over that set.
3. Candidates failing the **conformal / CVaR chance-constraint gate** are dropped:
   for each hazard family $\ell$, require $\mathbb P(Z_\ell > 0) \le \delta_\ell$
   and/or $\mathrm{CVaR}_{\alpha_\ell}(Z_\ell) \le \beta_\ell$, all on the
   *one-sided conformal* side, not the nominal point estimate.
4. Survivors are ranked **lexicographically** (hard safety → statistical safety →
   p95 queue wait & admission → GPU-hours → locality → fragmentation).
5. If epistemic disagreement / OOD / degraded-telemetry exceeds threshold →
   **abstain** or fall back to a conservative rule.

Formally this is a **shielded constrained belief-MDP** (doc M): hard constraints
are enforced *outside* the learning objective by restricting to the shielded
policy class $\Pi_K$; only modelled hazards live inside the risk-constrained
layer (doc M hard/soft separation theorem T1).

## Consequences

**Positive**

- **Constructive zero hard-violation:** the learned model cannot emit an action
  outside $A_{\text{exact}}$ (doc O claim L1, provable under implementation
  assumptions).
- Statistical safety is **certifiable** with finite-sample conformal bounds, and
  reportable separately (interval coverage, optimism-bias, false-safe rate).
- Abstention becomes a first-class, *rewarded* behaviour — the system is rewarded
  for *knowing when not to recommend*.

**Negative / costs**

- Requires a calibration corpus and a conformal calibration step
  (`uncertainty/conformal.py`).
- Lexicographic ranking needs an order-preserving scalarization for numeric regret
  (doc M, D2), and dominating weights chosen carefully on the bounded candidate set.

## Alternatives rejected

- **Single blended reward / mean-variance objective** (doc M, risk_objective):
  variance penalizes benign upside and does not isolate catastrophic tails;
  expectation-only ignores them entirely.
- **Dualizing hard constraints** into the Lagrangian: hard constraints are
  *filtered exactly*, never dualized (doc M, E6).

## References

- [Architecture → Exact kernel vs learned residual](../architecture/exact-kernel-vs-learned.md)
- [Constrained planning](../theory/constrained-planning.md),
  [Uncertainty & conformal](../theory/uncertainty-conformal.md)
- [Benchmark → Acceptance thresholds](../benchmark/acceptance.md)
