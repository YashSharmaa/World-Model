<!--
  GPU-Center World Model — ADR-0001.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# ADR-0001 — Hybrid: exact kernel + learned residual

- **Status:** Accepted
- **Date:** 2026-06-14
- **Author:** [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)
- **Deciders:** Yash Sharma

## Context

A world model for GPU-center operations must answer **intervention** questions
correctly (*what happens if I act?*), not merely produce plausible futures. Two
naive architectures were on the table:

1. a **pure end-to-end** learned model that also learns feasibility, and
2. a **purely analytic** queueing/physics model with no learned component.

The operational state and admissible actions are already standardized and exact
(Kueue quotas/borrowing, gang atomicity, Kubernetes DRA device allocation,
topology-aware scheduling, DCGM telemetry). Some dynamics, by contrast, are
genuinely uncertain (remaining runtime, hotspot emergence, device-failure
hazards, thermal/headroom residuals).

## Decision

Adopt a **hybrid** architecture that splits the state $S_t = (X_t, Z_t, C_t,
M_t)$:

- an **exact symbolic kernel** owns $X_t$ — hard, current-state feasibility, read
  directly from scheduler/topology/policy state and **never learned**;
- a **learned residual world model** owns $Z_t$ — uncertain dynamics, with
  calibrated uncertainty, predicting **innovations** $y_{\text{realized}} -
  y_{\text{analytic}}$ relative to an analytic spine.

The split is enforced by the boundary rule: *if a rule can be read directly from
scheduler/topology/device/policy state, it lives in the exact kernel.*

## Consequences

**Positive**

- Supports a **constructive zero-violation** guarantee (see
  [ADR-0002](0002-safety-as-a-gate.md)): the learned model ranks/forecasts only
  over the kernel-certified safe set.
- The fixed exact kernel **shrinks the learned hypothesis class**, so sample
  complexity is spent on real residuals, not re-derived scheduler math (doc O,
  GB3/T6).
- The analytic spine acts as a **control variate**, reducing variance in training
  and planning (doc K, EqP6).
- Auditable: every recommendation carries exact blockers and a why-not ledger.

**Negative / costs**

- Two code paths to maintain (symbolic kernel + learned backend) and a
  backend-neutral contract between them (`*/contracts.py`).
- The kernel must faithfully encode the intended hard semantics; a kernel bug is a
  safety bug (made explicit in the provable-vs-empirical ledger's assumptions).

## Alternatives rejected

- **Pure end-to-end latent dynamics** (doc K, R1): buries feasibility, breaks
  auditability, cannot support a constructive zero-violation theorem.
- **Purely analytic model** (doc K, R2): topology-conditioned throughput, startup
  maxima, failures, and headroom residuals are materially data-dependent.

## References

- [Architecture → Exact kernel vs learned residual](../architecture/exact-kernel-vs-learned.md)
- [Causal foundations](../theory/causal-foundations.md),
  [Queueing & control dynamics](../theory/queueing-dynamics.md)
