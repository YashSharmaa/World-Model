<!--
  GPU-Center World Model — ADR-0004.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# ADR-0004 — Shadow-mode only for v1

- **Status:** Accepted
- **Date:** 2026-06-14
- **Author:** [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)
- **Deciders:** Yash Sharma

## Context

The system reasons about safety-critical infrastructure (admission, placement,
power/thermal headroom). A natural temptation is to let it **actuate** the
scheduler directly. For a public v1 proof-of-concept, that would be premature and
would over-claim the project's guarantees.

## Decision

v1 is **offline replay / shadow mode only**. The system *advises*; it never acts.

- The action alphabet is advisory: `{hold, admit, place, admit_and_place}` plus
  `mask_domain` as an operator guardrail.
- Power and cooling are **read-only risk signals**, not control knobs. There is no
  direct cooling / DVFS / power-cap actuation.
- Counterfactual correctness is claimed **relative to the branch reference
  environment**, not as unrestricted real-world causality (doc O, claim L5).

## Consequences

**Positive**

- The *zero hard-violation* claim stays **narrow and honest**: it covers modelled
  current-state hard feasibility, under correct kernel semantics and atomic
  re-validation before any (future) actuation — not autonomous production control.
- The benchmark, not the live demo, is the source of truth; the demo only proves
  latency, stability, and safe behaviour under scripted interventions.
- Operator usefulness is measured behaviorally (decision-time, override rates),
  and explicitly marked **provisional** if no timed study is run (doc F).

**Negative / costs**

- No closed-loop control benefits in v1 (deferred to a later "read → advisory →
  guarded actuation" progression on the [roadmap](../index.md)).

## Always deferred for v1

Autonomous production actuation; direct cooling / DVFS control; cross-site carbon
optimization; full-facility CFD / electrical twins; any photorealistic or
video-first modeling.

## What this enables later (hooks kept open)

- A **guarded actuation** path (read → advisory → atomic re-validate → act).
- A **fairness ranking layer** (DRF / PF / NBS) over the safe set (v2 hooks already
  present; see [Fairness & mechanism design](../theory/fairness-mechanism.md)).
- **Reliability changepoint surveillance** wired into the abstention gate.

## References

- [Architecture → Overview](../architecture/overview.md) (scope & honesty)
- [Generalization & sim2real](../theory/generalization-sim2real.md) (the
  provable-vs-empirical ledger, claim L5)
- [Benchmark → Philosophy](../benchmark/philosophy.md)
