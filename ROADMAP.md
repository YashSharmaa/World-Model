<!-- Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/ · License: CC-BY-4.0 -->
# Roadmap

> Maintained by **Yash Sharma** — https://www.linkedin.com/in/yashsharmaa/

## Now (v0.1 — public proof-of-concept)
- Canonical schema contract + exact symbolic kernel (`hold` / `admit` / `place`).
- Synthetic causal-SCM data engine + tiny public demo pack with provenance.
- Residual world model (typed MPNN + JEPA latent + quantile/hazard heads) on MLX,
  with a NumPy fallback for non-Apple environments.
- Shielded, lexicographic, risk-sensitive planner + explanation ledger.
- FastAPI + SSE serving and the three.js Decision Studio.
- Safety-gated evaluation harness + provable-vs-empirical ledger.

## Next
- Real-cluster ingestion (live Kueue/DCGM) behind feature flags.
- Conformal calibration certificates on real held-out interventions.
- Health-aware drain / requeue module reusing the same latent.
- Reliability changepoint surveillance (CUSUM / SR / BOCPD) wired to abstention.

## Later
- Cooling-aware power-cap advisory (read → advisory → guarded actuation).
- Fairness layer (DRF / PF / NBS) as a v2 ranking objective over the safe set.
- Multi-Mac MLX scale-out; optional Swift/C++ packaging via exported functions.
- Broader datacenter digital-twin scope (cooling loops, power domains, carbon).

**Always deferred for v1:** autonomous production actuation, full-facility CFD,
photorealistic/video-first modeling.

## Contributor on-ramp (launch checklist)
Before the public announcement, the milestones, labels, and starter issues are
live so outside interest has somewhere to go (doc H launch checklist). The
starter issues are tracked in-repo at
[`scripts/good_first_issues.yml`](scripts/good_first_issues.yml) — 12 scoped
issues with at least one per major area, none requiring private data or a
production cluster — and are materialised into GitHub (with the
`good-first-issue` / `help-wanted` / `needs-design` / `blocked` / `release/v0.1.0`
labels) via [`scripts/seed_starter_issues.sh`](scripts/seed_starter_issues.sh).
See [CONTRIBUTING](CONTRIBUTING.md#good-first-issues) for the first-contribution
path.
