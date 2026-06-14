<!-- Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/ · License: CC-BY-4.0 -->
# Contributing

Thanks for your interest in the **GPU-Center World Model**, created by
[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/).

## First contribution path
1. `make dev` — install dev + serve + data + baselines extras.
2. `make pack` — build the tiny synthetic demo pack (no private data needed).
3. `make test` — run the test pyramid.
4. `make lint && make typecheck` — keep CI green.
5. Open a PR using the template; link an issue.

## The one rule that matters most
**Hard feasibility lives in the exact symbolic kernel; never in the learned
model.** If a rule can be read from scheduler / topology / device-allocation /
policy state, encode it symbolically. The learned model may only rank, forecast,
or abstain over the kernel-certified safe set. PRs that blur this boundary will be
asked to refactor.

## Style & checks
- Python: `ruff` (lint+format) and `pyright` (basic).
- Keep `schema/` dependency-light and free of ML/IO logic; schema changes are
  versioned (`SCHEMA_VERSION`).
- New modules carry the standard attribution header (see any source file).
- Add unit + property tests; property tests must cover resource conservation,
  gang atomicity, and deterministic branching.

## Good first issues
Look for the [`good-first-issue`](https://github.com/yashsharmaa/gpu-center-world-model/labels/good-first-issue)
label — typically a new scenario, a new report slice, or a UI panel polish. Avoid
tasks needing private data or production clusters.

The starter-issue catalog is **tracked in-repo** at
[`scripts/good_first_issues.yml`](scripts/good_first_issues.yml): 12 issues with
at least one per major area (docs, schema, data engine, evaluation, benchmarks,
web, serving, planner, uncertainty, reliability, CI, fairness). Maintainers
materialise it into real GitHub issues + the launch labels (`good-first-issue`,
`help-wanted`, `needs-design`, `blocked`, `release/v0.1.0`) with:

```bash
scripts/seed_starter_issues.sh            # dry run — prints what it would create
scripts/seed_starter_issues.sh --apply    # create labels + issues (needs `gh auth login`)
```

## Code of conduct
Be kind and constructive. See `CODE_OF_CONDUCT.md`.
