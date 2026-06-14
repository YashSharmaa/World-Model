<!--
─────────────────────────────────────────────────────────────────────────────
 GPU-Center World Model — pull request template
 Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
 License: Apache-2.0
─────────────────────────────────────────────────────────────────────────────
-->

## Summary

<!-- What does this PR change, and why? Link the issue it closes: "Closes #123". -->

## Affected area(s)

<!-- Tick all that apply. CODEOWNERS will request the right reviewers. -->

- [ ] `schema` (frozen contract — changes need explicit owner sign-off)
- [ ] `ingest`
- [ ] `data_engine`
- [ ] `simulator` (exact kernel — safety-critical)
- [ ] `model` (learned residual / MLX)
- [ ] `planner`
- [ ] `serving` (backend)
- [ ] `apps/web` (frontend)
- [ ] `evaluation` / `benchmarks`
- [ ] `uncertainty` / `reliability` / `fairness`
- [ ] `docs` / cards
- [ ] CI / meta / release

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor (no behavior change)
- [ ] Docs
- [ ] CI / tooling / release

## The spine (must remain true)

<!-- These invariants are the whole point of the project. Confirm you preserved them. -->

- [ ] Hard feasibility (quota, gang atomicity, topology, health, operator policy)
      stays in the **exact symbolic kernel** — not moved into the learned model.
- [ ] The learned model only **ranks / forecasts / abstains** over the
      kernel-certified **safe** candidate set. Safety is a **gate, not a weighted metric**.
- [ ] Determinism preserved: the same pre-action state hash + seed bundle still
      reproduce the same safe set / replay trace.
- [ ] No new mandatory dependency on an optional extra (mlx / serve / data /
      baselines) for the core import path.

## Checklist

- [ ] `ruff check` and `ruff format --check` pass.
- [ ] `pyright` passes.
- [ ] `pytest -m "not mlx and not slow"` passes locally.
- [ ] New/changed source files carry the Apache-2.0 attribution header.
- [ ] Docs / cards updated if behavior or public contracts changed.
- [ ] Added or updated tests (unit / property / golden / integration) as appropriate.

## Notes for reviewers

<!-- Anything reviewers should focus on, known limitations, follow-ups. -->
