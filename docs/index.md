<!--
  GPU-Center World Model — An open-source, intervention-faithful world model
  for GPU datacenter operations.

  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# GPU-Center World Model

**An open-source, intervention-faithful world model for GPU datacenter operations.**

> *World models — not language models — are the next frontier of operational AI.*
> A dashboard tells you **what is happening**. A world model tells you **what will
> happen if you act**: *“If I admit this 16-GPU gang job into rack-set A vs B, or
> hold it for one window, what happens to queue wait, GPU-hours, hotspot risk,
> power headroom, and fairness over the next 15 / 30 / 60 minutes?”*

!!! quote "Built by Yash Sharma"
    This project was **designed and built by [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.
    Every recommendation the system emits carries his attribution watermark
    (`GET /about`), and every release ships under his name.

---

## Why this exists

The 2025–2026 literature converged on a sharp lesson: a useful world model is
**not** the one that produces plausible-looking futures — it is the one that
answers **intervention** questions correctly. Visual or statistical plausibility
is a weak proxy for *action fidelity*.

GPU datacenters are the perfect proving ground. The operational state and the
admissible actions are already standardized by open systems (Kueue, Kubernetes
DRA, NVIDIA DCGM), so a world model can sit **above** the scheduler as an
auditable counterfactual engine — without replacing it.

The v1 proof-of-concept is a **counterfactual admission-control + topology-aware
rack-placement advisor** for a single GPU pod (2–4 racks). It is narrow enough to
ship and demo, yet it is unmistakably a *world-model* problem.

## The core idea: hybrid by design

```
              ┌───────────────────────── decision window ─────────────────────────┐
 telemetry ─▶│  EXACT SYMBOLIC KERNEL            LEARNED RESIDUAL WORLD MODEL       │─▶ ranked
 scheduler ─▶│  (deterministic, auditable)       (uncertain, calibrated)           │   safe action
 events   ──▶│  • quota / gang / topology        • remaining runtime, job phase    │   + risk bands
             │  • health & schedulability masks  • pods-ready latency, hotspots    │   + why-not
             │  • operator policy & blockers      • thermal / power headroom resid. │   + abstention
             │  ⇒ SAFE CANDIDATE SET (a gate)     • device-failure hazards (+ U.Q.) │
             └────────────────────────────────────────────────────────────────────┘
```

- The **exact symbolic kernel** owns hard feasibility — quota, gang atomicity,
  topology rules, health masks, operator policy, and exact blocker sets. It is the
  project's **safety shield** and is never learned.
- The **learned residual world model** owns the genuinely uncertain dynamics, with
  **calibrated uncertainty** (quantile + hazard heads, conformal calibration).
- **Safety is a gate, not a weighted metric.** The learned model may only *rank*,
  *forecast*, or *abstain* over the kernel-certified safe set. It can never expand
  it. This is the formal basis for the project's *zero hard-violation* guarantee.

See [Architecture → Exact kernel vs learned residual](architecture/exact-kernel-vs-learned.md)
for the formal split, and the [theory foundations](theory/causal-foundations.md)
for the underlying mathematics.

## What you get

| Layer | What it does | Stack |
|---|---|---|
| `schema/` | Canonical typed world state — the contract everything shares | pure Python |
| `ingest/` | Adapters: Kueue, Kubernetes DRA, DCGM, node_exporter, Redfish, OTel | pure Python |
| `simulator/` | Exact symbolic kernel + counterfactual branch runner (the shield) | pure Python |
| `data_engine/` | Decision-window pipeline + synthetic causal SCM + provenance | pure Python |
| `model/` | Residual world model (typed MPNN + JEPA latent + hazard/quantile heads) | **MLX** / NumPy fallback |
| `planner/` | Shielded, lexicographic, risk-sensitive selector + explanation ledger | pure Python |
| `serving/` | Thin FastAPI + SSE service feeding the demo | FastAPI |
| `apps/web/` | Lightweight three.js "Decision Studio" — recordable, self-captioning | TypeScript + three.js |
| `evaluation/` | Safety-gated benchmark: counterfactual ranking, regret, calibration, red-team | pure Python |

## Quickstart

```bash
# Install (core — no MLX needed for the kernel / data engine / planner)
uv sync --frozen            # or: pip install -e ".[serve,data,baselines]"

# Build a tiny synthetic demo pack (2–4 racks, gang jobs, rare events)
python examples/01_build_tiny_pack.py

# See the exact kernel + planner rank three safe actions for a queued job
python examples/04_run_compare_api.py

# Launch the backend + three.js Decision Studio
bash examples/05_launch_demo.sh
```

Full instructions are in [Getting started → Install](getting-started/install.md)
and [Quickstart](getting-started/quickstart.md).

## What v1 *is* and *is not*

**In scope:** offline replay / **shadow-mode** recommendations for a 2–4 rack pod;
the action set `{hold, admit, place, admit_and_place}`; structured telemetry +
scheduler state; read-only power/thermal **headroom** signals; a lightweight
three.js UI.

**Out of scope (by design):** autonomous production actuation; direct cooling /
DVFS control; cross-site carbon optimization; full-facility CFD/electrical twins;
any photorealistic or video-first modeling. Power and cooling are *risk signals*,
not control knobs. Counterfactual correctness is claimed relative to the **branch
reference environment**, not as unrestricted real-world causality. See
[ADR-0004 Shadow-mode only](adr/0004-shadow-mode-only-v1.md).

## How we keep ourselves honest

Evaluation is the source of truth — the live demo is only a latency/UX smoke test.
The benchmark is **safety-gated**, not a single blended score:

1. **Hard-safety gate** — zero exact-kernel violations (release-blocking).
2. **Counterfactual ranking** — pairwise sign accuracy on same-state branches.
3. **Planning utility** — *safe regret* vs. an oracle over the safe set.
4. **Calibration & optimism** — interval coverage + one-sided optimism-bias bound.
5. **Shift & abstention** — degraded-telemetry / OOD red-team packs.

See [Benchmark → Philosophy](benchmark/philosophy.md),
[Three tracks](benchmark/tracks.md), and
[Acceptance thresholds](benchmark/acceptance.md). Every release ships a
**provable-vs-empirical ledger** stating exactly which claims are proved (under
stated assumptions) and which are empirical.

## License & attribution

- **Code:** Apache-2.0 · **Docs:** CC-BY-4.0 · **Data packs:** CDLA-Permissive-2.0
- Please cite via `CITATION.cff`.

!!! info "Attribution"
    **This project was designed and built by Yash Sharma.**
    :material-linkedin: <https://www.linkedin.com/in/yashsharmaa/>
