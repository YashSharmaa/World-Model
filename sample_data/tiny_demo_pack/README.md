<!--
  GPU-Center World Model — An open-source, intervention-faithful world model
  for GPU datacenter operations.

  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
  License: Apache-2.0
-->

# Tiny Demo Pack

A minimal, Git-tracked fixture that lets a practitioner run the GPU-Center
World Model end to end — build decision windows, train the `mlx_smoke` config,
and launch the Decision Studio demo — **without downloading any release asset**.
It is intentionally tiny: only enough decision windows to exercise the pipeline
and prove shapes, masks, metrics, and the safety shield all line up.

This pack is *generated*, not hand-edited. It is materialised from the
checked-in scenario packs in [`scenarios/`](../../scenarios/) by the demo-pack
generator that [`examples/01_build_tiny_pack.py`](../../examples/01_build_tiny_pack.py)
calls, which in turn uses
[`scenario_loader`](../../src/gpu_center_world_model/data_engine/scenario_loader.py)
to compile each scenario YAML into an authoritative `WorldState` + pending
queue. Larger benchmark and demo corpora are published as GitHub Releases /
Zenodo archives, never committed here (see `.gitignore` → `sample_data/large/`).

## How to (re)build it

```bash
python examples/01_build_tiny_pack.py --out sample_data/tiny_demo_pack
```

The generator is deterministic: same scenarios + same seed bundle ⇒ byte-stable
windows and a stable per-window `state_hash`. Counterfactual branches in this
pack share one exogenous seed bundle so branch deltas estimate intervention
effects, not intervention-plus-noise (the project's core counterfactual rule).

## What's inside (once built)

| Artifact | Purpose |
| --- | --- |
| `decision_windows/` | Gold-layer `DecisionWindow` records: pre-action `state_hash`, candidate action ids, realized + counterfactual horizon URIs. |
| `label_packs/` | Multi-task, intervention-aware `LabelPack`s (KPI quantiles + hazards) per `(decision, action)`. |
| `provenance/manifest.json` | The `ProvenanceManifest` for this pack (see data card below). |
| `seed_registry.json` | Per-stream master seeds for reproducible branch replay. |

The schema for every artifact is the frozen contract under
[`schema/`](../../src/gpu_center_world_model/schema/) and
[`data_engine/contracts.py`](../../src/gpu_center_world_model/data_engine/contracts.py)
(`DecisionWindow`, `LabelPack`, `ProvenanceManifest`). Nothing in this pack
redefines those types.

## Source scenarios

The pack draws one or more decision windows from each shipped scenario family so
the smoke and demo paths see normal, stress, and rare-event behaviour:

- **normal** — `steady_pod`
- **stress** — `thermal_stress`, `scheduling_stress`
- **rare_events** — `gpu_fault_burst`, `cooling_derate`, `pods_ready_timeout`,
  `power_derate`, `queue_burst`, `maintenance_mask`, `telemetry_degradation`

Each scenario describes a 2–4 rack demo pod (racks → nodes → GPUs), a queue of
mixed gang jobs (1-GPU inference, 4-GPU fine-tune, 8/16-GPU training), an
operator policy, and any single rare-event injection plus a seed.

## Data card → provenance pointer

This pack is **synthetic** (`origin = sim_observational` + paired
`sim_counterfactual` branches); it contains no real cluster telemetry. The
authoritative, machine-readable data card is the
[`ProvenanceManifest`](../../src/gpu_center_world_model/data_engine/contracts.py)
written to `provenance/manifest.json` at build time. Per the data-card
discipline the project commits to, that manifest declares at minimum:

- `schema_version` and the canonical units / cadences (fast ≈ 15–30 s GPU
  telemetry, meso ≈ 30–60 s node/rack, sparse event lane for admissions /
  placements / evictions / drains / faults)
- `simulator_version` and `generator_commit`
- `seed_registry` (per-stream master seeds)
- `source_inventory` (the scenario packs that seeded each window)
- `calibration_window` (synthetic — fitted-posterior bounds, not a real window)
- `real_fraction` (0.0 for this pack)
- counterfactual availability (true: ≥ 3 branched safe actions per window)
- `known_invalid_uses`

### Known invalid uses

- **Not** a benchmark of real-world causal validity. Counterfactual truth here
  comes from the exact kernel plus a calibrated residual simulator under fixed
  seeds — a credible open benchmark, not a claim of universal real-world
  fidelity.
- **Not** sized for accuracy. The pack is for *pipeline validation and demo
  recording*; train and evaluate on the full release corpus for any quantitative
  claim.
- The `origin` / scenario metadata travels with the data for auditing but must
  **not** be used as a default model feature (it leaks generator identity rather
  than dynamics).

---

Authored by **Yash Sharma** · https://www.linkedin.com/in/yashsharmaa/ · Apache-2.0
