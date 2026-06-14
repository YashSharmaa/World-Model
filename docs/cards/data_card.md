<!--
  GPU-Center World Model — data card.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Data card — GPU-Center decision-window packs

> **Author / owner:** [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/) ·
> **Project:** GPU-Center World Model · **Schema version:** `1.0.0` ·
> **Data license:** CDLA-Permissive-2.0 (this card: CC-BY-4.0).

This card describes the **decision-window data packs** the model trains and is
evaluated on. The machine-readable form is `data_card.json`, emitted by
`data_engine/release_manifest.py`; a release is **only emitted when its provenance
manifest validates and (when a fidelity report is supplied) is certified**.

## Sources

| Source | Provides |
|---|---|
| **Synthetic causal-SCM engine** | the bulk of v1 windows — observational + counterfactual branches |
| **Kueue** | scheduler / queue / admission / eviction surfaces |
| **Kubernetes (DRA)** | device-allocation semantics (optional, feature-flagged) |
| **NVIDIA DCGM** | per-GPU power, temperature, util, XID/ECC, recovery action |
| **node_exporter** | host & sensor metrics |
| **Redfish** | out-of-band inventory, power, thermal metadata |
| **OpenTelemetry + Prometheus** | vendor-neutral collection backbone |

Each window records its **origin** (`real` / `sim_observational` /
`sim_counterfactual`) and traces back to its generator commit.

## Schema & unit

- **Schema version:** `1.0.0` (the canonical typed contract).
- **Unit:** the **decision window** (`DecisionWindow`) — a pre-action state hash,
  a seed bundle, candidate action ids, and URIs into the observational context,
  the realized horizon, and the counterfactual pack. Labels (`LabelPack`) are
  multi-task and intervention-aware: quantile KPI targets + hazard targets +
  explanation tags, per (decision, action, horizon).
- **Horizons:** 15 / 30 / 60 minutes (with a 5-minute fast probe).

## Cadences

Observation families arrive multi-rate and are **never** forced onto one
synchronous frame by zero-imputation:

| Family | Nominal cadence |
|---|---|
| GPU telemetry / profiling | 5–30 s |
| node/rack telemetry | 30 s – 1 min |
| placement state | event + 1 min |
| scheduler state | event + 5 min |
| exogenous context | 5–15 min |

## Counterfactual construction

Branches share **one exogenous seed bundle** (common random numbers) and are
rooted at the **same pre-action state hash**, so only the action structural
equation changes between factual and counterfactual branches (doc R, A4/A5). This
is what makes branch deltas estimate intervention effects rather than
intervention-plus-noise. `mask_domain` enters as an **exogenous guardrail /
scenario variable**, not as an actor action.

## Calibration window

A held-out, exchangeable **calibration window** (recorded in the provenance
manifest's `calibration_window`) supplies the conformal calibration corpus. Split
policy (`Split`): `train`, `val`, and disjoint test slices —
`test_temporal`, `test_topology`, `test_intervention`, `test_rare_event`,
`test_severe_shift`. Contiguous temporal splits prevent post-action leakage.

## Provenance & reproducibility

Every release carries a `ProvenanceManifest`: `schema_version`,
`generator_commit`, `simulator_version`, a **seed registry**, a source inventory,
the calibration window, the **real fraction**, and a list of **known-invalid
uses**. The author/url fields name **Yash Sharma**. Provenance-blind mixing is an
explicit anti-pattern; any pack that ships without lineage or certification is
treated as untrustworthy (doc R gating).

## Fidelity certification

Synthetic packs are validated by a fidelity suite before release: exact-kernel
conformance (a deterministic-zero audit), kernel two-sample / energy-distance /
classifier two-sample tests on path space, TSTR/TRTS predictive-utility tests, and
**counterfactual fidelity against held-out real interventions** (doubly-robust).
Poisoning detectors guard against the classic traps — channel-randomization that
breaks lag structure, cartoon anomalies, pre-decision leakage, action-ignorance,
generator-agreement traps, support-overlap failures, and benign-missingness. See
[Synthetic fidelity](../theory/synthetic-fidelity.md).

## Blind spots & limitations

- **Sim-to-real gap.** Synthetic data raises *effective sample size* but does not
  remove adaptation bias; it is real-equivalent only when the query-conditioned
  sim-to-real divergence is small (doc O, L8). The data card records the real
  fraction so readers can judge.
- **Latent non-identifiability.** Telemetry alone does not pin down the latent
  state; some structure is only identifiable up to action-sufficient symmetries.
- **Rare-event coverage.** Incident bundles are simulated; held-out *real*
  incidents are scarce, so rare-event claims lean more on simulator assumptions —
  stated explicitly in the provable-vs-empirical ledger.
- **MNAR missingness.** Real failures often surface first as missing/degraded
  telemetry; the data preserves missingness as informative state, but the exact
  MNAR mechanism is modelled, not observed.

## License & attribution

Data packs: **CDLA-Permissive-2.0**. Designed and built by **Yash Sharma**
(:material-linkedin: <https://www.linkedin.com/in/yashsharmaa/>); cite via
`CITATION.cff`.
