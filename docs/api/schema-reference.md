<!--
  GPU-Center World Model — schema dataclass reference (generated from source).
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Schema contract — dataclass reference (from source)

> Project by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.

The canonical, versioned schema contract (`SCHEMA_VERSION`) rendered directly from
the `gpu_center_world_model.schema` and `gpu_center_world_model.data_engine`
sources. Docstrings, fields, and type signatures are the single source of truth,
so this reference can never drift from the code (doc H: *generate from source*).

For the conceptual narrative — the four-block hybrid state
$S_t = (X_t, Z_t, C_t, M_t)$ and the pre-action `state_hash` — see
[State & hashing](../schema/state.md).

## Hybrid world state $S_t = (X_t, Z_t, C_t, M_t)$

::: gpu_center_world_model.schema.state.WorldState

::: gpu_center_world_model.schema.state.SymbolicState

::: gpu_center_world_model.schema.state.ResidualBelief

::: gpu_center_world_model.schema.state.ExogenousContext

::: gpu_center_world_model.schema.state.SourceState

## Actions (the kernel-checked action alphabet)

::: gpu_center_world_model.schema.actions.Hold

::: gpu_center_world_model.schema.actions.Admit

::: gpu_center_world_model.schema.actions.Place

::: gpu_center_world_model.schema.actions.AdmitAndPlace

::: gpu_center_world_model.schema.actions.MaskDomain

## Pre-action hashing (branch anchor)

::: gpu_center_world_model.schema.hashing.world_state_hash

::: gpu_center_world_model.schema.hashing.state_hash

## Data-engine contracts (decision window / labels / provenance)

::: gpu_center_world_model.data_engine.contracts.DecisionWindow

::: gpu_center_world_model.data_engine.contracts.LabelPack

::: gpu_center_world_model.data_engine.contracts.ProvenanceManifest
