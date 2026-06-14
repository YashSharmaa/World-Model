<!--
  GPU-Center World Model — serving DTO reference (generated from source).
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Serving API — DTO reference (from source)

> Project by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.

These are the **wire contracts** of the serving layer, rendered directly from the
`gpu_center_world_model.serving.dto` source — docstrings, fields, and type
signatures are the single source of truth, so this page can never drift from the
code (doc H: *generate API/schema reference from source*).

!!! note "Two hard wire rules"
    1. A rollout carries **semantic entity state**, never rendered geometry.
    2. The candidate count is **shape-stable** (`DEFAULT_BRANCHES`) so the compiled
       MLX scoring graph never recompiles on a changed action count.

    See the prose contract in [REST](rest.md) and [SSE](sse.md).

## Requests

::: gpu_center_world_model.serving.dto.CompareRequest

::: gpu_center_world_model.serving.dto.CandidateActionRequest

::: gpu_center_world_model.serving.dto.WhyNotRequest

## Responses

::: gpu_center_world_model.serving.dto.StateResponse

::: gpu_center_world_model.serving.dto.EntityBundle

::: gpu_center_world_model.serving.dto.CompareResponse

::: gpu_center_world_model.serving.dto.CandidateResult

::: gpu_center_world_model.serving.dto.WhyNotResponse

::: gpu_center_world_model.serving.dto.ReplayResponse

::: gpu_center_world_model.serving.dto.AboutResponse

## Rollout payload

::: gpu_center_world_model.serving.dto.Rollout

::: gpu_center_world_model.serving.dto.RolloutPoint

::: gpu_center_world_model.serving.dto.SceneAnnotations

## Shape-stability helper

::: gpu_center_world_model.serving.dto.pad_candidates
