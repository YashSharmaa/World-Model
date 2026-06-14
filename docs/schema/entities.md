<!--
  GPU-Center World Model — schema entities & relations.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Entities & relations

> Project by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.

The world is a **typed dynamical graph**. `schema/entities.py` defines the
canonical entity dataclasses; `schema/topology.py` defines the typed edges. Every
field maps to an open scheduler/telemetry surface so the representation stays
auditable. Each entity separates **static** (slowly/never changing) from
**dynamic** (evolving telemetry / scheduler state) fields.

## Entity types

The registry `ENTITY_TYPES` enumerates all thirteen canonical kinds:

`workload` · `podset` · `local_queue` · `cluster_queue` · `resource_flavor` ·
`topology_domain` · `node` · `gpu` · `switch` · `rack` · `power_domain` ·
`cooling_zone` · `operator_policy`.

### Workload-side

| Entity | Key role | Notable fields |
|---|---|---|
| `Workload` | the unit of admission and the primary planning object | `priority`, `topology_policy`, `comm_class`, `podsets`, `status`, `est_remaining_s`, `ready_fraction` |
| `PodSet` | a homogeneous group of pods — the **gang building block** | `count`, `gpu_req`, `podset_group`, `ready_count` |
| `LocalQueue` | tenant-facing submission queue → ClusterQueue | `pending_gpu`, `wait_time_p95_s`, `fairshare_weighted_share` |
| `ClusterQueue` | quota / borrowing / fair-sharing / preemption domain | `nominal_quota`, `borrowing_limit`, `lending_limit`, `weighted_share` |
| `ResourceFlavor` | maps workloads to a node subset by labels/taints | `node_label_constraints`, `topology_name` |

!!! note "Demand is computed, never re-learned"
    A workload's resource demand is computed from its PodSets as
    `sum(count × per-replica request)`. The exact kernel reads this arithmetic
    directly; it is never estimated statistically.

### Topology / infrastructure

| Entity | Key role | Notable dynamic fields |
|---|---|---|
| `TopologyDomain` | block / rack / node / compute-domain (TAS hierarchy) | `free_gpu`, `headroom_score`, `hotspot_score` |
| `Node` | a schedulable compute host | `ready`, `schedulable`, `node_health` |
| `GPU` | an allocatable accelerator (DCGM field semantics) | `util`, `temp_c`, `power_w`, `xid_code`, `health_state` |
| `Switch` | network bottleneck / topology boundary (ToR) | `port_util_p95`, `egress_hotspot_score` |
| `Rack` | primary visual & operational grouping | `rack_power_kw`, `rack_headroom_kw`, `inferred_thermal_margin_c` |
| `PowerDomain` | power-budget constraint aggregation | `cap_kw`, `used_kw`, `violation_risk` |
| `CoolingZone` | shared thermal state aggregation | `inlet_temp_c`, `return_temp_c`, `cooling_margin` |
| `OperatorPolicy` | human override / risk configuration (guardrails) | `allowed_actions`, `protected_domains`, `risk_thresholds` |

## Key enumerations

- `WorkloadStatus`: `pending → reserving → admitted → running → finished`, with
  `evicted` / `requeued` (Kueue `Workload` semantics).
- `TopologyLevel`: `block`, `rack`, `node`, `compute_domain` (TAS domains).
- `TopologyPolicy`: `required` (must fit one domain), `preferred` (prefer one,
  fall back), `unconstrained` (fill gaps, reduce fragmentation).
- `CommClass`: `none`, `light`, `allreduce_heavy`, `pipeline` — drives the network
  surrogate.
- `NodeHealth`: `ready`, `degraded`, `unschedulable`, `draining`, `down`.
- `DeviceHealth`: `healthy`, `degraded`, `intermittently_faulty`,
  `reset_recovery`, `masked` — the latent device-health states of the
  [reliability module](../theory/reliability-changepoint.md).

## Typed relations

`RelationType` is the canonical edge set; `RELATION_ENDPOINTS` records each edge's
(source kind, target kind):

| Relation | From → To |
|---|---|
| `SUBMITTED_TO` | workload → local_queue |
| `TARGETS` | local_queue → cluster_queue |
| `CONTAINS_PODSET` | workload → podset |
| `REQUESTS_FLAVOR` | podset → resource_flavor |
| `IN_DOMAIN` | node → topology_domain |
| `HOSTS` | node → gpu |
| `CONNECTED_TO` | node → switch |
| `CONTAINS_NODE` | rack → node |
| `POWERED_BY` | rack → power_domain |
| `COOLED_BY` | rack → cooling_zone |
| `RUNS_ON` | workload/podset → node/gpu |
| `PARENT_OF` | topology_domain → topology_domain |

A `Relation` stores parallel `src` / `dst` index lists plus optional edge
features. The tensorizer compiles these into one-hot **incidence matrices** $S_r$
(edge × source) and $T_r$ (edge × target) for batched message passing — this is
how the typed MPNN is realised without a sparse graph runtime. `HIERARCHY_UP`
fixes the upward pooling order `gpu → node → rack → topology_domain` used by
hierarchical PNA pooling. See [Relational architecture](../theory/relational-architecture.md).
