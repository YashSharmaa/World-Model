# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Prespecified Mondrian slices for shift-aware, slice-wise reporting.

Doc F is emphatic: averaging away the incidents that matter is the second way
teams fool themselves. The countermeasure is to *always publish slice-wise
metrics* over a small set of operationally meaningful, **prespecified** slice
families (prespecified = decided before looking at results, so this is not
post-hoc slicing / p-hacking). Doc O adds the sample-complexity requirement: each
major reported slice needs ~750 effective windows.

A "Mondrian" slicing partitions the evaluation set along a single categorical
axis into disjoint, non-overlapping cells (named after the painter's grids) —
each window lands in exactly one cell of each family. The five families here are
the ones doc F/C name as the axes that drive shift:

* ``action_family``     — which v1 action (hold/admit/place/admit_and_place);
* ``horizon``           — the prediction horizon (5/15/30/60 min);
* ``workload_family``   — communication / gang character of the workload;
* ``topology_family``   — depth/locality of the placement target;
* ``source_health``     — telemetry integrity bucket (observed/degraded/...).

Each :class:`SliceFamily` exposes a ``key_of`` callable that maps a window's
feature record (a plain dict the runner assembles) to its cell label. The runner
groups metrics by ``f"{family}={cell}"`` and the report renders one block per
slice. Keeping the labelling functions here (not in the runner) means the slice
contract is frozen and auditable independently of the scoring code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from ...schema.kpis import HORIZONS_S

#: The frozen v1 action alphabet used as the ``action_family`` slice cells.
ACTION_FAMILY_CELLS: tuple[str, ...] = (
    "hold",
    "admit",
    "place",
    "admit_and_place",
)

#: Communication / gang character buckets (doc C CommClass + gang size).
WORKLOAD_FAMILY_CELLS: tuple[str, ...] = (
    "single_pod",
    "small_gang",
    "allreduce_heavy",
    "pipeline",
    "other",
)

#: Topology depth / locality buckets (doc C TopologyLevel + required/preferred).
TOPOLOGY_FAMILY_CELLS: tuple[str, ...] = (
    "intra_node",
    "intra_rack",
    "cross_rack",
    "cross_block",
    "unconstrained",
)

#: Telemetry source-health buckets (doc T/U M_t observation-source state).
SOURCE_HEALTH_CELLS: tuple[str, ...] = (
    "observed",
    "stale",
    "delayed",
    "dropped",
)


def _horizon_cells() -> tuple[str, ...]:
    return tuple(f"h{int(h)}s" for h in HORIZONS_S)


@dataclass(slots=True, frozen=True)
class SliceFamily:
    """A single Mondrian slice axis: a name, its cells, and a labelling rule.

    ``key_of`` maps a window feature record to one of ``cells`` (or
    ``fallback`` when the record lacks the field). The labelling is pure and
    deterministic so two runs slice identically.
    """

    name: str
    cells: tuple[str, ...]
    feature: str
    fallback: str
    key_of: Callable[[Mapping[str, object]], str] = field(repr=False)

    def label(self, record: Mapping[str, object]) -> str:
        cell = self.key_of(record)
        return cell if cell in self.cells else self.fallback


def _action_key(record: Mapping[str, object]) -> str:
    val = str(record.get("action_family", record.get("action_type", "")))
    return val


def _horizon_key(record: Mapping[str, object]) -> str:
    h = record.get("horizon_s")
    if h is None:
        return ""
    return f"h{int(h)}s"


def _workload_key(record: Mapping[str, object]) -> str:
    return str(record.get("workload_family", ""))


def _topology_key(record: Mapping[str, object]) -> str:
    return str(record.get("topology_family", ""))


def _source_health_key(record: Mapping[str, object]) -> str:
    return str(record.get("source_health", record.get("source_health_bucket", "")))


#: The five prespecified Mondrian slice families (doc F / doc O slice plan).
SLICE_FAMILIES: tuple[SliceFamily, ...] = (
    SliceFamily(
        name="action_family",
        cells=ACTION_FAMILY_CELLS,
        feature="action_family",
        fallback="other",
        key_of=_action_key,
    ),
    SliceFamily(
        name="horizon",
        cells=_horizon_cells(),
        feature="horizon_s",
        fallback="other",
        key_of=_horizon_key,
    ),
    SliceFamily(
        name="workload_family",
        cells=WORKLOAD_FAMILY_CELLS,
        feature="workload_family",
        fallback="other",
        key_of=_workload_key,
    ),
    SliceFamily(
        name="topology_family",
        cells=TOPOLOGY_FAMILY_CELLS,
        feature="topology_family",
        fallback="unconstrained",
        key_of=_topology_key,
    ),
    SliceFamily(
        name="source_health",
        cells=SOURCE_HEALTH_CELLS,
        feature="source_health",
        fallback="observed",
        key_of=_source_health_key,
    ),
)

SLICE_FAMILIES_BY_NAME: dict[str, SliceFamily] = {f.name: f for f in SLICE_FAMILIES}

#: Doc O: each major reported slice needs ~750 effective windows for calibration.
MIN_EFFECTIVE_WINDOWS_PER_SLICE: int = 750


def assign_slices(record: Mapping[str, object]) -> dict[str, str]:
    """Return ``{family_name: "family=cell"}`` for one window feature record.

    The runner calls this once per window and uses the returned labels to bucket
    every metric, so the same window contributes to exactly one cell per family.
    """
    return {f.name: f"{f.name}={f.label(record)}" for f in SLICE_FAMILIES}


def all_slice_labels() -> dict[str, list[str]]:
    """Enumerate every prespecified slice label, for empty-slice reporting."""
    out: dict[str, list[str]] = {}
    for f in SLICE_FAMILIES:
        cells = list(f.cells)
        if f.fallback not in cells:
            cells.append(f.fallback)
        out[f.name] = [f"{f.name}={c}" for c in cells]
    return out


__all__ = [
    "ACTION_FAMILY_CELLS",
    "MIN_EFFECTIVE_WINDOWS_PER_SLICE",
    "SLICE_FAMILIES",
    "SLICE_FAMILIES_BY_NAME",
    "SOURCE_HEALTH_CELLS",
    "TOPOLOGY_FAMILY_CELLS",
    "WORKLOAD_FAMILY_CELLS",
    "SliceFamily",
    "all_slice_labels",
    "assign_slices",
]
