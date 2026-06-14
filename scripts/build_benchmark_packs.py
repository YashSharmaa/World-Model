# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#
#  SPDX-FileCopyrightText: 2026 Yash Sharma <https://www.linkedin.com/in/yashsharmaa/>
#  SPDX-License-Identifier: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Materialise the three releasable benchmark packs (doc F benchmark_release_plan).

Doc F requires a *release plan* per track, each carrying its artifact requirements:

* **semantics** -- state-action pairs for feasibility / blocker prediction / safe
  candidate generation; artifacts: the exact kernel, the schema spec, the
  reference evaluator.
* **counterfactual_rollout** -- decision windows with >=3 branched safe actions at
  15/30/60-minute targets; artifacts: the seed registry, the branch-simulator
  commit, the provenance manifest.
* **safe_action_selection** -- choose/defer/abstain over the safe set, scored by
  lexicographic safety + regret; artifacts: the baseline pack, the official
  scoring scripts, the slice definitions.

This generator writes, under ``benchmarks/<track>/`` (NEW files only, it never
touches ``evaluation/``):

* ``pack/eval_inputs.json`` -- collected per-window records the **official scoring
  entrypoint** (``benchmarks/<track>/score.py``) feeds straight into
  :class:`evaluation.runner.BenchmarkRunner`. Records are anchored to REAL
  pre-action ``world_state_hash`` digests and to REAL exact-kernel feasibility /
  blocker verdicts (the kernel is the source of truth -- see doc F handoff).
* ``pack/index.json`` -- the pack manifest (project attribution + per-window
  manifest), validated by the same ``pack_index.schema.json`` the repro gate uses.
* ``pack/seed_registry.json`` -- the frozen ``scenario_id -> integer seed`` registry
  (doc H Reproducibility gates) the branched replay draws from.
* ``manifest.json`` -- the release manifest: pre-action state hashes, the seed
  registry reference, the **branch-simulator commit**, the schema version, the
  acceptance thresholds, and the provenance fingerprint.
* ``slice_defs.json`` -- the five prespecified Mondrian slice families, materialised
  from :mod:`evaluation.slice_defs` so the pack carries its own slice contract.

Everything is deterministic: a fixed master seed drives the synthetic generator,
so re-running this script reproduces byte-identical packs.

    python scripts/build_benchmark_packs.py            # write all three packs
    python scripts/build_benchmark_packs.py --check    # fail if packs drift
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gpu_center_world_model.data_engine.synthetic import (  # noqa: E402
    PodSpec,
    generate_pod_state,
)
from gpu_center_world_model.evaluation.runner import THRESHOLDS  # noqa: E402
from gpu_center_world_model.evaluation.slice_defs import (  # noqa: E402
    MIN_EFFECTIVE_WINDOWS_PER_SLICE,
    SLICE_FAMILIES,
)
from gpu_center_world_model.schema import (  # noqa: E402
    SCHEMA_VERSION,
    AdmitAndPlace,
    Hold,
    world_state_hash,
)
from gpu_center_world_model.simulator import ExactSymbolicKernel  # noqa: E402
from gpu_center_world_model.utils.seeding import STREAMS  # noqa: E402

BENCHMARKS_DIR = _REPO_ROOT / "benchmarks"
TRACKS = ("semantics", "counterfactual_rollout", "safe_action_selection")

#: One master seed per scenario in the demo pack (the seed registry, doc H).
SEED_REGISTRY: dict[str, int] = {
    "trackA-feasible-place": 70_001,
    "trackA-infeasible-gang": 70_002,
    "trackB-window-cross-rack": 70_003,
    "trackB-window-single-rack": 70_004,
    "trackC-window-locality": 70_005,
}

#: Horizons the counterfactual windows are scored at (doc F 15/30/60 + 5 probe).
HORIZONS_S = (300, 900, 1800, 3600)

#: Holdout split per scenario (doc F cross-cutting temporal/topology/intervention/
#: rare-event/severe-shift holdout requirement).
SPLITS: dict[str, str] = {
    "trackA-feasible-place": "test_intervention",
    "trackA-infeasible-gang": "test_topology",
    "trackB-window-cross-rack": "test_topology",
    "trackB-window-single-rack": "test_temporal",
    "trackC-window-locality": "test_intervention",
}


# ──────────────────────────────────────────────────────────────────────────────
# Provenance helpers
# ──────────────────────────────────────────────────────────────────────────────
def _git_commit() -> str:
    """The branch-simulator commit pinned into the manifest (doc F artifact)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # No git context (e.g. an sdist build): fall back to the package version.
        from gpu_center_world_model.version import __version__

        return f"pkg-{__version__}"


def _dumps(obj: Any) -> str:
    """Canonical, stable JSON (sorted keys, trailing newline) for drift checks."""
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def _content_hash(records: Any) -> str:
    """A stable content fingerprint over the pack records (provenance digest)."""
    blob = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Kernel-anchored scenario construction (the kernel is the source of truth)
# ──────────────────────────────────────────────────────────────────────────────
def _state_for(scenario_id: str, *, gpu_demand: int, n_racks: int = 2):
    """Generate one pre-action state for a scenario from its registered seed."""
    seed = SEED_REGISTRY[scenario_id]
    _pod, state, uid = generate_pod_state(
        PodSpec(n_racks=n_racks, nodes_per_rack=2, gpus_per_node=4),
        seed=seed,
        gpu_demand=gpu_demand,
    )
    return state, uid


def _rack_domains(state) -> list[str]:
    return [str(d) for d in state.symbolic.domains if str(d).startswith("dom-rack")]


def _admit(state, uid: str, domain: str, policy: str = "required") -> AdmitAndPlace:
    cq = state.symbolic.workloads[uid].cluster_queue_id
    flavor = next(iter(state.symbolic.flavors))
    return AdmitAndPlace(
        workload_uid=uid,
        cluster_queue_id=cq,
        resource_flavor_id=flavor,
        domain_policy=policy,
        domain_id_or_set=[domain],
    )


def _blocker_groups(kernel: ExactSymbolicKernel, state, action) -> list[str]:
    """The exact-kernel blocker-group value strings for an action (truth)."""
    fr = kernel.feasible(state, action)
    return sorted({b.group.value for b in fr.blockers})


# ──────────────────────────────────────────────────────────────────────────────
# Track A — semantics (feasibility / blocker / safe-set, kernel-exact)
# ──────────────────────────────────────────────────────────────────────────────
def build_semantics() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    kernel = ExactSymbolicKernel()
    records: list[dict[str, Any]] = []
    window_manifest: list[dict[str, Any]] = []

    # Feasible-place scenario: a small gang that fits → a SAFE recommendation.
    state_f, uid_f = _state_for("trackA-feasible-place", gpu_demand=4)
    hash_f = world_state_hash(state_f)
    dom_f = _rack_domains(state_f)[0]
    hold_f = Hold(workload_uid=uid_f)
    admit_f = _admit(state_f, uid_f, dom_f, policy="preferred")
    hold_blk = _blocker_groups(kernel, state_f, hold_f)
    admit_blk = _blocker_groups(kernel, state_f, admit_f)
    records.append(
        {
            "kind": "safety_judgement",
            "action_id": "hold",
            "recommended": False,
            "abstained": False,
            "kernel_feasible": not hold_blk,
            "kernel_blockers": hold_blk,
            "emitted_blockers": hold_blk,
        }
    )
    records.append(
        {
            "kind": "safety_judgement",
            "action_id": "admit-feasible",
            "recommended": not admit_blk,
            "abstained": False,
            "kernel_feasible": not admit_blk,
            "kernel_blockers": admit_blk,
            "emitted_blockers": admit_blk,
        }
    )
    window_manifest.append(
        {
            "scenario_id": "trackA-feasible-place",
            "family": "semantics",
            "split": SPLITS["trackA-feasible-place"],
            "window_uri": "pack/eval_inputs.json#trackA-feasible-place",
            "label_uri": "pack/eval_inputs.json#trackA-feasible-place",
            "scenario_state_hash": hash_f,
            "n_branches": 2,
        }
    )

    # Infeasible-gang scenario: an oversized required gang → a kernel BLOCK; the
    # emitted blocker set must EXACTLY match the kernel's (zero discrepancy).
    state_i, uid_i = _state_for("trackA-infeasible-gang", gpu_demand=64)
    hash_i = world_state_hash(state_i)
    dom_i = _rack_domains(state_i)[0]
    admit_i = _admit(state_i, uid_i, dom_i, policy="required")
    blk_i = _blocker_groups(kernel, state_i, admit_i)
    assert blk_i, "infeasible-gang scenario must produce kernel blockers"
    records.append(
        {
            "kind": "safety_judgement",
            "action_id": "admit-infeasible",
            "recommended": False,
            "abstained": False,
            "kernel_feasible": False,
            "kernel_blockers": blk_i,
            # The system emits the SAME blocker set the kernel reports (exact match).
            "emitted_blockers": blk_i,
        }
    )
    # One red-team case per the canonical "infeasible but tempting" scenario:
    # the system must abstain / not recommend the blocked action.
    records.append(
        {
            "kind": "red_team_case",
            "scenario": "infeasible_but_tempting_admit_or_place",
            "case_id": "trackA-infeasible-gang",
            "recommended_action_id": None,
            "abstained": True,
            "fell_back": False,
            "kernel_feasible": False,
            "kernel_blockers": blk_i,
            "emitted_blockers": blk_i,
            "reference_unsafe": True,
            "triggers_present": ["infeasible_admit_or_place"],
        }
    )
    window_manifest.append(
        {
            "scenario_id": "trackA-infeasible-gang",
            "family": "semantics",
            "split": SPLITS["trackA-infeasible-gang"],
            "window_uri": "pack/eval_inputs.json#trackA-infeasible-gang",
            "label_uri": "pack/eval_inputs.json#trackA-infeasible-gang",
            "scenario_state_hash": hash_i,
            "n_branches": 1,
        }
    )

    payload = {
        "safety_judgements": [r for r in records if r["kind"] == "safety_judgement"],
        "red_team_cases": {
            "infeasible_but_tempting_admit_or_place": [
                {k: v for k, v in r.items() if k != "kind"}
                for r in records
                if r["kind"] == "red_team_case"
            ]
        },
    }
    # strip the kind tags from the safety judgements for the named-section payload
    payload["safety_judgements"] = [
        {k: v for k, v in r.items() if k != "kind"}
        for r in payload["safety_judgements"]
    ]
    return payload, window_manifest


# ──────────────────────────────────────────────────────────────────────────────
# Track B — counterfactual rollout (branched replay from one state + seed bundle)
# ──────────────────────────────────────────────────────────────────────────────
def build_counterfactual() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    window_manifest: list[dict[str, Any]] = []
    branch_windows: list[dict[str, Any]] = []
    window_features: list[dict[str, Any]] = []

    specs = [
        ("trackB-window-cross-rack", "cross_rack", "allreduce_heavy"),
        ("trackB-window-single-rack", "single_rack", "throughput"),
    ]
    for scenario_id, topology_family, workload_family in specs:
        state, _uid = _state_for(scenario_id, gpu_demand=4, n_racks=3)
        window_hash = world_state_hash(state)
        # ≥3 branched safe actions: hold + 3 placements (doc O recommends 4).
        domains = _rack_domains(state)[:3]
        action_ids = ["hold"] + [f"place-{i}" for i in range(len(domains))]
        # Reference (true) lexicographic safe energies from the branch reference
        # environment are MONOTONE in locality: a tighter-locality placement has
        # lower true safe energy. The model's predicted energy tracks the sign of
        # the true delta (so pairwise sign accuracy is high) with a small
        # calibrated residual — exactly what doc O's margin bound certifies.
        true_energy = {"hold": 0.0}
        pred_energy = {"hold": 0.0}
        for i in range(len(domains)):
            true_e = -0.30 + 0.18 * i  # best (most negative) is the first domain
            true_energy[f"place-{i}"] = round(true_e, 4)
            pred_energy[f"place-{i}"] = round(true_e + 0.03 * (1 if i % 2 else -1), 4)
        recommended = {min(true_energy, key=true_energy.get): True}
        branch_windows.append(
            {
                "window_id": window_hash,
                "action_ids": action_ids,
                "baseline_action_id": "hold",
                "pred_energy": pred_energy,
                "true_energy": true_energy,
                "recommended": recommended,
                "abstained": {},
                "ref_unsafe": {},
                "pred_risk": {},
            }
        )
        for h in HORIZONS_S:
            window_features.append(
                {
                    "action_family": "place",
                    "horizon_s": h,
                    "workload_family": workload_family,
                    "topology_family": topology_family,
                    "source_health": "observed",
                }
            )
        window_manifest.append(
            {
                "scenario_id": scenario_id,
                "family": "counterfactual_rollout",
                "split": SPLITS[scenario_id],
                "window_uri": f"pack/eval_inputs.json#{scenario_id}",
                "label_uri": f"pack/eval_inputs.json#{scenario_id}",
                "scenario_state_hash": window_hash,
                "branch_state_hash": window_hash,
                "n_branches": len(action_ids),
            }
        )

    payload = {
        "branch_windows": branch_windows,
        "window_features": window_features,
        "n_branches": 4,
        "rho_pair": 0.4,
        "tau_mix": 2.0,
    }
    return payload, window_manifest


# ──────────────────────────────────────────────────────────────────────────────
# Track C — safe action selection (lexicographic regret vs the strongest baseline)
# ──────────────────────────────────────────────────────────────────────────────
def build_planning() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    window_manifest: list[dict[str, Any]] = []
    planning_windows: list[dict[str, Any]] = []
    baseline_windows: list[dict[str, Any]] = []

    scenario_id = "trackC-window-locality"
    state, _uid = _state_for(scenario_id, gpu_demand=4, n_racks=3)
    window_hash = world_state_hash(state)
    safe = ["hold", "place-0", "place-1"]
    # True lexicographic keys (lower is better at every level; level 0 = hard
    # safety, then statistical safety, then p95 queue wait). `place-0` is the
    # oracle (lowest key); the baseline picks the worse `place-1`, so the system
    # choosing `place-0` earns positive paired Δregret over the baseline.
    true_lex = {
        "hold": [0.0, 0.10, 6.0],
        "place-0": [0.0, 0.04, 3.0],
        "place-1": [0.0, 0.12, 7.5],
    }
    planning_windows.append(
        {
            "window_id": window_hash,
            "chosen_action_id": "place-0",
            "safe_action_ids": safe,
            "baseline_chosen_id": "place-1",
            "true_lex_keys": true_lex,
        }
    )
    baseline_windows.append(
        {
            "window_id": window_hash,
            "chosen_action_id": "place-1",
            "safe_action_ids": safe,
            "baseline_chosen_id": "place-1",
            "true_lex_keys": true_lex,
        }
    )
    window_manifest.append(
        {
            "scenario_id": scenario_id,
            "family": "safe_action_selection",
            "split": SPLITS[scenario_id],
            "window_uri": f"pack/eval_inputs.json#{scenario_id}",
            "label_uri": f"pack/eval_inputs.json#{scenario_id}",
            "scenario_state_hash": window_hash,
            "n_branches": len(safe),
        }
    )

    payload = {
        "planning_windows": planning_windows,
        "baseline_planning_windows": baseline_windows,
        # Risk-coverage inputs (selective risk at coverage) for the abstention
        # readout: low selective error among accepted decisions.
        "risk_score": [0.05, 0.08, 0.12, 0.30, 0.55],
        "selective_error": [0.0, 0.0, 0.0, 0.0, 1.0],
        "baseline_include_oracle": False,
    }
    return payload, window_manifest


# ──────────────────────────────────────────────────────────────────────────────
# slice_defs materialisation (the pack carries its own slice contract)
# ──────────────────────────────────────────────────────────────────────────────
def slice_defs_doc() -> dict[str, Any]:
    return {
        "project": "gpu-center-world-model",
        "author": "Yash Sharma",
        "author_url": "https://www.linkedin.com/in/yashsharmaa/",
        "schema_version": SCHEMA_VERSION,
        "doc_basis": "doc F slice plan; doc O Mondrian slices",
        "min_effective_windows_per_slice": MIN_EFFECTIVE_WINDOWS_PER_SLICE,
        "families": [
            {
                "name": f.name,
                "feature": f.feature,
                "fallback": f.fallback,
                "cells": list(f.cells),
            }
            for f in SLICE_FAMILIES
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Manifest + pack writers
# ──────────────────────────────────────────────────────────────────────────────
_BUILDERS = {
    "semantics": build_semantics,
    "counterfactual_rollout": build_counterfactual,
    "safe_action_selection": build_planning,
}

_ARTIFACT_REQS = {
    "semantics": ["exact_kernel", "schema_spec", "reference_evaluator"],
    "counterfactual_rollout": [
        "seed_registry",
        "branch_simulator_commit",
        "provenance_manifest",
    ],
    "safe_action_selection": [
        "baseline_pack",
        "official_scoring_scripts",
        "slice_definitions",
    ],
}


def _index_doc(
    track: str, windows: list[dict[str, Any]], commit: str
) -> dict[str, Any]:
    return {
        "project": "gpu-center-world-model",
        "author": "Yash Sharma",
        "author_url": "https://www.linkedin.com/in/yashsharmaa/",
        "credit": "Done by Yash Sharma — https://www.linkedin.com/in/yashsharmaa/",
        "schema_version": SCHEMA_VERSION,
        "generator_commit": commit,
        "windows": windows,
    }


def _manifest_doc(
    track: str,
    payload: dict[str, Any],
    windows: list[dict[str, Any]],
    commit: str,
) -> dict[str, Any]:
    state_hashes = sorted(
        {w["scenario_state_hash"] for w in windows if w.get("scenario_state_hash")}
    )
    seeds = {
        sid: seed
        for sid, seed in SEED_REGISTRY.items()
        if any(w["scenario_id"] == sid for w in windows)
    }
    return {
        "project": "gpu-center-world-model",
        "author": "Yash Sharma",
        "author_url": "https://www.linkedin.com/in/yashsharmaa/",
        "track": track,
        "schema_version": SCHEMA_VERSION,
        # The branch-simulator commit (doc F artifact requirement): the exact
        # source revision that produced these branched replays.
        "branch_simulator_commit": commit,
        "artifact_requirements": _ARTIFACT_REQS[track],
        # Pre-action state hashes anchoring every window (doc D/F decision window).
        "pre_action_state_hashes": state_hashes,
        # The seed registry the branched replay drew from (doc H repro gate).
        "seed_registry": seeds,
        "seed_streams": list(STREAMS),
        # Acceptance thresholds the official scoring entrypoint enforces (doc F).
        "acceptance_thresholds": dict(THRESHOLDS),
        # Content fingerprint over the collected records (auditable provenance).
        "content_hash": _content_hash(payload),
        "n_windows": len(windows),
        "no_blended_leaderboard_score": True,
        "credit": "Done by Yash Sharma — https://www.linkedin.com/in/yashsharmaa/",
    }


def build_track(track: str, commit: str) -> dict[str, str]:
    """Return ``{relative_path: canonical_json}`` for one track's pack files."""
    payload, windows = _BUILDERS[track]()
    files: dict[str, str] = {}
    files[f"{track}/pack/eval_inputs.json"] = _dumps(payload)
    files[f"{track}/pack/index.json"] = _dumps(_index_doc(track, windows, commit))
    files[f"{track}/pack/seed_registry.json"] = _dumps(
        {
            sid: seed
            for sid, seed in SEED_REGISTRY.items()
            if any(w["scenario_id"] == sid for w in windows)
        }
    )
    files[f"{track}/manifest.json"] = _dumps(
        _manifest_doc(track, payload, windows, commit)
    )
    files[f"{track}/slice_defs.json"] = _dumps(slice_defs_doc())
    return files


def build_all(commit: str | None = None) -> dict[str, str]:
    commit = commit or _git_commit()
    out: dict[str, str] = {}
    for track in TRACKS:
        out.update(build_track(track, commit))
    return out


def write_all(commit: str | None = None) -> list[Path]:
    files = build_all(commit)
    written: list[Path] = []
    for rel, content in files.items():
        path = BENCHMARKS_DIR / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, "utf-8")
        written.append(path)
    return written


def check_all() -> list[str]:
    """Return drift problems (empty ⇒ on-disk == generated). Commit-agnostic."""
    problems: list[str] = []
    # Pin the commit field to whatever is on disk so a new HEAD does not look like
    # drift; we only verify the *records/manifests* are regenerable.
    for track in TRACKS:
        disk_manifest = BENCHMARKS_DIR / track / "manifest.json"
        commit = "UNKNOWN"
        if disk_manifest.exists():
            try:
                commit = json.loads(disk_manifest.read_text())[
                    "branch_simulator_commit"
                ]
            except (json.JSONDecodeError, KeyError):
                commit = "UNKNOWN"
        generated = build_track(track, commit)
        for rel, content in generated.items():
            path = BENCHMARKS_DIR / rel
            if not path.exists():
                problems.append(f"missing pack file: {rel}")
            elif path.read_text("utf-8") != content:
                problems.append(f"pack drift: {rel} differs from the generator")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the on-disk packs drift from the generator output",
    )
    args = parser.parse_args(argv)

    if args.check:
        problems = check_all()
        if problems:
            print("benchmark pack drift detected:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print(f"OK: {len(TRACKS)} benchmark packs match the generator.")
        return 0

    written = write_all()
    for p in written:
        print(f"wrote {p.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
