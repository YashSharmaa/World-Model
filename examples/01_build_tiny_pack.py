#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Example 01 — build the tiny, Git-tracked demo pack (Research doc H/G).

Compiles the checked-in scenario YAMLs into authoritative :class:`WorldState`s
via :mod:`data_engine.scenario_loader`, generates a paired synthetic
counterfactual decision-window pack per family (≥ 3 branches sharing one
exogenous seed bundle so branch deltas estimate *intervention effects*, not
intervention-plus-noise), and writes the whole thing as small, byte-stable JSON
under ``sample_data/tiny_demo_pack/`` with a full provenance manifest.

What it produces (each a frozen schema contract — nothing is redefined here):

* ``decision_windows/*.json``  — Gold :class:`DecisionWindow` records anchored by
  a pre-action ``state_hash``; candidate action ids are the branch ids.
* ``label_packs/*.json``       — multi-task :class:`LabelPack`s (KPI quantiles +
  hazards) per ``(decision, action, horizon)``.
* ``provenance/manifest.json`` — the :class:`ProvenanceManifest` data card.
* ``seed_registry.json``       — per-scenario master seeds for reproducible
  branch replay.
* ``index.json``               — a small human-readable table of contents.

Run::

    python examples/01_build_tiny_pack.py
    python examples/01_build_tiny_pack.py --out sample_data/tiny_demo_pack

The generator is deterministic: same scenarios + same seeds ⇒ byte-stable
windows and a stable per-window ``state_hash``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

# Example-local bootstrap (wires <repo>/src onto sys.path for a bare checkout).
from _bootstrap import CREDIT, SAMPLE_DATA_DIR, SCENARIOS_DIR, ensure_importable

ensure_importable()

from gpu_center_world_model._watermark import AUTHOR, AUTHOR_URL  # noqa: E402
from gpu_center_world_model.data_engine import (  # noqa: E402
    Origin,
    Split,
    build_manifest,
    build_window,
    generate_pack,
    generate_pod_state,
    load_scenario,
    transform_record,
    validate_manifest,
)
from gpu_center_world_model.schema.hashing import world_state_hash  # noqa: E402
from gpu_center_world_model.version import (  # noqa: E402
    SCHEMA_VERSION,
    __version__,
)

# A *tiny* curated slice — one window per family is enough to prove shapes,
# masks, metrics, and the safety shield all line up (doc H tiny-pack purpose).
# Each entry maps a checked-in scenario YAML to the synthetic SCENARIO_PACKS key
# whose residual SCM rolls out its counterfactual branches.
SCENARIO_SLICE: tuple[tuple[str, str, str, Split], ...] = (
    # (scenario yaml path, family, synthetic-pack key, split)
    ("normal/steady_pod.yaml", "normal", "normal", Split.TRAIN),
    ("stress/thermal_stress.yaml", "stress", "thermal_stress", Split.VAL),
    ("stress/scheduling_stress.yaml", "stress", "scheduling_stress", Split.TRAIN),
    (
        "rare_events/gpu_fault_burst.yaml",
        "rare_events",
        "fault_stress",
        Split.TEST_RARE_EVENT,
    ),
)

GENERATOR_COMMIT = f"tiny-demo-pack@{__version__}"


def _write_json(path: Path, obj: object) -> None:
    """Write a dataclass/dict as pretty, sorted, byte-stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def build_pack(out_dir: Path) -> dict[str, object]:
    """Build the tiny demo pack and return a summary dict for printing."""
    win_dir = out_dir / "decision_windows"
    lab_dir = out_dir / "label_packs"
    prov_dir = out_dir / "provenance"

    seed_registry: dict[str, int] = {}
    index_rows: list[dict[str, object]] = []
    transform_lines: list[str] = []
    total_branches = 0
    total_labels = 0

    for rel_path, family, pack_key, split in SCENARIO_SLICE:
        scenario_path = SCENARIOS_DIR / rel_path
        # 1) Compile the human-authored scenario into the authoritative world
        #    state + ordered pending decision queue (the task's required entry).
        loaded = load_scenario(scenario_path)
        seed = loaded.seed_bundle.master
        seed_registry[loaded.scenario_id] = seed
        scenario_hash = world_state_hash(loaded.state)

        # 2) Generate the paired counterfactual branches under the matching
        #    synthetic residual SCM, sharing the scenario's seed bundle (CRN).
        #    `generate_pack` builds its own demo pod from the SCM so the branch
        #    rollout has the per-GPU/rack streams the residual model needs; the
        #    pre-action context (queue, demand) mirrors the scenario's gang mix.
        pack = generate_pack(seed=seed, scenario=pack_key)

        # 3) Assemble the Gold decision window + label packs. The pre-action
        #    state used for the window's t0 / leakage guard is the synthetic
        #    pod state for the same seed/demand so timestamps line up with the
        #    branch rollout.
        _pod, pre_state, _uid = generate_pod_state(seed=seed, gpu_demand=16)
        window, labels = build_window(
            pre_state,
            pack,
            split=split,
            scenario_family=family,
            source_mix=Origin.SIM_COUNTERFACTUAL,
            generator_commit=GENERATOR_COMMIT,
            provenance_uri="provenance/manifest.json",
        )

        win_path = win_dir / f"{loaded.scenario_id}.json"
        lab_path = lab_dir / f"{loaded.scenario_id}.json"
        _write_json(win_path, window)
        _write_json(lab_path, [dataclasses.asdict(lp) for lp in labels])

        transform_lines.append(transform_record(window).to_json())
        total_branches += len(pack.branches)
        total_labels += len(labels)
        index_rows.append(
            {
                "scenario_id": loaded.scenario_id,
                "family": family,
                "split": split.value,
                "scenario_state_hash": scenario_hash,
                "branch_state_hash": pack.state_hash,
                "n_branches": len(pack.branches),
                "n_labels": len(labels),
                "n_gpus": loaded.n_gpus,
                "n_nodes": loaded.n_nodes,
                "pending_queue": list(loaded.pending_queue),
                "rare_event": (
                    loaded.rare_event.get("kind") if loaded.rare_event else None
                ),
                "window_uri": str(win_path.relative_to(out_dir)),
                "label_uri": str(lab_path.relative_to(out_dir)),
            }
        )

    # 4) Provenance manifest (the machine-readable data card) + seed registry.
    manifest = build_manifest(
        generator_commit=GENERATOR_COMMIT,
        seed_registry=seed_registry,
        real_fraction=0.0,
        calibration_window="synthetic (fitted-posterior bounds, not a real window)",
    )
    problems = validate_manifest(manifest)
    if problems:  # the release writer raises; here we fail loudly too
        raise SystemExit(f"provenance manifest invalid: {problems}")

    _write_json(prov_dir / "manifest.json", manifest)
    _write_json(out_dir / "seed_registry.json", seed_registry)
    (prov_dir / "transform_manifest.jsonl").write_text(
        "\n".join(transform_lines) + "\n", encoding="utf-8"
    )
    _write_json(
        out_dir / "index.json",
        {
            "project": "GPU-Center World Model",
            "author": AUTHOR,
            "author_url": AUTHOR_URL,
            "schema_version": SCHEMA_VERSION,
            "generator_commit": GENERATOR_COMMIT,
            "credit": CREDIT,
            "windows": index_rows,
        },
    )

    return {
        "n_scenarios": len(SCENARIO_SLICE),
        "n_branches": total_branches,
        "n_labels": total_labels,
        "seed_registry": seed_registry,
        "rows": index_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=SAMPLE_DATA_DIR / "tiny_demo_pack",
        help="output directory for the tiny pack",
    )
    args = parser.parse_args()
    out_dir: Path = args.out

    summary = build_pack(out_dir)

    print("=" * 78)
    print("GPU-Center World Model — tiny demo pack builder")
    print("=" * 78)
    print(f"output dir         : {out_dir}")
    print(f"scenarios compiled : {summary['n_scenarios']}")
    print(f"counterfactual br. : {summary['n_branches']}")
    print(f"label packs        : {summary['n_labels']}")
    print(f"seed registry      : {summary['seed_registry']}")
    print("-" * 78)
    print(f"{'scenario':<22}{'family':<12}{'split':<16}{'gpus':>5} {'branches':>9}")
    for row in summary["rows"]:  # type: ignore[index]
        print(
            f"{row['scenario_id']:<22}{row['family']:<12}"
            f"{row['split']:<16}{row['n_gpus']:>5} {row['n_branches']:>9}"
        )
    print("-" * 78)
    print("artifacts: decision_windows/, label_packs/, provenance/manifest.json,")
    print("           provenance/transform_manifest.jsonl, seed_registry.json,")
    print("           index.json")
    print("=" * 78)
    print(CREDIT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
