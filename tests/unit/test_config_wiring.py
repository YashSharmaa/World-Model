# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Scoped tests for REPRO-CONFIGS-WIRING (doc H configs/ author-without-Python).

Doc H ships ``configs/{train,eval,demo}`` as human-editable YAML so a recipe,
acceptance profile, or demo preset can be authored *without editing Python*.
These tests assert each shipped config **round-trips into the object it feeds**:

* ``configs/train/{baseline,mlx_smoke}.yaml`` -> :class:`TrainConfig` (the
  training entrypoint's hyper-parameter dataclass);
* ``configs/eval/benchmark.yaml`` -> the runner's flat acceptance ``THRESHOLDS``
  (replacing the hardcoded dict) plus the :class:`BenchmarkConfig` profile that
  ``run_suite`` consumes;
* ``configs/demo/decision_studio.yaml`` -> the serving/demo compare path's fixed
  branch count and decision horizon.

They also assert the additive contract: an absent key keeps the consuming
object's default (the public constructors stay byte-stable), and the loaded
benchmark thresholds actually flow into the runner's gating.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from gpu_center_world_model.evaluation.metrics.counterfactual import (  # noqa: E402
    CounterfactualReport,
)
from gpu_center_world_model.evaluation.runner import (  # noqa: E402
    THRESHOLDS,
    BenchmarkConfig,
    apply_acceptance,
    load_benchmark_config,
    load_benchmark_config_from_yaml,
    load_thresholds,
    run_benchmark,
)
from gpu_center_world_model.evaluation.suite_entry import run_suite  # noqa: E402
from gpu_center_world_model.model.train import (  # noqa: E402
    TrainConfig,
    resolve_repo_config,
)
from gpu_center_world_model.utils.config import load_config  # noqa: E402

# Repo root: tests/unit/ -> tests/ -> <repo>.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS = _REPO_ROOT / "configs"


# ──────────────────────────────────────────────────────────────────────────────
# The shipped configs exist where doc H places them, and resolve from the package
# ──────────────────────────────────────────────────────────────────────────────
def test_shipped_configs_present() -> None:
    for rel in (
        "train/baseline.yaml",
        "train/mlx_smoke.yaml",
        "eval/benchmark.yaml",
        "demo/decision_studio.yaml",
    ):
        assert (_CONFIGS / rel).is_file(), f"missing configs/{rel}"


def test_resolve_repo_config_finds_configs() -> None:
    # The package-relative resolver lands on the same files as the repo root.
    p = resolve_repo_config("configs/eval/benchmark.yaml")
    assert p.is_file()
    assert p == _CONFIGS / "eval" / "benchmark.yaml"
    # absolute paths pass through unchanged
    abs_p = (_CONFIGS / "train" / "baseline.yaml").resolve()
    assert resolve_repo_config(abs_p) == abs_p


# ──────────────────────────────────────────────────────────────────────────────
# configs/train/*.yaml  ->  TrainConfig  (the training entrypoint's dataclass)
# ──────────────────────────────────────────────────────────────────────────────
def test_train_baseline_roundtrips_into_trainconfig() -> None:
    raw = load_config(_CONFIGS / "train" / "baseline.yaml")
    cfg = TrainConfig.from_config(raw)
    assert isinstance(cfg, TrainConfig)
    # optimizer block maps faithfully onto the dataclass fields
    assert cfg.micro_batch == raw["optim"]["micro_batch_size"] == 12
    assert cfg.accum_steps == raw["optim"]["grad_accum_steps"] == 4
    assert cfg.lr == pytest.approx(raw["optim"]["learning_rate"])
    assert cfg.weight_decay == pytest.approx(raw["optim"]["weight_decay"])
    assert cfg.grad_clip == pytest.approx(raw["optim"]["grad_clip_norm"])
    assert cfg.seed == raw["seed"] == 20260614
    # EMA decay -> ema_end (cosine tau schedule end)
    assert cfg.ema_end == pytest.approx(raw["ema"]["decay"])
    # slow lane bounds the rollout-horizon curriculum
    assert cfg.h_slow == raw["temporal"]["t_slow"] == 12
    assert cfg.curriculum_h_end <= cfg.h_slow
    # warmup_frac is a fraction in [0, 1]
    assert 0.0 <= cfg.warmup_frac <= 1.0


def test_train_smoke_roundtrips_and_shrinks() -> None:
    raw = load_config(_CONFIGS / "train" / "mlx_smoke.yaml")
    cfg = TrainConfig.from_config(raw)
    assert cfg.micro_batch == 4
    assert cfg.accum_steps == 1
    assert cfg.seed == 7
    # the smoke config shrinks the slow lane, which must bound the horizon caps
    assert cfg.h_slow == 4
    assert cfg.rollout_horizon <= 4
    assert cfg.curriculum_h_end <= 4
    # round-trip is usable by the consuming entrypoint (from_yaml resolves + loads)
    cfg2 = TrainConfig.from_yaml("configs/train/mlx_smoke.yaml")
    assert cfg2 == cfg


def test_trainconfig_from_config_is_additive() -> None:
    # an empty / partial config preserves every dataclass default (byte-stable
    # public constructor) — only the supplied keys move.
    assert TrainConfig.from_config({}) == TrainConfig()
    base = TrainConfig()
    partial = TrainConfig.from_config({"seed": 99})
    assert partial.seed == 99
    # everything else is untouched
    assert partial.micro_batch == base.micro_batch
    assert partial.lr == base.lr
    assert partial.steps_per_stage == base.steps_per_stage


# ──────────────────────────────────────────────────────────────────────────────
# configs/eval/benchmark.yaml  ->  THRESHOLDS / BenchmarkConfig  (run_suite feed)
# ──────────────────────────────────────────────────────────────────────────────
def test_benchmark_yaml_roundtrips_into_thresholds() -> None:
    raw = load_config(_CONFIGS / "eval" / "benchmark.yaml")
    th = load_thresholds(raw)
    at = raw["acceptance_thresholds"]
    # every flat gating key maps from the nested doc-F structure
    assert th["min_sign_accuracy"] == at["counterfactual_ranking"][
        "overall_pairwise_sign_accuracy_min"
    ]
    assert th["min_sign_accuracy_non_abstained"] == at["counterfactual_ranking"][
        "non_abstained_sign_accuracy_min"
    ]
    assert th["min_coverage"] == at["counterfactual_ranking"]["min_coverage"]
    assert th["min_regret_reduction"] == at["planning_utility"][
        "safe_regret_reduction_vs_best_baseline_min"
    ]
    lo, hi = at["calibration_and_optimism"]["interval_coverage_90_band"]
    assert (th["coverage_band_lo"], th["coverage_band_hi"]) == (lo, hi)
    assert th["max_optimism_bias"] == at["calibration_and_optimism"][
        "non_abstained_optimism_bias_rate_max"
    ]
    assert th["min_abstain_under_shift"] == at["shift_and_abstention"][
        "abstain_or_fallback_rate_min"
    ]
    assert th["max_false_safe_rate_shift"] == at["shift_and_abstention"][
        "false_safe_rate_max"
    ]
    # the shipped benchmark.yaml encodes exactly the doc-F default profile
    assert th == THRESHOLDS


def test_benchmark_config_carries_slices_and_baselines() -> None:
    bc = load_benchmark_config_from_yaml()
    assert isinstance(bc, BenchmarkConfig)
    assert bc.raw.get("config_id") == "eval_benchmark"
    # slice families authored in the YAML are surfaced for the report
    assert "horizon_s" in bc.slices
    assert "stress_regime" in bc.slices
    # skeptical baselines + the brutal ablation set carry through
    assert "operations_grade" in bc.baselines
    assert "remove_counterfactual_rollout_loss" in bc.ablations


def test_load_thresholds_is_additive_override() -> None:
    # a config supplying only ONE gate overrides that key and keeps the rest.
    th = load_thresholds(
        {"acceptance_thresholds": {"counterfactual_ranking": {"min_coverage": 0.42}}}
    )
    assert th["min_coverage"] == 0.42
    # untouched keys keep the doc-F default
    assert th["min_sign_accuracy"] == THRESHOLDS["min_sign_accuracy"]
    # an empty config is the doc-F defaults verbatim
    assert load_thresholds({}) == THRESHOLDS
    assert load_benchmark_config(None).thresholds == THRESHOLDS


def test_loaded_thresholds_flow_into_runner_gating() -> None:
    # A config that *tightens* the sign-accuracy gate must change the pass/fail.
    cf = CounterfactualReport(
        n_windows=10,
        sign_accuracy=0.75,  # clears 0.70 default, fails a tightened 0.80
        sign_accuracy_pairs=100,
        sign_accuracy_non_abstained=0.85,
        sign_accuracy_non_abstained_pairs=20,
        delta_mae=0.1,
        delta_wape=0.1,
        top1_near_oracle=0.9,
        optimism_bias_rate=0.0,
        optimism_bias_n=10,
    )
    # default doc-F gate: passes (0.75 >= 0.70 and 0.85 >= 0.80)
    acc_default = apply_acceptance(
        safety_gate=None,
        red_team=None,
        counterfactual=cf,
        planning=None,
        abstention_shift=None,
        risk_coverage=None,
        counterfactual_coverage_value=0.80,
    )
    assert acc_default["counterfactual_ranking"]["passed"] is True
    # config-tightened gate (0.80 overall) flows through and now fails
    tight = load_thresholds(
        {
            "acceptance_thresholds": {
                "counterfactual_ranking": {
                    "overall_pairwise_sign_accuracy_min": 0.80
                }
            }
        }
    )
    acc_tight = apply_acceptance(
        safety_gate=None,
        red_team=None,
        counterfactual=cf,
        planning=None,
        abstention_shift=None,
        risk_coverage=None,
        counterfactual_coverage_value=0.80,
        thresholds=tight,
    )
    assert acc_tight["counterfactual_ranking"]["passed"] is False


def test_run_suite_consumes_benchmark_config(tmp_path: Path) -> None:
    # run_suite loads configs/eval/benchmark.yaml and stamps the report metadata
    # with the resolved thresholds + the config id (the replaced THRESHOLDS dict).
    import json

    out_dir = tmp_path / "report"
    md_path = run_suite(out=str(out_dir))
    assert Path(md_path).exists()
    report = json.loads((out_dir / "eval.json").read_text(encoding="utf-8"))
    assert report["metadata"]["benchmark_config_id"] == "eval_benchmark"
    assert report["metadata"]["thresholds"] == THRESHOLDS


def test_run_benchmark_publishes_benchmark_profile() -> None:
    bc = load_benchmark_config_from_yaml()
    report = run_benchmark(__empty_inputs(), config=bc).as_dict()
    profile = report["slices"]["_benchmark_profile"]
    assert profile["config_id"] == "eval_benchmark"
    assert "stress_regime" in profile["slices"]
    assert "operations_grade" in profile["baselines"]


# ──────────────────────────────────────────────────────────────────────────────
# configs/demo/decision_studio.yaml  ->  serving/demo compare path
# ──────────────────────────────────────────────────────────────────────────────
def test_demo_config_roundtrips_into_compare_path() -> None:
    from gpu_center_world_model.cli import (
        _default_candidates,
        _demo_branch_count,
        _load_demo_config,
    )
    from gpu_center_world_model.serving.app import demo_state_provider

    demo_cfg = _load_demo_config(None)
    assert demo_cfg.get("config_id") == "demo_decision_studio"
    # the FIXED branch count comes straight from the preset (doc G: exactly 3)
    branch_cap = _demo_branch_count(demo_cfg)
    assert branch_cap == demo_cfg["actions"]["count"] == 3
    # default horizon flows from the preset
    assert demo_cfg["default_horizon_s"] == 1800
    # the candidate builder honours the preset branch cap (hold + 2 domains)
    state = demo_state_provider()
    wl = next(iter(state.symbolic.workloads))
    cands = _default_candidates(state, wl, branch_cap)
    assert len(cands) == branch_cap
    assert cands[0].action_id == "hold"  # declared baseline (doc G)


def test_demo_branch_count_defaults_without_config() -> None:
    from gpu_center_world_model.cli import _demo_branch_count
    from gpu_center_world_model.serving.dto import DEFAULT_BRANCHES

    # an empty preset falls back to the serving default branch count
    assert _demo_branch_count({}) == DEFAULT_BRANCHES
    # candidates list form is honoured when count is absent
    assert _demo_branch_count({"actions": {"candidates": [1, 2]}}) == 2


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def __empty_inputs():
    from gpu_center_world_model.evaluation.runner import TrackInputs

    return TrackInputs()
