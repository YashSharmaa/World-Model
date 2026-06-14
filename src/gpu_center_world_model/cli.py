# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The ``gcwm`` command-line interface (argparse only, dependency-light).

Subcommands mirror the v0.1 operator/developer workflow (docs G, H):

* ``gcwm about``    — print the attribution BANNER and machine-readable watermark.
* ``gcwm compare``  — run the exact-kernel + reference-model compare path on a
  demo (or supplied) pod state and print the ranked safe candidates, exact
  blockers, and KPI deltas (doc H: "CLI replay that prints safe actions,
  blockers, and symbolic deltas").
* ``gcwm serve``    — launch the long-lived FastAPI service (needs the ``serve``
  extra). Imports of FastAPI/uvicorn are deferred to invocation time.
* ``gcwm pack``     — build a synthetic data pack (delegates to ``data_engine``
  if available; otherwise prints actionable guidance).
* ``gcwm eval``     — run the offline evaluation suite (delegates to
  ``evaluation`` if available; otherwise prints actionable guidance).

The CLI deliberately depends only on the standard library plus this package's
own dependency-light core, so ``gcwm about`` / ``gcwm compare`` work on bare CI.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from typing import Any

from ._watermark import BANNER, watermark
from .version import API_VERSION, SCHEMA_VERSION, __version__


# ──────────────────────────────────────────────────────────────────────────────
# gcwm about
# ──────────────────────────────────────────────────────────────────────────────
def _cmd_about(args: argparse.Namespace) -> int:
    if args.json:
        wm = watermark(as_dict=True)
        wm.update(
            {
                "version": __version__,
                "schema_version": SCHEMA_VERSION,
                "api_version": API_VERSION,
            }
        )
        print(json.dumps(wm, indent=2))
        return 0
    print(BANNER)
    print(f"\n  version       : {__version__}")
    print(f"  schema version: {SCHEMA_VERSION}")
    print(f"  api version   : {API_VERSION}")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# gcwm compare
# ──────────────────────────────────────────────────────────────────────────────
def _cmd_compare(args: argparse.Namespace) -> int:
    from .schema.hashing import state_hash as compute_state_hash
    from .serving import dto
    from .serving.api_v1 import EngineBundle, compare_actions
    from .serving.app import demo_state_provider

    state = demo_state_provider()
    sh = compute_state_hash(state.symbolic, state.context)
    engine = EngineBundle()

    # Demo preset (doc H configs/demo): the Decision Studio's fixed branch count,
    # decision horizon, and declared baseline come from the author-editable
    # ``configs/demo/decision_studio.yaml`` so the public 3-action / 30-60 min
    # demo can be re-scripted without editing Python (doc G shape contract). The
    # CLI flags still win when explicitly set; absent a config the doc-G defaults
    # (3 branches, 1800 s, baseline=hold) stand.
    demo_cfg = _load_demo_config(getattr(args, "config", None))
    branch_cap = _demo_branch_count(demo_cfg)
    horizon_s = (
        float(args.horizon_s)
        if args.horizon_s is not None
        else float(demo_cfg.get("default_horizon_s", 1800))
    )

    # Build the candidate action set. Default: hold + admit_and_place into each
    # rack domain for the first pending workload, capped at the demo branch count.
    wl_uid = args.workload or _first_pending_workload(state)
    candidates = _default_candidates(state, wl_uid, branch_cap)

    req = dto.CompareRequest(
        state_hash=sh,
        candidate_actions=candidates,
        horizon_s=horizon_s,
        include_rollouts=args.rollouts,
    )
    resp = compare_actions(state, req, engine)

    if args.json:
        print(json.dumps(dto.to_dict(resp), indent=2, default=str))
        return 0

    print(BANNER)
    print(f"\nstate_hash       : {resp.state_hash}")
    print(f"baseline action  : {resp.baseline_action_id}")
    print(f"recommended      : {resp.recommended_action_id}")
    print(f"abstain          : {resp.abstain}  ({resp.abstain_reason or '-'})")
    print(f"model / policy   : {resp.model_version} / {resp.policy_version}")
    print("\ncandidates (kernel safety is a gate; learned model only ranks):")
    for c in resp.candidates:
        if c.action_id.startswith("__pad"):
            continue
        flag = "SAFE " if c.safe else "BLOCK"
        rank = "-" if c.rank is None else str(c.rank)
        print(f"  [{flag}] rank={rank:<3} {c.action_id}")
        if not c.safe and c.exact_blockers:
            for b in c.exact_blockers:
                print(f"          blocker: {b}")
        if c.safe and c.delta_kpis:
            for k, v in sorted(c.delta_kpis.items()):
                print(f"          Δ {k}: {v}")
    return 0


def _first_pending_workload(state: Any) -> str:
    for uid, wl in state.symbolic.workloads.items():
        if wl.status.value == "pending":
            return uid
    return next(iter(state.symbolic.workloads), "")


def _default_candidates(state: Any, wl_uid: str, branch_cap: int | None = None):
    from .serving import dto
    from .serving.dto import DEFAULT_BRANCHES

    cap = DEFAULT_BRANCHES if branch_cap is None else max(1, int(branch_cap))
    wl = state.symbolic.workloads.get(wl_uid)
    cq = wl.cluster_queue_id if wl else ""
    cands = [
        dto.CandidateActionRequest(
            action_id="hold",
            type="hold",
            params={"workload_uid": wl_uid},
        )
    ]
    for dom_id in state.symbolic.domains:
        if len(cands) >= cap:
            break
        cands.append(
            dto.CandidateActionRequest(
                action_id=f"admit_place_{dom_id}",
                type="admit_and_place",
                params={
                    "workload_uid": wl_uid,
                    "cluster_queue_id": cq,
                    "resource_flavor_id": "",
                    "domain_policy": "required",
                    "domain_id_or_set": [dom_id],
                },
            )
        )
    return cands


def _load_demo_config(path: str | None) -> dict[str, Any]:
    """Load ``configs/demo/decision_studio.yaml`` (doc H configs/demo).

    Returns the parsed preset mapping, or an empty dict when the config is
    unavailable (so ``gcwm compare`` still runs on the doc-G defaults). ``path``
    overrides the default repo location.
    """
    try:
        from .model.train import resolve_repo_config
        from .utils.config import load_config
    except ImportError:  # pragma: no cover - dependency-light core only
        return {}
    target = resolve_repo_config(path or "configs/demo/decision_studio.yaml")
    try:
        raw = load_config(target)
    except (FileNotFoundError, RuntimeError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _demo_branch_count(demo_cfg: dict[str, Any]) -> int:
    """The FIXED branch count from the demo preset (doc G shape contract).

    Doc G fixes the Decision Studio at exactly three branches so the compiled MLX
    scorer never recompiles mid-presentation. The preset's ``actions.count`` (or
    the number of declared ``candidates``) drives the CLI candidate cap; absent a
    config the serving default (:data:`serving.dto.DEFAULT_BRANCHES`) stands.
    """
    from .serving.dto import DEFAULT_BRANCHES

    actions = demo_cfg.get("actions")
    if isinstance(actions, dict):
        if "count" in actions:
            return max(1, int(actions["count"]))
        cands = actions.get("candidates")
        if isinstance(cands, list) and cands:
            return max(1, len(cands))
    return DEFAULT_BRANCHES


# ──────────────────────────────────────────────────────────────────────────────
# gcwm serve
# ──────────────────────────────────────────────────────────────────────────────
def _cmd_serve(args: argparse.Namespace) -> int:
    from .serving.app import have_fastapi

    if not have_fastapi():
        print(
            "FastAPI is not installed. Install the serving extra:\n"
            "  pip install 'gpu-center-world-model[serve]'",
            file=sys.stderr,
        )
        return 2
    from .serving.app import run

    print(BANNER)
    print(f"\nServing on http://{args.host}:{args.port}  (Ctrl-C to stop)\n")
    run(host=args.host, port=args.port, log_level=args.log_level)
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# gcwm pack  (synthetic data engine — delegated)
# ──────────────────────────────────────────────────────────────────────────────
def _cmd_pack(args: argparse.Namespace) -> int:
    fn = _maybe_callable("gpu_center_world_model.data_engine", "build_pack")
    if fn is None:
        print(
            "The data engine pack builder is not available in this build.\n"
            "Expected `gpu_center_world_model.data_engine.build_pack(...)`.\n"
            "Install/enable the data engine, or run `gcwm compare` for a demo.",
            file=sys.stderr,
        )
        return 3
    out = fn(
        out_dir=args.out,
        scenario=args.scenario,
        seed=args.seed,
        n_windows=args.n_windows,
    )
    print(f"Wrote data pack: {out}")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# gcwm eval  (offline evaluation — delegated)
# ──────────────────────────────────────────────────────────────────────────────
def _cmd_eval(args: argparse.Namespace) -> int:
    fn = _maybe_callable("gpu_center_world_model.evaluation", "run_suite")
    if fn is None:
        print(
            "The evaluation suite runner is not available in this build.\n"
            "Expected `gpu_center_world_model.evaluation.run_suite(...)`.\n"
            "Install/enable the evaluation package to produce reports.",
            file=sys.stderr,
        )
        return 3
    report = fn(
        suite=args.suite,
        data=args.data,
        out=args.out,
        config=getattr(args, "config", None),
    )
    print(f"Evaluation report: {report}")
    return 0


def _maybe_callable(module: str, name: str):
    """Return ``module.name`` if importable & callable, else ``None``."""
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return None
    fn = getattr(mod, name, None)
    return fn if callable(fn) else None


# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gcwm",
        description=(
            "GPU-Center World Model — intervention-faithful world model for GPU "
            "datacenter operations. Done by Yash Sharma."
        ),
    )
    parser.add_argument("--version", action="version", version=f"gcwm {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_about = sub.add_parser("about", help="print the attribution banner")
    p_about.add_argument("--json", action="store_true", help="machine-readable")
    p_about.set_defaults(func=_cmd_about)

    p_cmp = sub.add_parser(
        "compare", help="rank safe candidate actions on a demo pod state"
    )
    p_cmp.add_argument("--workload", default=None, help="target workload uid")
    p_cmp.add_argument(
        "--horizon-s",
        dest="horizon_s",
        type=int,
        default=None,
        help="horizon (s); default from configs/demo/decision_studio.yaml",
    )
    p_cmp.add_argument(
        "--config",
        default=None,
        help="demo preset YAML (default: configs/demo/decision_studio.yaml)",
    )
    p_cmp.add_argument(
        "--no-rollouts",
        dest="rollouts",
        action="store_false",
        help="omit rollout points",
    )
    p_cmp.add_argument("--json", action="store_true", help="machine-readable")
    p_cmp.set_defaults(func=_cmd_compare, rollouts=True)

    p_serve = sub.add_parser("serve", help="launch the FastAPI service")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--log-level", default="info")
    p_serve.set_defaults(func=_cmd_serve)

    p_pack = sub.add_parser("pack", help="build a synthetic data pack")
    p_pack.add_argument("--out", default="./data_pack", help="output directory")
    p_pack.add_argument("--scenario", default="normal", help="scenario family")
    p_pack.add_argument("--seed", type=int, default=0)
    p_pack.add_argument("--n-windows", dest="n_windows", type=int, default=128)
    p_pack.set_defaults(func=_cmd_pack)

    p_eval = sub.add_parser("eval", help="run the offline evaluation suite")
    p_eval.add_argument("--suite", default="all", help="evaluation suite name")
    p_eval.add_argument("--data", default=None, help="data pack path")
    p_eval.add_argument("--out", default="./eval_report", help="report output")
    p_eval.add_argument(
        "--config",
        default=None,
        help="benchmark YAML (default: configs/eval/benchmark.yaml)",
    )
    p_eval.set_defaults(func=_cmd_eval)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``gcwm`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
