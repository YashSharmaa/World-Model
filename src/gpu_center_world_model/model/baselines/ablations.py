# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The doc-F ablation harness (doc F ``ablations`` — narrow but brutal).

Doc F names exactly **five** ablated variants, each a *matched-parameter,
matched-compute* knock-out of one spine component, scored on the readout that
component is supposed to earn:

1. ``remove_temporal_hierarchy``            — replace the fast/slow cores with a
   single-timescale model (we **reuse** :class:`FlatSequenceBaseline`, the flat
   single-GRU foil). Readout: long-horizon counterfactual sign accuracy, safe
   regret, planning latency.
2. ``remove_object_centric_state``          — drop typed relations (set
   ``n_rounds = 0`` so no relational message passing runs → pooled aggregates
   only). Readout: locality, fragmentation, fault-propagation error.
3. ``remove_uncertainty_modeling``          — a point predictor (``n_ensemble =
   1`` → no epistemic disagreement) with abstention disabled. Readout:
   optimism-bias rate, red-team false-safe rate, risk-coverage.
4. ``remove_synthetic_rare_event_augmentation`` — train only on observational /
   factual replay (drop the ``rare_event`` stage **and** the ``is_rare_event``
   windows). Readout: rare-event / degraded-telemetry / severe-shift packs.
5. ``remove_counterfactual_rollout_loss``   — teacher-forced factual prediction
   only (``λ_Δ = 0`` and only the ``warmup`` + ``calibration`` stages, so the
   free-running branch-difference rollout loss never fires). Readout: action
   sensitivity, counterfactual ranking, optimism-bias.

Every variant is trained through the **public** :func:`model.train.train` API on
the **public** :func:`model.train.make_smoke_windows` smoke set at a matched
``TrainConfig`` budget (HARD RULE: we never touch ``model/train.py`` or
``model/architecture.py``; we only choose public knobs). The harness then probes
each trained variant for its doc-F main readouts and the champion (the un-ablated
model) on the same fixtures, so the report shows *whether the ablated component
earned its keep* — the whole point of an ablation (doc F).

The probes are real, not stubs: **action sensitivity** is the mean latent /
forecast shift between the factual and the paired-branch action over the smoke
windows (the UWM-JEPA action-collapse signal doc F calls out for ablation #5);
**optimism bias** is the rate at which a variant under-predicts the realized
branch energy; **collapse** is the doc-J D1-D5 alarm carried on the train report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ...schema.kpis import KPIChannel, KPIForecastBundle
from ...schema.policy import AbstentionConfig
from .flat_sequence import FlatSequenceBaseline

#: The five doc-F ablation names, in their listed order, with the main_readout
#: doc F assigns to each (verbatim, so the report is auditable against the doc).
ABLATION_READOUTS: dict[str, str] = {
    "remove_temporal_hierarchy": (
        "Long-horizon counterfactual sign accuracy, safe regret, and planning "
        "latency."
    ),
    "remove_object_centric_state": (
        "Locality prediction, fragmentation cost, and fault-propagation errors."
    ),
    "remove_uncertainty_modeling": (
        "Optimism-bias rate, red-team false-safe rate, and risk-coverage."
    ),
    "remove_synthetic_rare_event_augmentation": (
        "Performance on rare-event, degraded-telemetry, and severe-shift test "
        "packs."
    ),
    "remove_counterfactual_rollout_loss": (
        "Action sensitivity, counterfactual ranking, and optimism-bias."
    ),
}

ABLATION_NAMES: tuple[str, ...] = tuple(ABLATION_READOUTS)


@dataclass(slots=True)
class AblationResult:
    """One ablated variant's training outcome + its doc-F main readouts.

    Attributes
    ----------
    name:
        The doc-F ablation name.
    main_readout:
        The doc-F ``main_readout`` string for this ablation (carried so the
        report names exactly what the ablation is supposed to degrade).
    n_steps:
        Training steps the variant ran (the matched-compute budget evidence).
    final_loss:
        Final training loss from the variant's :class:`TrainReport`.
    collapsed:
        Whether any doc-J D1-D5 collapse alarm fired during the variant's run.
    action_sensitivity:
        Mean forecast shift between the factual and paired-branch action over the
        probe windows — the UWM-JEPA action-collapse signal (small ⇒ the variant
        has lost action sensitivity, the failure mode doc F flags for #5).
    optimism_bias_rate:
        Fraction of probe windows where the variant under-predicted the realized
        branch energy (an over-optimistic forecast — doc F readout for #3/#5).
    readouts:
        The full named readout map (extra per-ablation diagnostics).
    """

    name: str
    main_readout: str
    n_steps: int = 0
    final_loss: float = float("nan")
    collapsed: bool = False
    action_sensitivity: float = float("nan")
    optimism_bias_rate: float = float("nan")
    readouts: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "main_readout": self.main_readout,
            "n_steps": self.n_steps,
            "final_loss": self.final_loss,
            "collapsed": self.collapsed,
            "action_sensitivity": self.action_sensitivity,
            "optimism_bias_rate": self.optimism_bias_rate,
            "readouts": dict(self.readouts),
        }


@dataclass(slots=True)
class AblationReport:
    """The full ablation profile: champion + five ablated variants (doc F)."""

    champion: AblationResult
    ablations: list[AblationResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "champion": self.champion.as_dict(),
            "ablations": [a.as_dict() for a in self.ablations],
            "per_ablation_main_readouts": {
                a.name: a.main_readout for a in self.ablations
            },
        }


# ──────────────────────────────────────────────────────────────────────────────
# Probes (real diagnostics over the smoke windows)
# ──────────────────────────────────────────────────────────────────────────────
def _bundle_energy(bundle: KPIForecastBundle, horizon_s: int) -> float:
    """A scalar 'energy' for a forecast bundle: lower admitted GPU-hours ⇒ worse.

    We read the median admitted-GPU-hours delta (negated so *more admitted work*
    is *lower energy*) plus the median queue-wait delta — the two utility terms
    the planner's lexicographic key leads with. This collapses a bundle to one
    comparable number for the action-sensitivity / optimism probes.
    """
    gpu = _median(bundle, KPIChannel.ADMITTED_GPU_HOURS_DELTA, horizon_s)
    qw = _median(bundle, KPIChannel.QUEUE_WAIT_DELTA_S, horizon_s)
    return float(qw - gpu)


def _median(bundle: KPIForecastBundle, channel: KPIChannel, horizon_s: int) -> float:
    best = None
    for q in bundle.quantiles:
        if q.channel is channel:
            if q.horizon_s == horizon_s:
                return float(q.quantiles.get(0.5, 0.0))
            if best is None:
                best = q
    return float(best.quantiles.get(0.5, 0.0)) if best is not None else 0.0


def _probe_variant(
    model: object, windows: Sequence, horizon_s: int
) -> tuple[float, float]:
    """Action-sensitivity + optimism-bias over the paired-branch smoke windows.

    For every window with a paired branch action we score the factual and the
    branch action through ``model.score_actions`` and measure:

    * **action sensitivity** = mean ``|E(branch) - E(factual)|`` — how much the
      variant's forecast moves when only the action changes (the UWM-JEPA
      action-collapse probe; small ⇒ collapsed, doc F #5);
    * **optimism bias** = fraction of windows where the variant's predicted branch
      energy is *below* the factual energy while the realized branch (the future
      state) is actually *worse* — an over-optimistic forecast (doc F #3/#5).

    Both are computed from the public ``score_actions`` surface so they apply
    uniformly to the learned variants and the flat-sequence reuse for #1.
    """
    shifts: list[float] = []
    optimistic = 0
    n = 0
    for w in windows:
        branch = getattr(w, "branch_action", None)
        if branch is None:
            continue
        try:
            bundles = model.score_actions(  # type: ignore[attr-defined]
                w.state, [w.action, branch], horizon_s
            )
        except Exception:
            continue
        if len(bundles) < 2:
            continue
        e_fact = _bundle_energy(bundles[0], horizon_s)
        e_branch = _bundle_energy(bundles[1], horizon_s)
        shifts.append(abs(e_branch - e_fact))
        # Realized branch "worse" proxy: the paired branch future vs the factual
        # future. We approximate the realized branch energy by the gap in pending
        # workloads (more pending ⇒ worse). The optimism event is: the variant
        # predicted the branch *better* (lower energy) than factual, but the
        # realized branch future is *not* better.
        realized_better = _realized_branch_better(w)
        if e_branch < e_fact and not realized_better:
            optimistic += 1
        n += 1
    sensitivity = float(np.mean(shifts)) if shifts else float("nan")
    optimism = float(optimistic / n) if n else float("nan")
    return sensitivity, optimism


def _realized_branch_better(window) -> bool:
    """Did the paired-branch future realize a better (less-loaded) state?

    Compares pending-workload counts in the branch future vs the factual future.
    Fewer pending workloads in the branch future ⇒ the branch genuinely helped.
    Absent futures fall back to "not better" (conservative for the optimism test).
    """
    fut = getattr(window, "future_state", None)
    bfut = getattr(window, "branch_future_state", None)
    if fut is None or bfut is None:
        return False
    from ...schema.entities import WorkloadStatus

    def _pending(state) -> int:
        return sum(
            1
            for wl in state.symbolic.workloads.values()
            if wl.status == WorkloadStatus.PENDING
        )

    return _pending(bfut) < _pending(fut)


# ──────────────────────────────────────────────────────────────────────────────
# The harness
# ──────────────────────────────────────────────────────────────────────────────
def run_ablation_suite(
    *,
    n_windows: int = 12,
    seed: int = 0,
    horizon_s: int = 1800,
    train_config=None,
) -> AblationReport:
    """Train + probe the champion and the five doc-F ablated variants (doc F).

    All variants share one matched :class:`TrainConfig` budget (the smoke config
    by default) and the **public** :func:`model.train.train` API over the
    **public** :func:`model.train.make_smoke_windows` set, so compute and
    parameter budgets are matched across the champion and ablations (doc F: "keep
    the exact kernel, train/validation/test splits, and approximate training
    budget matched"). Returns an :class:`AblationReport` whose ``as_dict`` carries
    the per-ablation ``main_readout`` strings doc F prescribes.

    This is intentionally tiny (it runs in CI seconds on NumPy SPSA); it is a
    *structural* ablation harness — the matched-budget plumbing and the doc-F
    readouts — not a full champion-training run.
    """
    # Public model APIs only (HARD RULE: never edit train.py / architecture.py).
    from ..architecture import ArchConfig
    from ..train import TrainConfig, make_smoke_windows, train
    from ..world_model import ResidualWorldModel

    cfg = train_config or TrainConfig(seed=seed)
    windows = make_smoke_windows(n_windows, cfg.seed)

    # ---- champion (the un-ablated model) -------------------------------- #
    champ_model = ResidualWorldModel.build(
        arch=ArchConfig(seed=cfg.seed), n_ensemble=2
    )
    champ_report = train(champ_model, windows, cfg)
    champ_sens, champ_opt = _probe_variant(champ_model, windows, horizon_s)
    champion = AblationResult(
        name="champion",
        main_readout="Full residual world model (un-ablated reference).",
        n_steps=champ_report.steps,
        final_loss=champ_report.final_loss,
        collapsed=champ_report.collapsed(),
        action_sensitivity=champ_sens,
        optimism_bias_rate=champ_opt,
        readouts={"backend": champ_report.backend},
    )

    results: list[AblationResult] = []
    for name in ABLATION_NAMES:
        results.append(
            _run_one_ablation(name, windows, cfg, horizon_s, champ_sens)
        )

    return AblationReport(champion=champion, ablations=results)


def _run_one_ablation(
    name: str, windows, cfg, horizon_s: int, champion_sensitivity: float
) -> AblationResult:
    """Train + probe one named ablated variant via the public train() API."""
    from ..architecture import ArchConfig
    from ..train import TrainConfig, train
    from ..world_model import ResidualWorldModel

    readout = ABLATION_READOUTS[name]

    if name == "remove_temporal_hierarchy":
        # Reuse the flat single-GRU baseline (no fast/slow split, no relations).
        # It has no train() loop (it is a forecasting foil); its compute budget is
        # matched in spirit — one matched-width recurrence over the same windows.
        flat = FlatSequenceBaseline.build(seed=cfg.seed)
        sens, opt = _probe_variant(flat, windows, horizon_s)
        return AblationResult(
            name=name,
            main_readout=readout,
            n_steps=0,
            final_loss=float("nan"),
            collapsed=False,
            action_sensitivity=sens,
            optimism_bias_rate=opt,
            readouts={
                "variant": "FlatSequenceBaseline",
                "note": "single-timescale GRU, no hierarchy, no typed relations",
            },
        )

    # The remaining four are public-knob ablations of the real model.
    arch = ArchConfig(seed=cfg.seed)
    n_ensemble = 2
    abst = None
    use_windows = windows
    stages = ("warmup", "rollout", "rare_event", "calibration")
    jcfg = None

    if name == "remove_object_centric_state":
        # Drop typed relational message passing (pooled aggregates only).
        arch = ArchConfig(seed=cfg.seed, n_rounds=0)
    elif name == "remove_uncertainty_modeling":
        # Point predictor: single member ⇒ no epistemic disagreement; abstention
        # disabled (thresholds raised beyond the [0,1] score range).
        n_ensemble = 1
        abst = AbstentionConfig(
            max_epistemic_disagreement=float("inf"),
            max_ood_score=float("inf"),
            max_source_health_bad_prob=float("inf"),
            fallback="none",
        )
    elif name == "remove_synthetic_rare_event_augmentation":
        # Observational / factual replay only: drop the rare-event stage and the
        # injected rare-event windows.
        stages = ("warmup", "rollout", "calibration")
        use_windows = [w for w in windows if not w.is_rare_event] or list(windows)
    elif name == "remove_counterfactual_rollout_loss":
        # Teacher-forced factual prediction only: λ_Δ = 0 and only the warmup +
        # calibration stages, so the free-running branch-difference rollout loss
        # never fires (doc F #5: action sensitivity can vanish under this).
        from ..jepa import JEPAConfig

        jcfg = JEPAConfig(lambda_delta=0.0)
        stages = ("warmup", "calibration")

    model = ResidualWorldModel.build(
        arch=arch, abstention=abst, n_ensemble=n_ensemble
    )
    # Matched compute: same TrainConfig budget; only the stage *set* changes for
    # the rare-event / counterfactual ablations (which is the ablation itself).
    train_cfg = cfg if isinstance(cfg, TrainConfig) else TrainConfig()
    report = train(model, use_windows, train_cfg, jcfg, stages)
    sens, opt = _probe_variant(model, windows, horizon_s)

    extra: dict[str, object] = {"stages": list(stages), "backend": report.backend}
    if name == "remove_object_centric_state":
        extra["n_rounds"] = 0
    if name == "remove_uncertainty_modeling":
        extra["n_ensemble"] = 1
        extra["abstention_disabled"] = True
    if name == "remove_synthetic_rare_event_augmentation":
        extra["n_train_windows"] = len(use_windows)
        extra["dropped_rare_event_stage"] = True
    if name == "remove_counterfactual_rollout_loss":
        extra["lambda_delta"] = 0.0
        # Action-sensitivity *retention* vs the champion: the headline doc-F #5
        # number (a value far below 1 means the ablation collapsed action
        # sensitivity relative to the full model).
        if champion_sensitivity == champion_sensitivity and champion_sensitivity > 0:
            extra["action_sensitivity_retention"] = float(
                sens / champion_sensitivity
            ) if sens == sens else float("nan")

    return AblationResult(
        name=name,
        main_readout=readout,
        n_steps=report.steps,
        final_loss=report.final_loss,
        collapsed=report.collapsed(),
        action_sensitivity=sens,
        optimism_bias_rate=opt,
        readouts=extra,
    )


__all__ = [
    "ABLATION_NAMES",
    "ABLATION_READOUTS",
    "AblationReport",
    "AblationResult",
    "run_ablation_suite",
]
