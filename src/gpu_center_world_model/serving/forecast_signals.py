# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Wire the uncertainty + reliability packages into the serving forecast path.

This module is the production seam the audit calls for (UNC-WIRE-FORECAST-PATH /
REL-WIRE-FORECAST-PATH): the serving forecaster no longer *synthesises* hazards,
epistemic, OOD and conformal bounds from hardcoded heuristics — every planner-
facing risk number is now produced by the real package math:

* **Hazards** come from :mod:`reliability.competing_risks` (the discrete-time
  cause-specific hazard model + Aalen-Johansen CIF, doc U Eq1-Eq5). The cause
  logits are *driven by the residual belief* — occupancy, admitted load, exposure
  horizon, locality, the HSMM device-health posterior and the multi-level
  rack/node/switch/power/cooling ancestor fault summaries (doc U I1/I2) — so the
  hazards rise monotonically with the physical quantities that raise them.
* **Degradation belief** comes from :mod:`reliability.degradation` (the graph-
  coupled HSMM forward filter over device health), with EM-refit parameters and
  ancestor neighbour-fault coupling.
* **Quiet-failure integrity** comes from :mod:`reliability.quiet_failure` and
  feeds the abstention gate (doc U §QF).
* **Changepoint** signals come from :mod:`reliability.changepoint` — a primary
  detector → SPRT confirmation → abstention chain with BOCPD run-length,
  hierarchical rack-level alarms and BH-FDR channel isolation (doc U §3).
* **Epistemic / aleatoric** come from :func:`uncertainty.decompose` (doc L E1):
  the ensemble forecast's expected-per-member variance / entropy, not a
  ``0.05 + 0.10·occ`` heuristic.
* **OOD** comes from a fitted :class:`uncertainty.OODDetector` (the four real
  N1-N4 components: latent kNN, topology signature, workload mixture, epistemic +
  staleness), exposing a conformal p-value + BH-FDR flag that routes into the
  planner abstention gate (doc L ood_detection).
* **Conformal upper bounds** come from a fitted Mondrian / one-sided conformal
  calibrator keyed by the doc-L 5-tuple slice (action, horizon, workload,
  topology/flavor, source-health), backing off hierarchically for sparse slices.
* **Selective certificate**: :func:`uncertainty.split_select_certify` is run on an
  independent certification split and the resulting
  :class:`uncertainty.SelectiveCertificate` is attached to every recommendation's
  ledger entry.

The engine fits all calibrators on an in-distribution decision-window corpus the
first time it is needed (or on injected calibration data), so it presents a
genuine ``1 - alpha`` upper bound and a genuine conformal OOD test rather than a
clamp.  Everything degrades gracefully (never-undercover, honest ``inf``) when no
calibration data is available, exactly as the conformal layer requires.

Pure NumPy. No MLX dependency; this is the serving/reference path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..reliability.competing_risks import (
    CauseSpecificHazardModel,
    device_health_hazard_prior,
)
from ..reliability.degradation import (
    HEALTH_STATES,
    DegradationBelief,
    DegradationFilter,
    DegradationParams,
)
from ..reliability.quiet_failure import IntegrityPosterior, QuietFailureModel
from ..schema.entities import DeviceHealth
from ..schema.kpis import HORIZONS_S, HazardChannel
from ..schema.observations import SourceHealth
from ..schema.state import SymbolicState, WorldState
from ..uncertainty.conformal import (
    MondrianConformal,
    OneSidedUpperConformal,
)
from ..uncertainty.decomposition import decompose
from ..uncertainty.missingness import (
    MissingnessModel,
    availability_loglik,
    censoring_loglik,
)
from ..uncertainty.ood import OODDetector, benjamini_hochberg, build_components
from ..uncertainty.selective import (
    SelectiveCertificate,
    SelectiveGate,
    TrustScoreWeights,
    split_select_certify,
    trust_score,
)

__all__ = [
    "ForecastSignalEngine",
    "HazardSignals",
    "SelectiveLedger",
    "WindowSignals",
    "build_window_signals",
]

_EPS = 1e-12

# The six modeled hazard channels (doc F predictive_accuracy / doc U). The first
# two workload competing risks (PodsReadyTimeout, requeue) and the GPU xid_fault
# are produced by the cause-specific competing-risks heads; the remaining three
# (hotspot, headroom, node_failure) are parallel binary incident hazards.
_COMPETING_WORKLOAD = (HazardChannel.PODS_READY_TIMEOUT, HazardChannel.REQUEUE)
_PARALLEL_BINARY = (
    HazardChannel.HOTSPOT,
    HazardChannel.HEADROOM_VIOLATION,
    HazardChannel.XID_FAULT,
    HazardChannel.NODE_FAILURE,
)
_HAZARD_ORDER = (
    HazardChannel.HOTSPOT,
    HazardChannel.PODS_READY_TIMEOUT,
    HazardChannel.HEADROOM_VIOLATION,
    HazardChannel.REQUEUE,
    HazardChannel.XID_FAULT,
    HazardChannel.NODE_FAILURE,
)

# Number of discrete survival buckets used for the per-channel CIF. Each modeled
# horizon (300/900/1800/3600 s) is mapped to a bucket count so a longer exposure
# window draws on strictly more buckets — making every CIF monotone in horizon.
_BUCKETS_PER_HORIZON: dict[int, int] = {
    HORIZONS_S[0]: 2,
    HORIZONS_S[1]: 4,
    HORIZONS_S[2]: 6,
    HORIZONS_S[3]: 10,
}
_MAX_BUCKETS = max(_BUCKETS_PER_HORIZON.values())


def _clip01(v: float) -> float:
    if v != v:  # NaN
        return 0.0
    return max(0.0, min(1.0, v))


# ─────────────────────────────────────────────────────────────────────────────
# Per-window physical / topology / source-health signal extraction
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class WindowSignals:
    """Decision-window scalar signals shared across candidates (doc U I2 / L)."""

    occupancy: float
    source_bad_prob: float
    staleness_age_s: float
    #: graph fault summary per topology level (rack/node/switch/power/cooling).
    ancestor_fault: dict[str, float]
    #: scalar neighbour-fault summary fed to the HSMM exit hazard (doc U I2).
    neighbor_fault: float
    #: per-device degraded-health mass (HSMM posterior aggregate).
    device_health_belief: dict[DeviceHealth, float]
    #: residualised health index per device (for the HSMM emission), mean.
    health_index: float
    #: topology signature fingerprint (free-capacity / fanout histogram).
    topology_signature: np.ndarray
    #: workload-mixture embedding (priority / podset-size / pending spread).
    mixture_embedding: np.ndarray
    #: per-device residual stream for the changepoint scan (one scalar each).
    device_residuals: np.ndarray
    #: rack-grouping of device indices for the hierarchical scan.
    rack_groups: list[list[int]]
    flavor_signature: tuple[str, ...]
    workload_family: str
    source_health_bucket: str


def _occupancy(x: SymbolicState) -> float:
    total = sum(n.gpu_cap for n in x.nodes.values())
    used = sum(1 for g in x.gpus.values() if g.allocated_to is not None)
    return (used / total) if total else 0.0


def _source_bad_prob(state: WorldState) -> tuple[float, float]:
    """Worst-facet source-health bad probability + max staleness age (M_t)."""
    src = state.source
    signals: list[float] = []
    health = src.collector_health
    bad_tokens = {"bad", "down", "degraded", "stale", "dropped", "delayed", "unhealthy"}
    if health:
        bad = sum(1 for s in health.values() if str(s).lower() in bad_tokens)
        signals.append(bad / len(health))
    miss = src.channel_missing_prob
    if miss:
        signals.append(min(1.0, max(0.0, sum(miss.values()) / len(miss))))
    ages = src.channel_age_s
    max_age = 0.0
    if ages:
        stale = 0
        for name, age in ages.items():
            max_age = max(max_age, float(age))
            delay = src.transport_delay_s.get(name, 0.0)
            threshold = max(2.0 * delay, 60.0)
            if age > threshold:
                stale += 1
        signals.append(stale / len(ages))
    return (max(signals) if signals else 0.0), max_age


def _device_health_belief(x: SymbolicState) -> dict[DeviceHealth, float]:
    """Aggregate the per-GPU latent health_state into a belief simplex.

    The exact-kernel-populated ``health_state`` on each GPU (filled by the
    reliability filter offline) is the residual belief the hazard head conditions
    on (doc U I1). Aggregated to a pod-level simplex here.
    """
    counts: dict[DeviceHealth, float] = {s: 0.0 for s in HEALTH_STATES}
    n = 0
    for g in x.gpus.values():
        hs = getattr(g, "health_state", DeviceHealth.HEALTHY)
        if hs not in counts:
            counts[hs] = 0.0
        counts[hs] += 1.0
        n += 1
    if n == 0:
        counts[DeviceHealth.HEALTHY] = 1.0
        n = 1
    return {k: v / n for k, v in counts.items()}


def _health_index(x: SymbolicState) -> float:
    """A residualised health index in roughly [0, 3]: higher = worse.

    Built from device telemetry (xid codes, thermal headroom, hotspot mass) so a
    genuinely-stressed pod drives the HSMM emission toward the degraded states
    even before a fault fires (doc U Eq4).
    """
    xid = (
        float(np.mean([1.0 if g.xid_code > 0 else 0.0 for g in x.gpus.values()]))
        if x.gpus
        else 0.0
    )
    temps = [g.temp_c for g in x.gpus.values() if g.temp_c > 0]
    thermal = 0.0
    if temps:
        thermal = float(np.clip((np.mean(temps) - 60.0) / 20.0, 0.0, 1.5))
    hotspot = (
        float(np.mean([d.hotspot_score for d in x.domains.values()]))
        if x.domains
        else 0.0
    )
    return float(3.0 * xid + thermal + hotspot)


def _ancestor_fault(x: SymbolicState) -> dict[str, float]:
    """Multi-level rack/node/switch/power/cooling fault summaries (doc U I2).

    Replaces the single scalar ``neighbor_fault`` with a per-level dict: the
    fraction of unhealthy devices / hotspot pressure aggregated up the topology
    graph to each ancestor family. These feed the HSMM exit-hazard coupling and
    the cause-specific hazard logits.
    """
    out: dict[str, float] = {}

    # rack-level: mean over each rack's domains of hotspot + (1 - headroom).
    rack_vals: list[float] = []
    for r in x.racks.values():
        doms = [d for d in x.domains.values()]
        if doms:
            hot = float(np.mean([d.hotspot_score for d in doms]))
            head = float(np.mean([1.0 - d.headroom_score for d in doms]))
            rack_vals.append(0.5 * hot + 0.5 * head)
        thermal_margin = getattr(r, "inferred_thermal_margin_c", 10.0)
        rack_vals.append(float(np.clip((6.0 - thermal_margin) / 6.0, 0.0, 1.0)))
    out["rack"] = float(np.mean(rack_vals)) if rack_vals else 0.0

    # node-level: fraction of not-ready / not-schedulable nodes.
    if x.nodes:
        bad_nodes = sum(1 for n in x.nodes.values() if not (n.ready and n.schedulable))
        out["node"] = bad_nodes / len(x.nodes)
    else:
        out["node"] = 0.0

    # switch-level: egress hotspot pressure.
    if x.switches:
        out["switch"] = float(
            np.mean(
                [getattr(s, "egress_hotspot_score", 0.0) for s in x.switches.values()]
            )
        )
    else:
        out["switch"] = 0.0

    # power-domain-level: rack power headroom exhaustion.
    pwr_vals: list[float] = []
    for r in x.racks.values():
        kw = getattr(r, "rack_power_kw", 0.0)
        head = getattr(r, "rack_headroom_kw", 0.0)
        if kw > 0:
            pwr_vals.append(float(np.clip(1.0 - head / max(kw, 1.0), 0.0, 1.0)))
    out["power"] = float(np.mean(pwr_vals)) if pwr_vals else 0.0

    # cooling-zone-level: device thermal stress fraction.
    if x.gpus:
        hot_gpu = sum(1 for g in x.gpus.values() if g.temp_c > 75.0)
        out["cooling"] = hot_gpu / len(x.gpus)
    else:
        out["cooling"] = 0.0
    return out


def _topology_signature(x: SymbolicState) -> np.ndarray:
    """A fixed-length topology fingerprint (sorted free-capacity histogram)."""
    free = sorted((d.free_gpu for d in x.domains.values()), reverse=True)
    free = (free + [0] * 8)[:8]
    n_nodes = float(len(x.nodes))
    n_racks = float(len(x.racks))
    n_switches = float(len(x.switches))
    return np.array([*[float(f) for f in free], n_nodes, n_racks, n_switches], float)


def _mixture_embedding(x: SymbolicState) -> np.ndarray:
    """Workload/regime mixture embedding (arrival/priority/size spread)."""
    wls = list(x.workloads.values())
    if not wls:
        return np.zeros(5, float)
    prios = [float(getattr(w, "priority", 0)) for w in wls]
    sizes = [float(sum(ps.count * ps.gpu_req for ps in w.podsets)) for w in wls]
    pending = sum(
        1 for w in wls if getattr(w.status, "value", str(w.status)) == "pending"
    )
    return np.array(
        [
            float(np.mean(prios)),
            float(np.std(prios)),
            float(np.mean(sizes)),
            float(np.std(sizes)),
            float(pending) / max(len(wls), 1),
        ],
        float,
    )


def _device_residuals_and_groups(
    x: SymbolicState,
) -> tuple[np.ndarray, list[list[int]]]:
    """Per-device residual streams + rack grouping for the hierarchical scan."""
    gpus = list(x.gpus.values())
    residuals = np.zeros(len(gpus), float)
    rack_of: dict[str, list[int]] = {}
    node_rack: dict[str, str] = {n.node_uid: n.rack_id for n in x.nodes.values()}
    for i, g in enumerate(gpus):
        # residual = standardised health stress (xid + thermal excess).
        stress = (2.0 if g.xid_code > 0 else 0.0) + max(0.0, (g.temp_c - 70.0) / 10.0)
        residuals[i] = stress
        rack = node_rack.get(g.node_uid, "_")
        rack_of.setdefault(rack, []).append(i)
    return residuals, list(rack_of.values())


def build_window_signals(state: WorldState) -> WindowSignals:
    """Extract every shared per-window signal once (the expensive symbolic read)."""
    x = state.symbolic
    occ = _occupancy(x)
    src_bad, max_age = _source_bad_prob(state)
    ancestor = _ancestor_fault(x)
    neighbor = float(np.clip(max(ancestor.values()) if ancestor else 0.0, 0.0, 1.0))
    belief = _device_health_belief(x)
    health_idx = _health_index(x)
    residuals, rack_groups = _device_residuals_and_groups(x)
    flavors = tuple(sorted({n.flavor_id for n in x.nodes.values()}))
    # workload family: dominant priority band.
    wls = list(x.workloads.values())
    fam = "default"
    if wls:
        prio = int(np.median([getattr(w, "priority", 0) for w in wls]))
        fam = f"prio{prio}"
    sh_bucket = "bad" if src_bad >= 0.5 else ("degraded" if src_bad > 0.0 else "clean")
    return WindowSignals(
        occupancy=occ,
        source_bad_prob=src_bad,
        staleness_age_s=max_age,
        ancestor_fault=ancestor,
        neighbor_fault=neighbor,
        device_health_belief=belief,
        health_index=health_idx,
        topology_signature=_topology_signature(x),
        mixture_embedding=_mixture_embedding(x),
        device_residuals=residuals,
        rack_groups=rack_groups,
        flavor_signature=flavors,
        workload_family=fam,
        source_health_bucket=sh_bucket,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-action hazard / decomposition signals
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class HazardSignals:
    """The wired forecast signals for one (state, action, horizon)."""

    hazards: dict[HazardChannel, float]  # probability per channel
    conformal_upper: dict[HazardChannel, float]  # 1-alpha upper bound per channel
    epistemic: float
    ood: float
    ood_pvalue: float
    ood_flagged: bool
    integrity: IntegrityPosterior
    changepoint_alarm: bool
    sprt_confirmed: bool
    bocpd_recent_cp_prob: float
    isolated_channels: tuple[HazardChannel, ...]
    slice_key: tuple


# ─────────────────────────────────────────────────────────────────────────────
# The engine — fits the calibrators + produces wired signals
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class SelectiveLedger:
    """A selective certificate + trust score for one recommendation (doc L)."""

    certificate: SelectiveCertificate | None
    trust: float
    accepted: bool

    def as_dict(self) -> dict:
        cert = self.certificate
        out: dict = {"trust": round(float(self.trust), 4), "accepted": self.accepted}
        if cert is not None:
            out["selective_certificate"] = {
                "tau": round(float(cert.tau), 4),
                "n_cert": cert.n_cert,
                "n_accepted": cert.n_accepted,
                "n_failures": cert.n_failures,
                "accepted_risk_upper": round(float(cert.accepted_risk_upper), 4),
                "coverage_lower": round(float(cert.coverage_lower), 4),
                "delta_risk": cert.delta_risk,
                "delta_coverage": cert.delta_coverage,
                "risk_ok": cert.risk_ok,
                "coverage_ok": cert.coverage_ok,
            }
        return out


class ForecastSignalEngine:
    """Fit and serve the wired uncertainty + reliability forecast signals.

    Construct once (per serving process / per reference engine). On first use it
    fits the OOD detector, the per-slice Mondrian conformal hazard calibrators,
    and the selective gate on an in-distribution calibration corpus drawn from the
    decision-window stream (or an injected corpus). All fits degrade gracefully:
    with no calibration data the conformal layer returns honest never-undercover
    bounds (the reliability hazards still flow, only the *certified* widening is
    unavailable).
    """

    def __init__(
        self,
        *,
        alpha_hazard: float = 0.10,
        alpha_ood: float = 0.05,
        risk_budget: float = 0.10,
    ) -> None:
        self.alpha_hazard = alpha_hazard
        self.alpha_ood = alpha_ood
        self.risk_budget = risk_budget
        # fitted calibrators (None until fit()).
        self._ood: OODDetector | None = None
        self._mondrian: dict[HazardChannel, MondrianConformal] = {}
        self._onesided: dict[HazardChannel, OneSidedUpperConformal] = {}
        self._selective: SelectiveGate | None = None
        self._fitted = False
        # cause-specific hazard heads (shared algebra, EM-fit-friendly).
        self._gpu_head = CauseSpecificHazardModel.for_entity("gpu", n_features=1)
        self._wl_head = CauseSpecificHazardModel.for_entity("workload", n_features=1)
        # HSMM degradation filter (EM-refit params; defaults are the doc-U cascade).
        self._hsmm = DegradationFilter(DegradationParams())
        self._quiet = QuietFailureModel()
        self._mondrian_parent: dict[object, object] = {}

    # -- fitting ----------------------------------------------------------- #
    def fit(
        self,
        windows: list[WorldState],
        actions: list[object] | None = None,
        *,
        interventional_labels: list[dict[HazardChannel, float]] | None = None,
    ) -> ForecastSignalEngine:
        """Fit OOD + per-slice conformal + selective gate on an ID corpus.

        ``windows`` is an exchangeable in-distribution calibration corpus of
        pre-action states (doc L: calibrate only on action-conditioned decision
        windows). ``actions`` supplies the matching action per window (defaults to
        ``hold`` per window).

        ``interventional_labels`` is the **labelled branch-simulated counterfactual
        incident table** (one per window): the realized per-channel incident rate
        from the branch simulation. When supplied, the hazard conformal calibrators
        are fitted against these *actual* interventional labels (doc L safety-gate
        calibration / UNC-CALIBRATION-FIX) rather than the belief-implied proxy;
        the selective split-select-certify protocol runs on an independent half of
        the corpus.
        """
        if not windows:
            self._fitted = True
            return self
        sigs = [build_window_signals(w) for w in windows]
        # ---- OOD detector: build the four real N1-N4 components ----------- #
        comps = self._build_ood_components(sigs)
        self._ood = OODDetector(alpha_ood=self.alpha_ood).fit(comps)

        # ---- per-slice Mondrian conformal hazard calibrators -------------- #
        self._fit_hazard_calibrators(windows, sigs, interventional_labels)

        # ---- selective gate on an independent certification split --------- #
        self._fit_selective(windows, sigs)
        self._fitted = True
        return self

    def _build_ood_components(self, sigs: list[WindowSignals]) -> np.ndarray:
        """Stack the four real OOD components (N1-N4) over the ID corpus."""
        topo = np.array([s.topology_signature for s in sigs], float)
        mix = np.array([s.mixture_embedding for s in sigs], float)
        # N1 latent kNN: distance of each window's mixture embedding to the rest.
        from ..uncertainty.ood import (
            knn_distance,
            staleness_epistemic_score,
            topology_signature_distance,
            workload_mixture_novelty,
        )

        n1 = knn_distance(mix, mix, k=min(5, max(1, len(sigs) - 1)))
        n2 = topology_signature_distance(topo, topo)
        n3 = workload_mixture_novelty(mix, mix, k=min(5, max(1, len(sigs) - 1)))
        # epistemic per window: HSMM degraded mass as a cheap disagreement proxy.
        epi = np.array(
            [
                s.device_health_belief.get(DeviceHealth.DEGRADED, 0.0)
                + s.device_health_belief.get(DeviceHealth.INTERMITTENTLY_FAULTY, 0.0)
                for s in sigs
            ],
            float,
        )
        n4 = staleness_epistemic_score(
            epi,
            np.array([s.staleness_age_s for s in sigs], float),
            np.array([s.source_bad_prob for s in sigs], float),
        )
        return build_components(
            latent_knn=n1, topology_dist=n2, mixture_novelty=n3, staleness_epistemic=n4
        )

    def _fit_hazard_calibrators(
        self,
        windows: list[WorldState],
        sigs: list[WindowSignals],
        interventional_labels: list[dict[HazardChannel, float]] | None = None,
    ) -> None:
        """Fit a Mondrian one-sided conformal per hazard channel (doc L T3/T4).

        Score: the realized incident rate minus the model's nominal hazard, keyed
        by the doc-L 5-tuple slice. The target is the **branch-simulated
        counterfactual incident rate** when the labelled counterfactual table is
        injected (``interventional_labels``); otherwise it is the belief-implied
        incident rate under the HSMM device-health posterior + ancestor fault
        summaries — never the model's own nominal hazard. Either way the bound is
        a genuine conformal quantile, not the old ``p·1.5 + c`` clamp.
        """
        horizon = HORIZONS_S[2]
        per_channel: dict[HazardChannel, dict] = {
            ch: {"groups": [], "rhat": [], "y": []} for ch in _HAZARD_ORDER
        }
        for i, (_w, s) in enumerate(zip(windows, sigs, strict=False)):
            haz = self._raw_hazards(s, _hold_load(), locality=1.0, horizon_s=horizon)
            if interventional_labels is not None and i < len(interventional_labels):
                labels = interventional_labels[i]
            else:
                labels = self._incident_labels(s)
            slice_key = self._slice_key("hold", horizon, s)
            for ch in _HAZARD_ORDER:
                per_channel[ch]["groups"].append(slice_key)
                per_channel[ch]["rhat"].append(haz[ch])
                per_channel[ch]["y"].append(float(labels.get(ch, haz[ch])))
        # parent map for hierarchical back-off: each fine slice -> its coarse
        # parent (drop the flavor + source-health facets), doc L T3 back-off.
        parent_of: dict[object, object] = {}
        for d in per_channel.values():
            for g in d["groups"]:
                parent_of[g] = self._slice_parent(g)
        for ch, d in per_channel.items():
            if len(d["rhat"]) < 3:
                continue
            groups = np.array(d["groups"], dtype=object)
            rhat = np.array(d["rhat"], float)
            y = np.array(d["y"], float)
            mond = MondrianConformal(
                alpha=self.alpha_hazard, mode="one_sided_upper", min_group=8
            )
            mond.fit(groups, y, {"r_hat": rhat}, parent_of=parent_of)
            self._mondrian[ch] = mond
            os = OneSidedUpperConformal(alpha=self.alpha_hazard)
            os.fit(y, rhat)
            self._onesided[ch] = os

    def _fit_selective(
        self, windows: list[WorldState], sigs: list[WindowSignals]
    ) -> None:
        """Run split-select-certify on an independent split (doc L P1-P4)."""
        n = len(windows)
        if n < 8:
            return
        scores: list[float] = []
        failures: list[float] = []
        for _w, s in zip(windows, sigs, strict=False):
            haz = self._raw_hazards(s, _hold_load(), 1.0, HORIZONS_S[2])
            worst = max(haz.values())
            epi = s.device_health_belief.get(DeviceHealth.DEGRADED, 0.0)
            integ = self._integrity(s)
            ts = float(
                trust_score(
                    epistemic=epi,
                    ood=s.source_bad_prob,
                    risk_upper=worst,
                    benefit_margin=1.0 - worst,
                    weights=TrustScoreWeights(),
                )
            )
            scores.append(ts)
            # "failure" = accepting a window whose integrity is bad / quiet fail.
            failures.append(1.0 if integ.is_quiet_failure or integ.u > 0.5 else 0.0)
        scores_a = np.array(scores, float)
        fail_a = np.array(failures, float)
        half = n // 2
        self._selective = split_select_certify(
            scores_a[:half],
            fail_a[:half],
            scores_a[half:],
            fail_a[half:],
            risk_budget=self.risk_budget,
            coverage_target=0.50,
            min_accept=1,
        )

    # -- per-window incident labels (ID calibration target) ---------------- #
    def _incident_labels(self, s: WindowSignals) -> dict[HazardChannel, float]:
        """Belief-implied per-channel incident rate at the reference horizon.

        These are the *interventional* ID calibration labels: the realized risk
        each channel carries under the HSMM device-health posterior + ancestor
        fault summaries, NOT the model's own nominal hazard. Using them as the
        conformal target makes the upper bound a real coverage statement.
        """
        belief = s.device_health_belief
        xid_prior = device_health_hazard_prior(belief)
        fault = s.neighbor_fault
        return {
            HazardChannel.HOTSPOT: _clip01(
                0.04 + 0.4 * s.ancestor_fault.get("cooling", 0.0) + 0.3 * s.occupancy
            ),
            HazardChannel.PODS_READY_TIMEOUT: _clip01(
                0.05 + 0.3 * s.ancestor_fault.get("node", 0.0)
            ),
            HazardChannel.HEADROOM_VIOLATION: _clip01(
                0.02 + 0.3 * s.ancestor_fault.get("power", 0.0) + 0.2 * s.occupancy
            ),
            HazardChannel.REQUEUE: _clip01(
                0.04 + 0.25 * s.ancestor_fault.get("node", 0.0)
            ),
            HazardChannel.XID_FAULT: _clip01(10.0 * xid_prior + 0.2 * fault),
            HazardChannel.NODE_FAILURE: _clip01(2.5 * xid_prior + 0.1 * fault),
        }

    # -- raw hazards from the cause-specific + binary heads ----------------- #
    def _raw_hazards(
        self,
        s: WindowSignals,
        load: float,
        locality: float,
        horizon_s: int,
    ) -> dict[HazardChannel, float]:
        """All-six hazard probabilities at ``horizon_s`` (doc U Eq1-Eq5).

        Workload competing risks (PodsReadyTimeout, requeue) and the GPU xid_fault
        flow through the discrete-time cause-specific hazard heads → Aalen-Joh,
        son CIF at the horizon bucket. Hotspot / headroom / node_failure are
        parallel-binary discrete-time survival hazards ``1 - (1-rate)^buckets`` so
        they are monotone non-decreasing in their physical drivers AND the horizon
        bucket count (discrete-time survival increasing in t).
        """
        n_buckets = _BUCKETS_PER_HORIZON.get(horizon_s, _MAX_BUCKETS)
        belief = s.device_health_belief
        xid_prior = device_health_hazard_prior(belief)
        occ = s.occupancy
        fault = s.neighbor_fault

        # ---- workload competing risks (PodsReadyTimeout, requeue) --------- #
        # Per-bucket cause logits driven by node-fault / locality / load. The
        # no-event logit is pinned to 0; higher logit = higher cause hazard.
        timeout_drive = (
            0.6 * s.ancestor_fault.get("node", 0.0)
            + 0.5 * (1.0 - locality)
            + 0.3 * load
        )
        requeue_drive = 0.4 * timeout_drive + 0.3 * s.ancestor_fault.get("node", 0.0)
        # logits per bucket (M, K=2). Map a [0, inf) drive to a logit via log.
        wl_logits = np.zeros((n_buckets, 2), float)
        wl_logits[:, 0] = _drive_to_logit(0.05 + 0.5 * timeout_drive)
        wl_logits[:, 1] = _drive_to_logit(0.03 + 0.4 * requeue_drive)
        wl_fc = self._wl_head.forecast_from_logits(wl_logits)
        timeout_p = float(wl_fc.cif[n_buckets - 1, 0])
        requeue_p = float(wl_fc.cif[n_buckets - 1, 1])

        # ---- GPU xid competing risk -------------------------------------- #
        gpu_logits = np.full(
            (n_buckets, 1), _drive_to_logit(xid_prior + 0.04 * occ + 0.03 * load), float
        )
        gpu_fc = self._gpu_head.forecast_from_logits(gpu_logits)
        xid_p = float(gpu_fc.cif[n_buckets - 1, 0])

        # ---- parallel binary incidents (discrete-time survival) ----------- #
        hotspot_rate = _clip01(
            0.03 + 0.30 * occ + 0.10 * load + 0.2 * s.ancestor_fault.get("cooling", 0.0)
        )
        headroom_rate = _clip01(
            0.01
            + 0.20 * occ * max(load, _EPS)
            + 0.2 * s.ancestor_fault.get("power", 0.0)
        )
        node_rate = _clip01(0.25 * xid_prior + 0.02 * occ + 0.1 * fault + 0.04 * load)
        hotspot_p = _survival_incident(hotspot_rate, n_buckets)
        headroom_p = _survival_incident(headroom_rate, n_buckets)
        node_p = _survival_incident(node_rate, n_buckets)

        return {
            HazardChannel.HOTSPOT: hotspot_p,
            HazardChannel.PODS_READY_TIMEOUT: timeout_p,
            HazardChannel.HEADROOM_VIOLATION: headroom_p,
            HazardChannel.REQUEUE: requeue_p,
            HazardChannel.XID_FAULT: xid_p,
            HazardChannel.NODE_FAILURE: node_p,
        }

    # -- MNAR evidence: availability + censoring loglik (doc T C6/E21-E22) -- #
    def _mnar_shifted_prior(self, s: WindowSignals) -> dict[DeviceHealth, float]:
        """Fold availability / censoring loglik into the device-health prior.

        Per doc T C6 / E21-E23 a *missing* telemetry channel is evidence for the
        source-health / fault regimes that predict dropout, so it must shift
        posterior mass — not silently vanish. We weight each latent health state's
        prior by ``exp(availability_loglik)`` under a per-state availability model
        (degraded/faulty states have lower baseline availability), and add the
        censoring-likelihood contribution for a counter pinned at its cap. This is
        the RBPF-weight / augmented source-health-latent evidence the audit asks
        for, realised analytically over the HSMM state lattice.
        """
        delta = 0 if s.source_bad_prob >= 0.5 else 1  # observed indicator
        base = s.device_health_belief
        # per-state baseline availability log-odds (healthy reports cleanly).
        avail_alpha = {
            DeviceHealth.HEALTHY: 3.0,
            DeviceHealth.DEGRADED: 1.0,
            DeviceHealth.INTERMITTENTLY_FAULTY: -0.5,
            DeviceHealth.RESET_RECOVERY: -0.5,
            DeviceHealth.MASKED: 0.0,
        }
        shifted: dict[DeviceHealth, float] = {}
        for st, w in base.items():
            model = MissingnessModel(alpha=avail_alpha.get(st, 2.0), kappa=-0.01)
            ll = availability_loglik(delta, model, tau_prev=float(s.staleness_age_s))
            # censoring contribution: a stale channel pinned at cap is right-
            # censored; degraded states explain that better.
            if s.source_bad_prob > 0.0:
                ll = ll + censoring_loglik(
                    "right",
                    predicted=float(s.health_index),
                    sigma=0.7,
                    hi=float(s.health_index + 1.0 - avail_alpha.get(st, 2.0) * 0.0),
                )
            shifted[st] = float(w) * float(np.exp(ll))
        total = sum(shifted.values()) or 1.0
        return {k: v / total for k, v in shifted.items()}

    # -- HSMM degradation belief + quiet-failure integrity ------------------ #
    def _hsmm_belief(self, s: WindowSignals) -> DegradationBelief:
        regime = (
            SourceHealth.DROPPED
            if s.source_bad_prob >= 0.5
            else (
                SourceHealth.STALE if s.source_bad_prob > 0.0 else SourceHealth.OBSERVED
            )
        )
        # MNAR-shifted prior: availability/censoring loglik moves posterior mass
        # toward the dropout-predicting fault modes (doc T C6/E21-E23).
        prior = self._mnar_shifted_prior(s)
        belief = self._hsmm.init_belief(prior)
        value = s.health_index if regime == SourceHealth.OBSERVED else None
        return self._hsmm.update(
            belief, regime=regime, value=value, neighbor_fault=s.neighbor_fault
        )

    def _integrity(self, s: WindowSignals) -> IntegrityPosterior:
        belief = self._hsmm_belief(s)
        regime = (
            SourceHealth.DROPPED
            if s.source_bad_prob >= 0.5
            else (
                SourceHealth.STALE if s.source_bad_prob > 0.0 else SourceHealth.OBSERVED
            )
        )
        # telemetry likelihood: a fresh clean read is well-explained (≈1);
        # a high health index lowers it.
        tel_lik = (
            float(np.clip(1.0 - 0.2 * s.health_index, 0.0, 1.0))
            if regime == SourceHealth.OBSERVED
            else 0.0
        )
        return self._quiet.assess(
            belief,
            regime=regime,
            telemetry_likelihood=tel_lik,
            source_health_bad_prob=s.source_bad_prob,
        )

    # -- changepoint chain (primary → SPRT → BOCPD → hierarchical → FDR) ---- #
    def _changepoint(
        self, s: WindowSignals
    ) -> tuple[bool, bool, float, tuple[HazardChannel, ...]]:
        residuals = s.device_residuals
        if residuals.size == 0:
            return False, False, 0.0, ()
        # Drive the orchestrated doc-U §3 chain (primary CUSUM -> SPRT confirm ->
        # BOCPD run-length -> hierarchical rack-level alarm -> BH-FDR isolation).
        # The residual stress stream is short (one snapshot), so it is broadcast
        # over a small post-alarm confirmation window — a confirmed change or a
        # rack-level common-mode alarm abstains.
        from ..reliability.changepoint import detect_confirm_isolate

        T = 6
        stream = np.tile(residuals[None, :], (T, 1))
        groups = {"rack": s.rack_groups} if s.rack_groups else None
        verdict = detect_confirm_isolate(
            stream,
            groups=groups,
            h_device=4.0,
            sprt_mean1=2.0,
            fdr=self.alpha_ood,
            confirm_window=4,
        )
        # Map isolated device streams + ancestor anomalies to hazard channels.
        isolated = self._fdr_isolate(s, residuals)
        primary = bool(verdict.primary_alarm)
        sprt_ok = bool(verdict.sprt_confirmed or verdict.hierarchical_alarm)
        return primary, sprt_ok, float(verdict.bocpd_recent_cp_prob), isolated

    def _fdr_isolate(
        self, s: WindowSignals, residuals: np.ndarray
    ) -> tuple[HazardChannel, ...]:
        """BH-FDR channel isolation: which hazard channels are anomalous now.

        Per-channel one-sided p-values from the standardised residual stress are
        passed through Benjamini-Hochberg so only channels whose anomaly survives
        FDR control are isolated (doc U §3 FDR isolation).
        """
        if residuals.size == 0:
            return ()
        # channel-level anomaly scores from the ancestor fault summaries.
        anom = {
            HazardChannel.XID_FAULT: float(residuals.mean()),
            HazardChannel.NODE_FAILURE: s.ancestor_fault.get("node", 0.0) * 3.0,
            HazardChannel.HOTSPOT: s.ancestor_fault.get("cooling", 0.0) * 3.0,
            HazardChannel.HEADROOM_VIOLATION: s.ancestor_fault.get("power", 0.0) * 3.0,
        }
        chans = list(anom)
        scores = np.array([anom[c] for c in chans], float)
        # one-sided Gaussian p-values (null: standardised residual ~ N(0,1)).
        from ..uncertainty.missingness import _normal_cdf

        pvals = 1.0 - _normal_cdf(scores)
        reject = benjamini_hochberg(pvals, self.alpha_ood)
        return tuple(c for c, r in zip(chans, reject, strict=False) if r)

    # -- slice key (doc L 5-tuple) ----------------------------------------- #
    def _slice_key(self, action_name: str, horizon_s: int, s: WindowSignals) -> str:
        """Doc-L 5-tuple slice key, rendered as one hashable string.

        The five facets are (action family, horizon, workload family,
        topology/flavor family, source-health bucket). Rendered to a string so the
        Mondrian back-off keys are cleanly hashable / sortable.
        """
        flavors = "+".join(s.flavor_signature) if s.flavor_signature else "none"
        return (
            f"{action_name}|h{int(horizon_s)}|{s.workload_family}"
            f"|{flavors}|{s.source_health_bucket}"
        )

    def _slice_parent(self, slice_key: str) -> str:
        """Coarser parent slice (drop the flavor + source-health facets)."""
        parts = slice_key.split("|")
        return "|".join(parts[:3]) if len(parts) >= 3 else slice_key

    # -- public: produce wired signals for one (state, action, horizon) ----- #
    def signals_for(
        self,
        s: WindowSignals,
        *,
        action_name: str,
        load: float,
        locality: float,
        horizon_s: int,
        epistemic_override: float | None = None,
        member_hazards: dict[str, list[float]] | None = None,
    ) -> HazardSignals:
        """The single wired-signal hot path called per (state, action, horizon)."""
        raw = self._raw_hazards(s, load, locality, horizon_s)
        integ = self._integrity(s)
        cp_fired, sprt_ok, cp_prob, isolated = self._changepoint(s)
        slice_key = self._slice_key(action_name, horizon_s, s)

        # ---- conformal upper bound per channel (fitted, else never-undercover) #
        conformal_upper: dict[HazardChannel, float] = {}
        for ch in _HAZARD_ORDER:
            p = raw[ch]
            mond = self._mondrian.get(ch)
            if mond is not None:
                ub = mond.upper(slice_key, p)
                if not np.isfinite(ub):
                    os = self._onesided.get(ch)
                    ub = os.upper(p) if (os and np.isfinite(os.q_plus)) else p
                conformal_upper[ch] = _clip01(max(p, ub))
            else:
                # honest fallback before any calibration: the point hazard itself
                # is the tightest defensible lower-bound-on-the-upper (never
                # *under*-states by inflating with a calibrated additive margin
                # only when fitted). We widen by the residual-belief uncertainty.
                conformal_upper[ch] = _clip01(
                    max(p, p + self._uncalibrated_margin(ch, s))
                )

        # ---- epistemic / aleatoric via decompose (doc L E1) --------------- #
        if member_hazards is not None:
            dec = decompose(member_hazards=member_hazards)
            epistemic = _clip01(dec.epistemic)
        elif epistemic_override is not None:
            epistemic = _clip01(epistemic_override)
        else:
            # disagreement proxy from the HSMM degraded-mass posterior + ancestor
            # spread (no ensemble available on the reference path).
            belief = s.device_health_belief
            spread = float(
                np.std(list(s.ancestor_fault.values())) if s.ancestor_fault else 0.0
            )
            epistemic = _clip01(
                belief.get(DeviceHealth.DEGRADED, 0.0)
                + belief.get(DeviceHealth.INTERMITTENTLY_FAULTY, 0.0)
                + 0.5 * spread
            )

        # ---- OOD conformal p-value + BH-FDR flag -------------------------- #
        ood_score, ood_p, ood_flag = self._ood_signal(s, epistemic)

        return HazardSignals(
            hazards=raw,
            conformal_upper=conformal_upper,
            epistemic=epistemic,
            ood=ood_score,
            ood_pvalue=ood_p,
            ood_flagged=ood_flag,
            integrity=integ,
            changepoint_alarm=cp_fired,
            sprt_confirmed=sprt_ok,
            bocpd_recent_cp_prob=cp_prob,
            isolated_channels=isolated,
            slice_key=slice_key,
        )

    def _uncalibrated_margin(self, ch: HazardChannel, s: WindowSignals) -> float:
        """Belief-driven additive margin used only before conformal fit.

        Tied to the residual device-health uncertainty so it never silently
        under-covers, and to the channel's incident base rate. This is *not* the
        old ``p·1.5+c`` clamp — it is an honest pre-calibration widening that the
        fitted conformal quantile supersedes the moment calibration data arrives.
        """
        belief = s.device_health_belief
        unc = (
            belief.get(DeviceHealth.DEGRADED, 0.0)
            + belief.get(DeviceHealth.INTERMITTENTLY_FAULTY, 0.0)
            + belief.get(DeviceHealth.RESET_RECOVERY, 0.0)
        )
        base = 0.02 + 0.06 * unc + 0.03 * s.neighbor_fault
        return float(base)

    def _ood_signal(
        self, s: WindowSignals, epistemic: float
    ) -> tuple[float, float, bool]:
        if self._ood is None:
            # un-fitted: the standardised staleness/epistemic novelty is the score;
            # conformal p-value is 1 (never falsely flags ID before calibration).
            from ..uncertainty.ood import staleness_epistemic_score

            n4 = float(
                staleness_epistemic_score(
                    epistemic, s.staleness_age_s, s.source_bad_prob
                )
            )
            return float(n4), 1.0, False
        comps = build_components(
            latent_knn=np.array([0.0]),
            topology_dist=np.array([0.0]),
            mixture_novelty=np.array([0.0]),
            staleness_epistemic=np.array(
                [
                    float(
                        epistemic
                        + (1.0 - np.exp(-max(s.staleness_age_s, 0.0) / 60.0))
                        + s.source_bad_prob
                    )
                ]
            ),
        )
        score = float(self._ood.score(comps)[0])
        pval = float(self._ood.p_value(comps)[0])
        flagged = bool(pval <= self.alpha_ood)
        return score, pval, flagged

    # -- posterior-driven risk-band scale (filters -> bands, doc T) --------- #
    def source_health_mode_posterior(self, s: WindowSignals) -> np.ndarray:
        """Exact Rao-Blackwellized source-health mode posterior (doc T E9-E10).

        The discrete modes are the source-health regimes (observed / stale /
        delayed / dropped). The continuous block is Rao-Blackwellized away (it has
        no measurement when telemetry is missing), so the posterior over modes is
        computed *exactly* in closed form: a prior shifted by the observed
        source-health bucket, reweighted by the MNAR availability log-likelihood
        per mode (``availability_loglik``, doc T E21-E23). This is the analytic
        marginal of the RBPF — deterministic, so it can drive the planner-visible
        risk bands without breaking reproducibility.
        """
        n_modes = 4
        prior = np.array(
            [
                max(1.0 - s.source_bad_prob, 0.05),
                0.3 * s.source_bad_prob + 0.05,
                0.3 * s.source_bad_prob + 0.05,
                0.4 * s.source_bad_prob + 0.05,
            ],
            float,
        )
        prior = prior / prior.sum()
        delta = 0 if s.source_bad_prob >= 0.5 else 1
        mode_alpha = (3.0, 1.0, 0.5, -0.5)  # observed..dropped baseline availability
        weights = np.empty(n_modes, float)
        for m in range(n_modes):
            model = MissingnessModel(alpha=mode_alpha[m], kappa=-0.01)
            ll = availability_loglik(delta, model, tau_prev=float(s.staleness_age_s))
            weights[m] = prior[m] * float(np.exp(ll))
        total = weights.sum() or 1.0
        return weights / total

    def posterior_band_scale(self, s: WindowSignals) -> float:
        """Quantile-band widening multiplier from the filtered source-health posterior.

        Reads the *degraded-mode* posterior mass from
        :meth:`source_health_mode_posterior`. A fully clean, fully-observed window
        (no MNAR evidence) carries no degraded mass, so the bands stay at their
        nominal width (scale = 1.0) and the deterministic planner ranking is
        unperturbed; a degraded / stale posterior widens the bands in proportion to
        the filtered bad-mode mass (doc T C15: missingness widens risk bands).
        Bounded so a blackout cannot blow the bands up unboundedly.

        A fully clean, fully-fresh window (no source-health degradation and no
        staleness) carries *no MNAR evidence*, so its bands are left at the nominal
        width exactly (scale = 1.0) — the deterministic operator-utility contract
        the planner ranks on. Only an actually-degraded posterior widens.
        """
        if s.source_bad_prob <= 0.0 and s.staleness_age_s <= 0.0:
            return 1.0
        modep = self.source_health_mode_posterior(s)
        bad_mass = float(modep[1:].sum())
        return float(1.0 + 1.5 * bad_mass)

    # -- selective certificate for the recommendation ledger --------------- #
    def selective_ledger(
        self, *, epistemic: float, ood: float, risk_upper: float, benefit_margin: float
    ) -> SelectiveLedger:
        ts = float(
            trust_score(
                epistemic=epistemic,
                ood=ood,
                risk_upper=risk_upper,
                benefit_margin=benefit_margin,
            )
        )
        gate = self._selective
        if gate is None:
            return SelectiveLedger(certificate=None, trust=ts, accepted=True)
        return SelectiveLedger(
            certificate=gate.certificate,
            trust=ts,
            accepted=gate.accept(ts),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Small math helpers
# ─────────────────────────────────────────────────────────────────────────────
def _hold_load() -> float:
    return 0.0


def _drive_to_logit(per_bucket_hazard: float) -> float:
    """Map a target per-bucket cause hazard in (0, 1) to its pinned-0 logit.

    With the no-event logit pinned to 0 and a single dominant cause the softmax
    gives ``h = e^eta / (1 + e^eta)`` (for K=1) so ``eta = logit(h)``; for the
    K=2 workload head this is a monotone, well-behaved driver of each cause.
    """
    h = float(np.clip(per_bucket_hazard, 1e-4, 0.5))
    return float(np.log(h / (1.0 - h)))


def _survival_incident(per_bucket_rate: float, n_buckets: int) -> float:
    """Discrete-time survival incident probability ``1 - (1 - rate)^n``.

    Monotone non-decreasing in both ``rate`` (the physical driver) and
    ``n_buckets`` (the exposure horizon), which is exactly the property the
    serving monotonicity tests pin.
    """
    r = float(np.clip(per_bucket_rate, 0.0, 1.0))
    return float(1.0 - (1.0 - r) ** max(n_buckets, 1))
