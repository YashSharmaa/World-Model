# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Two-tier network contention surrogate (design doc N network_contention_model).

Online planning needs *rates, queue proxies, and hotspot risk* on the exact path
graph — not packet reconstruction (doc N, ASSUMPTION A1). v1 therefore ships two
tiers:

* **Routine tier** — weighted max-min fair (progressive-filling) allocation on
  the exact routed topology graph (doc N, EQUATION N1 / RT1). Capacities and path
  sets stay exact symbolic objects from the canonical schema (A2); only the
  *demand* is phase-conditioned (A3). Weighted max-min fairness is preferred over
  a learned allocator because it matches the planner's safety-first semantics
  (INFERENCE I2).

* **Stress tier** — a hook for an *offline packet-teacher correction*: a learned
  residual ``Δ_k = f_corr(ψ_k)`` distilled from a Days-style packet simulator
  over the fluid base (doc N, EQUATION N4 / ST1/ST2). v1 ships an identity teacher
  plus an interface so a calibrated correction table can be loaded for stress
  scenario packs without changing the online code path (REJECTED R1).

Contention couples to placement: keeping a gang inside one rack/domain changes
the traffic graph and the bottleneck, which is exactly the locality signal the
planner must see (doc D correlation_rules #3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Link:
    """A capacitated, latency-bearing link on the routed topology graph (D1)."""

    link_id: str
    capacity: float  # c_e  (Gb/s)
    latency_s: float = 0.0  # l_e


@dataclass(slots=True)
class Commodity:
    """A communication episode k with a source/sink and allowed path set (D2)."""

    commodity_id: str
    demand: float  # b_k  (volume to move)
    weight: float = 1.0  # ω_k  (fair-share weight)
    # each path is the list of link_ids it traverses
    paths: list[list[str]] = field(default_factory=list)


@dataclass(slots=True)
class NetworkAllocation:
    """The routine-tier allocation result for one decision/step."""

    rate: dict[str, float]  # commodity_id -> r_k
    link_util: dict[str, float]  # link_id -> u_e ∈ [0, 1]
    transfer_time_s: dict[str, float]  # commodity_id -> τ_k
    bottleneck_links: list[str]  # saturated links (hotspot evidence)


def weighted_max_min_fair(
    links: list[Link],
    commodities: list[Commodity],
    *,
    eps: float = 1e-9,
    max_rounds: int | None = None,
) -> dict[str, float]:
    """Weighted max-min fair rates via progressive filling (doc N, EQUATION N1).

    Exact for the small demo topologies (doc N, RT1). Each round raises the fair
    *level* ``λ`` until some link saturates, freezes the commodities crossing it
    at ``r_k = ω_k λ``, removes their load from residual capacities, and repeats
    over the remaining commodities until all are frozen or starved.

    A commodity with multiple allowed paths splits its rate equally across its
    not-yet-saturated paths (a deterministic ECMP-style split); single-path
    commodities reduce to the textbook bottleneck algorithm.
    """
    cap = {link.link_id: float(link.capacity) for link in links}
    residual = dict(cap)
    rate: dict[str, float] = {c.commodity_id: 0.0 for c in commodities}
    active = {c.commodity_id: c for c in commodities}
    rounds = max_rounds if max_rounds is not None else len(commodities) + 2

    for _ in range(rounds):
        if not active:
            break
        # Live paths = paths whose every link still has residual capacity.
        live_paths: dict[str, list[list[str]]] = {}
        for cid, c in active.items():
            lp = [p for p in c.paths if all(residual.get(e, 0.0) > eps for e in p)]
            if lp:
                live_paths[cid] = lp
        starved = [cid for cid in active if cid not in live_paths]
        for cid in starved:
            active.pop(cid, None)
        if not active:
            break

        # Per-link weighted demand at unit fair level λ (rate = ω·λ split evenly).
        link_weight: dict[str, float] = {e: 0.0 for e in residual}
        for cid, paths in live_paths.items():
            w = active[cid].weight
            share = w / len(paths)
            for p in paths:
                for e in p:
                    link_weight[e] += share
        # Smallest λ that saturates any link given each commodity's *current* rate.
        dlambda = float("inf")
        for e, lw in link_weight.items():
            if lw <= eps:
                continue
            dlambda = min(dlambda, residual[e] / lw)
        if not np.isfinite(dlambda) or dlambda <= eps:
            break

        # Apply the increment, accumulate rates, decrement residual capacity.
        for cid, paths in live_paths.items():
            w = active[cid].weight
            add = w * dlambda
            rate[cid] += add
            share = add / len(paths)
            for p in paths:
                for e in p:
                    residual[e] -= share
        # Freeze commodities whose every live path now touches a saturated link.
        for cid in list(active):
            paths = live_paths.get(cid, [])
            if not paths or all(
                any(residual.get(e, 0.0) <= eps for e in p) for p in paths
            ):
                active.pop(cid, None)
    return rate


def allocate(links: list[Link], commodities: list[Commodity]) -> NetworkAllocation:
    """Full routine-tier allocation: rates, link utilisation, transfer times."""
    rate = weighted_max_min_fair(links, commodities)
    cap = {link.link_id: max(link.capacity, 1e-9) for link in links}
    lat = {link.link_id: link.latency_s for link in links}
    # Link utilisation u_e = Σ rate crossing e / c_e (EQUATION N2). With multi-
    # path commodities, the rate is split evenly across the commodity's paths.
    load = {link.link_id: 0.0 for link in links}
    for c in commodities:
        live = [p for p in c.paths if p]
        if not live:
            continue
        share = rate[c.commodity_id] / len(live)
        for p in live:
            for e in p:
                if e in load:
                    load[e] += share
    util = {e: min(load[e] / cap[e], 1.0) for e in cap}
    bottleneck = sorted(e for e, u in util.items() if u >= 0.999)

    transfer: dict[str, float] = {}
    for c in commodities:
        r = max(rate[c.commodity_id], 1e-9)
        # τ_k = b_k / r_k + Σ l_e + d_q(u_e)  (EQUATION N3); d_q is an M/M/1-ish
        # convex queue-delay proxy that explodes as util → 1.
        path = c.paths[0] if c.paths else []
        prop = sum(lat.get(e, 0.0) for e in path)
        q = sum(_queue_delay(util.get(e, 0.0)) for e in path)
        transfer[c.commodity_id] = c.demand / r + prop + q
    return NetworkAllocation(
        rate=rate,
        link_util=util,
        transfer_time_s=transfer,
        bottleneck_links=bottleneck,
    )


def _queue_delay(util: float, scale: float = 5e-4) -> float:
    """Lightweight convex queue-delay proxy d_q(u) = scale·u/(1−u) (EQUATION N3)."""
    u = min(max(util, 0.0), 0.999)
    return scale * u / (1.0 - u)


# ──────────────────────────────────────────────────────────────────────────────
# Stress-tier offline packet-teacher correction hook (doc N, ST1/ST2)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class PacketTeacherCorrection:
    """Distilled residual correction over the fluid base (doc N, EQUATION N4).

    A real teacher is a Days-style packet simulator run *offline*; the distilled
    artefact is a table mapping commodity feature buckets ψ_k (ECMP imbalance,
    phase synchrony, background load) to multiplicative rate / additive delay
    corrections. v1 ships an identity correction so the online path is unchanged
    until a stress pack supplies a calibrated table.
    """

    rate_factor: dict[str, float] = field(default_factory=dict)
    delay_add_s: dict[str, float] = field(default_factory=dict)

    def apply(self, base: NetworkAllocation) -> NetworkAllocation:
        """Apply the distilled correction to a routine-tier allocation."""
        rate = {cid: r * self.rate_factor.get(cid, 1.0) for cid, r in base.rate.items()}
        transfer = {
            cid: t + self.delay_add_s.get(cid, 0.0)
            for cid, t in base.transfer_time_s.items()
        }
        return NetworkAllocation(
            rate=rate,
            link_util=dict(base.link_util),
            transfer_time_s=transfer,
            bottleneck_links=list(base.bottleneck_links),
        )


def cross_rack_rate(
    commodities: list[Commodity], rack_of_link: dict[str, str]
) -> float:
    """Fraction of commodities whose chosen path crosses a rack boundary.

    Mirrors the :class:`~...schema.kpis.KPIChannel.CROSS_RACK_RATE` target: a
    topology-aware placement that keeps a gang in one rack drives this toward 0.
    """
    if not commodities:
        return 0.0
    crossing = 0
    for c in commodities:
        path = c.paths[0] if c.paths else []
        racks = {rack_of_link.get(e) for e in path if e in rack_of_link}
        if len({r for r in racks if r is not None}) > 1:
            crossing += 1
    return crossing / len(commodities)


# ──────────────────────────────────────────────────────────────────────────────
# Placement-conditioned commodity construction (doc N D1/D2/D3, doc D rule #3)
# ──────────────────────────────────────────────────────────────────────────────
#: Per-comm-class fair-share weight ω_k and per-replica demand volume b_k. The
#: weight orders the max-min fill (allreduce gangs contend hardest); the demand is
#: the volume each replica injects toward the gang's collective root (doc N D2).
#: **Uncalibrated priors** (doc D calibrated_bounds_only): qualitative class shape.
_COMM_CLASS_PROFILE: dict[str, tuple[float, float]] = {
    # comm_class: (fair_share_weight ω, per-replica demand volume b in Gb)
    "none": (0.2, 0.0),
    "light": (0.5, 4.0),
    "allreduce_heavy": (2.0, 64.0),
    "pipeline": (1.0, 24.0),
}

#: Intra-rack (ToR) and inter-rack (spine) link capacities in Gb/s (doc N D1).
#: NVLink/ToR fabric is fat; the spine crossing a rack boundary is the scarce
#: shared resource that creates the contention the planner must see (doc D #3).
_TOR_CAP_GBPS: float = 400.0
_SPINE_CAP_GBPS: float = 100.0


def pod_links(rack_ids: list[str]) -> tuple[list[Link], dict[str, str]]:
    """Build the two-tier routed link set for a pod (doc N D1 routed topology).

    One intra-rack ToR link per rack (``tor:<rack>``) plus one shared spine link
    (``spine``) every cross-rack path traverses. Returns the links and the
    ``rack_of_link`` map (the spine has no single rack scope) used to score
    cross-rack crossings (doc N D3 categorical attributes).
    """
    links = [Link(link_id=f"tor:{rid}", capacity=_TOR_CAP_GBPS) for rid in rack_ids]
    links.append(Link(link_id="spine", capacity=_SPINE_CAP_GBPS, latency_s=2e-5))
    rack_of_link = {f"tor:{rid}": rid for rid in rack_ids}
    return links, rack_of_link


def commodities_for_gang(
    gpu_racks: list[str],
    comm_class: str,
    *,
    workload_uid: str = "wl",
) -> list[Commodity]:
    """Build per-replica commodities for a placed gang (doc N D2/D3; doc D #3).

    Each placed GPU is one replica injecting ``b_k`` of collective traffic toward
    the gang's collective root (replica 0's rack). A replica co-located with the
    root crosses only its ToR link; a replica on a different rack crosses *both*
    its ToR link and the shared ``spine`` — so spreading a gang across racks loads
    the scarce spine and changes the bottleneck, exactly the placement-conditioned
    contention signal of doc D correlation_rule #3 (contention couples to
    placement). ``comm_class`` sets the fair-share weight ω_k and demand volume
    b_k via :data:`_COMM_CLASS_PROFILE`.
    """
    weight, demand = _COMM_CLASS_PROFILE.get(comm_class, _COMM_CLASS_PROFILE["light"])
    if not gpu_racks or demand <= 0.0:
        return []
    root_rack = gpu_racks[0]
    out: list[Commodity] = []
    for i, rack in enumerate(gpu_racks):
        if i == 0:
            continue  # the root does not send to itself
        if rack == root_rack:
            path = [f"tor:{rack}"]
        else:
            path = [f"tor:{rack}", "spine", f"tor:{root_rack}"]
        out.append(
            Commodity(
                commodity_id=f"{workload_uid}:r{i}",
                demand=demand,
                weight=weight,
                paths=[path],
            )
        )
    return out


def hotspot_risk(alloc: NetworkAllocation) -> float:
    """Network-hotspot hazard ∈ [0, 1] from the routine-tier allocation (doc N L3).

    The hotspot risk rises with the *peak* link utilisation and the presence of a
    saturated bottleneck link — the emergent switch-hotspot signal the doc-K Eq8
    stress proxy / doc-N L3 residual target tracks. It is a smooth, monotone map of
    the worst link utilisation (a convex penalty that climbs sharply as any link
    approaches saturation) plus a floor whenever a bottleneck link is fully
    saturated, so a gang spread across racks that saturates the shared spine
    reports materially higher hotspot risk than a rack-local placement. Replaces
    the old 0/1 cross-rack indicator with a contention-grounded continuous hazard.
    """
    if not alloc.link_util:
        return 0.0
    # A hotspot on the *shared inter-rack spine* is materially worse than one on a
    # local ToR: the spine is the scarce resource every cross-rack gang contends
    # for, so its utilisation is weighted higher (a cross-rack placement that jams
    # the spine is the canonical switch-hotspot, doc N L3 / doc K Eq8). Intra-rack
    # ToR saturation is a milder, contained hotspot.
    def _weight(link_id: str) -> float:
        return 1.0 if _is_spine(link_id) else 0.55

    weighted = max(
        _weight(lid) * u for lid, u in alloc.link_util.items()
    )
    # Convex climb in the (weighted) peak utilisation: 0 at u=0, → 1 as u → 1.
    risk = weighted / (2.0 - min(weighted, 0.999))
    return float(min(max(risk, 0.0), 1.0))


def _is_spine(link_id: str) -> bool:
    """True iff the link is an inter-rack spine crossing (vs an intra-rack ToR)."""
    return link_id == "spine" or link_id.startswith("spine")


__all__ = [
    "Commodity",
    "Link",
    "NetworkAllocation",
    "PacketTeacherCorrection",
    "allocate",
    "commodities_for_gang",
    "cross_rack_rate",
    "hotspot_risk",
    "pod_links",
    "weighted_max_min_fair",
]
