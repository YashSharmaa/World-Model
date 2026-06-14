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


__all__ = [
    "Commodity",
    "Link",
    "NetworkAllocation",
    "PacketTeacherCorrection",
    "allocate",
    "cross_rack_rate",
    "weighted_max_min_fair",
]
