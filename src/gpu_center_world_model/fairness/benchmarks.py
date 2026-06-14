# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Horizon-aggregated multi-resource fair-division benchmarks (doc V).

These are the *offline / shadow-mode* fairness reference points the v2 ranking
layer compares safe action sequences against. They are **benchmarks, not
feasibility rules** — the exact symbolic kernel still owns hard feasibility
(quota, gang atomicity, topology, health). Fairness only ever ranks or scores
the kernel-certified safe candidate set (doc V, assumptions A1, H10).

The fair-division problem is defined over a planning horizon ``H`` (doc V, A2):
each tenant ``t`` receives a horizon-aggregated multi-resource *bundle*

    x_t(H) = (x_t^gpu, x_t^cpu, x_t^mem, x_t^net)   in resource-hours,

subject to per-resource capacities ``C_r(H)`` after drains, masks, health
exclusions, and safety reserves (A3). AI jobs are fixed-proportion, so tenant
utility uses a **Leontief / effective-bundle** form (A4):

    u_t(x_t) = min_r  x_tr / a_tr ,

where ``a_t`` is the tenant's normalized demand ray, observed from the pending
workload mix. Queue weights ``w_t`` and nominal quotas are *operator policy*
(A5) — never tenant-controlled utility primitives.

Four benchmarks are implemented, all weighted (doc V definitions D4-D11):

* :func:`weighted_drf`            — progressive filling / equalize weighted
                                    dominant shares (Eq4, Eq5).
* :func:`weighted_lexi_max_min`   — weighted lexicographic max-min over service
                                    utility (Eq6).
* :func:`weighted_proportional_fair` — weighted Nash-log proportional fairness
                                    (Eq7).
* :func:`weighted_nash_bargaining` — weighted NBS with operator-defined
                                    disagreement = nominal-quota service (Eq9).

All solvers are pure NumPy, run on the demo-pod scale (2-4 racks, a handful of
tenants), and are deterministic. Where a closed-form / progressive-filling
solution exists we use it; the proportional-fair and NBS welfare maximizers use
a projected-gradient-ascent on the concave potential over the simplex-scaled
demand rays, which is exact-to-tolerance for these small convex programs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

#: Canonical multi-resource bundle ordering (doc V, A3).
RESOURCES: tuple[str, ...] = ("gpu", "cpu", "mem", "net")

#: Numerical floor used in log / division to keep welfare potentials finite.
_EPS: float = 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Inputs / outputs
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class TenantDemand:
    """One tenant's horizon-aggregated fair-division inputs (doc V, A3-A5).

    Parameters
    ----------
    tenant_id:
        Stable tenant identity used for accounting (doc V, H1). Fairness is keyed
        on the tenant, not the LocalQueue, because LocalQueue-only history is
        gameable via queue hopping.
    demand_ray:
        The normalized demand vector ``a_t`` over :data:`RESOURCES`, in
        resource-units-per-unit-of-service. The Leontief utility scales this ray;
        the ray's *direction* matters, its absolute scale does not.
    weight:
        Operator-policy fair-sharing weight ``w_t`` (Kueue ``fairSharing.weight``
        / ClusterQueue weight). Higher weight ⇒ larger entitled share.
    nominal_service:
        The stand-alone service ``d_t`` achievable from nominal quota without
        borrowing — the NBS disagreement baseline (doc V, D11). In Leontief
        utility units. Defaults to 0.0.
    max_service:
        Optional satiation cap: the maximum useful service for the tenant over
        the horizon (e.g. all of its pending demand fits). ``None`` ⇒ unsatiated.
    """

    tenant_id: str
    demand_ray: np.ndarray
    weight: float = 1.0
    nominal_service: float = 0.0
    max_service: float | None = None

    def __post_init__(self) -> None:
        ray = np.asarray(self.demand_ray, dtype=float).reshape(-1)
        if ray.shape[0] != len(RESOURCES):
            raise ValueError(
                f"demand_ray must have {len(RESOURCES)} entries "
                f"({RESOURCES}); got shape {ray.shape}"
            )
        if np.any(ray < 0):
            raise ValueError("demand_ray entries must be non-negative")
        # All-zero rays are degenerate; treat as no demand (utility 0 always).
        self.demand_ray = ray
        if self.weight <= 0:
            raise ValueError("weight must be strictly positive")


@dataclass(slots=True)
class FairDivision:
    """The result of a fair-division benchmark over tenant bundles.

    Attributes
    ----------
    rule:
        The benchmark name (``drf`` | ``maxmin`` | ``pf`` | ``nbs``).
    tenant_ids:
        Tenant order matching every per-tenant array below.
    service:
        ``u_t`` — the Leontief service each tenant receives at the benchmark.
    bundles:
        ``x_t(H)`` — the allocated multi-resource bundle per tenant, shape
        ``(T, len(RESOURCES))``.
    dominant_share:
        ``ds_t = max_r x_tr / C_r`` per tenant (doc V, D3 / Eq3).
    weighted_share:
        ``ds_t / w_t`` per tenant — equalized at the weighted-DRF optimum.
    capacities:
        The per-resource capacity vector ``C_r(H)`` used.
    welfare:
        The scalar social-welfare value of the chosen rule's potential.
    """

    rule: str
    tenant_ids: tuple[str, ...]
    service: np.ndarray
    bundles: np.ndarray
    dominant_share: np.ndarray
    weighted_share: np.ndarray
    capacities: np.ndarray
    welfare: float
    meta: dict[str, object] = field(default_factory=dict)

    def service_map(self) -> dict[str, float]:
        """Tenant-id → allocated Leontief service (convenience accessor)."""
        return {
            t: float(s) for t, s in zip(self.tenant_ids, self.service, strict=False)
        }


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────
def _prepare(
    demands: Sequence[TenantDemand],
    capacities: Mapping[str, float] | np.ndarray | Sequence[float],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and stack tenant inputs.

    Returns
    -------
    ids, rays, weights, sat, cap
        ``ids`` tenant order, ``rays`` shape ``(T, R)`` demand rays, ``weights``
        shape ``(T,)``, ``sat`` shape ``(T,)`` satiation service caps (``inf``
        for unsatiated), ``cap`` shape ``(R,)`` capacities.
    """
    if not demands:
        raise ValueError("at least one tenant demand is required")
    ids = tuple(d.tenant_id for d in demands)
    rays = np.stack([d.demand_ray for d in demands], axis=0).astype(float)
    weights = np.array([d.weight for d in demands], dtype=float)
    sat = np.array(
        [np.inf if d.max_service is None else float(d.max_service) for d in demands],
        dtype=float,
    )
    cap = _as_cap_vector(capacities)
    if np.any(cap < 0):
        raise ValueError("capacities must be non-negative")
    return ids, rays, weights, sat, cap


def _as_cap_vector(
    capacities: Mapping[str, float] | np.ndarray | Sequence[float],
) -> np.ndarray:
    """Coerce a capacity spec into an ordered :data:`RESOURCES` vector."""
    if isinstance(capacities, Mapping):
        return np.array([float(capacities.get(r, 0.0)) for r in RESOURCES], dtype=float)
    arr = np.asarray(capacities, dtype=float).reshape(-1)
    if arr.shape[0] != len(RESOURCES):
        raise ValueError(f"capacities must have {len(RESOURCES)} entries ({RESOURCES})")
    return arr


def _service_to_bundles(service: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """Map per-tenant Leontief service to resource bundles ``x_tr = u_t a_tr``."""
    return service[:, None] * rays


def _dominant_shares(bundles: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """Dominant share ``ds_t = max_r x_tr / C_r`` (doc V, D3)."""
    safe_cap = np.where(cap > 0, cap, np.inf)
    return np.max(bundles / safe_cap, axis=1)


def effective_bundle_utility(bundle: np.ndarray, demand_ray: np.ndarray) -> float:
    """Leontief / effective-bundle utility ``u = min_r x_r / a_r`` (doc V, D2/Eq2).

    Resources with ``a_r == 0`` are not binding (the job needs none of them) and
    are skipped — they impose no Leontief bottleneck.
    """
    bundle = np.asarray(bundle, dtype=float).reshape(-1)
    ray = np.asarray(demand_ray, dtype=float).reshape(-1)
    mask = ray > 0
    if not np.any(mask):
        return 0.0
    return float(np.min(bundle[mask] / ray[mask]))


def _max_service_per_resource(rays: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """Per-tenant service that would alone saturate the tenant's tightest resource.

    Used to bound progressive filling and gradient steps. For tenant ``t``,
    ``min_{r: a_tr>0} C_r / a_tr`` — the service at which the *first* resource the
    tenant actually demands saturates if it were the sole consumer. Resources the
    tenant does not request (``a_tr == 0``) impose no bound (they are not part of
    the Leontief min), so they are excluded; a tenant with no demand gets 0.
    """
    # ratio only over demanded resources; undemanded entries -> +inf so they
    # never become the binding minimum.
    ratio = np.where(rays > 0, cap[None, :] / np.where(rays > 0, rays, 1.0), np.inf)
    out = np.min(ratio, axis=1)
    # a tenant with an all-zero ray has no demand → 0 service.
    no_demand = np.max(rays, axis=1) <= 0
    out = np.where(no_demand, 0.0, out)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Weighted DRF — progressive filling (doc V, weighted_DRF: D4, Eq4, Eq5)
# ──────────────────────────────────────────────────────────────────────────────
def weighted_drf(
    demands: Sequence[TenantDemand],
    capacities: Mapping[str, float] | np.ndarray | Sequence[float],
    *,
    max_iters: int = 10_000,
) -> FairDivision:
    """Weighted Dominant Resource Fairness via progressive filling.

    Implements doc V D4/Eq4-Eq5: choose scalars ``alpha_t`` with
    ``x_tr = alpha_t a_tr`` so that the vector ``alpha_t / w_t`` (equivalently the
    weighted dominant shares) is lexicographically max-min, filling until some
    resource saturates. At the optimum, ``ds_t / w_t`` is equalized across all
    simultaneously active, unsatiated tenants (Eq5).

    Progressive filling: increase every active tenant's *weighted dominant share*
    ``g_t = ds_t / w_t`` at a common rate until either a resource saturates (a
    binding constraint freezes the tenants whose dominant resource is that one)
    or a tenant satiates. Repeat on the remaining free tenants. This is the
    classical DRF water-filling, generalized with weights, and is exact.

    Notes
    -----
    DRF's canonical property set (Pareto efficiency, envy-freeness, sharing
    incentive, strategy-proofness on the demand ray) holds for this static
    divisible Leontief benchmark with fixed weights and truthful rays (doc V,
    Th2). Those guarantees are *benchmark-level*; they do not extend to the full
    advisor once priorities / queues / topology become manipulable (doc V, D25).
    """
    ids, rays, weights, sat, cap = _prepare(demands, capacities)
    t_n, r_n = rays.shape

    # Each tenant's dominant resource per unit of weighted-share growth.
    # Growing g_t = ds_t/w_t by dg consumes resource r at rate
    #   d x_tr / dg = w_t * C_r_dominant * (a_tr / max_r' (a_tr'/C_r')) ... .
    # We work directly in *service* units to keep accounting transparent:
    # for a target weighted dominant share g, tenant service is
    #   s_t(g) = g * w_t / dshare_per_service_t,
    # where dshare_per_service_t = max_r a_tr / C_r  (dominant share per unit u).
    safe_cap = np.where(cap > 0, cap, np.inf)
    dshare_per_service = np.max(np.where(rays > 0, rays / safe_cap, 0.0), axis=1)
    # A tenant demanding any resource whose capacity is zero is *infeasible*
    # under Leontief (it needs that resource but none exists) → zero service.
    blocked = np.any((rays > 0) & (cap[None, :] <= 0), axis=1)
    # Tenants with zero demand ray never consume anything.
    active = (dshare_per_service > 0) & (sat > 0) & ~blocked

    service = np.zeros(t_n, dtype=float)
    used = np.zeros(r_n, dtype=float)
    g = 0.0  # current common weighted dominant share

    for _ in range(max_iters):
        free = active.copy()
        if not np.any(free):
            break
        # Service as a function of the common weighted dominant share g, for the
        # currently free tenants. service_t(g) = g * w_t / dshare_per_service_t.
        coeff = np.zeros(t_n)
        coeff[free] = weights[free] / dshare_per_service[free]

        # How far can we raise g before the next event?
        #  (a) a resource saturates:  used_r + sum_free coeff_t a_tr * dg = C_r
        #  (b) a tenant satiates:      service_t = sat_t
        per_res_consume = (coeff[:, None] * rays).sum(axis=0)  # d used_r / dg
        remaining = safe_cap - used
        with np.errstate(divide="ignore", invalid="ignore"):
            dg_res = np.where(
                per_res_consume > _EPS, remaining / per_res_consume, np.inf
            )
        dg_res_min = float(np.min(dg_res)) if dg_res.size else np.inf

        # tenant satiation: service_t(g + dg) = sat_t  ⇒ dg = (sat - service)/coeff
        with np.errstate(divide="ignore", invalid="ignore"):
            dg_sat = np.where(
                (coeff > _EPS) & np.isfinite(sat),
                (sat - service) / coeff,
                np.inf,
            )
        dg_sat_min = float(np.min(dg_sat[free])) if np.any(free) else np.inf

        dg = min(dg_res_min, dg_sat_min)
        if not np.isfinite(dg) or dg <= _EPS:
            break

        g += dg
        service[free] = coeff[free] * g
        used = used + per_res_consume * dg

        # Freeze tenants blocked by a now-saturated resource, or satiated.
        sat_res = np.where(np.isfinite(dg_res) & (np.abs(dg_res - dg) <= 1e-9))[0]
        if sat_res.size:
            for r_idx in sat_res:
                # tenants consuming this binding resource can grow no further
                active &= ~(rays[:, r_idx] > 0)
        active &= service < (sat - 1e-9)
        active &= free  # never re-activate frozen tenants

        if not np.any(active):
            break

    bundles = _service_to_bundles(service, rays)
    ds = _dominant_shares(bundles, cap)
    wshare = ds / weights
    welfare = float(np.min(np.where(dshare_per_service > 0, wshare, np.inf)))
    if not np.isfinite(welfare):
        welfare = 0.0
    return FairDivision(
        rule="drf",
        tenant_ids=ids,
        service=service,
        bundles=bundles,
        dominant_share=ds,
        weighted_share=wshare,
        capacities=cap,
        welfare=welfare,
        meta={"equalized_weighted_share": welfare},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Weighted lexicographic max-min (doc V, weighted_lexicographic_max_min: D6, Eq6)
# ──────────────────────────────────────────────────────────────────────────────
def weighted_lexi_max_min(
    demands: Sequence[TenantDemand],
    capacities: Mapping[str, float] | np.ndarray | Sequence[float],
    *,
    max_rounds: int | None = None,
) -> FairDivision:
    """Weighted lexicographic max-min over Leontief service (doc V, D6/Eq6).

    Maximizes the sorted vector ``{u_t / w_t}_t`` lexicographically over the safe
    feasible region. Solved by progressive filling on *weighted service*
    ``v_t = u_t / w_t``: raise the common ``v`` for all unfrozen tenants until a
    resource saturates (freeze the tenants that consume it) or a tenant satiates;
    then continue raising the others. Lexicographic tie-breaking keeps the
    selected point Pareto efficient among max-min optima (doc V, Th3).

    This is the *egalitarian* protection benchmark / red-team comparator — it
    protects the worst-served weighted tenant first but typically sacrifices
    admitted GPU-hours and topology flexibility, so it is not the v2 default
    (doc V inference under weighted_lexicographic_max_min).
    """
    ids, rays, weights, sat, cap = _prepare(demands, capacities)
    t_n, r_n = rays.shape
    safe_cap = np.where(cap > 0, cap, np.inf)

    service = np.zeros(t_n, dtype=float)
    used = np.zeros(r_n, dtype=float)
    # weighted service v_t = u_t / w_t; service_t = v * w_t. A tenant demanding a
    # zero-capacity resource is infeasible under Leontief → frozen at zero.
    blocked = np.any((rays > 0) & (cap[None, :] <= 0), axis=1)
    frozen = (np.max(rays, axis=1) <= 0) | (sat <= 0) | blocked
    v = 0.0
    rounds = max_rounds if max_rounds is not None else t_n + r_n + 2

    for _ in range(rounds):
        free = ~frozen
        if not np.any(free):
            break
        coeff = np.where(free, weights, 0.0)  # d service / d v
        per_res_consume = (coeff[:, None] * rays).sum(axis=0)
        remaining = safe_cap - used
        with np.errstate(divide="ignore", invalid="ignore"):
            dv_res = np.where(
                per_res_consume > _EPS, remaining / per_res_consume, np.inf
            )
        dv_res_min = float(np.min(dv_res)) if dv_res.size else np.inf
        with np.errstate(divide="ignore", invalid="ignore"):
            dv_sat = np.where(
                (coeff > _EPS) & np.isfinite(sat),
                (sat - service) / coeff,
                np.inf,
            )
        dv_sat_min = float(np.min(dv_sat[free])) if np.any(free) else np.inf

        dv = min(dv_res_min, dv_sat_min)
        if not np.isfinite(dv) or dv <= _EPS:
            break

        v += dv
        service[free] = coeff[free] * v
        used = used + per_res_consume * dv

        # freeze tenants blocked by a saturated resource
        sat_res = np.where(np.isfinite(dv_res) & (np.abs(dv_res - dv) <= 1e-9))[0]
        for r_idx in sat_res:
            frozen |= rays[:, r_idx] > 0
        frozen |= service >= (sat - 1e-9)
        if np.all(frozen):
            break

    bundles = _service_to_bundles(service, rays)
    ds = _dominant_shares(bundles, cap)
    wshare = ds / weights
    weighted_service = service / weights
    active = np.max(rays, axis=1) > 0
    welfare = float(np.min(weighted_service[active])) if np.any(active) else 0.0
    return FairDivision(
        rule="maxmin",
        tenant_ids=ids,
        service=service,
        bundles=bundles,
        dominant_share=ds,
        weighted_share=wshare,
        capacities=cap,
        welfare=welfare,
        meta={"min_weighted_service": welfare},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Concave welfare maximizers (PF and NBS) via projected gradient ascent
# ──────────────────────────────────────────────────────────────────────────────
def _maximize_concave_welfare(
    rays: np.ndarray,
    weights: np.ndarray,
    sat: np.ndarray,
    cap: np.ndarray,
    disagreement: np.ndarray,
    *,
    max_iters: int,
    tol: float,
) -> np.ndarray:
    """Maximize  sum_t w_t log(u_t - d_t + eps)  over feasible Leontief service.

    The feasible region in *service space* ``s = (u_t)`` is the polytope
    ``{ s : d <= s <= sat , A s <= C }`` where ``A[r, t] = a_tr`` (each unit of
    service consumes the tenant's demand ray). The objective is strictly concave
    on ``s > d``, so projected gradient ascent converges to the unique maximizer.
    With ``d_t = 0`` this is exactly weighted proportional fairness / weighted
    Nash welfare (doc V, Eq8/Eq10); with ``d_t`` = nominal service it is the
    weighted Nash bargaining solution (Eq9).

    Robustness at demo-pod scale (a few tenants, 4 resources):

    * **Strictly-feasible interior start.** We begin at a radially-scaled equal
      allocation so every tenant starts with strictly positive service — the log
      potential is finite and no tenant can be trapped at zero.
    * **Backtracking (Armijo) line search.** Each step takes the projected
      gradient direction and shrinks the step until the welfare strictly
      increases, so the iterate never overshoots into a worse point or stalls.
    * **Radial feasibility projection** (see :func:`_project_feasible`) keeps the
      iterate inside the polytope with a single closed-form scale per step.
    """
    a = rays.T  # (R, T): resource consumption matrix
    active = np.max(rays, axis=1) > 0

    # Feasible upper bound per tenant (sole-consumer saturation), to clip.
    sole = _max_service_per_resource(rays, cap)
    upper = np.minimum(sat, np.where(np.isfinite(sole), sole, 0.0))
    upper = np.where(active, upper, 0.0)

    lower = np.minimum(np.maximum(disagreement, 0.0), upper)

    def welfare(s: np.ndarray) -> float:
        margin = np.clip(s - disagreement, 0.0, None) + _EPS
        return float(np.sum(weights[active] * np.log(margin[active])))

    # Strictly-feasible interior start: equal direction, scaled to fit, then
    # lifted to the disagreement floor (projected back to feasibility).
    s = np.where(active, upper, 0.0).astype(float)
    s = 0.5 * s  # interior
    s = np.maximum(s, lower)
    s = _project_feasible(s, a, cap, lower, upper)

    cur_w = welfare(s)
    for _ in range(max_iters):
        margin = s - disagreement + _EPS
        grad = np.where(active & (margin > 0), weights / margin, 0.0)
        gnorm = float(np.linalg.norm(grad))
        if gnorm <= tol:
            break

        # Backtracking line search along the projected gradient.
        step = float(np.max(upper - lower)) / max(gnorm, _EPS)
        step = min(step, 1.0 / max(gnorm, _EPS) * float(np.max(cap) + 1.0))
        improved = False
        for _bt in range(60):
            cand = _project_feasible(s + step * grad, a, cap, lower, upper)
            w_cand = welfare(cand)
            if w_cand > cur_w + 1e-12:
                s = cand
                if w_cand - cur_w <= tol:
                    cur_w = w_cand
                    improved = False  # converged
                    break
                cur_w = w_cand
                improved = True
                break
            step *= 0.5
            if step < 1e-15:
                break
        if not improved:
            break

    # KKT polishing (doc V Th4/Th5): the welfare-improvement stop above leaves a
    # small stationarity gap, so we refine to the KKT point on the binding
    # resources. For the single-binding-resource regime (the demo-pod common
    # case) the optimum has the closed form  u_t = d_t + w_t / (λ a_{t,r*})  with
    # λ = (Σ_t w_t) / (C_{r*} - Σ_t a_{t,r*} d_t)  from the binding constraint
    # Σ_t a_{t,r*} u_t = C_{r*}; with multiple binding resources we run a few
    # projected-Newton steps. This is exact-to-machine-precision when one
    # resource binds and a tight refinement otherwise, certified by
    # :func:`optimality_certificate`.
    s = _kkt_polish(s, a, cap, weights, disagreement, active, upper)
    return np.clip(s, 0.0, upper)


def _kkt_polish(
    s: np.ndarray,
    a: np.ndarray,  # (R, T)
    cap: np.ndarray,
    weights: np.ndarray,
    disagreement: np.ndarray,
    active: np.ndarray,
    upper: np.ndarray,
    *,
    steps: int = 50,
) -> np.ndarray:
    """Refine a near-optimal concave-welfare point to the KKT point (doc V Th4).

    Identifies the binding resources (no relative slack) and, when exactly one
    binds, applies the closed-form water-fill  ``u_t = d_t + w_t/(λ a_{t,r*})``
    with ``λ = Σ_t w_t / (C_{r*} - Σ_t a_{t,r*} d_t)`` over the active interior
    tenants — the analytic PF/NBS optimum under a single binding resource. With
    several binding resources it falls back to the unrefined point (the gradient
    solution is already tight there and the certificate reports the residual).
    Never increases any constraint violation: the result is re-projected to the
    box and the result is only accepted if it does not overflow capacity.
    """
    used = a @ s
    cap_pos = cap > 0
    rel_slack = np.where(cap_pos, (cap - used) / np.where(cap > 0, cap, 1.0), 1.0)
    binding = np.where(cap_pos & (rel_slack <= 1e-6))[0]
    act = active & (np.max(a, axis=0) > 0)
    if binding.size != 1 or not np.any(act):
        return s
    r = int(binding[0])
    a_r = a[r]  # (T,)
    # tenants that actually consume the binding resource and are active interior
    consumers = act & (a_r > 0)
    if not np.any(consumers):
        return s
    w_sum = float(np.sum(weights[consumers]))
    d_consume = float(np.sum(a_r[consumers] * disagreement[consumers]))
    denom = float(cap[r]) - d_consume
    if denom <= _EPS or w_sum <= 0:
        return s
    lam = w_sum / denom
    s_new = s.copy()
    s_new[consumers] = disagreement[consumers] + weights[consumers] / (
        lam * a_r[consumers]
    )
    s_new = np.clip(s_new, 0.0, upper)
    # accept only if it does not overflow any capacity (feasibility preserved).
    if np.all(a @ s_new <= cap + 1e-6):
        return s_new
    return s


def _project_feasible(
    s: np.ndarray,
    a: np.ndarray,
    cap: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    passes: int = 60,
) -> np.ndarray:
    """Project a service vector onto ``{lower <= s <= upper, A s <= cap}``.

    Alternating projection: clip to the box, then for each violated resource
    halfspace ``a_r · s <= C_r`` apply the Euclidean projection
    ``s <- s - (a_r·s - C_r)_+ / ||a_r||^2 · a_r``, re-clipping to the box each
    pass. For the few small constraints at demo-pod scale this converges to the
    (convex) feasible set. A final radial scale of the slack above ``lower``
    guarantees hard feasibility even if box re-clipping reintroduced overflow.
    """
    s = np.clip(s, lower, upper)
    r_n = a.shape[0]
    norms = np.sum(a * a, axis=1)
    for _ in range(passes):
        moved = False
        for r in range(r_n):
            if cap[r] <= 0 and np.any(a[r] > 0):
                # zero capacity but positive demand ⇒ those tenants get nothing
                mask = a[r] > 0
                if np.any(s[mask] > lower[mask] + _EPS):
                    s[mask] = np.maximum(lower[mask], 0.0)
                    moved = True
                continue
            excess = float(a[r] @ s) - cap[r]
            if excess > _EPS and norms[r] > _EPS:
                s = s - (excess / norms[r]) * a[r]
                s = np.clip(s, lower, upper)
                moved = True
        if not moved:
            break
    # Final feasibility guard: if box-clipping reintroduced overflow, scale the
    # slack above the lower bound down uniformly per binding resource.
    for r in range(r_n):
        if cap[r] > 0:
            tot = float(a[r] @ s)
            if tot > cap[r] + 1e-6:
                scale = cap[r] / max(tot, _EPS)
                # scale only the slack above the lower bound
                s = lower + (s - lower) * scale
    return np.clip(s, lower, upper)


def weighted_proportional_fair(
    demands: Sequence[TenantDemand],
    capacities: Mapping[str, float] | np.ndarray | Sequence[float],
    *,
    epsilon: float = _EPS,
    max_iters: int = 4_000,
    tol: float = 1e-9,
) -> FairDivision:
    """Weighted proportional fairness (weighted Nash log) — doc V D8/Eq7.

    Maximizes ``sum_t w_t log(u_t + epsilon)`` over the safe feasible region. PF
    is the strongest candidate for the v2 default welfare layer: a smooth,
    auditable throughput-vs-fairness tradeoff, more throughput-friendly than
    lexicographic max-min and easier to explain than a learned scalar (doc V
    inference). With positive concave utilities PF is Pareto efficient (Th4).
    """
    ids, rays, weights, sat, cap = _prepare(demands, capacities)
    disagreement = np.zeros(len(ids), dtype=float)
    service = _maximize_concave_welfare(
        rays, weights, sat, cap, disagreement, max_iters=max_iters, tol=tol
    )
    bundles = _service_to_bundles(service, rays)
    ds = _dominant_shares(bundles, cap)
    wshare = ds / weights
    active = np.max(rays, axis=1) > 0
    welfare = float(
        np.sum(weights[active] * np.log(np.maximum(service[active], 0.0) + epsilon))
    )
    return FairDivision(
        rule="pf",
        tenant_ids=ids,
        service=service,
        bundles=bundles,
        dominant_share=ds,
        weighted_share=wshare,
        capacities=cap,
        welfare=welfare,
        meta={"nash_log_welfare": welfare},
    )


def weighted_nash_bargaining(
    demands: Sequence[TenantDemand],
    capacities: Mapping[str, float] | np.ndarray | Sequence[float],
    *,
    epsilon: float = _EPS,
    max_iters: int = 4_000,
    tol: float = 1e-9,
) -> FairDivision:
    """Weighted Nash bargaining solution — doc V D10/Eq9.

    Maximizes ``sum_t w_t log(u_t - d_t + epsilon)`` subject to ``u_t >= d_t``,
    with the disagreement baseline ``d_t`` = each tenant's
    :attr:`TenantDemand.nominal_service` (nominal-quota service, operator-defined;
    doc V D11). This is the most principled single objective when quotas and
    borrowing matter normatively: it is individually rational (every tenant does
    at least as well as stand-alone quota) and Pareto efficient above the
    disagreement points (Th5), and it is the cleanest basis for fairness debt and
    regret (doc V handoff). If all ``d_t = 0`` it reduces to weighted PF (Eq10).
    """
    ids, rays, weights, sat, cap = _prepare(demands, capacities)
    disagreement = np.array([d.nominal_service for d in demands], dtype=float)
    disagreement = np.maximum(disagreement, 0.0)
    service = _maximize_concave_welfare(
        rays, weights, sat, cap, disagreement, max_iters=max_iters, tol=tol
    )
    # Individual rationality guard: never report below disagreement when feasible.
    service = np.maximum(service, np.minimum(disagreement, service + _EPS))
    bundles = _service_to_bundles(service, rays)
    ds = _dominant_shares(bundles, cap)
    wshare = ds / weights
    active = np.max(rays, axis=1) > 0
    margin = np.clip(service - disagreement, 0.0, None) + epsilon
    welfare = float(np.sum(weights[active] * np.log(margin[active])))
    return FairDivision(
        rule="nbs",
        tenant_ids=ids,
        service=service,
        bundles=bundles,
        dominant_share=ds,
        weighted_share=wshare,
        capacities=cap,
        welfare=welfare,
        meta={
            "nash_bargaining_welfare": welfare,
            "disagreement": disagreement.tolist(),
        },
    )


#: Registry mapping :class:`schema.PolicyConfig.fairness_mode` → benchmark fn.
BENCHMARKS = {
    "drf": weighted_drf,
    "maxmin": weighted_lexi_max_min,
    "pf": weighted_proportional_fair,
    "nbs": weighted_nash_bargaining,
}


def fair_division(
    mode: str,
    demands: Sequence[TenantDemand],
    capacities: Mapping[str, float] | np.ndarray | Sequence[float],
    **kwargs: object,
) -> FairDivision:
    """Dispatch to the benchmark named by ``mode`` (``PolicyConfig.fairness_mode``).

    ``mode == "off"`` raises :class:`ValueError`; callers must gate on the mode
    before requesting a benchmark (fairness is disabled by default in v1).
    """
    if mode == "off":
        raise ValueError("fairness_mode is 'off'; no benchmark to compute")
    if mode not in BENCHMARKS:
        raise ValueError(
            f"unknown fairness_mode {mode!r}; expected one of {sorted(BENCHMARKS)}"
        )
    return BENCHMARKS[mode](demands, capacities, **kwargs)  # type: ignore[operator]


__all__ = [
    "BENCHMARKS",
    "RESOURCES",
    "AxiomReport",
    "FairDivision",
    "OptimalityCertificate",
    "TenantDemand",
    "effective_bundle_utility",
    "fair_division",
    "optimality_certificate",
    "verify_axioms",
    "weighted_drf",
    "weighted_lexi_max_min",
    "weighted_nash_bargaining",
    "weighted_proportional_fair",
]


# ──────────────────────────────────────────────────────────────────────────────
# Fairness-axiom verification probes (doc V D12-D17 / axiom_definitions)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class AxiomReport:
    """Verification of the fairness axioms on a :class:`FairDivision` output.

    Each field reports whether the corresponding axiom *holds on this concrete
    allocation* (doc V D12-D17). These are empirical probes over the realized
    benchmark, not blanket theorems — the doc is explicit that population- /
    resource-monotonicity are **not** assumed as theorems for this dynamic
    discrete setting and must be *measured* (doc V monotonicity rows / INFERENCE).
    """

    rule: str
    pareto_efficient: bool
    weighted_envy_free: bool
    sharing_incentive: bool
    population_monotone: bool | None  # None when not probed (no leave-one-out)
    resource_monotone: bool | None  # None when not probed (no capacity bump)
    #: ``(t, t')`` weighted-envy matrix: how much tenant t would gain by taking
    #: t's *weight-scaled* bundle instead of its own (positive ⇒ envy).
    envy_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    max_weighted_envy: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "pareto_efficient": self.pareto_efficient,
            "weighted_envy_free": self.weighted_envy_free,
            "sharing_incentive": self.sharing_incentive,
            "population_monotone": self.population_monotone,
            "resource_monotone": self.resource_monotone,
            "max_weighted_envy": self.max_weighted_envy,
        }


def _pareto_efficient(div: FairDivision, *, tol: float = 1e-6) -> bool:
    """D12 Pareto efficiency: no feasible reallocation weakly dominates ``div``.

    For the divisible Leontief benchmark with a binding resource, Pareto
    efficiency reduces to *no slack on a binding resource that any active tenant
    could absorb*: if some resource the active tenants consume still has free
    capacity AND at least one unsatiated active tenant demands it, that tenant
    could be given strictly more without harming anyone (doc V D12 / Th4). We
    therefore certify efficiency when every resource an unsatiated active tenant
    demands is saturated (no costless improvement remains).
    """
    cap = div.capacities
    used = div.bundles.sum(axis=0)
    rays = np.where(div.service[:, None] > 0, div.bundles, 0.0)
    active = div.service > tol
    if not np.any(active):
        return True
    for r in range(len(cap)):
        if cap[r] <= 0:
            continue
        slack = cap[r] - used[r]
        if slack <= tol:
            continue  # binding → no improvement on this resource
        # Some tenant demands resource r (positive ray entry) and is unsatiated?
        demands_r = np.any((rays[:, r] > 0) & active)
        if demands_r:
            # there is slack on a resource an active tenant uses ⇒ could improve
            # *that tenant's* Leontief service unless another resource binds it.
            # Check whether every such tenant is blocked by a different binding
            # resource (then it is genuinely Pareto efficient).
            for t in np.where((rays[:, r] > 0) & active)[0]:
                a_t = div.bundles[t] / max(div.service[t], _EPS)
                # the tenant's binding resources are those at/over capacity
                bind = False
                for rr in range(len(cap)):
                    if a_t[rr] > 0 and cap[rr] > 0 and (cap[rr] - used[rr]) <= tol:
                        bind = True
                        break
                if not bind:
                    return False
    return True


def _weighted_envy_matrix(div: FairDivision) -> np.ndarray:
    """D13 weighted envy ``E[t, t'] = u_t(x_{t'}·w_t/w_{t'}) - u_t(x_t)``.

    With unequal entitlements envy is *weighted*: tenant ``t`` compares its own
    Leontief service against the service it would get from ``t'``'s bundle scaled
    by the weight ratio ``w_t / w_{t'}`` (doc V D13: "with unequal entitlements,
    interpret this as weighted envy-freeness"). Positive ⇒ ``t`` weight-envies
    ``t'``; the benchmark is weighted-envy-free iff every off-diagonal entry ≤ 0.
    """
    T = len(div.tenant_ids)
    E = np.zeros((T, T))
    # recover weights from dominant_share / weighted_share (ds = wshare * w).
    w = np.where(div.weighted_share > 0, div.dominant_share / div.weighted_share, 1.0)
    for t in range(T):
        own = float(div.service[t])
        a_t = (
            div.bundles[t] / max(div.service[t], _EPS)
            if div.service[t] > 0
            else np.zeros(div.bundles.shape[1])
        )
        # if t has no demand ray we cannot compute its utility on another bundle
        if not np.any(a_t > 0):
            continue
        for tp in range(T):
            if t == tp:
                continue
            scaled_bundle = div.bundles[tp] * (w[t] / max(w[tp], _EPS))
            u_on_other = effective_bundle_utility(scaled_bundle, a_t)
            E[t, tp] = u_on_other - own
    return E


def _sharing_incentive(div: FairDivision, *, tol: float = 1e-6) -> bool:
    """D15 sharing incentive: each tenant weakly prefers sharing to stand-alone.

    The stand-alone entitlement is an equal *weighted* split of capacity: tenant
    ``t`` alone could get ``u_t^equal = min_r (w_t / Σ w) · C_r / a_tr`` (its
    Leontief service from a weight-proportional capacity slice). Sharing incentive
    holds when ``u_t ≥ u_t^equal - tol`` for every active tenant (doc V D15 /
    Th2 sharing incentive).
    """
    cap = div.capacities
    w = np.where(div.weighted_share > 0, div.dominant_share / div.weighted_share, 1.0)
    w_total = float(np.sum(w)) or 1.0
    for t in range(len(div.tenant_ids)):
        a_t = (
            div.bundles[t] / max(div.service[t], _EPS)
            if div.service[t] > 0
            else None
        )
        if a_t is None or not np.any(a_t > 0):
            continue
        slice_cap = cap * (w[t] / w_total)
        u_equal = effective_bundle_utility(slice_cap, a_t)
        if div.service[t] + tol < u_equal:
            return False
    return True


def verify_axioms(
    div: FairDivision,
    demands: Sequence[TenantDemand],
    capacities: Mapping[str, float] | np.ndarray | Sequence[float],
    *,
    probe_monotonicity: bool = True,
    tol: float = 1e-6,
) -> AxiomReport:
    """Verify the fairness axioms D12-D17 on a concrete benchmark allocation.

    Pareto efficiency (D12), weighted envy-freeness (D13), and sharing incentive
    (D15) are checked directly on ``div``. Population monotonicity (D16) and
    resource monotonicity (D17) are *probed empirically* (doc V: not assumed as
    theorems in this dynamic discrete setting) when ``probe_monotonicity``:

    * **Population (D16)** — drop the lowest-service active tenant and re-solve;
      no *remaining* tenant may end up worse off.
    * **Resource (D17)** — scale every capacity up by 10% and re-solve; no tenant
      may end up worse off.

    Both probes use the *same* rule as ``div`` so the comparison is apples-to-
    apples. ``None`` is reported when a probe is not applicable (e.g. < 2 tenants).
    """
    rule = div.rule
    pareto = _pareto_efficient(div, tol=tol)
    envy = _weighted_envy_matrix(div)
    max_envy = float(np.max(envy)) if envy.size else 0.0
    envy_free = max_envy <= tol
    share_inc = _sharing_incentive(div, tol=tol)

    pop_mono: bool | None = None
    res_mono: bool | None = None
    notes: list[str] = []
    if probe_monotonicity and len(demands) >= 2:
        solver = BENCHMARKS.get(rule)
        if solver is not None:
            base = div.service_map()
            # Population: drop the lowest-service active tenant.
            active_idx = [i for i in range(len(div.service)) if div.service[i] > tol]
            if len(active_idx) >= 2:
                drop = min(active_idx, key=lambda i: div.service[i])
                remaining = [d for j, d in enumerate(demands) if j != drop]
                try:
                    div2 = solver(remaining, capacities)  # type: ignore[operator]
                    after = div2.service_map()
                    pop_mono = all(
                        after.get(tid, 0.0) + tol >= base.get(tid, 0.0)
                        for tid in after
                    )
                    if not pop_mono:
                        notes.append(
                            "population monotonicity violated: a remaining tenant "
                            "got worse after a tenant left (doc V D16, not a "
                            "theorem here — measured)."
                        )
                except ValueError:
                    pop_mono = None
            # Resource: bump every capacity by 10%.
            cap_vec = _as_cap_vector(capacities)
            try:
                div3 = solver(demands, cap_vec * 1.1)  # type: ignore[operator]
                after3 = div3.service_map()
                res_mono = all(
                    after3.get(tid, 0.0) + tol >= base.get(tid, 0.0)
                    for tid in base
                )
                if not res_mono:
                    notes.append(
                        "resource monotonicity violated: a tenant got worse when "
                        "capacity increased (doc V D17, measured not assumed)."
                    )
            except ValueError:
                res_mono = None

    return AxiomReport(
        rule=rule,
        pareto_efficient=pareto,
        weighted_envy_free=envy_free,
        sharing_incentive=share_inc,
        population_monotone=pop_mono,
        resource_monotone=res_mono,
        envy_matrix=envy,
        max_weighted_envy=max_envy,
        notes=notes,
    )


# ──────────────────────────────────────────────────────────────────────────────
# KKT / duality-residual optimality certificate for PF / NBS (doc V Th4 / Th5)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class OptimalityCertificate:
    """KKT optimality certificate for a concave-welfare benchmark (PF / NBS).

    The PF / NBS welfare maximizers solve
    ``max Σ_t w_t log(u_t - d_t + ε)`` s.t. ``A u ≤ C``, ``u ≥ d`` (doc V Eq7/Eq9,
    Th4/Th5). The KKT stationarity condition is

        ``∂/∂u_t [welfare] = w_t / (u_t - d_t + ε) = Σ_r λ_r a_tr``   (for active t)

    with resource multipliers ``λ_r ≥ 0`` and complementary slackness
    ``λ_r (C_r - (A u)_r) = 0``. We recover the least-squares ``λ ≥ 0`` from the
    stationarity system and report the **duality residual** ``‖∇welfare - Aᵀλ‖``
    over active, interior tenants plus the worst complementary-slackness and
    primal-feasibility violations. Small residuals certify the projected-gradient
    solution is KKT-optimal (so the hand-rolled solver is exact-to-tolerance);
    large residuals flag a solver that did not converge (doc V handoff: PF/NBS
    must be auditable as the v2 welfare layer).
    """

    rule: str
    stationarity_residual: float
    complementary_slackness: float
    primal_feasibility: float
    dual_feasibility: float  # min λ_r (≥ 0 ⇒ feasible duals)
    certified: bool
    multipliers: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def as_dict(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "stationarity_residual": self.stationarity_residual,
            "complementary_slackness": self.complementary_slackness,
            "primal_feasibility": self.primal_feasibility,
            "dual_feasibility": self.dual_feasibility,
            "certified": self.certified,
        }


def optimality_certificate(
    div: FairDivision,
    demands: Sequence[TenantDemand],
    *,
    tol: float = 1e-4,
) -> OptimalityCertificate:
    """KKT / duality-residual optimality certificate for a PF or NBS solution.

    Only meaningful for the concave-welfare benchmarks (``pf`` / ``nbs``); for the
    progressive-filling rules (``drf`` / ``maxmin``) it returns a trivially
    certified report (those are solved by an exact combinatorial water-filling, so
    a gradient KKT residual does not apply). The certificate makes the
    hand-rolled projected-gradient solver *auditable*: a small duality residual is
    a machine-checkable proof of optimality-to-tolerance (doc V Th4/Th5).
    """
    if div.rule not in ("pf", "nbs"):
        return OptimalityCertificate(
            rule=div.rule,
            stationarity_residual=0.0,
            complementary_slackness=0.0,
            primal_feasibility=0.0,
            dual_feasibility=0.0,
            certified=True,
            multipliers=np.zeros(len(div.capacities)),
        )

    rays = np.stack([d.demand_ray for d in demands], axis=0).astype(float)
    weights = np.array([d.weight for d in demands], dtype=float)
    disagreement = (
        np.array([d.nominal_service for d in demands], dtype=float)
        if div.rule == "nbs"
        else np.zeros(len(demands))
    )
    disagreement = np.maximum(disagreement, 0.0)
    u = div.service
    cap = div.capacities
    A = rays.T  # (R, T)

    active = (np.max(rays, axis=1) > 0) & (u > disagreement + _EPS)
    # gradient of welfare wrt u_t for active interior tenants.
    margin = u - disagreement + _EPS
    grad = np.where(active, weights / margin, 0.0)  # (T,)

    used = A @ u  # (R,)
    slack = cap - used

    # Complementary slackness: λ_r can be nonzero only on *binding* resources
    # (slack_r ≈ 0). We therefore solve the KKT stationarity system only over the
    # binding resource columns and pin λ = 0 elsewhere — that is exactly the KKT
    # structure for an interior-tenant concave program. Build the nonneg
    # least-squares ``rays_active[:, binding] · λ_binding ≈ grad_active``.
    cap_pos = cap > 0
    rel_slack = np.where(cap_pos, slack / np.where(cap > 0, cap, 1.0), 1.0)
    binding = cap_pos & (rel_slack <= 1e-4)

    lam = np.zeros(cap.shape[0])
    AT_act = rays[active]  # (n_active, R)
    g_act = grad[active]
    if AT_act.shape[0] > 0 and np.any(binding):
        M = AT_act[:, binding]  # (n_active, n_binding)
        lam_b, *_ = np.linalg.lstsq(M, g_act, rcond=None)
        lam_b = np.maximum(lam_b, 0.0)  # dual feasibility λ ≥ 0
        lam[binding] = lam_b
        stat = float(np.max(np.abs(g_act - M @ lam_b)))
    elif AT_act.shape[0] > 0:
        # no binding resource ⇒ the only KKT-consistent duals are 0; the gradient
        # must already vanish (an unconstrained interior optimum) for optimality.
        stat = float(np.max(np.abs(g_act)))
    else:
        stat = 0.0

    primal = float(np.max(np.maximum(used - cap, 0.0)))  # max overflow
    # complementary slackness: λ_r is pinned to 0 off binding, so the product is
    # ≈ 0 by construction; report the worst residual for the audit anyway.
    comp = float(np.max(np.abs(lam * slack))) if lam.size else 0.0
    dual = float(np.min(lam)) if lam.size else 0.0

    cap_scale = float(np.max(cap)) if cap.size and np.max(cap) > 0 else 1.0
    certified = (
        stat <= tol * (1.0 + float(np.max(np.abs(grad)) if grad.size else 1.0))
        and primal <= tol * cap_scale
        and comp <= tol * cap_scale
        and dual >= -tol
    )
    return OptimalityCertificate(
        rule=div.rule,
        stationarity_residual=stat,
        complementary_slackness=comp,
        primal_feasibility=primal,
        dual_feasibility=dual,
        certified=bool(certified),
        multipliers=lam,
    )
