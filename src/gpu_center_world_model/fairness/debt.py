# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Long-run fairness DEBT to a fair benchmark, with exponential decay (doc V).

Dynamic fairness in this project is treated as *debt to a chosen fair benchmark*,
not as per-decision equalization (doc V, dynamic_fairness section). Per-decision
fairness is the wrong abstraction for holds, gangs, and topology-aware placement,
which create path dependence; the right object lives on the slow timescale and
mirrors Kueue's decayed historical usage.

For tenant ``t`` over horizon ``H`` (doc V, D29-D32):

* long-run service          ``U_t(H) = Σ_τ u_t(x_t(τ))``,
* fair benchmark path       ``B_t(H)`` (default: weighted NBS with disagreement
                            = nominal-quota service; fallback: weighted DRF over
                            borrowed capacity),
* fairness debt             ``Debt_t(H) = B_t(H) - U_t(H)``  (positive ⇒ the
                            tenant is *under-served* vs the fair benchmark),
* recursive decayed update  ``Debt_t(τ+1) = rho·Debt_t(τ) + b_t(τ) - u_t(τ)``
                            with ``rho = exp(-ln2 · Δτ / T_half)`` (Eq17),
* fair regret to benchmark  ``Reg_fair(H) = M(B(H)) - M(U(H))`` for a chosen SWF
                            ``M`` (D32).

The half-life ``T_half`` aligns with Kueue's ``usageHalfLifeTime`` when present
(doc V, A6 / H4), so the debt state decays the same way the platform decays its
own fair-sharing history.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .benchmarks import FairDivision

#: Kueue's default ``usageHalfLifeTime`` is operator-set; this is a sane fallback
#: (7 days, in seconds) used only when no value is supplied (doc V, H4).
DEFAULT_HALF_LIFE_S: float = 7.0 * 24.0 * 3600.0


def decay_factor(dt_s: float, half_life_s: float) -> float:
    """Exponential decay ``rho = exp(-ln2 · Δτ / T_half)`` (doc V, Eq17).

    ``half_life_s <= 0`` disables decay (``rho = 1``: debt never forgives), which
    is the conservative choice when no Kueue ``usageHalfLifeTime`` is configured.
    """
    if half_life_s <= 0:
        return 1.0
    if dt_s <= 0:
        return 1.0
    return math.exp(-math.log(2.0) * dt_s / half_life_s)


@dataclass(slots=True)
class TenantDebt:
    """Slow-timescale fairness-debt state for one tenant (doc V, H4).

    Mirrors the v1→v2 hook of dedicated state slots: ``fairness_debt``,
    ``disagreement_baseline``, ``decayed_service_history``, and
    ``last_fairness_update_ts``.
    """

    tenant_id: str
    debt: float = 0.0
    decayed_service: float = 0.0  # decayed cumulative realized service U_t
    decayed_benchmark: float = 0.0  # decayed cumulative benchmark service B_t
    disagreement_baseline: float = 0.0
    last_update_ts_s: float = 0.0
    half_life_s: float = DEFAULT_HALF_LIFE_S

    def positive_debt(self) -> float:
        """``Debt_t^+`` — the under-served part only (the planner's debt budget)."""
        return max(self.debt, 0.0)


@dataclass(slots=True)
class FairnessDebtTracker:
    """Accumulates per-tenant fairness debt across decision windows (doc V D31).

    The tracker is the slow-core memory: at each window it receives the realized
    Leontief service ``u_t`` per tenant and the fair-benchmark service ``b_t`` for
    the same window, decays the running state by the elapsed wall-clock interval,
    and updates ``Debt_t`` via Eq17. It is intentionally backend-neutral (plain
    floats) so it can be serialized into the schema's slow-state slots.
    """

    half_life_s: float = DEFAULT_HALF_LIFE_S
    tenants: dict[str, TenantDebt] = field(default_factory=dict)

    # -- registration -------------------------------------------------------
    def ensure_tenant(
        self,
        tenant_id: str,
        *,
        disagreement_baseline: float = 0.0,
        ts_s: float = 0.0,
    ) -> TenantDebt:
        """Return (creating if needed) the debt state for a tenant."""
        td = self.tenants.get(tenant_id)
        if td is None:
            td = TenantDebt(
                tenant_id=tenant_id,
                disagreement_baseline=disagreement_baseline,
                last_update_ts_s=ts_s,
                half_life_s=self.half_life_s,
            )
            self.tenants[tenant_id] = td
        return td

    # -- update -------------------------------------------------------------
    def update(
        self,
        realized_service: Mapping[str, float],
        benchmark_service: Mapping[str, float],
        *,
        ts_s: float,
        disagreement_baseline: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Apply one window's service to the decayed debt state (Eq17).

        Parameters
        ----------
        realized_service:
            ``tenant_id -> u_t(τ)`` actually delivered this window.
        benchmark_service:
            ``tenant_id -> b_t(τ)`` the fair benchmark would deliver this window.
        ts_s:
            Wall-clock time of this window; decay uses the elapsed interval since
            the tenant's previous update.
        disagreement_baseline:
            Optional refresh of each tenant's NBS disagreement floor.

        Returns
        -------
        ``tenant_id -> Debt_t`` after the update.
        """
        all_tenants = set(realized_service) | set(benchmark_service)
        out: dict[str, float] = {}
        for tid in all_tenants:
            base = (
                disagreement_baseline.get(tid, 0.0)
                if disagreement_baseline is not None
                else 0.0
            )
            td = self.ensure_tenant(tid, disagreement_baseline=base, ts_s=ts_s)
            if disagreement_baseline is not None and tid in disagreement_baseline:
                td.disagreement_baseline = float(disagreement_baseline[tid])

            dt = max(ts_s - td.last_update_ts_s, 0.0)
            rho = decay_factor(dt, td.half_life_s)

            u = float(realized_service.get(tid, 0.0))
            b = float(benchmark_service.get(tid, 0.0))

            # Eq17: Debt_{t+1} = rho * Debt_t + (b_t - u_t)
            td.debt = rho * td.debt + (b - u)
            td.decayed_service = rho * td.decayed_service + u
            td.decayed_benchmark = rho * td.decayed_benchmark + b
            td.last_update_ts_s = ts_s
            out[tid] = td.debt
        return out

    def update_from_benchmark(
        self,
        realized_service: Mapping[str, float],
        benchmark: FairDivision,
        *,
        ts_s: float,
    ) -> dict[str, float]:
        """Convenience: take ``b_t`` directly from a :class:`FairDivision` result.

        The benchmark's per-tenant disagreement (if it carries one, e.g. NBS)
        refreshes each tenant's baseline so the slow state stays aligned with the
        operator-defined entitlement (doc V, D11 / H4).
        """
        b_map = benchmark.service_map()
        disagreement = None
        d_list = benchmark.meta.get("disagreement")
        if isinstance(d_list, (list, tuple)) and len(d_list) == len(
            benchmark.tenant_ids
        ):
            disagreement = {
                tid: float(d)
                for tid, d in zip(benchmark.tenant_ids, d_list, strict=False)
            }
        return self.update(
            realized_service,
            b_map,
            ts_s=ts_s,
            disagreement_baseline=disagreement,
        )

    # -- queries ------------------------------------------------------------
    def debt(self, tenant_id: str) -> float:
        td = self.tenants.get(tenant_id)
        return td.debt if td is not None else 0.0

    def positive_debt(self, tenant_id: str) -> float:
        td = self.tenants.get(tenant_id)
        return td.positive_debt() if td is not None else 0.0

    def max_positive_debt(self) -> float:
        """``max_t Debt_t^+`` — the planner's worst-served-tenant debt budget."""
        return max((td.positive_debt() for td in self.tenants.values()), default=0.0)

    def debts(self) -> dict[str, float]:
        return {tid: td.debt for tid, td in self.tenants.items()}

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Serializable view for the explanation ledger / slow-state slots."""
        return {
            tid: {
                "debt": td.debt,
                "decayed_service": td.decayed_service,
                "decayed_benchmark": td.decayed_benchmark,
                "disagreement_baseline": td.disagreement_baseline,
                "last_update_ts_s": td.last_update_ts_s,
            }
            for tid, td in self.tenants.items()
        }


# ──────────────────────────────────────────────────────────────────────────────
# Fair regret (doc V, D32 / Eq32)
# ──────────────────────────────────────────────────────────────────────────────
def fair_regret(
    benchmark_welfare: float,
    realized_welfare: float,
) -> float:
    """Fair regret to a benchmark ``Reg_fair = M(B(H)) - M(U(H))`` (doc V, D32).

    ``M`` is the chosen SWF (DRF-equivalent service, PF Nash-log, or NBS welfare).
    Both arguments must be evaluated under the *same* welfare functional. Positive
    regret ⇒ the realized trajectory is less fair-efficient than the benchmark.
    """
    return float(benchmark_welfare - realized_welfare)


def welfare_of_service(
    service: Mapping[str, float] | Sequence[float] | np.ndarray,
    *,
    rule: str = "pf",
    weights: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
    disagreement: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
    epsilon: float = 1e-9,
) -> float:
    """Evaluate a realized service vector under a fairness SWF ``M`` (doc V, D32).

    Mirrors the welfare functional of each benchmark so realized trajectories can
    be scored on the *same* scale as the benchmark for regret:

    * ``rule="pf"``    →  ``Σ_t w_t log(u_t + ε)``           (Nash-log / PF).
    * ``rule="nbs"``   →  ``Σ_t w_t log(u_t - d_t + ε)``     (NBS welfare).
    * ``rule="maxmin"``→  ``min_t u_t / w_t``                (egalitarian).
    * ``rule="drf"``   →  ``min_t u_t / w_t`` (weighted-service proxy for the
                          equalized weighted dominant share).
    """
    u = _as_vector(service)
    w = _as_vector(weights, like=u) if weights is not None else np.ones_like(u)
    if rule in ("maxmin", "drf"):
        active = u > 0
        if not np.any(active):
            return 0.0
        return float(np.min(u[active] / w[active]))
    if rule == "pf":
        return float(np.sum(w * np.log(np.clip(u, 0.0, None) + epsilon)))
    if rule == "nbs":
        d = (
            _as_vector(disagreement, like=u)
            if disagreement is not None
            else np.zeros_like(u)
        )
        margin = np.clip(u - d, 0.0, None) + epsilon
        return float(np.sum(w * np.log(margin)))
    raise ValueError(f"unknown welfare rule {rule!r}")


def _as_vector(
    x: Mapping[str, float] | Sequence[float] | np.ndarray,
    *,
    like: np.ndarray | None = None,
) -> np.ndarray:
    if isinstance(x, Mapping):
        return np.array(list(x.values()), dtype=float)
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr


__all__ = [
    "DEFAULT_HALF_LIFE_S",
    "FairnessDebtTracker",
    "TenantDebt",
    "decay_factor",
    "fair_regret",
    "welfare_of_service",
]
