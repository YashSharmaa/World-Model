# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Abstention metrics: risk-coverage curves and false-safe rate under shift.

Doc F rewards the model for *knowing when not to recommend*. The two readouts:

* **Risk-coverage curve** — for a confidence / disagreement score, sweep the
  abstention threshold and trace the trade-off between *coverage* (fraction of
  cases the model answers) and *selective risk* (error on the answered subset).
  The area under the risk-coverage curve (AURC) and the selective risk at a
  target coverage are the headline numbers. Doc F acceptance for counterfactual
  ranking: >= 0.80 accuracy on the non-abstained subset while retaining >= 0.60
  coverage.

* **False-safe under shift** — on severe-shift / degraded-telemetry packs, the
  system should abstain or fall back conservatively in >= 80% of flagged cases,
  with a **false-safe rate < 1%** (doc F shift-and-abstention threshold; doc O
  E7 rule-of-three: ~300 zero-event cases certify < 1%).

Pure NumPy. The risk score convention is "higher = the model is more uncertain /
should abstain first"; errors are 0/1 indicators of a wrong (or unsafe) answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ArrayLike = np.ndarray | list[float]

# ``np.trapz`` was renamed to ``np.trapezoid`` in NumPy 2.0; support both so the
# harness imports cleanly across the pinned ``numpy>=1.26`` range.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]


def _as_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


@dataclass(slots=True)
class RiskCoveragePoint:
    """One point on the risk-coverage curve."""

    coverage: float
    selective_risk: float
    threshold: float


@dataclass(slots=True)
class RiskCoverageReport:
    """Risk-coverage curve summary for a selective predictor."""

    n: int
    aurc: float  # area under the risk-coverage curve (lower is better)
    selective_risk_at_target: float
    target_coverage: float
    curve: list[RiskCoveragePoint] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "aurc": self.aurc,
            "selective_risk_at_target": self.selective_risk_at_target,
            "target_coverage": self.target_coverage,
            "curve": [
                {
                    "coverage": p.coverage,
                    "selective_risk": p.selective_risk,
                    "threshold": p.threshold,
                }
                for p in self.curve
            ],
        }


def risk_coverage_curve(
    risk_score: ArrayLike,
    error: ArrayLike,
    *,
    target_coverage: float = 0.60,
) -> RiskCoverageReport:
    """Trace the risk-coverage curve by abstaining on the highest-risk cases.

    We sort by ascending ``risk_score`` (most-confident first) and, at each
    coverage level c = k/N, report the mean ``error`` over the k most-confident
    cases (the selective risk). AURC is the trapezoidal area under that curve.
    ``selective_risk_at_target`` is the selective risk when the model answers the
    ``target_coverage`` most-confident fraction — the doc-F 0.60-coverage point.
    """
    s, e = _as_array(risk_score), _as_array(error)
    n = s.size
    if n == 0:
        return RiskCoverageReport(
            n=0,
            aurc=float("nan"),
            selective_risk_at_target=float("nan"),
            target_coverage=target_coverage,
        )
    order = np.argsort(s, kind="stable")  # ascending risk = most confident first
    e_sorted = e[order]
    s_sorted = s[order]
    cum_err = np.cumsum(e_sorted)
    ks = np.arange(1, n + 1)
    coverages = ks / n
    selective_risk = cum_err / ks
    curve = [
        RiskCoveragePoint(
            coverage=float(coverages[i]),
            selective_risk=float(selective_risk[i]),
            threshold=float(s_sorted[i]),
        )
        for i in range(n)
    ]
    # AURC via trapezoid over coverage in [1/n, 1].
    aurc = float(_trapezoid(selective_risk, coverages))
    # selective risk at target coverage (nearest coverage >= target)
    idx = int(np.searchsorted(coverages, target_coverage, side="left"))
    idx = min(idx, n - 1)
    return RiskCoverageReport(
        n=n,
        aurc=aurc,
        selective_risk_at_target=float(selective_risk[idx]),
        target_coverage=target_coverage,
        curve=curve,
    )


@dataclass(slots=True)
class ShiftAbstentionReport:
    """Abstention behaviour on a flagged severe-shift / degraded-telemetry pack."""

    n_flagged: int
    abstain_or_fallback_count: int
    abstain_or_fallback_rate: float
    false_safe_count: int
    false_safe_rate: float
    false_safe_upper_95: float  # rule-of-three style one-sided 95% bound
    passed: bool
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_flagged": self.n_flagged,
            "abstain_or_fallback_count": self.abstain_or_fallback_count,
            "abstain_or_fallback_rate": self.abstain_or_fallback_rate,
            "false_safe_count": self.false_safe_count,
            "false_safe_rate": self.false_safe_rate,
            "false_safe_upper_95": self.false_safe_upper_95,
            "passed": self.passed,
            "failures": list(self.failures),
        }


def false_safe_under_shift(
    abstained_or_fellback: ArrayLike,
    was_false_safe: ArrayLike,
    *,
    min_abstain_rate: float = 0.80,
    max_false_safe_rate: float = 0.01,
) -> ShiftAbstentionReport:
    """Evaluate abstention/false-safe behaviour on flagged shift cases.

    Parameters
    ----------
    abstained_or_fellback:
        0/1 per flagged case — did the system abstain or fall back to the
        conservative rule baseline?
    was_false_safe:
        0/1 per flagged case — did the system confidently recommend something
        the reference branch shows to be unsafe (a false-safe)?

    Doc F thresholds: abstain/fallback in >= 80% of flagged cases; false-safe
    rate < 1%. ``false_safe_upper_95`` applies the rule-of-three (doc O E7):
    when zero false-safe events are seen in N cases, the 95% upper bound is ~3/N,
    so N ~= 300 certifies < 1%.
    """
    ab = _as_array(abstained_or_fellback)
    fs = _as_array(was_false_safe)
    n = ab.size
    if n == 0:
        return ShiftAbstentionReport(
            n_flagged=0,
            abstain_or_fallback_count=0,
            abstain_or_fallback_rate=float("nan"),
            false_safe_count=0,
            false_safe_rate=float("nan"),
            false_safe_upper_95=float("nan"),
            passed=False,
            failures=["no flagged cases"],
        )
    ab_count = int(np.sum(ab > 0.5))
    fs_count = int(np.sum(fs > 0.5))
    ab_rate = ab_count / n
    fs_rate = fs_count / n
    # one-sided 95% upper bound on the false-safe rate
    if fs_count == 0:
        upper95 = 3.0 / n  # rule of three
    else:
        # Wilson-style conservative upper bound for nonzero events
        z = 1.6448536269514722  # 95% one-sided
        phat = fs_rate
        denom = 1.0 + z * z / n
        centre = phat + z * z / (2 * n)
        margin = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
        upper95 = float((centre + margin) / denom)

    failures: list[str] = []
    if ab_rate < min_abstain_rate:
        failures.append(
            f"abstain/fallback rate {ab_rate:.3f} below {min_abstain_rate:.2f}"
        )
    if fs_rate > max_false_safe_rate:
        failures.append(
            f"false-safe rate {fs_rate:.4f} exceeds {max_false_safe_rate:.4f}"
        )
    return ShiftAbstentionReport(
        n_flagged=n,
        abstain_or_fallback_count=ab_count,
        abstain_or_fallback_rate=ab_rate,
        false_safe_count=fs_count,
        false_safe_rate=fs_rate,
        false_safe_upper_95=upper95,
        passed=len(failures) == 0,
        failures=failures,
    )


__all__ = [
    "RiskCoveragePoint",
    "RiskCoverageReport",
    "ShiftAbstentionReport",
    "false_safe_under_shift",
    "risk_coverage_curve",
]
