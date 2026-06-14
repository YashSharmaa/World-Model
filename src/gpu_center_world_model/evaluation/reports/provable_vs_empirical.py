# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The provable-vs-empirical claim ledger (doc O, L1-L12).

Doc O's handoff requires: "Publish the provable-versus-empirical ledger with
every release." This module encodes the twelve claims L1-L12 verbatim in
*structure* (claim id, headline, status, required assumptions, what is proved,
what remains empirical), and provides a generator that *binds* each empirical /
conditional claim to the measured result it depends on, so a release report makes
crystal-clear which numbers are mathematically guaranteed (under stated
assumptions) and which are merely measured on this run.

This is the antidote to the single-blended-score trap (doc F): a reader can see
at a glance that "no kernel-unsafe recommendation" (L1) is provable under
implementation assumptions, while "meets acceptance thresholds" (L9) is purely
empirical and only as trustworthy as the sample sizes behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ClaimStatus(str, Enum):
    """The provability status taxonomy used by doc O's ledger."""

    PROVABLE_UNDER_IMPLEMENTATION = "provable under implementation assumptions"
    PROVABLE_CONDITIONAL = "provable conditional"
    PROVABLE_TASK_EMPIRICAL_PERF = (
        "provable as a task formulation; empirical as a performance claim"
    )
    PROVABLE_FORM_EMPIRICAL_CONSTANTS = (
        "provable conditional for the form of the bound; empirical for the constants"
    )
    PROVABLE_NEGATIVE = "provable negative statement"
    ASSUMED_LIMITED = "assumed and explicitly limited"
    EMPIRICAL = "empirical"
    EMPIRICAL_STRONGLY_SUPPORTED = "empirical but strongly supported"
    DESIGN_CLAIM = "theoretically motivated design claim with strong empirical support"


@dataclass(slots=True)
class LedgerClaim:
    """One row of the provable-vs-empirical ledger (doc O L1-L12)."""

    claim_id: str
    headline_claim: str
    status: ClaimStatus
    required_assumptions: list[str] = field(default_factory=list)
    what_is_proved: str = ""
    what_remains_empirical: str = ""
    # filled in by the generator: the measured result this claim leans on
    empirical_binding: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "headline_claim": self.headline_claim,
            "status": self.status.value,
            "required_assumptions": list(self.required_assumptions),
            "what_is_proved": self.what_is_proved,
            "what_remains_empirical": self.what_remains_empirical,
            "empirical_binding": self.empirical_binding,
        }


# The twelve claims, transcribed from doc O ``provable_empirical_assumed_ledger``.
_LEDGER: tuple[LedgerClaim, ...] = (
    LedgerClaim(
        claim_id="L1",
        headline_claim="No exact-kernel-unsafe recommendation is emitted.",
        status=ClaimStatus.PROVABLE_UNDER_IMPLEMENTATION,
        required_assumptions=[
            "Every candidate action passes through the exact symbolic kernel "
            "before ranking.",
            "The kernel faithfully encodes the intended hard constraints and "
            "operator policy masks.",
            "The implementation has no software bugs.",
        ],
        what_is_proved="The learned model cannot emit an action outside the "
        "kernel-safe set.",
        what_remains_empirical="Whether a kernel-safe action is operationally "
        "wise under hidden dynamics.",
    ),
    LedgerClaim(
        claim_id="L2",
        headline_claim="Held-out generalization gap shrinks at an n_eff^(-1/2) "
        "rate for the residual world model.",
        status=ClaimStatus.PROVABLE_CONDITIONAL,
        required_assumptions=["[A1]-[A4]", "[A7]-[A9]"],
        what_is_proved="GB1 to GB3 imply this rate up to dependence, "
        "graph-sensitivity, and PAC-Bayes terms.",
        what_remains_empirical="The numerical size of Xi_G, KL(Q||P), and "
        "dependence constants.",
    ),
    LedgerClaim(
        claim_id="L3",
        headline_claim="Counterfactual ranking is the right formal target, not "
        "teacher-forced one-step error.",
        status=ClaimStatus.PROVABLE_TASK_EMPIRICAL_PERF,
        required_assumptions=["[A10]-[A11]"],
        what_is_proved="T7 shows that small enough pairwise gap error implies "
        "correct ranking.",
        what_remains_empirical="Whether the trained model achieves the needed "
        "pairwise-gap accuracy on held-out windows.",
    ),
    LedgerClaim(
        claim_id="L4",
        headline_claim="Calibrated domain randomization is preferable to blind "
        "broad randomization.",
        status=ClaimStatus.PROVABLE_CONDITIONAL,
        required_assumptions=[
            "[A15]",
            "correct or approximately correct simulator family",
            "identifiability and regularity conditions of ODR",
        ],
        what_is_proved="ODR consistency plus T12 imply that fitting "
        "randomization to offline real data shrinks expected parameter "
        "misspecification, while widening the radius beyond calibrated "
        "uncertainty increases the shift term.",
        what_remains_empirical="Actual misspec_q, L_nu,q, and simulator-family "
        "misspecification.",
    ),
    LedgerClaim(
        claim_id="L5",
        headline_claim="Counterfactual correctness is only relative to the "
        "branch reference environment.",
        status=ClaimStatus.ASSUMED_LIMITED,
        required_assumptions=[
            "The branch simulator is treated as the reference environment.",
        ],
        what_is_proved="Nothing stronger than simulator-relative correctness.",
        what_remains_empirical="External causal validity to production infrastructure.",
    ),
    LedgerClaim(
        claim_id="L6",
        headline_claim="Multi-step rollout error can be bounded and mitigated by "
        "hierarchy and uncertainty propagation.",
        status=ClaimStatus.PROVABLE_FORM_EMPIRICAL_CONSTANTS,
        required_assumptions=["[A16]", "calibrated predictive uncertainty"],
        what_is_proved="T13 to T16 give explicit growth laws and regret and "
        "false-optimism controls.",
        what_remains_empirical="Realized Lipschitz constants, calibration error, "
        "and observed long-horizon gains.",
    ),
    LedgerClaim(
        claim_id="L7",
        headline_claim="The latent state is not uniquely identifiable from "
        "telemetry alone.",
        status=ClaimStatus.PROVABLE_NEGATIVE,
        required_assumptions=[
            "single-view partial observability without extra structure",
        ],
        what_is_proved="Only symmetry or gauge-invariant and action-sufficient "
        "aspects are identifiable in general.",
        what_remains_empirical="How much action-sufficient structure the chosen "
        "model recovers.",
    ),
    LedgerClaim(
        claim_id="L8",
        headline_claim="Synthetic data can increase effective sample size.",
        status=ClaimStatus.PROVABLE_CONDITIONAL,
        required_assumptions=[
            "stable weighting",
            "small enough query-conditioned sim-to-real divergence",
            "no causal or semantic anti-patterns",
        ],
        what_is_proved="Variance can shrink through n_ESS, but only until "
        "adaptation bias dominates.",
        what_remains_empirical="The real-equivalent contribution of each "
        "synthetic pack.",
    ),
    LedgerClaim(
        claim_id="L9",
        headline_claim="The system meets the release acceptance thresholds.",
        status=ClaimStatus.EMPIRICAL,
        required_assumptions=[
            "sufficient held-out sample sizes",
            "no leakage",
            "successful calibration and training",
        ],
        what_is_proved="Only how many held-out windows are needed to estimate "
        "the metrics with stated confidence.",
        what_remains_empirical="The metrics themselves.",
    ),
    LedgerClaim(
        claim_id="L10",
        headline_claim="Operator usefulness improves median decision time by at "
        "least 20 percent with no safety loss.",
        status=ClaimStatus.EMPIRICAL,
        required_assumptions=[
            "counterbalanced scenario study",
            "representative operators",
        ],
        what_is_proved="Nothing from the learning theory alone.",
        what_remains_empirical="Human reliance, overrides, and speed gains.",
    ),
    LedgerClaim(
        claim_id="L11",
        headline_claim="The world model should be query-conditioned and "
        "intervention-faithful rather than observation-fidelity-first.",
        status=ClaimStatus.DESIGN_CLAIM,
        required_assumptions=[
            "planning objective is lexicographic KPI and risk energy"
        ],
        what_is_proved="Approximate value-equivalence arguments show why "
        "preserving action-relevant query answers is sufficient for planning.",
        what_remains_empirical="Whether the chosen query heads preserve all "
        "action-relevant distinctions in this domain.",
    ),
    LedgerClaim(
        claim_id="L12",
        headline_claim="Large passive pretraining plus limited "
        "action-conditioned post-training is a plausible data-efficient route.",
        status=ClaimStatus.EMPIRICAL_STRONGLY_SUPPORTED,
        required_assumptions=[
            "transferable self-supervised pretraining exists in the telemetry domain",
            "post-training covers the action families of interest",
        ],
        what_is_proved="No domain-specific sample-complexity theorem from "
        "current sources.",
        what_remains_empirical="Whether GPU-center telemetry follows the same "
        "transfer pattern seen in recent world-model work.",
    ),
)


def ledger_claims() -> tuple[LedgerClaim, ...]:
    """Return fresh copies of the twelve doc-O ledger claims (L1-L12)."""
    return tuple(
        LedgerClaim(
            claim_id=c.claim_id,
            headline_claim=c.headline_claim,
            status=c.status,
            required_assumptions=list(c.required_assumptions),
            what_is_proved=c.what_is_proved,
            what_remains_empirical=c.what_remains_empirical,
        )
        for c in _LEDGER
    )


#: Maps each claim to the harness result keys it leans on (for binding).
_CLAIM_BINDINGS: dict[str, tuple[str, ...]] = {
    "L1": ("safety_gate",),
    "L3": ("counterfactual",),
    "L6": ("counterfactual", "calibration"),
    "L9": (
        "safety_gate",
        "counterfactual",
        "planning",
        "calibration",
        "abstention",
        "sample_complexity",
    ),
}


def generate_ledger(
    results: dict[str, object] | None = None,
) -> list[LedgerClaim]:
    """Produce the ledger, binding empirical/conditional claims to results.

    ``results`` is the harness result dict (e.g. the runner's ``as_dict`` output
    with keys ``safety_gate``, ``counterfactual``, ``planning``, ``calibration``,
    ``abstention``, ``sample_complexity``). For each claim with a binding, the
    relevant measured sub-results are attached as ``empirical_binding`` so the
    reader sees exactly which numbers back an empirical claim. Provable-form
    claims keep ``empirical_binding=None`` for the parts that are mathematically
    guaranteed.
    """
    claims = list(ledger_claims())
    if not results:
        return claims
    for claim in claims:
        keys = _CLAIM_BINDINGS.get(claim.claim_id)
        if not keys:
            continue
        binding: dict[str, object] = {}
        for k in keys:
            if k in results:
                binding[k] = results[k]
        claim.empirical_binding = binding or None
    return claims


def ledger_as_dict(
    results: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return the full ledger as a JSON-serialisable dict (for the release report)."""
    claims = generate_ledger(results)
    return {
        "source": "doc O — theory & sim2real (L1-L12)",
        "note": (
            "Safety is a gate, not a weighted metric: L1 is provable under "
            "implementation assumptions; L9/L10 are purely empirical and only as "
            "strong as the sample sizes behind them."
        ),
        "claims": [c.as_dict() for c in claims],
    }


__all__ = [
    "ClaimStatus",
    "LedgerClaim",
    "generate_ledger",
    "ledger_as_dict",
    "ledger_claims",
]
