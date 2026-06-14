export const meta = {
  name: 'world-model-completion-audit',
  description: 'Audit GPU-Center World Model code against its 24 research docs; verify each gap; synthesize a completion plan',
  phases: [
    { title: 'Audit', detail: 'one agent per module lane reads its governing research doc(s) + code and reports gaps' },
    { title: 'Verify', detail: 'adversarially re-check each lane’s findings against the actual code' },
    { title: 'Synthesize', detail: 'dedupe + prioritize all verified gaps into a master completion plan' },
  ],
}

const ROOT = (args && args.root) || '/Users/yash/Documents/GitHub/World-Model'
const SRC = `${ROOT}/src/gpu_center_world_model`
const RD = `${ROOT}/Research_Docs`

// Research doc absolute paths (filenames with spaces preserved exactly).
const DOC = {
  A: `${RD}/A - Open-Source World Model Design for GPU Center Operations.md`,
  B: `${RD}/B - Choosing the Right V1 World Model for GPU Center Operations.md`,
  C: `${RD}/C - World Representation for a GPU Center World Model.md`,
  D: `${RD}/D - Synthetic Data Engine for a GPU Center World Model.md`,
  E: `${RD}/E - Apple MLX implementation plan for a structured GPU-center world model.md`,
  F: `${RD}/F - Evaluation plan for an open source GPU center operations world model.md`,
  G: `${RD}/G - Lightweight Front-End Design for a GPU Pod World Model Demo.md`,
  H: `${RD}/H - Final.md`,
  I: `${RD}/I_causal_probabilistic_foundations.md`,
  J: `${RD}/J_learning_objective_and_differentiable_rollout.md`,
  K: `${RD}/K_queueing_and_control_dynamics.md`,
  L: `${RD}/L_uncertainty_conformal_guarantees.md`,
  M: `${RD}/M_constrained_risk_sensitive_planning.md`,
  N: `${RD}/N_physical_surrogate_system_id.md`,
  O: `${RD}/O_theory_and_sim2real_guarantees.md`,
  P: `${RD}/P_representation_information_theory.md`,
  Q: `${RD}/Q_relational_architecture_expressivity.md`,
  R: `${RD}/R_synthetic_generative_process_and_fidelity.md`,
  S: `${RD}/S_combinatorial_exact_kernel_and_shield.md`,
  T: `${RD}/T_bayesian_filtering_state_estimation.md`,
  U: `${RD}/U_reliability_changepoint_detection.md`,
  V: `${RD}/V_fairness_mechanism_design.md`,
}

// Consolidated into 8 broad lanes (lower concurrent burst → avoids the
// server-side rate limit that killed the 17-lane version). Each agent has its
// own large context window, so a broad lane is fine.
const LANES = [
  { key: 'schema_kernel', docs: [DOC.C, DOC.I, DOC.S], paths: [`${SRC}/schema/`, `${SRC}/simulator/exact_kernel/`, `${SRC}/simulator/contracts.py`],
    focus: 'SCHEMA (doc C/I): every entity table, relation edge, observation family, action vocab, tensor caps (J_max=64,N_max=32,G_max=256,R_max=4...), latent dims, KPI defs, state hashing, doc I Definitions 1-7. EXACT KERNEL (doc S): feasibility ILP E1-E14 (action-type, queue/flavor, topology selection, gang all-or-nothing, node capacities, protected masks, quota+borrowing+lending, admission checks, podset-group); safe candidate set; why-not group-minimal infeasibility certificates (IIS / deletion-filter / feasibility-relaxation deficits); shield projection Pi_s (nearest-feasible, idempotent, minimal-perturbation); zero-violation conditions C1-C4 incl exact-integer post-check; flow-vs-MILP hybrid; pruning (dominance/symmetry/lower-bounds). Flag any entity/field/edge/constraint/certificate/pruning rule simplified or absent.' },
  { key: 'data_engine', docs: [DOC.D, DOC.R, DOC.N, DOC.K], paths: [`${SRC}/simulator/branch_runner.py`, `${SRC}/simulator/rare_events.py`, `${SRC}/data_engine/`],
    focus: 'BRANCH/RARE (doc D/K/S): deterministic branch replay from pre-action state hash + fixed exogenous seed bundle (same state+seed+action => same outcome); rare-event library (gpu_fault_burst, pods_ready_timeout, cooling_derate, power_derate, queue_burst, maintenance_mask/drain, telemetry_degradation); queueing (doc K correlated arrivals, fork-join startup max, virtual queues, fluid/diffusion). DATA ENGINE (doc D): bronze/silver/gold layers (metric_sample, scheduler_event, intervention_event, decision_window, label_pack); windows anchored by state hash; 3 cadence lanes; tensorization fixed buckets+masks; provenance manifest + data card; splits (temporal/topology/intervention/rare-event holdouts) + counterfactual harness (>=3 actions/window, shared seed). SYNTHETIC (doc N/R): anchored RC thermal surrogate (E1-E7, ZOH, stability S1-S6, Laplacian energy conservation, dissipative residual), switching semi-Markov power source (P1-P5, saturation+ramp), two-tier network max-min (N1-N4), Xid/ECC faults; SCM latent parents (workload_phase, placement_context, device_health_state, cooling_zone_state, operator_policy_mode); calibrated-randomization tiers; fidelity metrics. Flag any layer/schema-field/split/surrogate/latent-parent crude or missing.' },
  { key: 'model', docs: [DOC.Q, DOC.J, DOC.P, DOC.E], paths: [`${SRC}/model/architecture.py`, `${SRC}/model/world_model.py`, `${SRC}/model/heads.py`, `${SRC}/model/jepa.py`, `${SRC}/model/backend.py`, `${SRC}/model/contracts.py`, `${SRC}/model/train.py`, `${SRC}/model/serialization.py`],
    focus: 'ARCH (doc Q): typed fixed-incidence MPNN (Eq1-8 gather/message/aggregate/GRU update, hierarchical PNA pooling sum/mean/std/max/count, downward broadcast, FiLM, tiny candidate-set attention, per-type permutation equivariance). LEARNING (doc J): JEPA EMA-target+stop-grad+typed masking, collapse-prevention VICReg-lite (var/cov), counterfactual rollout loss (absolute + branch-difference), head losses (quantile pinball+non-crossing, discrete-time competing-risk hazard, abstention), consistency loss, group-wise uncertainty weighting; TRAIN: 4 stages, free-running rollout (no teacher forcing), horizon curriculum + truncated BPTT, collapse diagnostics D1-D5, EMA schedule, calibration freeze, SPSA-numpy vs MLX nn.value_and_grad+AdamW, multi-file checkpoint. doc P latent dims; doc E bucketed shapes, MLX/numpy parity, the dangling model.mlx_backend reference in backend.py. Flag any loss/pooling/equivariance/head/training-stage/diagnostic absent or simplified, and any mlx-only path with no numpy counterpart.' },
  { key: 'uncertainty_reliability', docs: [DOC.L, DOC.T, DOC.U], paths: [`${SRC}/uncertainty/`, `${SRC}/model/calibration/`, `${SRC}/reliability/`],
    focus: 'CONFORMAL (doc L): split-conformal, CQR, Mondrian group-conditional, one-sided upper, beta small-sample correction, weighted conformal (covariate shift), ACI/BC-ACI temporal, selective split-select-certify (Clopper-Pearson), safety-gate chance constraint, optimism one-sided bound, OOD (kNN/topology/regime/epistemic fuse + conformal p-values + FDR), epistemic BatchEnsemble. FILTERS (doc T): sqrt EKF/UKF, Rao-Blackwellized PF, EnKF, RB by kernel, GRU-as-amortized-filter, informative MNAR missingness, observability Gramian, consistency diagnostics (NIS/NEES/whiteness/ESS), offline RTS/ensemble smoothing. RELIABILITY (doc U): discrete-time cause-specific competing-risks hazards (softmax link, CIF Aalen-Johansen recursion, censored likelihood), graph-coupled hidden semi-Markov degradation (dwell-time, transition law, obs-degradation channel), quiet-failure obs model (MNAR regimes observed/stale/delayed/dropped, integrity posterior u_t), quickest-change detection (CUSUM, Shiryaev-Roberts, SPRT, BOCPD run-length, multichannel/graph scan+laplacian), operating-point (budget threshold, uncertainty gate, PR-space), RUL conformal (IPCW). Flag any method/filter/detector/diagnostic missing or stubbed.' },
  { key: 'planner_fairness', docs: [DOC.M, DOC.S, DOC.V], paths: [`${SRC}/planner/`, `${SRC}/fairness/`],
    focus: 'PLANNER (doc M): shielded constrained belief-MDP; lexicographic energy (hard-safety > statistical-safety > p95 wait > admission > GPU-hours > locality > fragmentation); CVaR + chance-constraint risk gate; enumerate safe candidates; score via model rollout; abstain on OOD/disagreement; why-not; explanation ledger; amortization (proposal/value/abstention heads); enum-now-search-later (MPC/CEM/MCTS). FAIRNESS (doc V — user wants ALL of it implemented, not deferred to v2): fair-division benchmarks (weighted DRF, lexicographic max-min, proportional fairness, Nash bargaining), fairness axioms, fairness-throughput frontier, fairness-drift KPI (cohort+admission, Eq12-14), incentives/strategyproofness (manipulation surfaces, restricted SP conditions), price-of-anarchy empirical estimator, dynamic fairness debt (recursive decay), hooks H1-H10, AND wiring fairness into the planner objective. Flag any planner stage / risk-constraint / fairness benchmark/KPI/hook absent.' },
  { key: 'eval_serving_ingest', docs: [DOC.F, DOC.O, DOC.G, DOC.D], paths: [`${SRC}/evaluation/`, `${SRC}/serving/`, `${SRC}/ingest/`, `${ROOT}/benchmarks/`],
    focus: 'EVAL (doc F/O): metric families (safety_gate, predictive WAPE/pinball/coverage/Brier, counterfactual sign-accuracy/optimism-bias, planning safe-regret/lexicographic, operator_usefulness); 6 baselines wired into runner; ablations (remove hierarchy/object-centric/uncertainty/synthetic/counterfactual-loss); acceptance thresholds; red-team suite (7 scenarios); 3 benchmark tracks (semantics/counterfactual_rollout/safe_action_selection); slices; sample-complexity formulas; provable-vs-empirical ledger. SERVING (doc G): 5 endpoints (state/current, actions/compare, actions/why-not, replays/{id}, stream SSE w/ 6 event types), DTO shapes, rollout object, cache, compile-stable shapes. INGEST (doc D/H): 7 adapters (kueue, kubernetes DRA, dcgm, node_exporter, redfish, otel_prom, offline_snapshots) + base — each must actually normalize source→canonical schema. Flag any metric/ablation/red-team/track/endpoint/SSE-event/DTO-field/adapter missing or not wired.' },
  { key: 'frontend', docs: [DOC.G], paths: [`${ROOT}/apps/web/src/`, `${ROOT}/apps/web/package.json`, `${ROOT}/apps/web/index.html`],
    focus: 'three.js Decision Studio (doc G): 4 views (Decision Studio, Counterfactual Compare, Replay+Calibration, Inspect Drawer); scene graph (rack/node InstancedMesh, gpu slots, domain/power/cooling hulls, branch ghosts, network links); animation rules (grounded vs predicted, blocker lock, branch cross-fade, workload commit motion, occupancy fill, risk halo, uncertainty, replay divergence); overlays (legend, queue strip, recommendation panel, why-not card, inspect drawer, timeline, branch delta ribbon, provenance header); API client hitting all endpoints + SSE; golden-path 8-beat demo. Read every file under src/. Flag whether each view/overlay/animation/scene-node/api-call is actually implemented vs a scaffold/placeholder. Do NOT run npm.' },
  { key: 'repo_infra', docs: [DOC.H, DOC.F], paths: [`${ROOT}/Makefile`, `${ROOT}/pyproject.toml`, `${ROOT}/.github/`, `${ROOT}/examples/`, `${ROOT}/configs/`, `${ROOT}/scenarios/`, `${ROOT}/sample_data/`, `${ROOT}/docs/`, `${SRC}/cli.py`],
    focus: 'Doc H: CI workflows (lint/type/test/docs/security, arm64 macOS MLX smoke); examples 01..05 — READ each and trace whether it imports real functions and would run end-to-end; configs (train/eval/demo) present+complete; scenarios (normal/stress/rare_events) authored; sample_data tiny pack present; docs (getting-started/architecture/schema/api/benchmark/cards data+model/adr/theory) — flag empty/placeholder docs; CLI subcommands (about/compare/serve/pack/eval) complete; reproducibility (lockfile, seed registry). Flag any example that won’t run, missing/empty config/scenario/doc, or CLI gap.' },
]

const FINDINGS_SCHEMA = {
  type: 'object',
  required: ['lane', 'files_reviewed', 'findings', 'lane_summary'],
  properties: {
    lane: { type: 'string' },
    files_reviewed: { type: 'array', items: { type: 'string' } },
    lane_summary: { type: 'string', description: 'overall maturity of this lane in 2-4 sentences' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'file', 'kind', 'title', 'description', 'doc_basis', 'severity', 'suggested_fix'],
        properties: {
          id: { type: 'string', description: 'short stable slug e.g. exact_kernel-missing-iis' },
          file: { type: 'string', description: 'repo-relative path' },
          lines: { type: 'string', description: 'line range or symbol name, "" if file-level' },
          kind: { type: 'string', enum: ['missing', 'stub', 'placeholder', 'simplified', 'not_wired', 'stale_reference', 'mlx_only_no_fallback', 'broken', 'doc_mismatch', 'untested'] },
          title: { type: 'string' },
          description: { type: 'string', description: 'what is wrong / absent, concretely' },
          doc_basis: { type: 'string', description: 'which doc + section/equation requires this' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          suggested_fix: { type: 'string' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
      },
    },
  },
}

const VERIFY_SCHEMA = {
  type: 'object',
  required: ['lane', 'verified'],
  properties: {
    lane: { type: 'string' },
    verified: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'file', 'title', 'kind', 'severity', 'verdict', 'verifier_note', 'suggested_fix'],
        properties: {
          id: { type: 'string' },
          file: { type: 'string' },
          lines: { type: 'string' },
          kind: { type: 'string' },
          title: { type: 'string' },
          description: { type: 'string' },
          doc_basis: { type: 'string' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          suggested_fix: { type: 'string' },
          verdict: { type: 'string', enum: ['confirmed', 'false_positive', 'already_implemented_elsewhere', 'partially_true'] },
          verifier_note: { type: 'string', description: 'evidence from the actual code for the verdict (cite file:line)' },
        },
      },
    },
  },
}

phase('Audit')
const verifiedLanes = await pipeline(
  LANES,
  // Stage 1: audit
  (lane) => agent(
    `You are auditing one module lane of the GPU-Center World Model repo for COMPLETENESS against its governing research design docs. ` +
    `The repo root is ${ROOT}.\n\n` +
    `STEP 1 — Read the governing research doc(s) for this lane (use the Read tool):\n${lane.docs.map((d) => `  - ${d}`).join('\n')}\n\n` +
    `STEP 2 — Read EVERY source file under these paths and grep within them:\n${lane.paths.map((p) => `  - ${p}`).join('\n')}\n\n` +
    `LANE FOCUS: ${lane.focus}\n\n` +
    `Your job: find EVERY place where the code is incomplete relative to the docs. Report: stubs; placeholders; "simplified"/"approximate"/"for now"/"reference-only" shortcuts; functions that return hardcoded/constant values where a real computation is specified; capabilities named in the docs but ABSENT from code; pieces implemented but NOT WIRED into the pipeline (computed but never called; defined but never registered; generator never invoked; metric never aggregated); dangling references to non-existent modules/functions; MLX-only or numpy-only code paths missing their counterpart; broken logic; doc mismatches; and claimed features with no test.\n\n` +
    `CRITICAL INSTRUCTION: The user wants EVERYTHING fully implemented. The docs often say a capability is "v2", "do-later", "deferred", "future work", or "shadow-mode only" — TREAT THOSE AS GAPS TO REPORT, not as acceptable omissions. If the docs describe it, the code must implement it.\n\n` +
    `Be concrete: cite file paths and line ranges/symbol names. Do NOT report cosmetic/style nits. Focus on functional completeness. Do NOT modify any files — this is read-only audit. Return findings for lane "${lane.key}".`,
    { label: `audit:${lane.key}`, phase: 'Audit', schema: FINDINGS_SCHEMA }
  ),
  // Stage 2: adversarial verify
  (auditResult, lane) => {
    if (!auditResult || !auditResult.findings || auditResult.findings.length === 0) {
      return { lane: lane.key, verified: [] }
    }
    return agent(
      `You are an adversarial verifier. Another agent audited the "${lane.key}" lane of the GPU-Center World Model repo (root ${ROOT}) and produced the findings below. ` +
      `For EACH finding, open the cited file(s) and the lane's source paths (${lane.paths.join(', ')}) and decide whether the gap is REAL. ` +
      `Re-grep the whole ${SRC} tree if needed to check whether a "missing" thing is actually implemented in a different file (verdict: already_implemented_elsewhere). ` +
      `Default to skepticism: only mark "confirmed" if you can point to the actual code (or its absence) proving the gap. Mark "false_positive" if the auditor misread, "partially_true" if the gap is real but smaller/larger than claimed (adjust severity), or "already_implemented_elsewhere" if it exists somewhere the auditor missed. ` +
      `Cite file:line evidence in verifier_note for every verdict. Keep the original suggested_fix (refine if needed). Governing doc(s): ${lane.docs.join(', ')}.\n\n` +
      `Reminder: deliberate "v2/do-later/deferred" scope cuts in the docs ARE confirmed gaps (the user wants everything done) — do not downgrade them to false_positive on scope grounds.\n\n` +
      `FINDINGS TO VERIFY (JSON):\n${JSON.stringify(auditResult.findings, null, 2)}`,
      { label: `verify:${lane.key}`, phase: 'Verify', schema: VERIFY_SCHEMA }
    )
  }
)

const allVerified = verifiedLanes.filter(Boolean).flatMap((l) => (l.verified || []).map((v) => ({ ...v, lane: l.lane })))
const confirmed = allVerified.filter((v) => v.verdict === 'confirmed' || v.verdict === 'partially_true')
log(`Audit complete: ${allVerified.length} findings checked, ${confirmed.length} confirmed/partial across ${verifiedLanes.filter(Boolean).length} lanes`)

phase('Synthesize')
const SYNTH_SCHEMA = {
  type: 'object',
  required: ['executive_summary', 'completion_plan'],
  properties: {
    executive_summary: { type: 'string', description: 'overall state of the repo and the shape of remaining work, 1 paragraph' },
    maturity_by_lane: { type: 'array', items: { type: 'object', required: ['lane', 'maturity_pct', 'note'], properties: { lane: { type: 'string' }, maturity_pct: { type: 'number' }, note: { type: 'string' } } } },
    completion_plan: {
      type: 'array',
      description: 'deduplicated, prioritized work items to make the repo fully complete',
      items: {
        type: 'object',
        required: ['item_id', 'title', 'lane', 'files', 'what_to_build', 'doc_basis', 'severity', 'effort', 'depends_on'],
        properties: {
          item_id: { type: 'string' },
          title: { type: 'string' },
          lane: { type: 'string' },
          files: { type: 'array', items: { type: 'string' } },
          what_to_build: { type: 'string', description: 'concrete description of the implementation needed' },
          doc_basis: { type: 'string' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          effort: { type: 'string', enum: ['small', 'medium', 'large'] },
          depends_on: { type: 'array', items: { type: 'string' }, description: 'other item_ids this depends on' },
        },
      },
    },
  },
}

const synthesis = await agent(
  `You are the lead engineer synthesizing a completion plan for the GPU-Center World Model repo (root ${ROOT}). ` +
  `Below are ${confirmed.length} CONFIRMED/PARTIAL gaps (already adversarially verified) from a module-by-module audit against the 24 research docs. ` +
  `Deduplicate overlapping items, merge near-duplicates, and produce a single prioritized completion plan that, if executed, makes the repo fully implement everything the research docs describe (no stubs/placeholders/scope-cuts). ` +
  `Order by severity then dependency. Each item must name concrete files and a concrete implementation. Also give a maturity_pct (0-100) per lane and a crisp executive summary.\n\n` +
  `Group/merge sensibly but do NOT drop any confirmed gap. If two lanes report the same root cause, merge into one item listing all files.\n\n` +
  `CONFIRMED GAPS (JSON):\n${JSON.stringify(confirmed, null, 2)}`,
  { label: 'synthesize-plan', phase: 'Synthesize', schema: SYNTH_SCHEMA }
)

return {
  lanes_audited: verifiedLanes.filter(Boolean).length,
  findings_checked: allVerified.length,
  confirmed_count: confirmed.length,
  synthesis,
  confirmed_gaps: confirmed,
}
