// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Typed client for the six frontend-facing backend contracts (design doc G):
//
//    GET  /api/v1/state/current          -> StateSnapshotDTO
//    POST /api/v1/actions/compare        -> CompareResultDTO
//    POST /api/v1/actions/why-not        -> WhyNotDTO
//    GET  /api/v1/replays/{scenario_id}  -> ReplayDTO
//    GET  /api/v1/replays                -> ReplayIndexDTO  (catalog)
//    GET  /api/v1/stream                 -> SSE EventSource (StreamEvent)
//
//  The DTOs mirror the frozen Python schema surface:
//    - actions.py  : ActionType / Hold / Admit / Place / AdmitAndPlace / MaskDomain
//    - kpis.py     : KPIChannel / HazardChannel / quantiles / horizons
//    - entities.py : Rack / Node / GPU / Workload / LocalQueue / TopologyDomain ...
//    - simulator/contracts.py : Blocker / BlockerGroup / WhyNot certificate
//    - planner/contracts.py   : PlannerResult provenance (model/policy version)
//
//  Transport rule (doc G): ordinary request/response + Server-Sent Events. The
//  backend returns SEMANTIC DIFFS and ROLLOUT POINTS, never rendered frames —
//  the three.js layer owns all geometry and materials.
//
//  When the backend is offline, every call degrades to deterministic mock data
//  so the recorded demo is reproducible on any machine.

import {
  buildMockSnapshot,
  buildMockCompare,
  buildMockWhyNot,
  buildMockReplay,
  buildMockReplayIndex,
  buildMockStateDelta,
} from "./mock";

// ──────────────────────────────────────────────────────────────────────────
// Action alphabet (mirrors schema/actions.py ActionType)
// ──────────────────────────────────────────────────────────────────────────
export type ActionType = "hold" | "admit" | "place" | "admit_and_place" | "mask_domain";

/** Stable per-candidate command sent to POST /actions/compare. */
export interface CandidateAction {
  action_id: string;
  type: ActionType;
  /** Free-form command params (workload_uid, cluster_queue_id, domain set …). */
  params: Record<string, unknown>;
}

// ──────────────────────────────────────────────────────────────────────────
// KPI / hazard channels (mirror schema/kpis.py enums)
// ──────────────────────────────────────────────────────────────────────────
export type KPIChannel =
  | "queue_wait_delta_s"
  | "admitted_gpu_hours_delta"
  | "topology_fragmentation_delta"
  | "rack_power_kw"
  | "rack_thermal_headroom_c"
  | "completion_time_delta_s"
  | "slowdown"
  | "cross_rack_rate";

export type HazardChannel =
  | "hotspot"
  | "pods_ready_timeout"
  | "requeue"
  | "xid_fault"
  | "headroom_violation"
  | "node_failure";

/** Blocker families (mirror simulator/contracts.py BlockerGroup). */
export type BlockerGroup =
  | "action_whitelist"
  | "status_precondition"
  | "admission_checks"
  | "queue_flavor_eligibility"
  | "protected_domain_mask"
  | "topology_rule"
  | "health_schedulability"
  | "node_capacities"
  | "quota"
  | "borrowing_lending"
  | "gang_atomicity";

/** Default horizons (mirror schema/kpis.py HORIZONS_S). */
export const HORIZONS_S = [300, 900, 1800, 3600] as const;
/** Default quantile grid (mirror schema/kpis.py QUANTILES). */
export const QUANTILES = [0.05, 0.1, 0.5, 0.9, 0.95] as const;

// ──────────────────────────────────────────────────────────────────────────
// Entity DTOs (mirror schema/entities.py; flattened for the renderer)
// ──────────────────────────────────────────────────────────────────────────
export interface RackDTO {
  rack_id: string;
  block_id: string;
  power_domain_id: string;
  cooling_zone_id: string;
  designed_gpu_cap: number;
  rack_power_kw: number;
  rack_headroom_kw: number;
  inferred_thermal_margin_c: number;
  /** 0..1 learned hotspot risk used to drive halo thickness (not exact). */
  hotspot_score?: number;
}

export interface NodeDTO {
  node_uid: string;
  rack_id: string;
  block_id: string;
  switch_id: string;
  flavor_id: string;
  gpu_cap: number;
  ready: boolean;
  schedulable: boolean;
  running_pods: number;
  node_health: "ready" | "degraded" | "unschedulable" | "draining" | "down";
}

export interface GPUDTO {
  gpu_uuid: string;
  node_uid: string;
  /** workload_uid this slot is allocated to, or null when free. */
  allocated_to: string | null;
  util: number;
  temp_c: number;
  power_w: number;
  xid_code: number;
  health_state: "healthy" | "degraded" | "intermittently_faulty" | "reset_recovery" | "masked";
}

export interface WorkloadDTO {
  workload_uid: string;
  tenant_id: string;
  local_queue_id: string;
  cluster_queue_id: string;
  priority: number;
  topology_policy: "required" | "preferred" | "unconstrained";
  comm_class: "none" | "light" | "allreduce_heavy" | "pipeline";
  status: "pending" | "reserving" | "admitted" | "running" | "finished" | "evicted" | "requeued";
  gpu_req: number;
  age_s: number;
  est_remaining_s: number;
  ready_fraction: number;
  admitted_domain_id: string | null;
}

export interface QueueDTO {
  name: string;
  tenant_id: string;
  cluster_queue_ref: string;
  pending_ct: number;
  pending_gpu: number;
  wait_time_p50_s: number;
  wait_time_p95_s: number;
}

export interface TopologyDomainDTO {
  domain_id: string;
  level: "block" | "rack" | "node" | "compute_domain";
  parent_domain_id: string | null;
  free_gpu: number;
  headroom_score: number;
  hotspot_score: number;
}

export interface RelationDTO {
  type: string;
  src_kind: string;
  dst_kind: string;
  /** source id -> target id pairs, already de-indexed for the renderer. */
  pairs: Array<[string, string]>;
}

export interface CurrentKPIsDTO {
  queue_wait_p95_s: number;
  admitted_gpu_hours: number;
  topology_locality: number;
  violation_probability: number;
  pending_workloads: number;
  free_gpu: number;
}

export interface FreshnessDTO {
  /** age-of-information per observation family (seconds). */
  scheduler_state_age_s: number;
  placement_state_age_s: number;
  gpu_telemetry_age_s: number;
  /** worst-case telemetry source health for the provenance header badge. */
  worst_source_health: "observed" | "stale" | "delayed" | "dropped";
}

// ──────────────────────────────────────────────────────────────────────────
// Contract 1: GET /api/v1/state/current
// ──────────────────────────────────────────────────────────────────────────
export interface StateSnapshotDTO {
  state_hash: string;
  generated_at: string; // ISO-8601
  model_version: string;
  policy_version: string;
  horizon_options_s: number[];
  entities: {
    racks: RackDTO[];
    nodes: NodeDTO[];
    gpus: GPUDTO[];
    workloads: WorkloadDTO[];
    queues: QueueDTO[];
    topology_domains: TopologyDomainDTO[];
  };
  relations: RelationDTO[];
  current_kpis: CurrentKPIsDTO;
  freshness: FreshnessDTO;
  /** Project attribution — travels with every payload (planner CREDIT). */
  credit: string;
}

// ──────────────────────────────────────────────────────────────────────────
// rollout object (doc G reusable nested payload)
// ──────────────────────────────────────────────────────────────────────────
export interface RolloutPoint {
  t_s: number;
  /** per-rack occupancy fraction 0..1, keyed by rack_id (semantic diff only). */
  rack_occupancy: Record<string, number>;
  /** per-rack learned hotspot risk 0..1 (drives halo growth). */
  rack_hotspot: Record<string, number>;
  /** workload_uid -> domain_id it is committed to at this timestep (ghost). */
  workload_placement: Record<string, string>;
  /** queue depth (pending workload count) at this timestep. */
  queue_depth: number;
  events: RolloutEvent[];
  uncertainty: { epistemic: number; ood: number };
}

export interface RolloutEvent {
  t_s: number;
  kind:
    | "admit"
    | "place"
    | "pods_ready"
    | "timeout"
    | "requeue"
    | "finish"
    | "hotspot"
    | "xid_fault";
  entity_id: string;
  detail: string;
}

export interface SceneAnnotations {
  recommended_domain_ids: string[];
  blocked_domain_ids: string[];
  highlight_workload_ids: string[];
}

export interface RolloutDTO {
  branch_id: string;
  action_label: string;
  timestep_s: number;
  points: RolloutPoint[];
  scene_annotations: SceneAnnotations;
}

// ──────────────────────────────────────────────────────────────────────────
// Forecast detail (mirror kpis.py QuantileForecast / HazardForecast)
// ──────────────────────────────────────────────────────────────────────────
export interface QuantileForecastDTO {
  channel: KPIChannel;
  horizon_s: number;
  quantiles: Record<string, number>; // "0.5" -> value
}

export interface HazardForecastDTO {
  channel: HazardChannel;
  horizon_s: number;
  probability: number;
  conformal_upper: number; // one-sided (1-alpha) upper bound (doc L)
}

// ──────────────────────────────────────────────────────────────────────────
// Contract 2: POST /api/v1/actions/compare
// ──────────────────────────────────────────────────────────────────────────
export interface CompareRequest {
  state_hash: string;
  candidate_actions: CandidateAction[];
  horizon_s: number;
  include_rollouts: boolean;
}

export interface CandidateResultDTO {
  action_id: string;
  /** Exact-kernel verdict (the safety GATE, never a weighted score). */
  safe: boolean;
  exact_blockers: string[];
  /** Lexicographic rank among safe survivors, or null if filtered/blocked. */
  rank: number | null;
  /** KPI deltas vs the declared baseline action (median of quantile heads). */
  delta_kpis: Partial<Record<KPIChannel, number>>;
  uncertainty: {
    epistemic_score: number;
    ood_score: number;
    abstain: boolean;
    abstain_reason: string | null;
    quantiles: QuantileForecastDTO[];
    hazards: HazardForecastDTO[];
  };
  rollout: RolloutDTO | null;
}

export interface CompareResultDTO {
  state_hash: string;
  horizon_s: number;
  baseline_action_id: string;
  recommended_action_id: string | null;
  abstain: boolean;
  abstain_reason: string | null;
  candidates: CandidateResultDTO[];
  model_version: string;
  policy_version: string;
  credit: string;
}

// ──────────────────────────────────────────────────────────────────────────
// Contract 3: POST /api/v1/actions/why-not
// ──────────────────────────────────────────────────────────────────────────
export interface WhyNotRequest {
  state_hash: string;
  candidate_action: CandidateAction;
}

export interface BlockerDTO {
  group: BlockerGroup;
  detail: string;
  deficit: Record<string, number>;
}

export interface WhyNotDTO {
  action_id: string;
  safe: boolean;
  /** Minimal group-level infeasibility certificate (doc S, IIS analogue). */
  minimal_groups: BlockerGroup[];
  exact_blockers: BlockerDTO[];
  predicted_risks: string[];
  rejected_domain_ids: string[];
  policy_reasons: string[];
  supporting_entities: string[];
}

// ──────────────────────────────────────────────────────────────────────────
// Contract 4: GET /api/v1/replays/{scenario_id}  (+ catalog)
// ──────────────────────────────────────────────────────────────────────────
export interface ReplayIndexEntry {
  scenario_id: string;
  title: string;
  description: string;
  duration_s: number;
}

export interface ReplayIndexDTO {
  scenarios: ReplayIndexEntry[];
}

export interface DecisionWindowDTO {
  t_s: number;
  workload_uid: string;
  label: string;
}

export interface ReplayDTO {
  scenario_id: string;
  title: string;
  initial_state: StateSnapshotDTO;
  /** the predicted branch the model produced at decision time. */
  predicted: RolloutDTO;
  /** the realized history (same shape) used for calibration overlay. */
  realized: RolloutDTO;
  decision_windows: DecisionWindowDTO[];
}

// ──────────────────────────────────────────────────────────────────────────
// Contract 6: GET /api/v1/stream  (SSE event union)
// ──────────────────────────────────────────────────────────────────────────
export type StreamEvent =
  | { type: "snapshot"; snapshot: StateSnapshotDTO }
  | { type: "state_delta"; state_hash: string; patch: Partial<StateSnapshotDTO> }
  | { type: "branch_ready"; state_hash: string; action_id: string }
  | { type: "policy_changed"; policy_version: string }
  | { type: "replay_loaded"; scenario_id: string }
  | { type: "backend_health"; healthy: boolean; detail: string };

// ──────────────────────────────────────────────────────────────────────────
// Live-backend normalization (anti-corruption boundary)
// ──────────────────────────────────────────────────────────────────────────
//  The live FastAPI backend emits a slightly different wire shape than the
//  canonical DTOs above: the planner serialises uncertainty as
//  `{epistemic, ood, bands:{ch:{p05,p50,p95}}, hazards:{ch:{p,conformal_upper}}}`
//  (dicts, not the `{epistemic_score, ood_score, quantiles[], hazards[]}` the
//  panels read), the rollout points carry list-of-row sub-states
//  (`rack_state`/`workload_state`/`queue_state`), per-candidate abstain is
//  derived from the planner risk gate, and `credit` is not on the wire. These
//  adapters map the live payloads into the canonical DTOs so the SAME panel +
//  scene code runs on live data and on the deterministic mock. They are
//  idempotent: a canonical payload (e.g. the mock) passes through unchanged.

const LIVE_CREDIT =
  "GPU-Center World Model · Built by Yash Sharma — linkedin.com/in/yashsharmaa";

const asRecord = (v: unknown): Record<string, unknown> =>
  v && typeof v === "object" ? (v as Record<string, unknown>) : {};
const num = (v: unknown, d = 0): number =>
  typeof v === "number" && Number.isFinite(v) ? v : d;
const str = (v: unknown, d = ""): string => (typeof v === "string" ? v : d);
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const strArr = (v: unknown): string[] =>
  arr(v).filter((x): x is string => typeof x === "string");
const numRecord = (v: unknown): Record<string, number> => {
  const out: Record<string, number> = {};
  for (const [k, val] of Object.entries(asRecord(v))) {
    if (typeof val === "number" && Number.isFinite(val)) out[k] = val;
  }
  return out;
};

// kind -> (entities array key, id field) so we can de-index the live edge lists.
const ENTITY_KEY: Record<string, string> = {
  rack: "racks",
  node: "nodes",
  gpu: "gpus",
  workload: "workloads",
  queue: "queues",
  topology_domain: "topology_domains",
};
const ID_FIELD: Record<string, string> = {
  rack: "rack_id",
  node: "node_uid",
  gpu: "gpu_uuid",
  workload: "workload_uid",
  queue: "name",
  topology_domain: "domain_id",
};

function idsForKind(entities: Record<string, unknown>, kind: string): string[] {
  const key = ENTITY_KEY[kind];
  if (!key) return [];
  const idf = ID_FIELD[kind] ?? "id";
  return arr(entities[key]).map((e) => str(asRecord(e)[idf]));
}

function normalizeRelations(raw: unknown, entities: Record<string, unknown>): RelationDTO[] {
  return arr(raw).map((rel) => {
    const r = asRecord(rel);
    const type = str(r.type);
    const src_kind = str(r.src_kind);
    const dst_kind = str(r.dst_kind);
    // canonical (mock) already carries id-paired `pairs`; pass it through.
    if (Array.isArray(r.pairs)) {
      return { type, src_kind, dst_kind, pairs: r.pairs as Array<[string, string]> };
    }
    // live: `edges` are [srcIndex, dstIndex] into the entity arrays — de-index.
    const srcIds = idsForKind(entities, src_kind);
    const dstIds = idsForKind(entities, dst_kind);
    const pairs: Array<[string, string]> = [];
    for (const e of arr(r.edges)) {
      const pair = arr(e);
      const sid = srcIds[num(pair[0], -1)];
      const did = dstIds[num(pair[1], -1)];
      if (sid != null && did != null) pairs.push([sid, did]);
    }
    return { type, src_kind, dst_kind, pairs };
  });
}

function normalizeSnapshot(raw: unknown): StateSnapshotDTO {
  const r = asRecord(raw);
  const entities = asRecord(r.entities);
  return {
    ...(r as unknown as StateSnapshotDTO),
    relations: normalizeRelations(r.relations, entities),
    credit: str(r.credit) || LIVE_CREDIT,
  };
}

function normalizeQuantiles(u: Record<string, unknown>, horizonS: number): QuantileForecastDTO[] {
  if (Array.isArray(u.quantiles)) return u.quantiles as QuantileForecastDTO[];
  const bands = asRecord(u.bands);
  return Object.entries(bands).map(([channel, b]) => {
    const bb = asRecord(b);
    return {
      channel: channel as KPIChannel,
      horizon_s: horizonS,
      quantiles: { "0.05": num(bb.p05), "0.5": num(bb.p50), "0.95": num(bb.p95) },
    };
  });
}

function normalizeHazards(u: Record<string, unknown>, horizonS: number): HazardForecastDTO[] {
  if (Array.isArray(u.hazards)) return u.hazards as HazardForecastDTO[];
  const h = asRecord(u.hazards);
  return Object.entries(h).map(([channel, v]) => {
    const hv = asRecord(v);
    return {
      channel: channel as HazardChannel,
      horizon_s: horizonS,
      probability: num(hv.probability ?? hv.p),
      conformal_upper: num(hv.conformal_upper ?? hv.p),
    };
  });
}

function normalizeUncertainty(
  rawC: Record<string, unknown>,
  horizonS: number,
  resultAbstainReason: string | null,
): CandidateResultDTO["uncertainty"] {
  const u = asRecord(rawC.uncertainty);
  let abstain: boolean;
  let abstain_reason: string | null;
  if (typeof u.abstain === "boolean") {
    abstain = u.abstain;
    abstain_reason = typeof u.abstain_reason === "string" ? u.abstain_reason : null;
  } else {
    // live: a safe candidate that fails the planner risk gate is abstained.
    abstain = rawC.safe === true && rawC.passes_risk_gate === false;
    abstain_reason = abstain
      ? (resultAbstainReason ?? "exceeds the uncertainty / risk gate")
      : null;
  }
  return {
    epistemic_score: num(u.epistemic_score ?? u.epistemic),
    ood_score: num(u.ood_score ?? u.ood),
    abstain,
    abstain_reason,
    quantiles: normalizeQuantiles(u, horizonS),
    hazards: normalizeHazards(u, horizonS),
  };
}

function normalizeAnnot(v: unknown): SceneAnnotations {
  const a = asRecord(v);
  return {
    recommended_domain_ids: strArr(a.recommended_domain_ids),
    blocked_domain_ids: strArr(a.blocked_domain_ids),
    highlight_workload_ids: strArr(a.highlight_workload_ids),
  };
}

function normalizeRolloutPoint(p: unknown, annot: SceneAnnotations): RolloutPoint {
  const r = asRecord(p);
  // canonical (mock) already keys rack_occupancy by id -> number.
  if (r.rack_occupancy && !Array.isArray(r.rack_occupancy)) return r as unknown as RolloutPoint;
  const rack_occupancy: Record<string, number> = {};
  const rack_hotspot: Record<string, number> = {};
  for (const row of arr(r.rack_state)) {
    const rr = asRecord(row);
    const id = str(rr.rack_id);
    if (id) {
      rack_occupancy[id] = num(rr.occupancy);
      rack_hotspot[id] = num(rr.hotspot_score);
    }
  }
  const workload_placement: Record<string, string> = {};
  const dom = annot.recommended_domain_ids[0];
  if (dom) {
    for (const row of arr(r.workload_state)) {
      const id = str(asRecord(row).workload_uid);
      if (id) workload_placement[id] = dom;
    }
  }
  const queue_depth = Array.isArray(r.queue_state)
    ? r.queue_state.length
    : num(r.queue_depth);
  const u = asRecord(r.uncertainty);
  return {
    t_s: num(r.t_s),
    rack_occupancy,
    rack_hotspot,
    workload_placement,
    queue_depth,
    events: arr(r.events) as RolloutEvent[],
    uncertainty: { epistemic: num(u.epistemic), ood: num(u.ood) },
  };
}

function normalizeRollout(ro: unknown): RolloutDTO | null {
  if (!ro || typeof ro !== "object") return null;
  const r = asRecord(ro);
  const annot = normalizeAnnot(r.scene_annotations);
  return {
    branch_id: str(r.branch_id),
    action_label: str(r.action_label),
    timestep_s: num(r.timestep_s, 60),
    points: arr(r.points).map((p) => normalizeRolloutPoint(p, annot)),
    scene_annotations: annot,
  };
}

function normalizeCandidate(
  raw: unknown,
  horizonS: number,
  resultAbstainReason: string | null,
): CandidateResultDTO {
  const c = asRecord(raw);
  return {
    action_id: str(c.action_id),
    safe: c.safe === true,
    exact_blockers: strArr(c.exact_blockers),
    rank: typeof c.rank === "number" ? c.rank : null,
    delta_kpis: numRecord(c.delta_kpis) as Partial<Record<KPIChannel, number>>,
    uncertainty: normalizeUncertainty(c, horizonS, resultAbstainReason),
    rollout: normalizeRollout(c.rollout),
  };
}

function normalizeCompare(raw: unknown, req: CompareRequest): CompareResultDTO {
  const r = asRecord(raw);
  const horizonS = num(r.horizon_s, req.horizon_s);
  const abstainReason = typeof r.abstain_reason === "string" ? r.abstain_reason : null;
  return {
    state_hash: str(r.state_hash) || req.state_hash,
    horizon_s: horizonS,
    baseline_action_id: str(r.baseline_action_id),
    recommended_action_id:
      typeof r.recommended_action_id === "string" ? r.recommended_action_id : null,
    abstain: r.abstain === true,
    abstain_reason: abstainReason,
    candidates: arr(r.candidates).map((c) =>
      normalizeCandidate(c, horizonS, abstainReason),
    ),
    model_version: str(r.model_version),
    policy_version: str(r.policy_version),
    credit: str(r.credit) || LIVE_CREDIT,
  };
}

function parseBlocker(b: unknown): BlockerDTO {
  if (typeof b === "string") {
    const i = b.indexOf(":");
    const group = (i >= 0 ? b.slice(0, i) : "topology_rule").trim() as BlockerGroup;
    const detail = (i >= 0 ? b.slice(i + 1) : b).trim();
    return { group, detail, deficit: {} };
  }
  const o = asRecord(b);
  return {
    group: (str(o.group) || "topology_rule") as BlockerGroup,
    detail: str(o.detail),
    deficit: numRecord(o.deficit),
  };
}

function normalizeWhyNot(raw: unknown, req: WhyNotRequest): WhyNotDTO {
  const r = asRecord(raw);
  const riskNotes = arr(r.risk_notes).map((n) =>
    typeof n === "string" ? n : str(asRecord(n).note) || JSON.stringify(n),
  );
  return {
    action_id: req.candidate_action.action_id,
    safe: r.safe === true,
    minimal_groups: strArr(r.minimal_groups) as BlockerGroup[],
    exact_blockers: arr(r.exact_blockers).map(parseBlocker),
    predicted_risks: strArr(r.predicted_risks),
    rejected_domain_ids: strArr(r.rejected_domain_ids),
    policy_reasons: r.policy_reasons ? strArr(r.policy_reasons) : riskNotes,
    supporting_entities: strArr(r.supporting_entities),
  };
}

function emptyRollout(): RolloutDTO {
  return {
    branch_id: "",
    action_label: "",
    timestep_s: 60,
    points: [],
    scene_annotations: {
      recommended_domain_ids: [],
      blocked_domain_ids: [],
      highlight_workload_ids: [],
    },
  };
}

function normalizeReplay(raw: unknown): ReplayDTO {
  const r = asRecord(raw);
  return {
    scenario_id: str(r.scenario_id),
    title: str(r.title),
    initial_state: normalizeSnapshot(r.initial_state),
    predicted: normalizeRollout(r.predicted) ?? emptyRollout(),
    realized: normalizeRollout(r.realized) ?? emptyRollout(),
    decision_windows: arr(r.decision_windows) as DecisionWindowDTO[],
  };
}

// ──────────────────────────────────────────────────────────────────────────
// Client implementation
// ──────────────────────────────────────────────────────────────────────────
export interface ClientOptions {
  /** Base URL prefix; "/api/v1" by default (proxied by Vite to the backend). */
  baseUrl?: string;
  /** When true, never hit the network — always use deterministic mock data. */
  forceMock?: boolean;
  /** Per-request timeout in ms before falling back to mock data. */
  timeoutMs?: number;
}

/**
 * Typed client mirroring the six doc-G contracts. Every method resolves even
 * when the backend is offline: it falls back to deterministic mock data so the
 * Decision Studio is recordable and reproducible standalone.
 */
export class WorldModelClient {
  private readonly baseUrl: string;
  private readonly forceMock: boolean;
  private readonly timeoutMs: number;
  /** Live flag flipped by the last network attempt (drives the health badge). */
  public backendOnline = false;

  constructor(opts: ClientOptions = {}) {
    this.baseUrl = opts.baseUrl ?? "/api/v1";
    this.forceMock = opts.forceMock ?? false;
    this.timeoutMs = opts.timeoutMs ?? 2500;
  }

  /** GET /api/v1/state/current — authoritative semantic snapshot for rendering. */
  async getCurrentState(): Promise<StateSnapshotDTO> {
    return this.getOrMock("/state/current", () => buildMockSnapshot(), normalizeSnapshot);
  }

  /** POST /api/v1/actions/compare — rank candidates + return branch rollouts. */
  async compareActions(req: CompareRequest): Promise<CompareResultDTO> {
    return this.postOrMock(
      "/actions/compare",
      req,
      () => buildMockCompare(req),
      (raw) => normalizeCompare(raw, req),
    );
  }

  /** POST /api/v1/actions/why-not — exact blockers + predicted downside. */
  async whyNot(req: WhyNotRequest): Promise<WhyNotDTO> {
    return this.postOrMock(
      "/actions/why-not",
      req,
      () => buildMockWhyNot(req),
      (raw) => normalizeWhyNot(raw, req),
    );
  }

  /** GET /api/v1/replays — replay scenario catalog. */
  async listReplays(): Promise<ReplayIndexDTO> {
    return this.getOrMock("/replays", () => buildMockReplayIndex());
  }

  /** GET /api/v1/replays/{scenario_id} — one replayable scenario. */
  async getReplay(scenarioId: string): Promise<ReplayDTO> {
    return this.getOrMock(
      `/replays/${encodeURIComponent(scenarioId)}`,
      () => buildMockReplay(scenarioId),
      normalizeReplay,
    );
  }

  /**
   * GET /api/v1/stream — open the one-way SSE channel. Returns an EventSource
   * plus an unsubscribe. When the backend is offline the callback is wired to a
   * synthetic delta loop so the live header stays animated in the demo.
   */
  openStream(onEvent: (ev: StreamEvent) => void): () => void {
    if (this.forceMock || typeof EventSource === "undefined") {
      return this.openMockStream(onEvent);
    }
    let closed = false;
    let source: EventSource | null = null;
    try {
      source = new EventSource(`${this.baseUrl}/stream`);
      source.onmessage = (msg) => {
        try {
          onEvent(JSON.parse(msg.data) as StreamEvent);
        } catch {
          /* ignore malformed frames */
        }
      };
      source.onerror = () => {
        // Backend dropped the channel: fall back to the synthetic loop so the
        // recorded demo never shows a dead header.
        if (!closed) {
          source?.close();
          this.backendOnline = false;
          this.openMockStream(onEvent);
        }
      };
    } catch {
      return this.openMockStream(onEvent);
    }
    return () => {
      closed = true;
      source?.close();
    };
  }

  // ── internals ───────────────────────────────────────────────────────────
  // `normalize` maps the LIVE backend wire shape into the canonical DTO; the
  // mock already returns the canonical shape, so it is never normalized.
  private async getOrMock<T>(
    path: string,
    mock: () => T,
    normalize?: (raw: unknown) => T,
  ): Promise<T> {
    if (this.forceMock) return mock();
    try {
      const res = await this.fetchWithTimeout(`${this.baseUrl}${path}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      this.backendOnline = true;
      const raw: unknown = await res.json();
      return normalize ? normalize(raw) : (raw as T);
    } catch {
      this.backendOnline = false;
      return mock();
    }
  }

  private async postOrMock<T>(
    path: string,
    body: unknown,
    mock: () => T,
    normalize?: (raw: unknown) => T,
  ): Promise<T> {
    if (this.forceMock) return mock();
    try {
      const res = await this.fetchWithTimeout(`${this.baseUrl}${path}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      this.backendOnline = true;
      const raw: unknown = await res.json();
      return normalize ? normalize(raw) : (raw as T);
    } catch {
      this.backendOnline = false;
      return mock();
    }
  }

  private fetchWithTimeout(url: string, init?: RequestInit): Promise<Response> {
    const ctrl = new AbortController();
    const id = setTimeout(() => ctrl.abort(), this.timeoutMs);
    return fetch(url, { ...init, signal: ctrl.signal }).finally(() =>
      clearTimeout(id),
    );
  }

  /**
   * Synthetic SSE loop used when the backend is offline. Emits a PARTIAL
   * `state_delta` each tick — only the handful of entities/KPIs that actually
   * change (telemetry on a couple of GPUs, the research queue's live wait, and
   * the pod free-GPU count) — exactly the semantic-diff shape the doc-G transport
   * promises, so the store's merge-by-id path is exercised on real partial data.
   */
  private openMockStream(onEvent: (ev: StreamEvent) => void): () => void {
    this.backendOnline = false;
    let tick = 0;
    onEvent({ type: "backend_health", healthy: false, detail: "mock data" });
    const handle = setInterval(() => {
      tick += 1;
      const delta = buildMockStateDelta(tick);
      onEvent({ type: "state_delta", state_hash: delta.state_hash, patch: delta.patch });
    }, 4000);
    return () => clearInterval(handle);
  }
}

/** Default singleton client wired to the Vite /api proxy. */
export const apiClient = new WorldModelClient();
