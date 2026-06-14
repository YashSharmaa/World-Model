// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Deterministic mock backend for the Decision Studio.
//
//  This is NOT the world model — it is a faithful *shape* of the six doc-G
//  contracts so the front-end is recordable and reproducible when the Python
//  serving layer is absent. It encodes the demo pod (2-4 racks) and the golden
//  path from doc G: one pending 16-GPU workload, three safe actions (hold,
//  admit to domain A, admit to domain B), and counterfactual rollouts that make
//  the world-model story legible — same state, different action, different
//  future.
//
//  The numbers are illustrative but internally consistent (occupancy never
//  exceeds capacity; hotspot risk grows monotonically when a rack fills; hold
//  releases capacity as a short job finishes). All randomness is seeded by a
//  tiny deterministic PRNG so recordings are byte-stable across runs.

import type {
  CompareRequest,
  CompareResultDTO,
  CandidateResultDTO,
  GPUDTO,
  NodeDTO,
  QueueDTO,
  RackDTO,
  ReplayDTO,
  ReplayIndexDTO,
  RolloutDTO,
  RolloutPoint,
  StateSnapshotDTO,
  TopologyDomainDTO,
  WhyNotDTO,
  WhyNotRequest,
  WorkloadDTO,
  KPIChannel,
} from "./client";
import { HORIZONS_S } from "./client";

export const MODEL_VERSION = "wm-v1.0.0-mock";
export const POLICY_VERSION = "policy-default-v1";
export const CREDIT =
  "GPU-Center World Model · Done by Yash Sharma — https://www.linkedin.com/in/yashsharmaa/";

/** Deterministic 4-rack demo pod constants. */
const RACK_IDS = ["rack-a1", "rack-a2", "rack-b1", "rack-b2"] as const;
const NODES_PER_RACK = 4;
const GPUS_PER_NODE = 8; // 32 GPUs/rack, 128 GPU pod
const FOCUS_WORKLOAD = "wl-train-7b";

/** Mulberry32 — small deterministic PRNG (seeded). */
function prng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function rackToDomain(rackId: string): string {
  return `dom-${rackId}`;
}

function domainToRack(domainId: string): string {
  return domainId.replace(/^dom-/, "");
}

/**
 * Baseline occupancy fraction per rack at tick 0. SINGLE source of truth shared
 * by the snapshot, the rollout builder, and the compare feasibility math so the
 * mock backend is internally consistent (a domain's reported `free_gpu` is the
 * same number the why-not deficit is computed against).
 */
const BASE_FILL: Record<string, number> = {
  "rack-a1": 0.18,
  "rack-a2": 0.62,
  "rack-b1": 0.45,
  "rack-b2": 0.84,
};

/** GPUs free in a rack domain at tick 0 (mirrors buildMockSnapshot). */
function rackFreeGpu(rackId: string): number {
  const cap = NODES_PER_RACK * GPUS_PER_NODE;
  const fill = BASE_FILL[rackId] ?? 0.5;
  return Math.round((1 - fill) * cap);
}

/**
 * Schedulable (gang-eligible) GPUs free in a rack domain. Mirrors the snapshot's
 * degraded-node masking: rack-b2's node n3 is degraded (Xid 79) and
 * unschedulable, so its 8 GPUs cannot host any part of a required gang.
 */
function rackSchedulableFreeGpu(rackId: string): number {
  const free = rackFreeGpu(rackId);
  const maskedByDegradedNode = rackId === "rack-b2" ? GPUS_PER_NODE : 0;
  return Math.max(0, free - maskedByDegradedNode);
}

// ──────────────────────────────────────────────────────────────────────────
// GET /state/current
// ──────────────────────────────────────────────────────────────────────────
export function buildMockSnapshot(tick = 0): StateSnapshotDTO {
  const rnd = prng(1337 + tick);
  const racks: RackDTO[] = [];
  const nodes: NodeDTO[] = [];
  const gpus: GPUDTO[] = [];
  const domains: TopologyDomainDTO[] = [];

  // Baseline occupancy: rack-a1 lightly loaded (clean block), a2 busy,
  // b1 half, b2 nearly full (tight headroom). Drives the golden-path story.
  const baseFill = BASE_FILL;

  for (const rackId of RACK_IDS) {
    const fill = Math.min(0.95, baseFill[rackId] + 0.03 * Math.sin(tick / 3));
    const hotspot = Math.max(0, fill - 0.55) * 1.4;
    const block = rackId.startsWith("rack-a") ? "block-a" : "block-b";
    racks.push({
      rack_id: rackId,
      block_id: block,
      power_domain_id: `pdu-${block}`,
      cooling_zone_id: `cool-${block}`,
      designed_gpu_cap: NODES_PER_RACK * GPUS_PER_NODE,
      rack_power_kw: 18 + fill * 22,
      rack_headroom_kw: Math.max(0.5, 14 - fill * 13),
      inferred_thermal_margin_c: Math.max(1, 12 - fill * 11),
      hotspot_score: Math.min(1, hotspot),
    });
    domains.push({
      domain_id: rackToDomain(rackId),
      level: "rack",
      parent_domain_id: block,
      free_gpu: Math.round((1 - fill) * NODES_PER_RACK * GPUS_PER_NODE),
      headroom_score: 1 - fill,
      hotspot_score: Math.min(1, hotspot),
    });

    for (let n = 0; n < NODES_PER_RACK; n++) {
      const nodeUid = `${rackId}-n${n}`;
      const degraded = rackId === "rack-b2" && n === 3;
      nodes.push({
        node_uid: nodeUid,
        rack_id: rackId,
        block_id: block,
        switch_id: `tor-${rackId}`,
        flavor_id: "h100-80g",
        gpu_cap: GPUS_PER_NODE,
        ready: !degraded,
        schedulable: !degraded,
        running_pods: Math.round(fill * GPUS_PER_NODE),
        node_health: degraded ? "degraded" : "ready",
      });
      for (let g = 0; g < GPUS_PER_NODE; g++) {
        const slotIdx = n * GPUS_PER_NODE + g;
        const allocated = slotIdx < fill * NODES_PER_RACK * GPUS_PER_NODE;
        const faulty = degraded && g === 0;
        gpus.push({
          gpu_uuid: `${nodeUid}-g${g}`,
          node_uid: nodeUid,
          allocated_to: allocated ? `wl-bg-${rackId}` : null,
          util: allocated ? 0.6 + rnd() * 0.35 : 0,
          temp_c: 38 + fill * 30 + (faulty ? 14 : 0),
          power_w: allocated ? 480 + rnd() * 180 : 60,
          xid_code: faulty ? 79 : 0,
          health_state: faulty ? "intermittently_faulty" : "healthy",
        });
      }
    }
  }

  const workloads: WorkloadDTO[] = [
    {
      workload_uid: FOCUS_WORKLOAD,
      tenant_id: "tenant-research",
      local_queue_id: "lq-research",
      cluster_queue_id: "cq-shared",
      priority: 100,
      topology_policy: "required",
      comm_class: "allreduce_heavy",
      status: "pending",
      gpu_req: 16,
      age_s: 540 + tick * 12,
      est_remaining_s: 5400,
      ready_fraction: 0,
      admitted_domain_id: null,
    },
    {
      workload_uid: "wl-infer-batch",
      tenant_id: "tenant-prod",
      local_queue_id: "lq-prod",
      cluster_queue_id: "cq-shared",
      priority: 60,
      topology_policy: "preferred",
      comm_class: "light",
      status: "pending",
      gpu_req: 8,
      age_s: 120,
      est_remaining_s: 1800,
      ready_fraction: 0,
      admitted_domain_id: null,
    },
    {
      workload_uid: "wl-short-eval",
      tenant_id: "tenant-research",
      local_queue_id: "lq-research",
      cluster_queue_id: "cq-shared",
      priority: 40,
      topology_policy: "unconstrained",
      comm_class: "none",
      status: "running",
      gpu_req: 8,
      age_s: 2400,
      est_remaining_s: 480, // finishes soon -> "hold" frees a clean block
      ready_fraction: 1,
      admitted_domain_id: rackToDomain("rack-a1"),
    },
  ];

  const queues: QueueDTO[] = [
    {
      name: "lq-research",
      tenant_id: "tenant-research",
      cluster_queue_ref: "cq-shared",
      pending_ct: 1,
      pending_gpu: 16,
      wait_time_p50_s: 420,
      wait_time_p95_s: 1260,
    },
    {
      name: "lq-prod",
      tenant_id: "tenant-prod",
      cluster_queue_ref: "cq-shared",
      pending_ct: 1,
      pending_gpu: 8,
      wait_time_p50_s: 90,
      wait_time_p95_s: 300,
    },
  ];

  const totalCap = RACK_IDS.length * NODES_PER_RACK * GPUS_PER_NODE;
  const usedGpu = gpus.filter((g) => g.allocated_to).length;

  return {
    state_hash: `0x${(0xabc123 + tick * 0x1111).toString(16)}`,
    generated_at: new Date(Date.UTC(2026, 5, 14, 9, 0, tick % 60)).toISOString(),
    model_version: MODEL_VERSION,
    policy_version: POLICY_VERSION,
    horizon_options_s: [...HORIZONS_S],
    entities: { racks, nodes, gpus, workloads, queues, topology_domains: domains },
    relations: [
      {
        type: "POWERED_BY",
        src_kind: "rack",
        dst_kind: "power_domain",
        pairs: racks.map((r) => [r.rack_id, r.power_domain_id]),
      },
      {
        type: "CONNECTED_TO",
        src_kind: "node",
        dst_kind: "switch",
        pairs: nodes.map((n) => [n.node_uid, n.switch_id]),
      },
      // ToR-to-ToR spine fabric: every rack's switch links to the next, plus a
      // cross-block link, so the renderer can build inter-rack link geometry from
      // the relation array (the topology QUALITY the demo cares about) rather
      // than an index heuristic. switch_id mirrors `tor-${rack_id}`.
      {
        type: "LINKED_TO",
        src_kind: "switch",
        dst_kind: "switch",
        pairs: RACK_IDS.flatMap((rid, i): Array<[string, string]> => {
          const next = RACK_IDS[(i + 1) % RACK_IDS.length];
          return next !== rid ? [[`tor-${rid}`, `tor-${next}`]] : [];
        }),
      },
    ],
    current_kpis: {
      queue_wait_p95_s: 1260,
      admitted_gpu_hours: 142.5,
      topology_locality: 0.78,
      violation_probability: 0.04,
      pending_workloads: workloads.filter((w) => w.status === "pending").length,
      free_gpu: totalCap - usedGpu,
    },
    freshness: {
      scheduler_state_age_s: 12,
      placement_state_age_s: 4,
      gpu_telemetry_age_s: 8,
      worst_source_health: "observed",
    },
    credit: CREDIT,
  };
}

// ──────────────────────────────────────────────────────────────────────────
// SSE state_delta (partial merge-by-id payload)
// ──────────────────────────────────────────────────────────────────────────
/**
 * Build a PARTIAL `state_delta` for the offline SSE loop: only the entities and
 * KPIs that actually move tick-to-tick are included, so the frontend store can
 * exercise a real merge-by-id (doc G transport: "return only changed entities").
 *
 * What changes each tick:
 *   - the two head GPUs of rack-a1 breathe util / temp / power (live telemetry)
 *   - the research queue's live p50/p95 wait times drift
 *   - the pod-level free-GPU KPI + freshness age tick
 * Everything else (the other ~126 GPUs, every rack/node/domain) is OMITTED and
 * must be preserved by the store's by-id merge.
 */
export function buildMockStateDelta(tick: number): {
  state_hash: string;
  patch: Partial<StateSnapshotDTO>;
} {
  const rnd = prng(7919 + tick);
  const phase = Math.sin(tick / 2);

  // A couple of live GPU telemetry updates on rack-a1's first node.
  const changedGpus: GPUDTO[] = [0, 1].map((g) => ({
    gpu_uuid: `rack-a1-n0-g${g}`,
    node_uid: "rack-a1-n0",
    allocated_to: "wl-bg-rack-a1",
    util: 0.55 + 0.4 * rnd(),
    temp_c: 46 + 6 * phase + 4 * rnd(),
    power_w: 470 + 160 * rnd(),
    xid_code: 0,
    health_state: "healthy",
  }));

  // The research queue's live wait times drift around the snapshot values.
  const changedQueue: QueueDTO = {
    name: "lq-research",
    tenant_id: "tenant-research",
    cluster_queue_ref: "cq-shared",
    pending_ct: 1,
    pending_gpu: 16,
    wait_time_p50_s: Math.max(60, Math.round(420 + 90 * phase)),
    wait_time_p95_s: Math.max(180, Math.round(1260 + 180 * phase)),
  };

  const base = buildMockSnapshot(0);
  const usedGpu = base.entities.gpus.filter((g) => g.allocated_to).length;

  return {
    state_hash: `0x${(0xabc123 + tick * 0x1111).toString(16)}`,
    patch: {
      state_hash: `0x${(0xabc123 + tick * 0x1111).toString(16)}`,
      generated_at: new Date(Date.UTC(2026, 5, 14, 9, 0, tick % 60)).toISOString(),
      entities: {
        // Only the changed families are present; the store merges them BY ID and
        // leaves every other rack / node / GPU / domain untouched.
        gpus: changedGpus,
        queues: [changedQueue],
      } as StateSnapshotDTO["entities"],
      current_kpis: {
        ...base.current_kpis,
        queue_wait_p95_s: changedQueue.wait_time_p95_s,
        free_gpu: base.entities.racks.length * NODES_PER_RACK * GPUS_PER_NODE - usedGpu,
      },
      freshness: {
        scheduler_state_age_s: 8 + (tick % 5),
        placement_state_age_s: 3 + (tick % 3),
        gpu_telemetry_age_s: 4 + (tick % 4),
        worst_source_health: "observed",
      },
    },
  };
}

// ──────────────────────────────────────────────────────────────────────────
// rollout builder
// ──────────────────────────────────────────────────────────────────────────
function buildRollout(
  branchId: string,
  label: string,
  horizonS: number,
  opts: {
    targetRack: string | null; // null => hold
    holdFreesRack?: string; // rack that frees up when a short job finishes
  },
): RolloutDTO {
  const step = 60;
  const points: RolloutPoint[] = [];
  const baseFill = BASE_FILL;

  for (let t = 0; t <= horizonS; t += step) {
    const frac = t / horizonS;
    const occ: Record<string, number> = {};
    const hot: Record<string, number> = {};
    const placement: Record<string, string> = {};
    const events: RolloutPoint["events"] = [];

    for (const r of RACK_IDS) {
      let f = baseFill[r];
      if (opts.targetRack === r) {
        // workload commits over the first ~3 min, then occupancy holds.
        f = Math.min(0.97, f + Math.min(1, t / 180) * (16 / 32));
        if (t === 60)
          events.push({
            t_s: t,
            kind: "admit",
            entity_id: FOCUS_WORKLOAD,
            detail: `admitted into ${r}`,
          });
        if (t === 180)
          events.push({
            t_s: t,
            kind: "pods_ready",
            entity_id: FOCUS_WORKLOAD,
            detail: "all pods ready",
          });
      }
      if (opts.holdFreesRack === r) {
        // short eval job finishes ~8 min in, releasing a clean block.
        if (t >= 480) f = Math.max(0.05, baseFill[r] - 0.25);
        if (t === 480)
          events.push({
            t_s: t,
            kind: "finish",
            entity_id: "wl-short-eval",
            detail: `released a clean block in ${r}`,
          });
      }
      occ[r] = f;
      hot[r] = Math.min(1, Math.max(0, f - 0.6) * 1.6 * (0.6 + 0.4 * frac));
      if (opts.targetRack === r && f > 0.85 && t > horizonS * 0.5)
        events.push({
          t_s: t,
          kind: "hotspot",
          entity_id: r,
          detail: "thermal headroom tightening",
        });
    }

    if (opts.targetRack)
      placement[FOCUS_WORKLOAD] = rackToDomain(opts.targetRack);

    points.push({
      t_s: t,
      rack_occupancy: occ,
      rack_hotspot: hot,
      workload_placement: placement,
      queue_depth: opts.targetRack ? 1 : 2,
      events,
      uncertainty: {
        epistemic: 0.12 + 0.18 * frac,
        ood: 0.05 + 0.05 * frac,
      },
    });
  }

  return {
    branch_id: branchId,
    action_label: label,
    timestep_s: step,
    points,
    scene_annotations: {
      recommended_domain_ids: opts.targetRack
        ? [rackToDomain(opts.targetRack)]
        : opts.holdFreesRack
          ? [rackToDomain(opts.holdFreesRack)]
          : [],
      blocked_domain_ids: [],
      highlight_workload_ids: [FOCUS_WORKLOAD],
    },
  };
}

// ──────────────────────────────────────────────────────────────────────────
// POST /actions/compare
// ──────────────────────────────────────────────────────────────────────────
/**
 * Abstain thresholds (mirror docs G / L). When a candidate's calibrated
 * epistemic or OOD score crosses these, the forecast is treated as too
 * uncertain to recommend on and the candidate ABSTAINS (banded chart + abstain
 * chip), rather than being silently down-weighted.
 */
const EPISTEMIC_ABSTAIN = 0.45;
const OOD_ABSTAIN = 0.35;

/**
 * Build a five-point quantile fan { p05, p10, p50, p90, p95 } around a median.
 * `spread` is the half-width of the p05-p95 ribbon; the p10/p90 heads sit at
 * ~0.55 of that half-width so the ribbon reads as a calibrated fan, not a box.
 * The fan is the data the recommendation band chart draws (doc G uncertainty
 * display: "confidence bands in charts").
 */
function spreadQuantiles(
  median: number,
  spread: number,
): Record<string, number> {
  const half = Math.abs(spread);
  return {
    "0.05": median - half,
    "0.1": median - half * 0.55,
    "0.5": median,
    "0.9": median + half * 0.55,
    "0.95": median + half,
  };
}

export function buildMockCompare(req: CompareRequest): CompareResultDTO {
  const horizon = req.horizon_s || 1800;

  // Resolve the focused workload's required gang size from the deterministic
  // tick-0 snapshot (kept consistent with what the renderer fetched), so the
  // gang-atomicity feasibility test below uses the SAME numbers the UI shows.
  const snap0 = buildMockSnapshot(0);
  const reqWorkloadUid =
    (req.candidate_actions[0]?.params.workload_uid as string | undefined) ??
    FOCUS_WORKLOAD;
  const gpuReq =
    snap0.entities.workloads.find((w) => w.workload_uid === reqWorkloadUid)
      ?.gpu_req ?? 16;

  const candidates: CandidateResultDTO[] = req.candidate_actions.map((c) => {
    const domainId = (c.params.domain_id_or_set as string[] | undefined)?.[0] ?? null;
    const rack = c.type === "hold" || !domainId ? null : domainToRack(domainId);
    const isHold = c.type === "hold";
    const domainPolicy = (c.params.domain_policy as string | undefined) ?? "required";

    // Exact-kernel feasibility (doc S gang_atomicity + health_schedulability):
    // a required/preferred gang must fit one rack domain's SCHEDULABLE free GPUs.
    const schedFree = rack != null ? rackSchedulableFreeGpu(rack) : Infinity;
    const gangConstrained = domainPolicy === "required" || domainPolicy === "preferred";
    const blocked = !isHold && gangConstrained && schedFree < gpuReq;

    // Branch into rack-b cluster is feasible-but-worse-locality; the blocked
    // branch lands in the tightest rack and exercises the why-not path.
    const worseLocality = rack?.startsWith("rack-b") ?? false;

    if (blocked) {
      // BLOCKED candidate: no rank, no forward motion, exact blockers populated.
      const epistemic = 0.31;
      const oodScore = 0.12;
      return {
        action_id: c.action_id,
        safe: false,
        exact_blockers: ["gang_atomicity", "node_capacities", "health_schedulability"],
        rank: null,
        delta_kpis: {
          // Deltas are still reported for context but the gate already rejected.
          queue_wait_delta_s: -420,
          admitted_gpu_hours_delta: 24,
          topology_fragmentation_delta: 0.22,
          rack_thermal_headroom_c: -5.5,
          completion_time_delta_s: 540,
          cross_rack_rate: 0.41,
        },
        uncertainty: {
          epistemic_score: epistemic,
          ood_score: oodScore,
          abstain: false,
          abstain_reason: null,
          quantiles: [
            // Five-point quantile fan (p05/p10/p50/p90/p95) so the recommendation
            // band chart can draw a real p05-p95 ribbon around the p50 median.
            {
              channel: "queue_wait_delta_s",
              horizon_s: horizon,
              quantiles: spreadQuantiles(-420, 160),
            },
            {
              channel: "rack_thermal_headroom_c",
              horizon_s: horizon,
              quantiles: spreadQuantiles(-5.5, 2.2),
            },
          ],
          hazards: [
            {
              channel: "hotspot",
              horizon_s: horizon,
              probability: 0.22,
              conformal_upper: 0.28,
            },
            {
              channel: "pods_ready_timeout",
              horizon_s: horizon,
              probability: 0.09,
              conformal_upper: 0.14,
            },
          ],
        },
        // Blocked branches carry NO rollout: motion stops at the blocker and the
        // branch gets a lock badge instead of a soft-failure animation (doc G).
        rollout: req.include_rollouts
          ? {
              branch_id: c.action_id,
              action_label: `Admit → ${rack} (blocked)`,
              timestep_s: 60,
              points: [],
              scene_annotations: {
                recommended_domain_ids: [],
                blocked_domain_ids: [domainId!],
                highlight_workload_ids: [reqWorkloadUid],
              },
            }
          : null,
      };
    }

    const deltaKpis: Partial<Record<KPIChannel, number>> = isHold
      ? {
          queue_wait_delta_s: 480,
          admitted_gpu_hours_delta: 0,
          topology_fragmentation_delta: -0.12,
          completion_time_delta_s: 480,
          cross_rack_rate: 0,
        }
      : {
          queue_wait_delta_s: -420,
          admitted_gpu_hours_delta: 24,
          topology_fragmentation_delta: worseLocality ? 0.18 : 0.02,
          rack_thermal_headroom_c: worseLocality ? -4.5 : -1.5,
          completion_time_delta_s: worseLocality ? 360 : -120,
          cross_rack_rate: worseLocality ? 0.34 : 0.02,
        };

    // Fit margin: how comfortably the gang fits the chosen feasible domain.
    // A domain that BARELY fits (free_gpu just above gpu_req) drives the forecast
    // toward saturation, where the learned model extrapolates out-of-distribution
    // — so OOD/epistemic rise as the margin shrinks (doc L calibrated abstain).
    const freeGpu = rack != null ? rackFreeGpu(rack) : Infinity;
    const fitMargin = isHold
      ? 1
      : Math.max(0, Math.min(1, (freeGpu - gpuReq) / Math.max(1, gpuReq)));
    const tightness = 1 - fitMargin; // 0 = roomy, 1 = exactly at capacity

    const epistemic = isHold
      ? 0.14
      : (worseLocality ? 0.28 : 0.14) + 0.22 * tightness;
    const oodScore = isHold
      ? 0.05
      : (worseLocality ? 0.1 : 0.06) + 0.4 * tightness;
    const hotspotProb = worseLocality ? 0.22 : 0.07;

    // Abstain GATE: if calibrated epistemic / OOD uncertainty crosses the
    // thresholds, this candidate abstains (doc L) — banded chart + abstain chip,
    // and it cannot win the recommendation.
    const abstain =
      epistemic >= EPISTEMIC_ABSTAIN || oodScore >= OOD_ABSTAIN;
    const abstainReason = abstain
      ? `epistemic ${epistemic.toFixed(2)} / ood ${oodScore.toFixed(2)} exceed abstain threshold`
      : null;

    return {
      action_id: c.action_id,
      safe: true, // kernel-certified feasible
      exact_blockers: [],
      rank: abstain ? null : isHold ? 3 : worseLocality ? 2 : 1,
      delta_kpis: deltaKpis,
      uncertainty: {
        epistemic_score: epistemic,
        ood_score: oodScore,
        abstain,
        abstain_reason: abstainReason,
        quantiles: [
          // Quantile spread widens with epistemic uncertainty so the band chart
          // visibly fans out for the tighter / worse-locality candidates.
          {
            channel: "queue_wait_delta_s",
            horizon_s: horizon,
            quantiles: spreadQuantiles(
              deltaKpis.queue_wait_delta_s ?? 0,
              150 + 600 * epistemic,
            ),
          },
          {
            channel: "rack_thermal_headroom_c",
            horizon_s: horizon,
            quantiles: spreadQuantiles(
              deltaKpis.rack_thermal_headroom_c ?? 0,
              1.5 + 5 * epistemic,
            ),
          },
        ],
        hazards: [
          {
            channel: "hotspot",
            horizon_s: horizon,
            probability: hotspotProb,
            conformal_upper: hotspotProb + 0.06,
          },
          {
            channel: "pods_ready_timeout",
            horizon_s: horizon,
            probability: worseLocality ? 0.09 : 0.03,
            conformal_upper: worseLocality ? 0.14 : 0.06,
          },
        ],
      },
      rollout: req.include_rollouts
        ? buildRollout(
            c.action_id,
            c.type === "hold" ? "Hold" : `Admit → ${rack}`,
            horizon,
            isHold
              ? { targetRack: null, holdFreesRack: "rack-a1" }
              : { targetRack: rack },
          )
        : null,
    };
  });

  // Recommendation: best lexicographic rank among candidates that are SAFE,
  // not abstaining, and ranked. Blocked / abstaining candidates are excluded by
  // the gate, never by a soft penalty.
  const safeRanked = candidates
    .filter((c) => c.safe && !c.uncertainty.abstain && c.rank !== null)
    .sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99));

  // Whole-compare abstain: if no safe ranked survivor remains (all blocked or
  // abstaining), the compare itself abstains rather than inventing a winner.
  const anyAbstain = candidates.some((c) => c.uncertainty.abstain);
  const compareAbstain = safeRanked.length === 0;
  const abstainReason = compareAbstain
    ? anyAbstain
      ? "all feasible candidates exceed the uncertainty abstain threshold"
      : "no feasible candidate survived the exact kernel"
    : null;

  return {
    state_hash: req.state_hash,
    horizon_s: horizon,
    baseline_action_id:
      req.candidate_actions.find((c) => c.type === "hold")?.action_id ??
      req.candidate_actions[0]?.action_id ??
      "baseline",
    recommended_action_id: safeRanked[0]?.action_id ?? null,
    abstain: compareAbstain,
    abstain_reason: abstainReason,
    candidates,
    model_version: MODEL_VERSION,
    policy_version: POLICY_VERSION,
    credit: CREDIT,
  };
}

// ──────────────────────────────────────────────────────────────────────────
// POST /actions/why-not
// ──────────────────────────────────────────────────────────────────────────
export function buildMockWhyNot(req: WhyNotRequest): WhyNotDTO {
  // The "blocked" certificate demonstrates the doc-S grammar: a required gang
  // cannot fit one rack domain (gang atomicity + node-capacity deficit) and a
  // degraded node masks part of the block. The deficits are computed from the
  // ACTUAL submitted action (its target domain + the workload's gang size) and
  // the same deterministic capacity model the snapshot/compare use, so the card
  // is consistent with the rest of the demo rather than hardcoded.
  const action = req.candidate_action;
  const domainId =
    (action.params.domain_id_or_set as string[] | undefined)?.[0] ?? "dom-rack-b2";
  const rack = domainToRack(domainId);
  const policy = (action.params.domain_policy as string | undefined) ?? "required";

  const snap0 = buildMockSnapshot(0);
  const wlUid = (action.params.workload_uid as string | undefined) ?? FOCUS_WORKLOAD;
  const gpuReq =
    snap0.entities.workloads.find((w) => w.workload_uid === wlUid)?.gpu_req ?? 16;

  const freeGpu = rackFreeGpu(rack);
  const schedFree = rackSchedulableFreeGpu(rack);
  const gangDeficit = Math.max(1, gpuReq - freeGpu);
  const degradedNode = rack === "rack-b2";

  const exactBlockers: WhyNotDTO["exact_blockers"] = [
    {
      group: "gang_atomicity",
      detail: `topology_policy=${policy}: ${gpuReq}-GPU gang must fit one rack domain; ${domainId} free=${freeGpu}`,
      deficit: { gpu: gangDeficit },
    },
    {
      group: "node_capacities",
      detail: `${domainId} schedulable free=${schedFree} < required ${gpuReq}`,
      deficit: { schedulable_gpu: Math.max(1, gpuReq - schedFree) },
    },
  ];
  const minimalGroups: WhyNotDTO["minimal_groups"] = ["gang_atomicity", "node_capacities"];
  if (degradedNode) {
    exactBlockers.push({
      group: "health_schedulability",
      detail: `node ${rack}-n3 is degraded (Xid 79) and unschedulable`,
      deficit: { schedulable_gpu: GPUS_PER_NODE },
    });
    minimalGroups.push("health_schedulability");
  }

  return {
    action_id: action.action_id,
    safe: false,
    minimal_groups: minimalGroups,
    exact_blockers: exactBlockers,
    predicted_risks: [
      "hotspot probability 0.22 (conformal upper 0.28) over 30 min",
      "pods-ready timeout probability 0.09 — cross-rack spillover likely",
    ],
    rejected_domain_ids: [domainId],
    policy_reasons: [`topology_policy=${policy} forbids cross-domain split`],
    supporting_entities: degradedNode
      ? [rack, `${rack}-n3`, `tor-${rack}`]
      : [rack, `tor-${rack}`],
  };
}

// ──────────────────────────────────────────────────────────────────────────
// GET /replays  +  /replays/{id}
// ──────────────────────────────────────────────────────────────────────────
export function buildMockReplayIndex(): ReplayIndexDTO {
  return {
    scenarios: [
      {
        scenario_id: "golden-path-admit-a",
        title: "Golden path · Admit to clean block",
        description:
          "Predicted vs realized for admitting the 16-GPU job into rack-a1.",
        duration_s: 1800,
      },
      {
        scenario_id: "hold-then-place",
        title: "Hold one window, then place",
        description:
          "Short job finishes; a cleaner topology block opens; calibration overlay.",
        duration_s: 3600,
      },
    ],
  };
}

export function buildMockReplay(scenarioId: string): ReplayDTO {
  const horizon = scenarioId === "hold-then-place" ? 3600 : 1800;
  const targetRack = scenarioId === "hold-then-place" ? null : "rack-a1";
  const predicted = buildRollout(
    "predicted",
    "Predicted branch",
    horizon,
    scenarioId === "hold-then-place"
      ? { targetRack: null, holdFreesRack: "rack-a1" }
      : { targetRack },
  );
  // Realized diverges: a hotspot emerges ~12 min in that the predictor under-
  // called, so the realized hotspot curve sits above predicted after t=720.
  const realized = buildRollout(
    "realized",
    "Realized history",
    horizon,
    scenarioId === "hold-then-place"
      ? { targetRack: null, holdFreesRack: "rack-a1" }
      : { targetRack },
  );
  // Realized diverges from prediction at t=720 across THREE channels so the
  // replay error badges (queue wait, hotspot, headroom) each carry real signal:
  //   - hotspot the predictor UNDER-called (realized curve sits above predicted)
  //   - occupancy realized slightly higher (extra spillover pods landed)
  //   - queue_depth stays one deeper than predicted (a job requeued, didn't drain)
  const divergeRack = targetRack ?? "rack-a1";
  for (const p of realized.points) {
    if (p.t_s >= 720) {
      p.rack_hotspot[divergeRack] = Math.min(
        1,
        (p.rack_hotspot[divergeRack] ?? 0) + 0.22,
      );
      p.rack_occupancy[divergeRack] = Math.min(
        0.99,
        (p.rack_occupancy[divergeRack] ?? 0) + 0.06,
      );
      p.queue_depth = p.queue_depth + 1;
      if (p.t_s === 720)
        p.events.push({
          t_s: p.t_s,
          kind: "hotspot",
          entity_id: divergeRack,
          detail: "realized hotspot exceeded prediction (under-called)",
        });
    }
  }

  return {
    scenario_id: scenarioId,
    title:
      scenarioId === "hold-then-place"
        ? "Hold one window, then place"
        : "Golden path · Admit to clean block",
    initial_state: buildMockSnapshot(0),
    predicted,
    realized,
    decision_windows: [
      { t_s: 0, workload_uid: FOCUS_WORKLOAD, label: "Admission decision" },
      { t_s: 720, workload_uid: FOCUS_WORKLOAD, label: "Divergence point" },
    ],
  };
}
