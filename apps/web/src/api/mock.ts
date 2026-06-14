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
  const baseFill: Record<string, number> = {
    "rack-a1": 0.18,
    "rack-a2": 0.62,
    "rack-b1": 0.45,
    "rack-b2": 0.84,
  };

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
  const baseFill: Record<string, number> = {
    "rack-a1": 0.18,
    "rack-a2": 0.62,
    "rack-b1": 0.45,
    "rack-b2": 0.84,
  };

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
export function buildMockCompare(req: CompareRequest): CompareResultDTO {
  const horizon = req.horizon_s || 1800;
  const candidates: CandidateResultDTO[] = req.candidate_actions.map((c) => {
    const target =
      (c.params.domain_id_or_set as string[] | undefined)?.[0]?.replace(
        "dom-",
        "",
      ) ?? null;
    const isHold = c.type === "hold";
    const rack = isHold ? null : (target ?? "rack-a1");

    // Branch B (rack-b cluster) is feasible but worse locality + higher risk.
    const worseLocality = rack?.startsWith("rack-b") ?? false;

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

    const epistemic = worseLocality ? 0.31 : 0.16;
    const hotspotProb = worseLocality ? 0.22 : 0.07;

    return {
      action_id: c.action_id,
      safe: true, // all three demo candidates are kernel-certified safe
      exact_blockers: [],
      rank: isHold ? 3 : worseLocality ? 2 : 1,
      delta_kpis: deltaKpis,
      uncertainty: {
        epistemic_score: epistemic,
        ood_score: 0.08,
        abstain: false,
        abstain_reason: null,
        quantiles: [
          {
            channel: "queue_wait_delta_s",
            horizon_s: horizon,
            quantiles: {
              "0.05": (deltaKpis.queue_wait_delta_s ?? 0) - 120,
              "0.5": deltaKpis.queue_wait_delta_s ?? 0,
              "0.95": (deltaKpis.queue_wait_delta_s ?? 0) + 180,
            },
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

  const safeRanked = candidates
    .filter((c) => c.safe && !c.uncertainty.abstain && c.rank !== null)
    .sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99));

  return {
    state_hash: req.state_hash,
    horizon_s: horizon,
    baseline_action_id:
      req.candidate_actions.find((c) => c.type === "hold")?.action_id ??
      req.candidate_actions[0]?.action_id ??
      "baseline",
    recommended_action_id: safeRanked[0]?.action_id ?? null,
    abstain: false,
    abstain_reason: null,
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
  // The mock "blocked" certificate demonstrates the doc-S grammar: a required
  // 16-GPU gang cannot fit in the nearly-full rack-b2 domain (gang atomicity +
  // node capacity deficit), and a degraded node masks part of the block.
  const action = req.candidate_action;
  return {
    action_id: action.action_id,
    safe: false,
    minimal_groups: ["gang_atomicity", "node_capacities"],
    exact_blockers: [
      {
        group: "gang_atomicity",
        detail:
          "required topology policy: 16-GPU gang must fit one rack domain; dom-rack-b2 free=5",
        deficit: { gpu: 11 },
      },
      {
        group: "node_capacities",
        detail: "node rack-b2-n3 is degraded (Xid 79) and unschedulable",
        deficit: { schedulable_gpu: 8 },
      },
    ],
    predicted_risks: [
      "hotspot probability 0.22 (conformal upper 0.28) over 30 min",
      "pods-ready timeout probability 0.09 — cross-rack spillover likely",
    ],
    rejected_domain_ids: ["dom-rack-b2"],
    policy_reasons: ["topology_policy=required forbids cross-domain split"],
    supporting_entities: ["rack-b2", "rack-b2-n3", "tor-rack-b2"],
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
  for (const p of realized.points) {
    if (p.t_s >= 720 && targetRack) {
      p.rack_hotspot[targetRack] = Math.min(
        1,
        (p.rack_hotspot[targetRack] ?? 0) + 0.22,
      );
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
