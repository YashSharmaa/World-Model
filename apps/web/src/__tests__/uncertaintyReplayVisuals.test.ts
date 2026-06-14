// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Scoped tests for FE-UNCERTAINTY-REPLAY-VISUALS (no framework in this lane;
//  same self-contained assert() harness as compareBlockedGoldenPath.test.ts).
//  Build verifies with `tsc --noEmit`; this file exercises the PURE logic that
//  backs the new visuals so the numbers stay correct independent of the DOM:
//
//   1. The compare mock returns a five-point p05/p10/p50/p90/p95 quantile fan on
//      multiple channels, ordered, so the recommendation band chart can draw a
//      real ribbon.
//   2. branchUncertainty() combines epistemic + ood + relative quantile spread
//      into a bounded 0..1 ring driver, and grows with a tighter forecast.
//   3. baselineOccupancyByRack() reads grounded occupancy straight from the
//      snapshot GPUs (the "now" the grounded pod must hold in both modes), and a
//      rollout's predicted occupancy sits ABOVE that baseline (ghost overlay, not
//      a promotion of the predicted future to solid).
//   4. The replay mock's realized trace diverges from predicted at t=720 across
//      the hotspot / occupancy / queue channels so the calibration error badges
//      and the timeline divergence split have real signal.

import { buildMockCompare, buildMockSnapshot, buildMockReplay } from "../api/mock";
import type { CompareRequest } from "../api/client";
import { branchUncertainty } from "../scene/branches";

let failures = 0;
function assert(cond: boolean, msg: string): void {
  if (!cond) {
    failures += 1;
    // eslint-disable-next-line no-console
    console.error(`  ✗ ${msg}`);
  } else {
    // eslint-disable-next-line no-console
    console.log(`  ✓ ${msg}`);
  }
}

function compareReq(feasible: string, blocked: string): CompareRequest {
  return {
    state_hash: "0xtest",
    horizon_s: 1800,
    include_rollouts: true,
    candidate_actions: [
      { action_id: "hold", type: "hold", params: { workload_uid: "wl-train-7b" } },
      {
        action_id: "admit-a",
        type: "admit_and_place",
        params: {
          workload_uid: "wl-train-7b",
          domain_policy: "required",
          domain_id_or_set: [feasible],
        },
      },
      {
        action_id: "admit-b",
        type: "admit_and_place",
        params: {
          workload_uid: "wl-train-7b",
          domain_policy: "required",
          domain_id_or_set: [blocked],
        },
      },
    ],
  };
}

/** Mirror of scene/racks baselineOccupancyByRack but driven by gpu_uuid prefix
 *  (so the test does not need to build a three.js InstancedMesh / RacksResult). */
function baselineFromGpus(
  gpus: { gpu_uuid: string; allocated_to: string | null; node_uid: string }[],
  nodeToRack: Map<string, string>,
): Record<string, number> {
  const alloc: Record<string, number> = {};
  const total: Record<string, number> = {};
  for (const g of gpus) {
    const rack = nodeToRack.get(g.node_uid);
    if (!rack) continue;
    total[rack] = (total[rack] ?? 0) + 1;
    if (g.allocated_to != null) alloc[rack] = (alloc[rack] ?? 0) + 1;
  }
  const occ: Record<string, number> = {};
  for (const r of Object.keys(total)) occ[r] = (alloc[r] ?? 0) / Math.max(1, total[r]);
  return occ;
}

function main(): void {
  // ── 1: five-point quantile fan on multiple channels ─────────────────────
  console.log("compare mock returns an ordered p05-p95 quantile fan");
  const cmp = buildMockCompare(compareReq("dom-rack-a1", "dom-rack-b2"));
  const admitA = cmp.candidates.find((c) => c.action_id === "admit-a")!;
  const qs = admitA.uncertainty.quantiles;
  assert(qs.length >= 2, "feasible candidate carries multiple forecast channels");
  const qwait = qs.find((q) => q.channel === "queue_wait_delta_s")!;
  const heads = ["0.05", "0.1", "0.5", "0.9", "0.95"];
  assert(
    heads.every((h) => qwait.quantiles[h] != null),
    "queue-wait forecast has all five quantile heads (p05/p10/p50/p90/p95)",
  );
  assert(
    qwait.quantiles["0.05"] <= qwait.quantiles["0.5"] &&
      qwait.quantiles["0.5"] <= qwait.quantiles["0.95"],
    "quantile fan is monotonically ordered p05 ≤ p50 ≤ p95",
  );
  assert(
    qs.some((q) => q.channel === "rack_thermal_headroom_c"),
    "thermal-headroom band channel is present for the band chart",
  );

  // ── 2: branchUncertainty bounded + grows with a tighter forecast ────────
  console.log("branchUncertainty combines epistemic + ood + spread, bounded 0..1");
  const uA = branchUncertainty(admitA);
  assert(uA >= 0 && uA <= 1, "branchUncertainty is bounded to 0..1");
  // A tight, worse-locality feasible target (dom-rack-b1, free 18) has higher
  // epistemic/ood + wider band, so its ring uncertainty exceeds the roomy admit-a.
  const cmpTight = buildMockCompare(compareReq("dom-rack-a1", "dom-rack-b1"));
  const tight = cmpTight.candidates.find((c) => c.action_id === "admit-b")!;
  const uTight = branchUncertainty(tight);
  assert(
    uTight > uA,
    "a tighter / worse-locality candidate drives a thicker uncertainty ring",
  );

  // ── 3: grounded baseline vs ghost overlay ───────────────────────────────
  console.log("grounded baseline is read from the snapshot; prediction sits above it");
  const snap = buildMockSnapshot(0);
  const nodeToRack = new Map(snap.entities.nodes.map((n) => [n.node_uid, n.rack_id]));
  const baseline = baselineFromGpus(snap.entities.gpus, nodeToRack);
  // rack-a1 baseline fill is 0.18 in the mock; grounded occupancy should match.
  assert(
    Math.abs((baseline["rack-a1"] ?? 0) - 0.18) < 0.02,
    "baseline occupancy for rack-a1 ≈ 0.18 (read from grounded GPU allocation)",
  );
  // The recommended branch's rollout predicts ADMITTING into rack-a1, so its
  // predicted occupancy at the end of horizon must EXCEED the grounded baseline
  // (the extra fill is the ghost overlay, never promoted to the solid base).
  const rolloutA = admitA.rollout!;
  const lastPoint = rolloutA.points.at(-1)!;
  assert(
    (lastPoint.rack_occupancy["rack-a1"] ?? 0) > (baseline["rack-a1"] ?? 0),
    "predicted rack-a1 occupancy exceeds the grounded baseline (ghost overlay only)",
  );

  // ── 4: replay realized diverges from predicted across three channels ────
  console.log("replay realized trace diverges from predicted at t=720");
  const replay = buildMockReplay("golden-path-admit-a");
  const at = (pts: typeof replay.realized.points, t: number) => {
    let chosen = pts[0];
    for (const p of pts) {
      if (p.t_s <= t) chosen = p;
      else break;
    }
    return chosen;
  };
  // The admit lands in rack-a1, so divergence shows on rack-a1 specifically —
  // compared PER-RACK (a fixed busy rack elsewhere must not mask it), matching
  // the calibration panel + timeline divergence detection.
  const RACK = "rack-a1";
  const predLate = at(replay.predicted.points, 1200);
  const realLate = at(replay.realized.points, 1200);
  assert(
    (realLate.rack_hotspot[RACK] ?? 0) > (predLate.rack_hotspot[RACK] ?? 0),
    "realized rack-a1 hotspot exceeds predicted after divergence (under-called)",
  );
  assert(
    (realLate.rack_occupancy[RACK] ?? 0) > (predLate.rack_occupancy[RACK] ?? 0),
    "realized rack-a1 occupancy exceeds predicted after divergence (extra spillover)",
  );
  assert(
    realLate.queue_depth > predLate.queue_depth,
    "realized queue depth stays deeper than predicted after divergence",
  );
  const predEarly = at(replay.predicted.points, 300);
  const realEarly = at(replay.realized.points, 300);
  assert(
    Math.abs((realEarly.rack_hotspot[RACK] ?? 0) - (predEarly.rack_hotspot[RACK] ?? 0)) < 0.06,
    "predicted and realized agree on rack-a1 BEFORE the divergence timestamp",
  );

  if (failures > 0) {
    // eslint-disable-next-line no-console
    console.error(`\nFAILED: ${failures} assertion(s) failed.`);
    process.exit(1);
  }
  // eslint-disable-next-line no-console
  console.log("\nAll FE-UNCERTAINTY-REPLAY-VISUALS assertions passed.");
}

main();
