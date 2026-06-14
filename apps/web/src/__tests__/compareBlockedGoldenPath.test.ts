// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Scoped tests for FE-COMPARE-BLOCKED-GOLDENPATH. No test framework is wired in
//  this lane (the build verifies with `tsc --noEmit`), so this file is a
//  self-contained, runnable harness: a tiny assert() + a main() that exits
//  non-zero on failure. Run with a TypeScript-aware runtime, e.g.:
//
//      node --experimental-strip-types src/__tests__/compareBlockedGoldenPath.test.ts
//
//  It exercises:
//   1. buildMockCompare derives a BLOCKED candidate (dom-rack-b2) with
//      safe:false, rank:null, and populated exact_blockers.
//   2. The feasible admit candidate is safe + ranked + recommended.
//   3. buildMockWhyNot computes the gang-atomicity deficit from the SUBMITTED
//      action's domain + gpu_req (not hardcoded), and matches the snapshot.
//   4. The abstain gate fires when epistemic/ood cross the threshold.
//   5. The Store derives hold + feasible-admit + blocked-admit from the snapshot
//      topology_domains, and exposes the full CandidateAction for why-not.

import {
  buildMockCompare,
  buildMockSnapshot,
  buildMockWhyNot,
} from "../api/mock";
import { WorldModelClient, type CompareRequest } from "../api/client";
import { Store } from "../state/store";

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

function compareRequestFor(
  workloadUid: string,
  feasibleDomain: string,
  blockedDomain: string,
): CompareRequest {
  return {
    state_hash: "0xtest",
    horizon_s: 1800,
    include_rollouts: true,
    candidate_actions: [
      { action_id: "hold", type: "hold", params: { workload_uid: workloadUid } },
      {
        action_id: "admit-a",
        type: "admit_and_place",
        params: {
          workload_uid: workloadUid,
          domain_policy: "required",
          domain_id_or_set: [feasibleDomain],
        },
      },
      {
        action_id: "admit-b",
        type: "admit_and_place",
        params: {
          workload_uid: workloadUid,
          domain_policy: "required",
          domain_id_or_set: [blockedDomain],
        },
      },
    ],
  };
}

async function main(): Promise<void> {
  const snap = buildMockSnapshot(0);
  const focus = "wl-train-7b";

  // ── 1+2: blocked + feasible compare candidates ──────────────────────────
  console.log("buildMockCompare derives blocked + feasible candidates");
  const cmp = buildMockCompare(
    compareRequestFor(focus, "dom-rack-a1", "dom-rack-b2"),
  );
  const hold = cmp.candidates.find((c) => c.action_id === "hold")!;
  const admitA = cmp.candidates.find((c) => c.action_id === "admit-a")!;
  const admitB = cmp.candidates.find((c) => c.action_id === "admit-b")!;

  assert(admitB.safe === false, "admit-b (dom-rack-b2) is safe:false");
  assert(admitB.rank === null, "blocked candidate has rank:null");
  assert(
    admitB.exact_blockers.length > 0 &&
      admitB.exact_blockers.includes("gang_atomicity"),
    "blocked candidate has populated exact_blockers incl. gang_atomicity",
  );
  assert(
    admitB.rollout != null && admitB.rollout.points.length === 0,
    "blocked candidate carries NO rollout points (motion stops at blocker)",
  );
  assert(
    admitB.rollout?.scene_annotations.blocked_domain_ids[0] === "dom-rack-b2",
    "blocked candidate annotates blocked_domain_ids = dom-rack-b2",
  );
  assert(admitA.safe === true && admitA.rank === 1, "admit-a is safe rank #1");
  assert(hold.safe === true, "hold is safe (baseline)");
  assert(
    cmp.recommended_action_id === "admit-a",
    "recommended_action_id is the feasible admit (admit-a)",
  );
  assert(cmp.abstain === false, "compare does not abstain (a safe winner exists)");

  // ── 3: why-not deficit derived from the submitted action ────────────────
  console.log("buildMockWhyNot derives deficit from the submitted action");
  const why = buildMockWhyNot({
    state_hash: snap.state_hash,
    candidate_action: {
      action_id: "admit-b",
      type: "admit_and_place",
      params: {
        workload_uid: focus,
        domain_policy: "required",
        domain_id_or_set: ["dom-rack-b2"],
      },
    },
  });
  assert(why.safe === false, "why-not certificate is safe:false");
  assert(
    why.rejected_domain_ids[0] === "dom-rack-b2",
    "why-not rejects the submitted domain dom-rack-b2",
  );
  // dom-rack-b2: fill 0.84, cap 32 → free 5, gpuReq 16 → gang deficit 11.
  const gang = why.exact_blockers.find((b) => b.group === "gang_atomicity")!;
  assert(gang.deficit.gpu === 11, "gang-atomicity deficit = 11 GPU (16 req − 5 free)");
  assert(
    why.minimal_groups.includes("gang_atomicity") &&
      why.minimal_groups.includes("health_schedulability"),
    "minimal certificate includes gang_atomicity + health_schedulability (degraded node)",
  );
  const domb2 = snap.entities.topology_domains.find(
    (d) => d.domain_id === "dom-rack-b2",
  )!;
  assert(
    domb2.free_gpu === 5,
    "snapshot dom-rack-b2 free_gpu = 5 (consistent with the why-not deficit)",
  );

  // A FEASIBLE why-not query returns no capacity deficit for the gang fit.
  const whyA = buildMockWhyNot({
    state_hash: snap.state_hash,
    candidate_action: {
      action_id: "admit-a",
      type: "admit_and_place",
      params: {
        workload_uid: focus,
        domain_policy: "required",
        domain_id_or_set: ["dom-rack-a1"],
      },
    },
  });
  assert(
    whyA.rejected_domain_ids[0] === "dom-rack-a1",
    "why-not echoes whatever domain was submitted (dom-rack-a1)",
  );

  // ── 4: abstain gate fires from a threshold-crossing forecast ────────────
  console.log("abstain gate fires when epistemic/ood cross the threshold");
  // dom-rack-b1 free = round((1-0.45)*32) = 18, just above the 16-GPU gang →
  // FEASIBLE but tight (fit margin 2/16) AND worse-locality rack-b: the forecast
  // is pushed toward saturation, so epistemic/ood cross the abstain threshold.
  const cmpB1 = buildMockCompare(
    compareRequestFor(focus, "dom-rack-a1", "dom-rack-b1"),
  );
  const b1 = cmpB1.candidates.find((c) => c.action_id === "admit-b")!;
  assert(b1.safe === true, "dom-rack-b1 is kernel-feasible (free 18 ≥ 16)");
  assert(
    b1.uncertainty.abstain === true,
    "tight feasible candidate ABSTAINS — its forecast crosses the threshold",
  );
  assert(
    b1.rank === null,
    "an abstaining candidate is unranked (cannot win the recommendation)",
  );
  assert(
    b1.uncertainty.abstain_reason != null,
    "abstain carries a reason for the explicit abstain chip",
  );
  // The roomy feasible target (dom-rack-a1, free 26) stays confident.
  const a1Cand = cmpB1.candidates.find((c) => c.action_id === "admit-a")!;
  assert(
    a1Cand.uncertainty.abstain === false && a1Cand.rank === 1,
    "the roomy feasible target stays confident (no abstain) and wins",
  );

  // ── 5: Store derives the candidate alphabet from topology_domains ───────
  console.log("Store.runCompare derives candidates from the snapshot");
  const client = new WorldModelClient({ forceMock: true });
  const store = new Store(client);
  await store.boot();
  assert(store.get().focusedWorkload === focus, "boot focuses the pending 16-GPU workload");
  await store.runCompare();
  const sc = store.get().compare!;
  const sBlocked = sc.candidates.find((c) => !c.safe);
  assert(sBlocked != null, "store compare contains a blocked candidate");
  assert(
    store.candidateAction(sBlocked!.action_id)?.type === "admit_and_place",
    "store exposes the full CandidateAction (type) for the blocked action",
  );
  const blockedDomain = (
    store.candidateAction(sBlocked!.action_id)?.params.domain_id_or_set as
      | string[]
      | undefined
  )?.[0];
  assert(
    blockedDomain === "dom-rack-b2",
    "the derived blocked candidate targets the drained domain dom-rack-b2",
  );
  const recCand = sc.candidates.find(
    (c) => c.action_id === sc.recommended_action_id,
  );
  assert(
    recCand?.safe === true && recCand.rank === 1,
    "store recommendation is a safe rank #1 feasible admit",
  );

  // ── golden-path driver methods drive store state ────────────────────────
  console.log("golden-path driver methods mutate store state");
  store.playBranch(sc.recommended_action_id!);
  assert(
    store.get().mode === "counterfactual-compare" &&
      store.get().pinnedActions[0] === sc.recommended_action_id,
    "playBranch fronts the chosen branch and switches to compare mode",
  );
  store.freezeAt();
  assert(
    store.get().playing === false &&
      store.get().cursorS === store.get().horizonS,
    "freezeAt stops playback and parks the cursor at the horizon",
  );
  await store.showWhyNot(sBlocked!.action_id);
  assert(
    store.get().whyNotActionId === sBlocked!.action_id &&
      store.get().pinnedActions.includes(sBlocked!.action_id),
    "showWhyNot opens the card and pins the blocked branch (for the lock badge)",
  );
  store.resolveToRecommendation();
  assert(
    store.get().mode === "decision-studio" &&
      store.get().whyNotActionId === null &&
      store.get().pinnedActions[0] === sc.recommended_action_id,
    "resolveToRecommendation closes why-not and fronts the recommended branch",
  );

  store.dispose();

  if (failures > 0) {
    // eslint-disable-next-line no-console
    console.error(`\nFAILED: ${failures} assertion(s) failed.`);
    process.exit(1);
  }
  // eslint-disable-next-line no-console
  console.log("\nAll FE-COMPARE-BLOCKED-GOLDENPATH assertions passed.");
}

void main();
