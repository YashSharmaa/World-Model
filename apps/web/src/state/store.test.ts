// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Vitest unit tests for the client store reducers, the scene occupancy fill
//  math, and the branch lock-badge logic. The store is the authoritative
//  read-model the renderer interpolates over (doc G "Timeline authority"); these
//  tests pin its reducer contract and the two scene-side derivations that the
//  store's data feeds (applyOccupancy fill, rebuildBranches lock badge).

import { afterEach, describe, expect, it, vi } from "vitest";
import { Store } from "./store";
import { WorldModelClient } from "../api/client";
import { applyOccupancy, buildRacks } from "../scene/racks";
import { createBranchesRoot, rebuildBranches } from "../scene/branches";
import { branchUncertainty } from "../scene/branches";
import { createMaterials } from "../scene/materials";
import { buildMockSnapshot } from "../api/mock";
import type {
  CandidateResultDTO,
  RolloutDTO,
  RolloutPoint,
} from "../api/client";

afterEach(() => {
  vi.restoreAllMocks();
});

function freshStore(): Store {
  return new Store(new WorldModelClient({ forceMock: true }));
}

// ──────────────────────────────────────────────────────────────────────────
// Store reducer: runCompare derives the candidate alphabet from the snapshot
// ──────────────────────────────────────────────────────────────────────────
describe("Store.runCompare", () => {
  it("derives hold + feasible-admit + blocked-admit and pins safe first", async () => {
    const store = freshStore();
    await store.boot();
    expect(store.get().focusedWorkload).toBe("wl-train-7b");

    await store.runCompare();
    const cmp = store.get().compare!;
    expect(cmp).not.toBeNull();
    // hold baseline is always present.
    expect(store.candidateAction("hold")?.type).toBe("hold");
    // The derived alphabet contains a blocked admit (the why-not / lock path).
    const blocked = cmp.candidates.find((c) => !c.safe);
    expect(blocked).toBeDefined();
    // Safe survivors are pinned before the blocked branch (doc G stability).
    const firstPinned = store.get().pinnedActions[0];
    const firstCand = cmp.candidates.find((c) => c.action_id === firstPinned)!;
    expect(firstCand.safe).toBe(true);
    // The recommended action is a safe rank #1 admit.
    const rec = cmp.candidates.find(
      (c) => c.action_id === cmp.recommended_action_id,
    )!;
    expect(rec.safe).toBe(true);
    expect(rec.rank).toBe(1);
    // runCompare resets the shared cursor and clears any open why-not.
    expect(store.get().cursorS).toBe(0);
    expect(store.get().whyNotActionId).toBeNull();
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Store reducer: togglePinned (add/remove, capped at MAX_PINNED = 3)
// ──────────────────────────────────────────────────────────────────────────
describe("Store.togglePinned", () => {
  it("adds, removes, and caps the pinned set at three", async () => {
    const store = freshStore();
    await store.boot();
    await store.runCompare();
    // Start from a known empty pin set so the cap is exercised deterministically.
    for (const id of [...store.get().pinnedActions]) store.togglePinned(id);
    expect(store.get().pinnedActions).toEqual([]);

    store.togglePinned("a");
    store.togglePinned("b");
    store.togglePinned("c");
    expect(store.get().pinnedActions).toEqual(["a", "b", "c"]);

    // A fourth pin is rejected — the candidate count stays shape-stable (MLX).
    store.togglePinned("d");
    expect(store.get().pinnedActions).toEqual(["a", "b", "c"]);

    // Toggling an existing id removes it.
    store.togglePinned("b");
    expect(store.get().pinnedActions).toEqual(["a", "c"]);
    // ...and now a new id fits again.
    store.togglePinned("d");
    expect(store.get().pinnedActions).toEqual(["a", "c", "d"]);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Store reducer: loadReplay swaps snapshot, mode, and horizon from the replay
// ──────────────────────────────────────────────────────────────────────────
describe("Store.loadReplay", () => {
  it("loads a replay, switches to replay-calibration, and sets the horizon", async () => {
    const store = freshStore();
    await store.loadReplay("thermal-runaway");
    const s = store.get();
    expect(s.mode).toBe("replay-calibration");
    expect(s.replay?.scenario_id).toBe("thermal-runaway");
    // The snapshot is the replay's recorded initial state.
    expect(s.snapshot).toBe(s.replay!.initial_state);
    // The horizon is the last predicted rollout point's t_s.
    const lastT = s.replay!.predicted.points.at(-1)!.t_s;
    expect(s.horizonS).toBe(lastT);
    expect(s.cursorS).toBe(0);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Store reducer: advance interpolates the cursor and stops at the horizon
// ──────────────────────────────────────────────────────────────────────────
describe("Store.advance", () => {
  it("only advances while playing and clamps + stops at the horizon", async () => {
    const store = freshStore();
    await store.boot();
    store.setHorizon(100);

    // Not playing -> advance is a no-op.
    store.advance(10);
    expect(store.get().cursorS).toBe(0);

    store.setPlaying(true);
    store.advance(40);
    expect(store.get().cursorS).toBe(40);
    store.advance(40);
    expect(store.get().cursorS).toBe(80);

    // Overrunning the horizon clamps to the horizon AND stops playback.
    store.advance(50);
    expect(store.get().cursorS).toBe(100);
    expect(store.get().playing).toBe(false);

    // Once stopped, further advances are no-ops.
    store.advance(10);
    expect(store.get().cursorS).toBe(100);
  });

  it("setCursor clamps into [0, horizon]", async () => {
    const store = freshStore();
    await store.boot();
    store.setHorizon(50);
    store.setCursor(-10);
    expect(store.get().cursorS).toBe(0);
    store.setCursor(999);
    expect(store.get().cursorS).toBe(50);
    store.setCursor(25);
    expect(store.get().cursorS).toBe(25);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// applyOccupancy fill math: round(occ * slots) tiles recoloured "filled"
// ──────────────────────────────────────────────────────────────────────────
describe("applyOccupancy fill math", () => {
  function buildRackResult() {
    const snap = buildMockSnapshot(0);
    const mats = createMaterials();
    return {
      result: buildRacks(
        snap.entities.racks,
        snap.entities.nodes,
        snap.entities.gpus,
        mats,
      ),
      rackId: snap.entities.racks[0].rack_id,
    };
  }

  it("recolours round(occupancy * slotCount) tiles as filled", () => {
    const { result, rackId } = buildRackResult();
    const indices: number[] = [];
    result.slotIndex.forEach((s, i) => {
      if (s.rackId === rackId) indices.push(i);
    });
    expect(indices.length).toBeGreaterThan(0);

    // Apply 50% occupancy, grounded (not a ghost).
    applyOccupancy(result, rackId, 0.5, 0.0, false);

    // Inspect the instanceColor buffer: an "empty" slot is PALETTE.slotEmpty.
    const colorAttr = result.slotMesh.instanceColor!;
    const empties = indices.filter((idx) => {
      const r = colorAttr.getX(idx);
      const g = colorAttr.getY(idx);
      const b = colorAttr.getZ(idx);
      // slotEmpty (#1b2436) is a dark blue; a filled tile is brighter.
      return r < 0.12 && g < 0.18 && b < 0.25;
    }).length;
    const expectedFilled = Math.round(0.5 * indices.length);
    expect(indices.length - empties).toBe(expectedFilled);
  });

  it("full occupancy fills every slot; zero fills none", () => {
    const { result, rackId } = buildRackResult();
    const indices: number[] = [];
    result.slotIndex.forEach((s, i) => {
      if (s.rackId === rackId) indices.push(i);
    });
    const colorAttr = result.slotMesh.instanceColor!;
    const countEmpty = () =>
      indices.filter((idx) => {
        const r = colorAttr.getX(idx);
        const g = colorAttr.getY(idx);
        const b = colorAttr.getZ(idx);
        return r < 0.12 && g < 0.18 && b < 0.25;
      }).length;

    applyOccupancy(result, rackId, 1.0, 0.0, false);
    expect(countEmpty()).toBe(0); // all filled

    applyOccupancy(result, rackId, 0.0, 0.0, false);
    expect(countEmpty()).toBe(indices.length); // all empty
  });
});

// ──────────────────────────────────────────────────────────────────────────
// rebuildBranches lock-badge logic: a blocked (safe:false) branch lights its
// lock badge; a safe branch hides it (doc G: kernel verdict is a GATE).
// ──────────────────────────────────────────────────────────────────────────
describe("rebuildBranches lock badge", () => {
  function candidate(id: string, safe: boolean): CandidateResultDTO {
    const rollout: RolloutDTO = {
      branch_id: id,
      action_label: id,
      timestep_s: 60,
      points: [] as RolloutPoint[],
      scene_annotations: {
        recommended_domain_ids: ["dom-rack-a1"],
        blocked_domain_ids: [],
        highlight_workload_ids: [],
      },
    };
    return {
      action_id: id,
      safe,
      exact_blockers: safe ? [] : ["gang_atomicity"],
      rank: safe ? 1 : null,
      delta_kpis: {},
      uncertainty: {
        epistemic_score: 0.1,
        ood_score: 0.05,
        abstain: false,
        abstain_reason: null,
        quantiles: [],
        hazards: [],
      },
      rollout,
    };
  }

  it("shows the lock badge only for the blocked branch", () => {
    const snap = buildMockSnapshot(0);
    const mats = createMaterials();
    const racks = buildRacks(
      snap.entities.racks,
      snap.entities.nodes,
      snap.entities.gpus,
      mats,
    );
    const branches = createBranchesRoot();
    const safeCand = candidate("admit-a", true);
    const blockedCand = candidate("admit-b", false);

    rebuildBranches(
      branches,
      [safeCand, blockedCand],
      ["admit-a", "admit-b"],
      racks.racks,
      mats,
    );

    const safeGhost = branches.ghosts.get("admit-a")!;
    const blockedGhost = branches.ghosts.get("admit-b")!;
    expect(safeGhost.blocked).toBe(false);
    expect(safeGhost.lockBadge.visible).toBe(false);
    expect(blockedGhost.blocked).toBe(true);
    // The lock badge is lit precisely because the exact kernel marked it unsafe.
    expect(blockedGhost.lockBadge.visible).toBe(true);
  });

  it("rebuild tears down old ghosts (only the current pinned set survives)", () => {
    const snap = buildMockSnapshot(0);
    const mats = createMaterials();
    const racks = buildRacks(
      snap.entities.racks,
      snap.entities.nodes,
      snap.entities.gpus,
      mats,
    );
    const branches = createBranchesRoot();
    rebuildBranches(branches, [candidate("a", true)], ["a"], racks.racks, mats);
    expect([...branches.ghosts.keys()]).toEqual(["a"]);
    // Rebuild with a different pinned set: the stale ghost is removed.
    rebuildBranches(branches, [candidate("b", false)], ["b"], racks.racks, mats);
    expect([...branches.ghosts.keys()]).toEqual(["b"]);
  });

  it("branchUncertainty grows with epistemic + ood and saturates at 1", () => {
    const low = branchUncertainty(candidate("lo", true));
    const hiCand = candidate("hi", true);
    hiCand.uncertainty.epistemic_score = 0.9;
    hiCand.uncertainty.ood_score = 0.9;
    const hi = branchUncertainty(hiCand);
    expect(hi).toBeGreaterThan(low);
    expect(hi).toBeLessThanOrEqual(1);
    expect(low).toBeGreaterThanOrEqual(0);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// sampleRollout interpolation: the renderer samples the rollout point that is
// authoritative for the shared cursor t (doc G "Timeline authority": the scene
// only ever interpolates over backend rollout points, never free-runs). The
// sampler is a step/floor sampler — it returns the last point whose t_s <= t —
// because each RolloutPoint carries the discrete world state stamped at t_s.
//
// NOTE: the production `sampleRollout` lives in `src/main.ts`, which is a
// non-exported function in a module that self-invokes `main()` on import (it
// bootstraps the whole three.js app). It therefore cannot be imported into a
// jsdom unit test without a src change (outside this test lane's owned set —
// reported as a finding). We pin the SAME floor-sample contract here and run it
// against the REAL mock rollout points produced by the client's compare
// contract, asserting the production preconditions the renderer relies on:
// monotonic 60s-spaced t_s, correct floor selection at the boundaries, clamp at
// the horizon, and the admit branch's monotone occupancy ramp the scene plays.
// ──────────────────────────────────────────────────────────────────────────
describe("sampleRollout interpolation (renderer floor-sample contract)", () => {
  // Mirror of the renderer's sampler (src/main.ts): last point with t_s <= t.
  function sampleRollout<T extends { t_s: number }>(
    points: T[],
    t: number,
  ): T | null {
    if (!points.length) return null;
    let chosen = points[0];
    for (const p of points) {
      if (p.t_s <= t) chosen = p;
      else break;
    }
    return chosen;
  }

  async function safeRolloutPoints(): Promise<RolloutPoint[]> {
    const client = new WorldModelClient({ forceMock: true });
    const cmp = await client.compareActions({
      state_hash: "0xtest",
      candidate_actions: [
        { action_id: "hold", type: "hold", params: { workload_uid: "wl-train-7b" } },
        {
          action_id: "admit-a",
          type: "admit_and_place",
          params: {
            workload_uid: "wl-train-7b",
            domain_policy: "required",
            domain_id_or_set: ["dom-rack-a1"],
          },
        },
      ],
      horizon_s: 1800,
      include_rollouts: true,
    });
    // The admit branch (not hold) commits the gang into the target rack, so its
    // rollout carries the rising-occupancy ramp the renderer animates.
    const admit = cmp.candidates.find(
      (c) => c.action_id === "admit-a" && (c.rollout?.points.length ?? 0) > 0,
    );
    expect(admit).toBeDefined();
    expect(admit!.safe).toBe(true);
    return admit!.rollout!.points;
  }

  it("the real mock rollout points are monotonically increasing in t_s (60s step)", async () => {
    const points = await safeRolloutPoints();
    expect(points.length).toBeGreaterThan(2);
    expect(points[0].t_s).toBe(0);
    for (let i = 1; i < points.length; i++) {
      expect(points[i].t_s).toBeGreaterThan(points[i - 1].t_s);
      expect(points[i].t_s - points[i - 1].t_s).toBe(60);
    }
  });

  it("returns the floor point: exact hit, between samples, before-first, past-horizon", async () => {
    const points = await safeRolloutPoints();
    const last = points.at(-1)!;

    // Exact grid time -> that exact point.
    expect(sampleRollout(points, 120)?.t_s).toBe(120);
    // Between two grid points (90s) -> the earlier one (floor at 60), never the
    // future point: the scene must not render a state the model has not stamped.
    expect(sampleRollout(points, 90)?.t_s).toBe(60);
    // 119s is still the t=60 sample; 120s flips to the t=120 sample.
    expect(sampleRollout(points, 119)?.t_s).toBe(60);
    // Before the first sample (negative cursor) -> clamp to the first point.
    expect(sampleRollout(points, -5)?.t_s).toBe(0);
    // At and beyond the horizon -> the last (terminal) point, never overshoot.
    expect(sampleRollout(points, last.t_s)?.t_s).toBe(last.t_s);
    expect(sampleRollout(points, last.t_s + 10_000)?.t_s).toBe(last.t_s);
  });

  it("an empty rollout (blocked branch) samples to null — motion stops", () => {
    expect(sampleRollout([] as RolloutPoint[], 60)).toBeNull();
  });

  it("sampling the admit branch yields a monotone occupancy ramp on the target rack", async () => {
    const points = await safeRolloutPoints();
    // The admit branch commits the workload into exactly one rack domain; that
    // rack's occupancy is non-decreasing over the rollout (mock invariant the
    // renderer animates as the loader arm fills slots — doc G occupancy fill).
    const target = Object.keys(points.at(-1)!.rack_occupancy).find(
      (r) =>
        (points.at(-1)!.rack_occupancy[r] ?? 0) >
        (points[0].rack_occupancy[r] ?? 0),
    );
    expect(target).toBeDefined();

    let prev = -Infinity;
    for (let t = 0; t <= points.at(-1)!.t_s; t += 30) {
      const occ = sampleRollout(points, t)!.rack_occupancy[target!];
      expect(occ).toBeGreaterThanOrEqual(prev);
      prev = occ;
    }
    // The terminal occupancy is strictly above the starting occupancy (the gang
    // landed), and the sampler never reads past the model's last stamped state.
    expect(prev).toBeGreaterThan(points[0].rack_occupancy[target!]);
  });
});
