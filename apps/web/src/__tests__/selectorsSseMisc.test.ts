// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Scoped tests for FE-SELECTORS-SSE-MISC (same self-contained assert() harness;
//  build verifies with `tsc --noEmit`). It exercises the PURE logic behind the
//  selectors + SSE wiring:
//
//   1. buildMockStateDelta emits a PARTIAL patch — only the handful of changed
//      entities/KPIs, never a whole-array snapshot — so the store's merge-by-id
//      path is meaningful.
//   2. A merge-by-id of that partial delta into the live snapshot updates the
//      changed GPUs/queue IN PLACE while preserving every other entity (the bug
//      the item targets: a shallow whole-array replace would clobber the rest).
//   3. The replay catalog from GET /replays carries BOTH recorded scenarios so
//      the second one is loadable through the picker.
//   4. Store.setHorizonAndCompare changes the horizon AND re-runs the comparison
//      at the new horizon (rollouts recomputed for the selected window).

import {
  buildMockSnapshot,
  buildMockStateDelta,
  buildMockReplayIndex,
} from "../api/mock";
import { WorldModelClient } from "../api/client";
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

/** Mirror of the store's private mergeById (kept in lock-step by the tests). */
function mergeById<T extends Record<string, unknown>>(
  base: T[],
  patch: T[] | undefined,
  key: string,
): T[] {
  if (!patch || !patch.length) return base;
  const byId = new Map<unknown, T>(base.map((e) => [e[key], e]));
  for (const e of patch) byId.set(e[key], e);
  return [...byId.values()];
}

async function main(): Promise<void> {
  // ── 1: buildMockStateDelta is a PARTIAL patch ───────────────────────────
  console.log("buildMockStateDelta emits a partial (changed-only) patch");
  const delta = buildMockStateDelta(3);
  assert(delta.patch.entities != null, "delta carries an entities patch");
  const pe = delta.patch.entities!;
  assert(
    (pe.gpus?.length ?? 0) === 2,
    "delta patches exactly the 2 changed GPUs (not all 128)",
  );
  assert(
    (pe.queues?.length ?? 0) === 1,
    "delta patches exactly the 1 changed queue",
  );
  assert(
    pe.racks === undefined && pe.nodes === undefined && pe.topology_domains === undefined,
    "unchanged entity families are OMITTED from the delta",
  );
  assert(
    delta.patch.current_kpis != null && delta.patch.freshness != null,
    "delta carries the live KPI + freshness scalars",
  );

  // ── 2: merge-by-id preserves un-patched entities ────────────────────────
  console.log("merge-by-id updates changed entities and preserves the rest");
  const snap = buildMockSnapshot(0);
  const baseGpuCount = snap.entities.gpus.length;
  const mergedGpus = mergeById(
    snap.entities.gpus as unknown as Record<string, unknown>[],
    pe.gpus as unknown as Record<string, unknown>[] | undefined,
    "gpu_uuid",
  );
  assert(
    mergedGpus.length === baseGpuCount,
    "GPU count is unchanged after the merge (no clobbering, no duplication)",
  );
  // the two patched GPUs took the delta's new telemetry…
  const g0 = mergedGpus.find((g) => g.gpu_uuid === "rack-a1-n0-g0")!;
  const deltaG0 = pe.gpus!.find((g) => g.gpu_uuid === "rack-a1-n0-g0")!;
  assert(
    (g0.temp_c as number) === deltaG0.temp_c,
    "the patched GPU took the delta's new temperature",
  );
  // …and an un-patched GPU on another rack is the ORIGINAL object.
  const originalFar = snap.entities.gpus.find((g) => g.gpu_uuid === "rack-b2-n0-g0")!;
  const mergedFar = mergedGpus.find((g) => g.gpu_uuid === "rack-b2-n0-g0")!;
  assert(
    mergedFar === (originalFar as unknown as Record<string, unknown>),
    "an un-patched far GPU is preserved untouched (identity held)",
  );
  const mergedQueues = mergeById(
    snap.entities.queues as unknown as Record<string, unknown>[],
    pe.queues as unknown as Record<string, unknown>[] | undefined,
    "name",
  );
  assert(
    mergedQueues.length === snap.entities.queues.length,
    "queue count unchanged after merge (the prod queue is preserved)",
  );

  // ── 3: replay catalog carries both scenarios ────────────────────────────
  console.log("replay catalog exposes both recorded scenarios for the picker");
  const idx = buildMockReplayIndex();
  assert(idx.scenarios.length >= 2, "catalog lists at least two replay scenarios");
  assert(
    idx.scenarios.some((s) => s.scenario_id === "hold-then-place"),
    "the SECOND scenario (hold-then-place) is present and loadable",
  );

  // ── 4: setHorizonAndCompare re-runs the comparison ──────────────────────
  console.log("Store.setHorizonAndCompare re-compares at the new horizon");
  const store = new Store(new WorldModelClient({ forceMock: true }));
  await store.boot();
  await store.runCompare();
  const beforeHorizon = store.get().compare!.horizon_s;
  assert(beforeHorizon === store.get().horizonS, "compare ran at the current horizon");
  const newHorizon = beforeHorizon === 3600 ? 900 : 3600;
  await store.setHorizonAndCompare(newHorizon);
  assert(store.get().horizonS === newHorizon, "horizon switched to the picked value");
  assert(
    store.get().compare!.horizon_s === newHorizon,
    "the comparison was re-run at the NEW horizon (rollouts recomputed)",
  );
  // the rollout points now span the new horizon.
  const rec = store
    .get()
    .compare!.candidates.find((c) => c.action_id === store.get().compare!.recommended_action_id);
  const lastT = rec?.rollout?.points.at(-1)?.t_s ?? 0;
  assert(lastT === newHorizon, "the recomputed rollout spans the new horizon");
  store.dispose();

  if (failures > 0) {
    // eslint-disable-next-line no-console
    console.error(`\nFAILED: ${failures} assertion(s) failed.`);
    process.exit(1);
  }
  // eslint-disable-next-line no-console
  console.log("\nAll FE-SELECTORS-SSE-MISC assertions passed.");
}

void main();
