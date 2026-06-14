// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Vitest unit tests for the typed backend client (doc G six contracts). The
//  contract under test: EVERY method resolves even when the backend is offline —
//  it falls back to deterministic mock data so the Decision Studio is recordable
//  and reproducible standalone, and flips `backendOnline` to drive the header
//  badge. We assert the fallback on each of the six contracts plus the SSE
//  stream's degrade-to-synthetic-loop behaviour.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  WorldModelClient,
  type CompareRequest,
  type StreamEvent,
} from "./client";

const COMPARE_REQ: CompareRequest = {
  state_hash: "0xtest",
  candidate_actions: [
    { action_id: "hold", type: "hold", params: { workload_uid: "wl-train-7b" } },
  ],
  horizon_s: 1800,
  include_rollouts: true,
};

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

// ──────────────────────────────────────────────────────────────────────────
// forceMock: every contract returns deterministic mock data, no network
// ──────────────────────────────────────────────────────────────────────────
describe("forceMock never touches the network", () => {
  it("getCurrentState returns the deterministic mock snapshot", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const client = new WorldModelClient({ forceMock: true });
    const snap = await client.getCurrentState();
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(snap.entities.workloads.some((w) => w.status === "pending")).toBe(true);
    expect(snap.credit).toBeTruthy();
  });

  it("compareActions / whyNot / listReplays / getReplay all resolve from mock", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const client = new WorldModelClient({ forceMock: true });

    const cmp = await client.compareActions(COMPARE_REQ);
    expect(cmp.candidates.length).toBeGreaterThan(0);

    const why = await client.whyNot({
      state_hash: "0xtest",
      candidate_action: {
        action_id: "admit-b",
        type: "admit_and_place",
        params: {
          workload_uid: "wl-train-7b",
          domain_policy: "required",
          domain_id_or_set: ["dom-rack-b2"],
        },
      },
    });
    expect(typeof why.safe).toBe("boolean");

    const index = await client.listReplays();
    expect(index.scenarios.length).toBeGreaterThan(0);

    const replay = await client.getReplay(index.scenarios[0].scenario_id);
    expect(replay.predicted.points.length).toBeGreaterThan(0);
    expect(replay.realized.points.length).toBeGreaterThan(0);

    expect(fetchSpy).not.toHaveBeenCalled();
  });
});

// ──────────────────────────────────────────────────────────────────────────
// network failure -> fall back to mock + flip backendOnline=false
// ──────────────────────────────────────────────────────────────────────────
describe("network failure degrades to mock data on each GET/POST contract", () => {
  beforeEach(() => {
    // Every fetch rejects (backend offline). The client must still resolve.
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("ECONNREFUSED"));
  });

  it("getCurrentState falls back and marks the backend offline", async () => {
    const client = new WorldModelClient({ timeoutMs: 50 });
    const snap = await client.getCurrentState();
    expect(snap.entities.racks.length).toBeGreaterThan(0);
    expect(client.backendOnline).toBe(false);
  });

  it("compareActions falls back to a deterministic mock compare", async () => {
    const client = new WorldModelClient({ timeoutMs: 50 });
    const cmp = await client.compareActions(COMPARE_REQ);
    expect(cmp.candidates.find((c) => c.action_id === "hold")).toBeDefined();
    expect(client.backendOnline).toBe(false);
  });

  it("whyNot falls back to a deterministic mock certificate", async () => {
    const client = new WorldModelClient({ timeoutMs: 50 });
    const why = await client.whyNot({
      state_hash: "0xtest",
      candidate_action: {
        action_id: "admit-b",
        type: "admit_and_place",
        params: {
          workload_uid: "wl-train-7b",
          domain_policy: "required",
          domain_id_or_set: ["dom-rack-b2"],
        },
      },
    });
    expect(why.action_id).toBe("admit-b");
    expect(client.backendOnline).toBe(false);
  });

  it("getReplay falls back to a deterministic mock replay", async () => {
    const client = new WorldModelClient({ timeoutMs: 50 });
    const replay = await client.getReplay("thermal-runaway");
    expect(replay.scenario_id).toBe("thermal-runaway");
    expect(client.backendOnline).toBe(false);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// HTTP non-2xx is treated as an outage (fall back, do not surface a 500 body)
// ──────────────────────────────────────────────────────────────────────────
describe("non-2xx responses are treated as an outage", () => {
  it("a 500 response falls back to mock data", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("boom", { status: 500 }),
    );
    const client = new WorldModelClient({ timeoutMs: 50 });
    const snap = await client.getCurrentState();
    // Falls back rather than throwing or returning the error body.
    expect(snap.entities.workloads.length).toBeGreaterThan(0);
    expect(client.backendOnline).toBe(false);
  });

  it("a 200 JSON response is passed through and marks the backend online", async () => {
    const payload = { state_hash: "0xlive", entities: { racks: [] } };
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    const client = new WorldModelClient({ timeoutMs: 50 });
    const snap = await client.getCurrentState();
    expect(snap.state_hash).toBe("0xlive");
    expect(client.backendOnline).toBe(true);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// Contract 6: SSE stream fallback to the synthetic loop
// ──────────────────────────────────────────────────────────────────────────
describe("openStream falls back to the synthetic delta loop", () => {
  it("with forceMock it emits backend_health(false) then state_delta ticks", () => {
    vi.useFakeTimers();
    const client = new WorldModelClient({ forceMock: true });
    const events: StreamEvent[] = [];
    const unsub = client.openStream((ev) => events.push(ev));

    // The synthetic loop emits an immediate offline backend_health frame.
    expect(events[0]).toEqual({
      type: "backend_health",
      healthy: false,
      detail: "mock data",
    });
    expect(client.backendOnline).toBe(false);

    // Advancing the timers drives partial state_delta ticks (semantic diffs).
    vi.advanceTimersByTime(4000);
    const delta = events.find((e) => e.type === "state_delta");
    expect(delta).toBeDefined();
    if (delta && delta.type === "state_delta") {
      // The delta is PARTIAL — only changed entity families are present.
      expect(delta.patch.entities?.gpus?.length).toBeGreaterThan(0);
      expect(delta.state_hash).toMatch(/^0x/);
    }
    unsub();
  });

  it("the unsubscribe handle stops the synthetic loop", () => {
    vi.useFakeTimers();
    const client = new WorldModelClient({ forceMock: true });
    const events: StreamEvent[] = [];
    const unsub = client.openStream((ev) => events.push(ev));
    const countAfterBoot = events.length;
    unsub();
    vi.advanceTimersByTime(20000);
    // No further ticks once unsubscribed.
    expect(events.length).toBe(countAfterBoot);
  });
});
