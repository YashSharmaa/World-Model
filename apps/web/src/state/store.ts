// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Client state store. A tiny observable container (no framework) holding the
//  authoritative snapshot, the active mode, the focused workload, the pinned
//  comparison branches, the shared time cursor, and the latest provenance.
//
//  It consumes GET /api/v1/state/current on boot and then the GET
//  /api/v1/stream SSE channel for deltas (doc G transport rule). Scene motion is
//  always driven by backend timestamps / rollout points the store exposes — the
//  renderer only interpolates between them, never free-runs a local simulation
//  (doc G "Timeline authority").

import {
  WorldModelClient,
  type CandidateAction,
  type CompareResultDTO,
  type ReplayDTO,
  type StateSnapshotDTO,
  type StreamEvent,
} from "../api/client";

export type StudioMode =
  | "decision-studio"
  | "counterfactual-compare"
  | "replay-calibration"
  | "inspect-drawer";

/** A picked semantic entity from the Raycaster (instanceId -> identity). */
export interface Selection {
  kind: "rack" | "node" | "gpu" | "workload" | "domain";
  id: string;
}

export interface StoreState {
  mode: StudioMode;
  snapshot: StateSnapshotDTO | null;
  compare: CompareResultDTO | null;
  /** action_ids pinned for counterfactual compare (max 3 — doc G stability). */
  pinnedActions: string[];
  focusedWorkload: string | null;
  selection: Selection | null;
  whyNotActionId: string | null;
  replay: ReplayDTO | null;
  horizonS: number;
  /** shared time cursor in seconds across all branches (single time axis). */
  cursorS: number;
  /** wall-clock play state for the timeline (backend-stamped interpolation). */
  playing: boolean;
  backendOnline: boolean;
  health: string;
}

type Listener = (s: StoreState) => void;

const MAX_PINNED = 3; // doc G: keep candidate count shape-stable for MLX compile.

export class Store {
  private state: StoreState;
  private listeners = new Set<Listener>();
  private unsubStream: (() => void) | null = null;

  constructor(private readonly client: WorldModelClient) {
    this.state = {
      mode: "decision-studio",
      snapshot: null,
      compare: null,
      pinnedActions: [],
      focusedWorkload: null,
      selection: null,
      whyNotActionId: null,
      replay: null,
      horizonS: 1800,
      cursorS: 0,
      playing: false,
      backendOnline: false,
      health: "connecting…",
    };
  }

  // ── subscription ──────────────────────────────────────────────────────
  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    fn(this.state);
    return () => this.listeners.delete(fn);
  }

  get(): Readonly<StoreState> {
    return this.state;
  }

  private set(patch: Partial<StoreState>): void {
    this.state = { ...this.state, ...patch };
    for (const fn of this.listeners) fn(this.state);
  }

  // ── lifecycle ─────────────────────────────────────────────────────────
  /** Fetch the initial snapshot and open the SSE stream. */
  async boot(): Promise<void> {
    const snapshot = await this.client.getCurrentState();
    const focus =
      snapshot.entities.workloads.find((w) => w.status === "pending")
        ?.workload_uid ?? null;
    this.set({
      snapshot,
      focusedWorkload: focus,
      backendOnline: this.client.backendOnline,
      health: this.client.backendOnline ? "backend online" : "mock data (offline)",
    });
    this.openStream();
  }

  private openStream(): void {
    this.unsubStream?.();
    this.unsubStream = this.client.openStream((ev: StreamEvent) => {
      switch (ev.type) {
        case "snapshot":
          this.set({ snapshot: ev.snapshot });
          break;
        case "state_delta":
          if (this.state.snapshot && ev.patch.entities) {
            this.set({ snapshot: { ...this.state.snapshot, ...ev.patch } });
          }
          break;
        case "policy_changed":
          if (this.state.snapshot)
            this.set({
              snapshot: {
                ...this.state.snapshot,
                policy_version: ev.policy_version,
              },
            });
          break;
        case "backend_health":
          this.set({
            backendOnline: ev.healthy,
            health: ev.healthy ? "backend online" : `mock data — ${ev.detail}`,
          });
          break;
        case "branch_ready":
        case "replay_loaded":
        default:
          break;
      }
    });
  }

  dispose(): void {
    this.unsubStream?.();
    this.listeners.clear();
  }

  // ── mode & selection ──────────────────────────────────────────────────
  setMode(mode: StudioMode): void {
    this.set({ mode });
  }

  setSelection(sel: Selection | null): void {
    const patch: Partial<StoreState> = { selection: sel };
    if (sel?.kind === "workload") patch.focusedWorkload = sel.id;
    this.set(patch);
  }

  setFocusedWorkload(uid: string | null): void {
    this.set({ focusedWorkload: uid, compare: null, pinnedActions: [] });
  }

  setHorizon(horizonS: number): void {
    this.set({ horizonS });
  }

  // ── compare / branches ────────────────────────────────────────────────
  /**
   * Build the kernel-safe candidate alphabet for the focused workload and ask
   * the backend to rank + roll them out. In v1 the mock backend certifies three
   * safe actions; the real serving layer enumerates the kernel's safe set.
   */
  async runCompare(): Promise<void> {
    const snap = this.state.snapshot;
    const wl = this.state.focusedWorkload;
    if (!snap || !wl) return;

    const candidates: CandidateAction[] = [
      { action_id: "hold", type: "hold", params: { workload_uid: wl, hold_windows: 1 } },
      {
        action_id: "admit-a",
        type: "admit_and_place",
        params: {
          workload_uid: wl,
          cluster_queue_id: "cq-shared",
          resource_flavor_id: "h100-80g",
          domain_policy: "required",
          domain_id_or_set: ["dom-rack-a1"],
        },
      },
      {
        action_id: "admit-b",
        type: "admit_and_place",
        params: {
          workload_uid: wl,
          cluster_queue_id: "cq-shared",
          resource_flavor_id: "h100-80g",
          domain_policy: "required",
          domain_id_or_set: ["dom-rack-b1"],
        },
      },
    ];

    const compare = await this.client.compareActions({
      state_hash: snap.state_hash,
      candidate_actions: candidates,
      horizon_s: this.state.horizonS,
      include_rollouts: true,
    });
    this.set({
      compare,
      pinnedActions: compare.candidates.slice(0, MAX_PINNED).map((c) => c.action_id),
      cursorS: 0,
    });
  }

  togglePinned(actionId: string): void {
    const cur = this.state.pinnedActions;
    if (cur.includes(actionId)) {
      this.set({ pinnedActions: cur.filter((a) => a !== actionId) });
    } else if (cur.length < MAX_PINNED) {
      this.set({ pinnedActions: [...cur, actionId] });
    }
  }

  // ── why-not ───────────────────────────────────────────────────────────
  async showWhyNot(actionId: string): Promise<void> {
    this.set({ whyNotActionId: actionId });
  }

  clearWhyNot(): void {
    this.set({ whyNotActionId: null });
  }

  // ── replay ────────────────────────────────────────────────────────────
  async loadReplay(scenarioId: string): Promise<void> {
    const replay = await this.client.getReplay(scenarioId);
    this.set({
      replay,
      mode: "replay-calibration",
      snapshot: replay.initial_state,
      horizonS: replay.predicted.points.at(-1)?.t_s ?? this.state.horizonS,
      cursorS: 0,
    });
  }

  // ── timeline ──────────────────────────────────────────────────────────
  setCursor(tS: number): void {
    this.set({ cursorS: Math.max(0, Math.min(this.state.horizonS, tS)) });
  }

  setPlaying(playing: boolean): void {
    this.set({ playing });
  }

  /** Advance the cursor by dt seconds of scaled playback (interpolation only). */
  advance(dtScaledS: number): void {
    if (!this.state.playing) return;
    let next = this.state.cursorS + dtScaledS;
    if (next >= this.state.horizonS) {
      next = this.state.horizonS;
      this.set({ playing: false });
    }
    this.set({ cursorS: next });
  }
}
