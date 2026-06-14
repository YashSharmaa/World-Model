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
  /**
   * The exact CandidateAction commands sent to /actions/compare for the active
   * compare result, keyed by action_id. The why-not card looks up the full
   * (type + params) command here instead of re-synthesising it, so the exact
   * blocker certificate is computed for the real submitted action.
   */
  compareCandidates: Record<string, CandidateAction>;
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
      compareCandidates: {},
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
    this.set({
      focusedWorkload: uid,
      compare: null,
      compareCandidates: {},
      pinnedActions: [],
      whyNotActionId: null,
      cursorS: 0,
    });
  }

  setHorizon(horizonS: number): void {
    this.set({ horizonS });
  }

  // ── compare / branches ────────────────────────────────────────────────
  /**
   * Build the kernel-safe candidate ALPHABET for the focused workload directly
   * from the snapshot — `hold` plus one `admit_and_place` per rack domain in
   * `topology_domains` — and ask the backend to rank + roll them out.
   *
   * The candidate alphabet is derived, never hardcoded:
   *   - `hold` (the declared baseline) is always present.
   *   - the highest-headroom feasible rack domain becomes the primary admit
   *     target (clean block, recommended branch).
   *   - a deliberately INFEASIBLE rack domain is also included so the exact
   *     kernel's BLOCKED / why-not path is exercised. A domain is infeasible
   *     here when it cannot host the workload's whole required gang
   *     (`free_gpu < gpu_req`) — e.g. the nearly-full / drained `dom-rack-b2`.
   *
   * Candidate count is kept shape-stable at three (doc G / MLX compile rule):
   * baseline hold + one feasible admit + one blocked admit.
   */
  async runCompare(): Promise<void> {
    const snap = this.state.snapshot;
    const wl = this.state.focusedWorkload;
    if (!snap || !wl) return;

    const workload = snap.entities.workloads.find((w) => w.workload_uid === wl);
    const gpuReq = workload?.gpu_req ?? 16;
    const domainPolicy = workload?.topology_policy ?? "required";

    // Rack-level domains only (a gang lands in exactly one rack domain when the
    // topology policy is `required` — doc S gang_atomicity).
    const rackDomains = snap.entities.topology_domains.filter(
      (d) => d.level === "rack",
    );

    const buildAdmit = (actionId: string, domainId: string): CandidateAction => ({
      action_id: actionId,
      type: "admit_and_place",
      params: {
        workload_uid: wl,
        cluster_queue_id: workload?.cluster_queue_id ?? "cq-shared",
        resource_flavor_id: "h100-80g",
        domain_policy: domainPolicy,
        domain_id_or_set: [domainId],
      },
    });

    // Feasible target: the rack domain with the most free GPUs that can host the
    // whole gang. Falls back to the highest-headroom domain if none fit.
    const feasible = rackDomains
      .filter((d) => d.free_gpu >= gpuReq)
      .sort((a, b) => b.free_gpu - a.free_gpu);
    const feasibleDomain =
      feasible[0]?.domain_id ??
      [...rackDomains].sort((a, b) => b.headroom_score - a.headroom_score)[0]
        ?.domain_id ??
      "dom-rack-a1";

    // Infeasible target: prefer the canonical drained rack `dom-rack-b2` when it
    // is present and genuinely cannot host the gang; otherwise pick the tightest
    // rack domain that still fails the gang-atomicity capacity test.
    const blockedCandidates = rackDomains
      .filter((d) => d.free_gpu < gpuReq && d.domain_id !== feasibleDomain)
      .sort((a, b) => a.free_gpu - b.free_gpu);
    const blockedDomain =
      blockedCandidates.find((d) => d.domain_id === "dom-rack-b2")?.domain_id ??
      blockedCandidates[0]?.domain_id ??
      "dom-rack-b2";

    const candidates: CandidateAction[] = [
      { action_id: "hold", type: "hold", params: { workload_uid: wl, hold_windows: 1 } },
      buildAdmit("admit-a", feasibleDomain),
      buildAdmit("admit-b", blockedDomain),
    ];

    const compare = await this.client.compareActions({
      state_hash: snap.state_hash,
      candidate_actions: candidates,
      horizon_s: this.state.horizonS,
      include_rollouts: true,
    });

    // Pin the safe survivors first so the blocked branch never crowds out a
    // playable ghost, while still keeping the blocked branch pinnable for its
    // lock badge. Keep the candidate map so why-not can recover type + params.
    const safeFirst = [...compare.candidates].sort((a, b) => {
      if (a.safe !== b.safe) return a.safe ? -1 : 1;
      return (a.rank ?? 99) - (b.rank ?? 99);
    });
    this.set({
      compare,
      compareCandidates: Object.fromEntries(candidates.map((c) => [c.action_id, c])),
      pinnedActions: safeFirst.slice(0, MAX_PINNED).map((c) => c.action_id),
      cursorS: 0,
      whyNotActionId: null,
    });
  }

  /** The exact CandidateAction command submitted for `actionId`, or null. */
  candidateAction(actionId: string): CandidateAction | null {
    return this.state.compareCandidates[actionId] ?? null;
  }

  togglePinned(actionId: string): void {
    const cur = this.state.pinnedActions;
    if (cur.includes(actionId)) {
      this.set({ pinnedActions: cur.filter((a) => a !== actionId) });
    } else if (cur.length < MAX_PINNED) {
      this.set({ pinnedActions: [...cur, actionId] });
    }
  }

  /**
   * Bring one branch to the front of the pinned set so the grounded scene plays
   * THAT action's rollout (golden-path beats "play A", "play B", "play hold").
   * Switches to counterfactual-compare so the renderer drives occupancy from
   * `pinnedActions[0]` (fixed camera, shared cursor — doc G compare rule), then
   * rewinds the cursor and plays from t=0.
   */
  playBranch(actionId: string): void {
    if (!this.state.compare) return;
    const exists = this.state.compare.candidates.some(
      (c) => c.action_id === actionId,
    );
    if (!exists) return;
    const rest = this.state.pinnedActions.filter((a) => a !== actionId);
    const ordered = [actionId, ...rest].slice(0, MAX_PINNED);
    this.set({
      mode: "counterfactual-compare",
      pinnedActions: ordered,
      cursorS: 0,
      playing: true,
      whyNotActionId: null,
    });
  }

  /**
   * Freeze the shared time cursor at the horizon (or a given time) so the delta
   * ribbon can be read at the end-of-horizon outcome — golden-path beat "Freeze
   * and compare". Stops playback; the scene holds on the frozen rollout point.
   */
  freezeAt(tS: number = this.state.horizonS): void {
    this.set({
      playing: false,
      cursorS: Math.max(0, Math.min(this.state.horizonS, tS)),
    });
  }

  // ── why-not ───────────────────────────────────────────────────────────
  /**
   * Open the why-not card for a rejected action (golden-path beat "Show
   * why-not"). The rejected branch is also pinned so its scene-graph lock badge
   * lights up alongside the card (doc G: "Rejected option gets a lock badge and
   * a why-not card listing exact blockers and predicted downside").
   */
  async showWhyNot(actionId: string): Promise<void> {
    const cur = this.state.pinnedActions;
    const pinned = cur.includes(actionId)
      ? cur
      : [...cur, actionId].slice(0, MAX_PINNED);
    this.set({ whyNotActionId: actionId, pinnedActions: pinned });
  }

  clearWhyNot(): void {
    this.set({ whyNotActionId: null });
  }

  /**
   * Golden-path beat "Resolve to recommendation": close the why-not card, return
   * to decision-studio (where the renderer brightens the recommended branch and
   * dims the alternatives), re-pin the recommended branch first, and replay it
   * from t=0 so the demo ends on the bright recommended future.
   */
  resolveToRecommendation(): void {
    const cmp = this.state.compare;
    const recId = cmp?.recommended_action_id ?? null;
    const ordered = recId
      ? [
          recId,
          ...this.state.pinnedActions.filter((a) => a !== recId),
        ].slice(0, MAX_PINNED)
      : this.state.pinnedActions;
    this.set({
      mode: "decision-studio",
      whyNotActionId: null,
      pinnedActions: ordered,
      cursorS: 0,
      playing: true,
    });
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
