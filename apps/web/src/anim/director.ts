// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  CinematicDirector — a custom, robust camera sequencer (NO Theatre studio,
//  to keep the bundle lean). It LAYERS ON TOP of routes/modes.ts CameraDirector:
//
//    - The mode CameraDirector owns the camera by default (mode-preset lerp +
//      limited user orbit). That behaviour is preserved.
//    - When a cinematic shot or the golden-path beat sequence is engaged, the
//      CinematicDirector takes over with eased (smoothstep/cubic) transitions,
//      then RELEASES control back to the mode director when it finishes — so the
//      two never fight for the camera.
//
//  Named shots: establishing wide, pod 3/4, rack close-up, compare side-by-side.
//  Beat sequencer: plays the golden-path storyboard (reveal -> branch A ->
//  branch B -> hold -> compare -> why-not -> recommendation) timed off the
//  store's shared cursor via beat callbacks. Gentle idle drift keeps the frame
//  alive during holds.

import { PerspectiveCamera, Vector3 } from "three";
import type { Store } from "../state/store";

export interface Shot {
  name: string;
  position: Vector3;
  target: Vector3;
}

/** Named cinematic shots framing the demo pod. */
export const SHOTS: Record<string, Shot> = {
  establishing: {
    name: "Establishing wide",
    position: new Vector3(16, 12, 18),
    target: new Vector3(0, 2.2, 0),
  },
  pod34: {
    name: "Pod 3/4",
    position: new Vector3(11, 8, 12),
    target: new Vector3(0, 2.4, 0),
  },
  rackCloseup: {
    name: "Rack close-up",
    position: new Vector3(5.5, 4.5, 6.0),
    target: new Vector3(0.5, 2.6, 0),
  },
  compareSideBySide: {
    name: "Compare side-by-side",
    position: new Vector3(0, 9.5, 16),
    target: new Vector3(0, 2.6, 0),
  },
  recommendation: {
    name: "Recommendation hero",
    position: new Vector3(-4, 6, 11),
    target: new Vector3(0, 2.5, 0),
  },
};

/** A single beat in the golden-path sequence. */
export interface Beat {
  /** seconds from the start of the sequence at which this beat fires. */
  at: number;
  /** the shot to transition into. */
  shot: keyof typeof SHOTS;
  /** transition duration (seconds). */
  duration: number;
  /** optional caption / side-effect callback when the beat begins. */
  onBegin?: () => void;
}

/** smoothstep easing (0..1) — soft in/out, no overshoot. */
function smootherstep(t: number): number {
  const x = Math.min(1, Math.max(0, t));
  return x * x * x * (x * (x * 6 - 15) + 10);
}

export class CinematicDirector {
  /** when false the mode CameraDirector owns the camera (default). */
  private engaged = false;

  // active transition
  private fromPos = new Vector3();
  private fromTarget = new Vector3();
  private toPos = new Vector3();
  private toTarget = new Vector3();
  private transT = 0;
  private transDur = 0;

  // live target the camera looks at (lerped) + idle drift accumulator.
  private liveTarget = new Vector3(0, 2.2, 0);
  private driftTime = 0;

  // golden-path sequence state.
  private beats: Beat[] = [];
  private seqTime = 0;
  private seqActive = false;
  private nextBeat = 0;

  constructor(private readonly camera: PerspectiveCamera) {}

  /** True while the cinematic layer is driving the camera. */
  isEngaged(): boolean {
    return this.engaged || this.seqActive;
  }

  /** True while the golden-path beat SEQUENCE is still playing (not just a cut). */
  isSequenceActive(): boolean {
    return this.seqActive;
  }

  /** Cut to a named shot with an eased transition; engages cinematic control. */
  cutTo(shot: keyof typeof SHOTS, duration = 1.6): void {
    const s = SHOTS[shot];
    this.beginTransition(s.position, s.target, duration);
    this.engaged = true;
  }

  /** Release camera control back to the mode CameraDirector. */
  release(): void {
    this.engaged = false;
    this.seqActive = false;
  }

  /**
   * Start the golden-path beat sequence. Beats are timed in seconds of the
   * sequence clock (which advances with update's dt). Pass beats from
   * {@link defaultGoldenPath} (which binds the store side-effects) or a custom
   * storyboard.
   */
  playGoldenPath(beats: Beat[]): void {
    this.beats = [...beats].sort((a, b) => a.at - b.at);
    this.seqTime = 0;
    this.nextBeat = 0;
    this.seqActive = true;
    this.engaged = true;
  }

  /** Stop the sequence (camera holds, then mode director resumes on release). */
  stopGoldenPath(): void {
    this.seqActive = false;
  }

  private beginTransition(pos: Vector3, target: Vector3, duration: number): void {
    this.fromPos.copy(this.camera.position);
    this.fromTarget.copy(this.liveTarget);
    this.toPos.copy(pos);
    this.toTarget.copy(target);
    this.transT = 0;
    this.transDur = Math.max(0.001, duration);
  }

  /**
   * Advance the cinematic camera. Returns true if it OWNED the camera this
   * frame (so main.ts can skip the mode CameraDirector.tick), false if control
   * is yielded back to the mode director.
   */
  update(dt: number): boolean {
    // drive the beat sequence first (may begin a transition).
    if (this.seqActive) {
      this.seqTime += dt;
      while (
        this.nextBeat < this.beats.length &&
        this.seqTime >= this.beats[this.nextBeat].at
      ) {
        const b = this.beats[this.nextBeat];
        const s = SHOTS[b.shot];
        this.beginTransition(s.position, s.target, b.duration);
        b.onBegin?.();
        this.nextBeat += 1;
      }
      // sequence finishes a beat or two after the last beat begins.
      if (
        this.nextBeat >= this.beats.length &&
        this.transT >= this.transDur &&
        this.beats.length > 0 &&
        this.seqTime > this.beats[this.beats.length - 1].at + 4
      ) {
        this.seqActive = false;
      }
    }

    if (!this.isEngaged()) return false;

    // eased transition between shots.
    this.transT = Math.min(this.transDur, this.transT + dt);
    const t = smootherstep(this.transT / this.transDur);
    this.camera.position.lerpVectors(this.fromPos, this.toPos, t);
    this.liveTarget.lerpVectors(this.fromTarget, this.toTarget, t);

    // gentle idle drift once settled, so holds never look frozen.
    this.driftTime += dt;
    if (this.transT >= this.transDur) {
      const driftX = Math.sin(this.driftTime * 0.18) * 0.5;
      const driftY = Math.cos(this.driftTime * 0.13) * 0.25;
      this.camera.position.x = this.toPos.x + driftX;
      this.camera.position.y = this.toPos.y + driftY;
    }

    this.camera.lookAt(this.liveTarget);
    return true;
  }
}

/** Optional per-beat caption sink (drives the on-screen muted caption). */
export type CaptionSink = (text: string | null) => void;

/**
 * The built-in golden-path storyboard — the eight doc-G beats, with each
 * `onBegin` wired to drive REAL store state so the camera move and the decision
 * narrative stay in lockstep:
 *
 *   1 current pod state   setFocusedWorkload(focus) + runCompare()
 *   2 surface the choice  (candidate chips are now populated by the compare)
 *   3 play branch A       playBranch(feasible / recommended)
 *   4 play branch B       reset cursor + playBranch(blocked) → lock badge
 *   5 play hold           reset cursor + playBranch(hold)
 *   6 freeze and compare  freezeAt(horizon) → delta ribbon reads the outcome
 *   7 show why-not        showWhyNot(blocked) → lock badge + why-not card
 *   8 resolve             resolveToRecommendation() → bright recommended branch
 *
 * The feasible / blocked / hold action ids are LOOKED UP from the live compare
 * result at beat time (never hardcoded), so the storyboard tracks whatever
 * candidate alphabet `runCompare` derived from the snapshot.
 */
export function defaultGoldenPath(store: Store, caption?: CaptionSink): Beat[] {
  const recommendedId = (): string | null =>
    store.get().compare?.recommended_action_id ?? null;
  const blockedId = (): string | null =>
    store.get().compare?.candidates.find((c) => !c.safe)?.action_id ?? null;
  const holdId = (): string | null =>
    store.get().compare?.candidates.find(
      (c) => store.candidateAction(c.action_id)?.type === "hold",
    )?.action_id ?? "hold";

  const cap = (text: string): void => caption?.(text);

  return [
    {
      at: 0.0,
      shot: "establishing",
      duration: 2.2,
      onBegin: () => {
        const focus =
          store.get().focusedWorkload ??
          store.get().snapshot?.entities.workloads.find(
            (w) => w.status === "pending",
          )?.workload_uid ??
          null;
        store.setFocusedWorkload(focus);
        void store.runCompare();
        cap("Current pod state");
      },
    },
    {
      at: 4.0,
      shot: "pod34",
      duration: 2.0,
      onBegin: () => {
        store.setMode("decision-studio");
        cap("Three safe choices");
      },
    },
    {
      at: 9.0,
      shot: "rackCloseup",
      duration: 1.8,
      onBegin: () => {
        const a = recommendedId();
        if (a) store.playBranch(a);
        cap("Admit now, domain A");
      },
    },
    {
      at: 16.0,
      shot: "compareSideBySide",
      duration: 2.0,
      onBegin: () => {
        const b = blockedId();
        store.setCursor(0);
        if (b) store.playBranch(b);
        cap("Feasible, but worse locality");
      },
    },
    {
      at: 23.0,
      shot: "pod34",
      duration: 2.0,
      onBegin: () => {
        const h = holdId();
        store.setCursor(0);
        if (h) store.playBranch(h);
        cap("Wait one window");
      },
    },
    {
      at: 29.0,
      shot: "compareSideBySide",
      duration: 1.6,
      onBegin: () => {
        store.freezeAt();
        cap("Compare futures");
      },
    },
    {
      at: 37.0,
      shot: "recommendation",
      duration: 1.8,
      onBegin: () => {
        const b = blockedId();
        if (b) void store.showWhyNot(b);
        cap("Rejected for exact blockers and higher risk");
      },
    },
    {
      at: 42.0,
      shot: "recommendation",
      duration: 2.0,
      onBegin: () => {
        store.resolveToRecommendation();
        cap("Predict before you place");
      },
    },
  ];
}
