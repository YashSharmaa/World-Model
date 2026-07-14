// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  The four v1 modes (doc G), implemented as a SINGLE-SHELL application with
//  mode shifts — not a navigation-heavy product, and never a scene remount:
//
//    decision-studio        : default. Pick a workload, compare safe actions,
//                             play one branch.
//    counterfactual-compare : same camera, up to 3 ghost branches on ONE shared
//                             time cursor (fixed camera + cross-fade).
//    replay-calibration     : predicted branch overlaid on realized history for a
//                             recorded scenario.
//    inspect-drawer         : a slide-over detail mode that keeps the camera
//                             fixed and the scene stable.
//
//  This module also owns the fixed camera PRESETS (recordability: limited orbit
//  only, no free-fly that would interrupt the causal story — doc G).

import { PerspectiveCamera, Vector3 } from "three";
import type { StudioMode } from "../state/store";

export interface CameraPreset {
  name: string;
  position: Vector3;
  target: Vector3;
}

/** Fixed presets framing the 2-4 rack demo pod for recordable shots. */
export const CAMERA_PRESETS: Record<StudioMode, CameraPreset> = {
  "decision-studio": {
    name: "Pod overview",
    position: new Vector3(11, 9, 13),
    target: new Vector3(0, 2, 0),
  },
  "counterfactual-compare": {
    name: "Compare (fixed)",
    position: new Vector3(0, 10, 16),
    target: new Vector3(0, 2.5, 0),
  },
  "replay-calibration": {
    name: "Replay",
    position: new Vector3(-9, 8, 14),
    target: new Vector3(0, 2, 0),
  },
  "inspect-drawer": {
    name: "Inspect (offset)",
    position: new Vector3(13, 7, 9),
    target: new Vector3(-2, 2, 0),
  },
};

/**
 * Smoothly drive the camera toward the active mode's preset. Camera motion is
 * a slow lerp; limited orbit is layered on top by main.ts. Compare mode pins
 * the camera so all branches read against the same framing.
 */
export class CameraDirector {
  private current: CameraPreset;
  constructor(private readonly camera: PerspectiveCamera) {
    this.current = CAMERA_PRESETS["decision-studio"];
    this.snapTo(this.current);
  }

  goTo(mode: StudioMode): void {
    this.current = CAMERA_PRESETS[mode];
  }

  /** Lerp the camera toward the active preset each frame. */
  tick(orbitOffset: { x: number; y: number }): void {
    const p = this.current;
    // limited orbit: small azimuth/elevation offset around the preset.
    const radius = p.position.distanceTo(p.target);
    const baseAngle = Math.atan2(p.position.z - p.target.z, p.position.x - p.target.x);
    const angle = baseAngle + orbitOffset.x * 0.6;
    const height = p.position.y + orbitOffset.y * 4;
    const desired = new Vector3(
      p.target.x + Math.cos(angle) * radius,
      Math.max(2, height),
      p.target.z + Math.sin(angle) * radius,
    );
    this.camera.position.lerp(desired, 0.08);
    this.camera.lookAt(p.target);
  }

  private snapTo(p: CameraPreset): void {
    this.camera.position.copy(p.position);
    this.camera.lookAt(p.target);
  }
}
