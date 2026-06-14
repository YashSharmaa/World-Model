// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  The shared material palette — the single place the demo's visual grammar
//  lives (doc G "visual states"). The grammar separates EXACT semantics from
//  LEARNED forecasts so the recording teaches itself even when muted:
//
//    grounded (now)      -> opaque, fully saturated         (#58c4ff)
//    predicted (future)  -> translucent ghost               (#c9a4ff)
//    alternative branch  -> distinct ghost tint             (#ffcf6b)
//    exact-blocked       -> hard red, motion stops          (#ff6b7a)
//    safe / certified    -> green accent                    (#5fe3a1)
//    low confidence      -> desaturated grey, wider halo    (#8a93a6)
//
//  Materials are shared/instanced where possible to keep draw calls minimal;
//  this is a semantic read model, not a game engine.

import {
  Color,
  DoubleSide,
  LineBasicMaterial,
  MeshBasicMaterial,
  MeshStandardMaterial,
} from "three";

export const PALETTE = {
  grounded: new Color("#58c4ff"),
  predicted: new Color("#c9a4ff"),
  alt: new Color("#ffcf6b"),
  blocked: new Color("#ff6b7a"),
  safe: new Color("#5fe3a1"),
  lowConf: new Color("#8a93a6"),
  rackFrame: new Color("#2a3550"),
  slotEmpty: new Color("#1b2436"),
  slotFilled: new Color("#3f8fd0"),
  hot: new Color("#ff8a4c"),
  floor: new Color("#070a11"),
  switchMesh: new Color("#6f7da0"),
  domainHull: new Color("#58c4ff"),
  powerRail: new Color("#ffcf6b"),
  coolingPlane: new Color("#5fe3a1"),
} as const;

/** Material set shared across the scene graph; created once at boot. */
export interface MaterialSet {
  rackFrame: MeshStandardMaterial;
  nodeShelf: MeshStandardMaterial;
  /** instanced GPU slot tiles — colour comes from per-instance attributes. */
  gpuSlot: MeshStandardMaterial;
  rackHalo: MeshBasicMaterial;
  switchMesh: MeshStandardMaterial;
  interRackLine: LineBasicMaterial;
  domainHull: MeshBasicMaterial;
  powerRail: MeshBasicMaterial;
  coolingPlane: MeshBasicMaterial;
  workloadCapsule: MeshStandardMaterial;
  ghostCapsule: MeshStandardMaterial;
  allocationPath: LineBasicMaterial;
  selectionOutline: MeshBasicMaterial;
  floor: MeshStandardMaterial;
}

export function createMaterials(): MaterialSet {
  return {
    rackFrame: new MeshStandardMaterial({
      color: PALETTE.rackFrame,
      metalness: 0.3,
      roughness: 0.7,
      transparent: true,
      opacity: 0.92,
    }),
    nodeShelf: new MeshStandardMaterial({
      color: PALETTE.slotEmpty,
      metalness: 0.2,
      roughness: 0.85,
    }),
    // Instanced GPU slots use vertexColors so each tile carries its own state
    // (empty / filled / hot) without a separate material per slot.
    gpuSlot: new MeshStandardMaterial({
      vertexColors: true,
      metalness: 0.1,
      roughness: 0.6,
      emissive: new Color("#000000"),
    }),
    rackHalo: new MeshBasicMaterial({
      color: PALETTE.hot,
      transparent: true,
      opacity: 0.0, // animated up as learned hotspot risk rises (risk emergence)
      side: DoubleSide,
      depthWrite: false,
    }),
    switchMesh: new MeshStandardMaterial({
      color: PALETTE.switchMesh,
      metalness: 0.5,
      roughness: 0.4,
    }),
    interRackLine: new LineBasicMaterial({
      color: PALETTE.grounded,
      transparent: true,
      opacity: 0.45,
    }),
    domainHull: new MeshBasicMaterial({
      color: PALETTE.domainHull,
      transparent: true,
      opacity: 0.06,
      side: DoubleSide,
      depthWrite: false,
    }),
    powerRail: new MeshBasicMaterial({
      color: PALETTE.powerRail,
      transparent: true,
      opacity: 0.12,
      depthWrite: false,
    }),
    coolingPlane: new MeshBasicMaterial({
      color: PALETTE.coolingPlane,
      transparent: true,
      opacity: 0.08,
      side: DoubleSide,
      depthWrite: false,
    }),
    workloadCapsule: new MeshStandardMaterial({
      color: PALETTE.grounded,
      metalness: 0.2,
      roughness: 0.5,
      emissive: PALETTE.grounded.clone().multiplyScalar(0.25),
    }),
    // Predicted / counterfactual ghosts: translucent (doc G grounded-vs-predicted).
    ghostCapsule: new MeshStandardMaterial({
      color: PALETTE.predicted,
      transparent: true,
      opacity: 0.42,
      metalness: 0.0,
      roughness: 0.9,
      depthWrite: false,
      emissive: PALETTE.predicted.clone().multiplyScalar(0.3),
    }),
    allocationPath: new LineBasicMaterial({
      color: PALETTE.safe,
      transparent: true,
      opacity: 0.8,
    }),
    selectionOutline: new MeshBasicMaterial({
      color: PALETTE.safe,
      transparent: true,
      opacity: 0.9,
      side: DoubleSide,
      depthWrite: false,
    }),
    floor: new MeshStandardMaterial({
      color: PALETTE.floor,
      metalness: 0.0,
      roughness: 1.0,
    }),
  };
}

/** Tint a clone of the ghost material for a specific branch (alt vs predicted). */
export function branchGhostColor(branchIndex: number): Color {
  const ramp = [PALETTE.predicted, PALETTE.alt, PALETTE.safe];
  return ramp[branchIndex % ramp.length].clone();
}
