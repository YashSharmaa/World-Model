// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  workload_root — the moving layer (doc G scene graph):
//    queue_lane           : a staging rail off to the side of the pod
//    active_workload_tokens : capsules for pending/running workloads
//    allocation_paths     : a line from the queue lane to the placed domain
//
//  Workloads are capsules, not avatars. Queue-to-rack movement is a short
//  segmented commit (500-900 ms) at the placement event, not continuous travel
//  (doc G "Workload movement"). Grounded tokens are opaque; predicted/branch
//  ghosts live in branches.ts.

import {
  BufferGeometry,
  CapsuleGeometry,
  Group,
  Line,
  Mesh,
  MeshStandardMaterial,
  Vector3,
} from "three";
import type { WorkloadDTO } from "../api/client";
import { PALETTE, type MaterialSet } from "./materials";
import type { RackView } from "./racks";

/**
 * Fixed commit-tween duration (seconds). Queue-to-rack movement is a short
 * segmented commit of 500-900 ms at the placement event — NOT continuous travel
 * mapped to scene-cursor seconds (doc G animation_rules "Workload movement").
 */
export const COMMIT_TWEEN_S = 0.7; // 700 ms, inside the 500-900 ms band

/** A spawned workload token with its lane home and current target. */
export interface WorkloadToken {
  workloadUid: string;
  mesh: Mesh;
  lanePos: Vector3;
  gpuReq: number;
  /** active wall-clock commit tween, or null when settled / idle. */
  commit: {
    rack: RackView;
    /** wall-clock seconds elapsed since the tween was triggered. */
    elapsed: number;
  } | null;
}

export interface WorkloadResult {
  root: Group;
  queueLane: Group;
  tokens: Map<string, WorkloadToken>;
  allocationPaths: Group;
}

const LANE_X = -7.5;
const LANE_Z_STEP = 1.1;

/** Build the queue lane + one capsule per pending/running workload. */
export function buildWorkloads(
  workloads: WorkloadDTO[],
  mats: MaterialSet,
): WorkloadResult {
  const root = new Group();
  root.name = "workload_root";

  const queueLane = new Group();
  queueLane.name = "queue_lane";
  root.add(queueLane);

  const allocationPaths = new Group();
  allocationPaths.name = "allocation_paths";
  root.add(allocationPaths);

  const tokens = new Map<string, WorkloadToken>();

  // Show pending first (decision targets), then running.
  const ordered = [...workloads].sort((a, b) => {
    const rank = (w: WorkloadDTO) => (w.status === "pending" ? 0 : 1);
    return rank(a) - rank(b) || b.priority - a.priority;
  });

  ordered.forEach((wl, i) => {
    // Capsule length encodes GPU ask; thin radius keeps it a "card in motion".
    const len = 0.4 + Math.min(2.0, wl.gpu_req / 16);
    const geo = new CapsuleGeometry(0.22, len, 4, 8);
    const mesh = new Mesh(geo, mats.workloadCapsule.clone());
    mesh.rotation.z = Math.PI / 2;
    const lanePos = new Vector3(LANE_X, 1.0 + i * 0.05, (i - ordered.length / 2) * LANE_Z_STEP);
    mesh.position.copy(lanePos);
    mesh.userData = { kind: "workload", id: wl.workload_uid };
    // Running workloads already committed read dimmer (grounded but settled).
    if (wl.status !== "pending") {
      (mesh.material as { opacity: number; transparent: boolean }).opacity = 0.6;
      (mesh.material as { transparent: boolean }).transparent = true;
    }
    queueLane.add(mesh);
    tokens.set(wl.workload_uid, {
      workloadUid: wl.workload_uid,
      mesh,
      lanePos,
      gpuReq: wl.gpu_req,
      commit: null,
    });
  });

  return { root, queueLane, tokens, allocationPaths };
}

/**
 * Draw / update an allocation path from a workload's lane home to the centre of
 * its placed rack. Called when a rollout commits a placement.
 */
export function setAllocationPath(
  result: WorkloadResult,
  workloadUid: string,
  rack: RackView | null,
  mats: MaterialSet,
): void {
  // Clear any prior path for this workload.
  const existing = result.allocationPaths.getObjectByName(`path_${workloadUid}`);
  if (existing) result.allocationPaths.remove(existing);
  const token = result.tokens.get(workloadUid);
  if (!token || !rack) return;

  const from = token.lanePos.clone();
  const to = rack.center.clone();
  const mid = from.clone().lerp(to, 0.5).add(new Vector3(0, 1.6, 0));
  const geo = new BufferGeometry().setFromPoints([from, mid, to]);
  const line = new Line(geo, mats.allocationPath);
  line.name = `path_${workloadUid}`;
  result.allocationPaths.add(line);
}

/**
 * Emphasise the workloads named in a rollout's `highlight_workload_ids`
 * scene-annotation (doc G scene_annotations): the highlighted tokens get a
 * brighter emissive pulse so the decision target reads at a glance, while every
 * other token is restored to its calm grounded emissive. `phase` is a 0..1
 * animation phase for the pulse.
 */
export function applyWorkloadHighlight(
  result: WorkloadResult,
  highlightIds: string[],
  phase = 0,
): void {
  const highlighted = new Set(highlightIds);
  const pulse = 0.25 + 0.75 * Math.abs(Math.sin(phase * Math.PI * 2));
  for (const token of result.tokens.values()) {
    const mat = token.mesh.material as MeshStandardMaterial;
    if (highlighted.has(token.workloadUid)) {
      mat.emissive.copy(PALETTE.grounded).multiplyScalar(0.35 + 0.55 * pulse);
    } else {
      mat.emissive.copy(PALETTE.grounded).multiplyScalar(0.25);
    }
    mat.needsUpdate = true;
  }
}

/**
 * Trigger (or retarget) a token's FIXED-DURATION commit tween toward its placed
 * rack. The actual easing happens in {@link updateCommitTweens} off a wall-clock
 * delta, so the move always takes {@link COMMIT_TWEEN_S} regardless of the scene
 * cursor speed (doc G workload movement: a 500-900 ms segmented commit, not
 * continuous cursor-mapped travel). Passing `rack=null` parks the token back in
 * its queue lane and cancels any tween.
 */
export function commitToken(
  result: WorkloadResult,
  workloadUid: string,
  rack: RackView | null,
): void {
  const token = result.tokens.get(workloadUid);
  if (!token) return;
  if (!rack) {
    token.commit = null;
    token.mesh.position.copy(token.lanePos);
    return;
  }
  // (Re)start the tween only when the target rack changes or none is running.
  if (!token.commit || token.commit.rack !== rack) {
    token.commit = { rack, elapsed: 0 };
  }
}

/**
 * Advance every active commit tween by `dt` wall-clock seconds and ease the
 * token from its lane home to its placed rack over a fixed {@link COMMIT_TWEEN_S}
 * window. Tokens with no active tween are left at their lane home.
 */
export function updateCommitTweens(result: WorkloadResult, dt: number): void {
  for (const token of result.tokens.values()) {
    if (!token.commit) continue;
    token.commit.elapsed += dt;
    const p = Math.max(0, Math.min(1, token.commit.elapsed / COMMIT_TWEEN_S));
    // ease-out cubic for a snappy, settled commit.
    const eased = 1 - Math.pow(1 - p, 3);
    token.mesh.position.lerpVectors(token.lanePos, token.commit.rack.center, eased);
  }
}
