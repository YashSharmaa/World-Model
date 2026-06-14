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
  Vector3,
} from "three";
import type { WorkloadDTO } from "../api/client";
import type { MaterialSet } from "./materials";
import type { RackView } from "./racks";

/** A spawned workload token with its lane home and current target. */
export interface WorkloadToken {
  workloadUid: string;
  mesh: Mesh;
  lanePos: Vector3;
  gpuReq: number;
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
    tokens.set(wl.workload_uid, { workloadUid: wl.workload_uid, mesh, lanePos, gpuReq: wl.gpu_req });
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
 * Animate a token's commit toward its placed rack as a fraction 0..1 of the
 * short commit window (the segmented move, not continuous travel).
 */
export function commitToken(
  result: WorkloadResult,
  workloadUid: string,
  rack: RackView | null,
  progress: number,
): void {
  const token = result.tokens.get(workloadUid);
  if (!token) return;
  if (!rack) {
    token.mesh.position.copy(token.lanePos);
    return;
  }
  const p = Math.max(0, Math.min(1, progress));
  // ease-out for a snappy commit
  const eased = 1 - Math.pow(1 - p, 3);
  token.mesh.position.lerpVectors(token.lanePos, rack.center, eased);
}
