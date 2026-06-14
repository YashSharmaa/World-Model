// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  branch_ghost_group[action_id] (doc G scene graph). Each compared action gets
//  its OWN translucent ghost layer for the predicted future:
//    predicted_workload_ghosts  : a ghost capsule landing in the placed domain
//    predicted_rack_state_overlays : ghost occupancy is recoloured on the shared
//                                    slot mesh by the renderer, not duplicated
//    branch_label_anchor        : where the HTML branch label is pinned
//    lock_badge                 : a hard lock primitive shown when the exact
//                                 kernel marks the branch infeasible
//
//  Animation rules implemented here (doc G):
//    - grounded opaque, predicted translucent ghosts
//    - EXACT-BLOCKED: motion stops immediately, a lock badge appears (not a soft
//      failure animation) — the kernel verdict is a GATE, not a soft score
//    - branch compare: fixed camera, ghost cross-fade on one shared time cursor

import {
  CapsuleGeometry,
  Group,
  Mesh,
  MeshBasicMaterial,
  TorusGeometry,
  Vector3,
} from "three";
import type { CandidateResultDTO } from "../api/client";
import { branchGhostColor, PALETTE, type MaterialSet } from "./materials";
import type { RackView } from "./racks";

export interface BranchGhost {
  actionId: string;
  group: Group;
  ghostCapsule: Mesh;
  lockBadge: Mesh;
  labelAnchor: Vector3;
  blocked: boolean;
  /** rack the ghost lands in (null for hold / blocked). */
  targetRack: RackView | null;
}

export interface BranchesResult {
  root: Group;
  ghosts: Map<string, BranchGhost>;
}

/** Container group attached under workload_root by the renderer. */
export function createBranchesRoot(): BranchesResult {
  const root = new Group();
  root.name = "branch_ghost_group_container";
  return { root, ghosts: new Map() };
}

/** Rebuild the ghost layers for the current compare result + pinned set. */
export function rebuildBranches(
  result: BranchesResult,
  candidates: CandidateResultDTO[],
  pinned: string[],
  rackViews: RackView[],
  mats: MaterialSet,
): void {
  // tear down old ghosts
  for (const g of result.ghosts.values()) result.root.remove(g.group);
  result.ghosts.clear();

  const viewByDomain = new Map(rackViews.map((v) => [`dom-${v.rackId}`, v]));

  pinned.forEach((actionId, branchIndex) => {
    const cand = candidates.find((c) => c.action_id === actionId);
    if (!cand) return;

    const group = new Group();
    group.name = `branch_ghost_group_${actionId}`;

    const annot = cand.rollout?.scene_annotations;
    const targetDomain = annot?.recommended_domain_ids?.[0] ?? null;
    const targetRack = targetDomain ? (viewByDomain.get(targetDomain) ?? null) : null;

    // predicted_workload_ghosts — a translucent capsule, branch-tinted.
    const ghostMat = mats.ghostCapsule.clone();
    ghostMat.color.copy(branchGhostColor(branchIndex));
    ghostMat.emissive.copy(branchGhostColor(branchIndex)).multiplyScalar(0.3);
    const ghost = new Mesh(new CapsuleGeometry(0.24, 1.0, 4, 8), ghostMat);
    ghost.rotation.z = Math.PI / 2;
    ghost.visible = false; // revealed as the cursor advances
    group.add(ghost);

    // lock_badge — a red torus "lock" ring shown only when exact-blocked.
    const lockMat = new MeshBasicMaterial({
      color: PALETTE.blocked,
      transparent: true,
      opacity: 0.95,
      depthWrite: false,
    });
    const lock = new Mesh(new TorusGeometry(0.45, 0.12, 8, 24), lockMat);
    lock.visible = !cand.safe;
    const labelAnchor = targetRack
      ? targetRack.center.clone().add(new Vector3(0, 3.2, 0))
      : new Vector3(-7.5, 3.0 + branchIndex * 0.6, 0);
    lock.position.copy(labelAnchor);
    group.add(lock);

    result.root.add(group);
    result.ghosts.set(actionId, {
      actionId,
      group,
      ghostCapsule: ghost,
      lockBadge: lock,
      labelAnchor,
      blocked: !cand.safe,
      targetRack,
    });
  });
}

/**
 * Drive all ghosts to a shared time cursor (0..horizon). For an exact-blocked
 * branch the ghost never moves — motion STOPS at the blocker and the lock badge
 * stays lit (hard state, not a soft fail). `fade` cross-fades focus between
 * branches in compare mode.
 */
export function updateBranches(
  result: BranchesResult,
  cursorS: number,
  horizonS: number,
  focusedActionId: string | null,
): void {
  const lanePos = new Vector3(-7.5, 1.2, 0);
  for (const ghost of result.ghosts.values()) {
    const isFocus = focusedActionId == null || focusedActionId === ghost.actionId;
    const baseOpacity = isFocus ? 0.5 : 0.14; // cross-fade dim for non-focused

    if (ghost.blocked) {
      // motion stops immediately; only the lock badge animates (gentle pulse).
      ghost.ghostCapsule.visible = false;
      ghost.lockBadge.visible = true;
      ghost.lockBadge.rotation.z += 0.02;
      (ghost.lockBadge.material as MeshBasicMaterial).opacity =
        0.6 + 0.35 * Math.abs(Math.sin(cursorS / 60));
      continue;
    }

    ghost.lockBadge.visible = false;
    if (!ghost.targetRack) {
      // hold: ghost lingers in the queue lane (translucent), no commit.
      ghost.ghostCapsule.visible = true;
      ghost.ghostCapsule.position.copy(lanePos);
      (ghost.ghostCapsule.material as { opacity: number }).opacity = baseOpacity * 0.7;
      continue;
    }

    // commit happens over the first 180 s; afterward the ghost sits in the rack.
    const commitWindow = Math.min(180, horizonS);
    const p = Math.max(0, Math.min(1, cursorS / commitWindow));
    const eased = 1 - Math.pow(1 - p, 3);
    ghost.ghostCapsule.visible = cursorS > 0;
    ghost.ghostCapsule.position.lerpVectors(lanePos, ghost.targetRack.center, eased);
    (ghost.ghostCapsule.material as { opacity: number }).opacity = baseOpacity;
  }
}
