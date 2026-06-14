// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  createLoaderArm — the headline 3D-skeleton feature: a REAL rigged,
//  articulated robotic loader arm built as a three.js SkinnedMesh.
//
//  Rig (bone chain, base -> tip):
//     base  -> column -> shoulder -> elbow -> wrist -> gripper
//  A capsule/segment mesh is skinned to these bones: each vertex gets up to two
//  skinIndices/skinWeights computed from its Y position along the arm so the
//  tube bends smoothly across joints (no hard segment seams). The gripper has a
//  small two-finger claw parented near the tip bone.
//
//  Clips are authored PROGRAMMATICALLY with Quaternion/Vector keyframe tracks:
//     idle    : gentle continuous sway (shoulder/elbow breathing)
//     reach   : bend toward a target rack slot
//     place   : lower + open the gripper at the slot
//     retract : return to the idle pose
//  An AnimationMixer drives them; playPlace(targetWorldPos) sequences
//  reach -> place -> retract -> idle so the arm visibly performs a "place" when
//  the planner commits an admit_and_place (dramatising the world-model place
//  intervention). Subtle and elegant — it is a read-model flourish, not a toy.

import {
  AnimationClip,
  AnimationMixer,
  type AnimationAction,
  Bone,
  BoxGeometry,
  Color,
  CylinderGeometry,
  Euler,
  Float32BufferAttribute,
  Group,
  LoopOnce,
  LoopRepeat,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  Quaternion,
  QuaternionKeyframeTrack,
  Skeleton,
  SkinnedMesh,
  Uint16BufferAttribute,
  Vector3,
  VectorKeyframeTrack,
} from "three";

export interface LoaderArm {
  /** root Object3D to add to the scene (or world_root). */
  object3D: Object3D;
  mixer: AnimationMixer;
  /** advance the rig (call each frame with delta seconds). */
  update(dt: number): void;
  /** play reach->place->retract toward a WORLD-space target, then idle. */
  playPlace(targetWorldPos: Vector3): void;
  /** return to the idle loop. */
  toIdle(): void;
  dispose(): void;
}

// Heights of each joint along +Y (the rest pose stands the arm upright).
const Y_BASE = 0.0;
const Y_COLUMN = 0.6;
const Y_SHOULDER = 2.0;
const Y_ELBOW = 3.4;
const Y_WRIST = 4.4;
const Y_GRIP = 5.0;
const ARM_RADIUS = 0.16;

/** Build the rigged loader arm. */
export function createLoaderArm(): LoaderArm {
  const object3D = new Group();
  object3D.name = "loader_arm";

  // ── bone chain ─────────────────────────────────────────────────────────────
  // Each bone is positioned RELATIVE to its parent (local space), which is what
  // Skeleton expects. We keep references for clip authoring.
  const base = new Bone();
  base.name = "arm_base";
  base.position.set(0, Y_BASE, 0);

  const column = new Bone();
  column.name = "arm_column";
  column.position.set(0, Y_COLUMN - Y_BASE, 0);

  const shoulder = new Bone();
  shoulder.name = "arm_shoulder";
  shoulder.position.set(0, Y_SHOULDER - Y_COLUMN, 0);

  const elbow = new Bone();
  elbow.name = "arm_elbow";
  elbow.position.set(0, Y_ELBOW - Y_SHOULDER, 0);

  const wrist = new Bone();
  wrist.name = "arm_wrist";
  wrist.position.set(0, Y_WRIST - Y_ELBOW, 0);

  const gripper = new Bone();
  gripper.name = "arm_gripper";
  gripper.position.set(0, Y_GRIP - Y_WRIST, 0);

  base.add(column);
  column.add(shoulder);
  shoulder.add(elbow);
  elbow.add(wrist);
  wrist.add(gripper);

  const bones: Bone[] = [base, column, shoulder, elbow, wrist, gripper];
  const skeleton = new Skeleton(bones);

  // ── skinned tube geometry (one cylinder skinned across the chain) ──────────
  // A tall cylinder spanning base..gripper; we assign each vertex to the two
  // nearest joints by Y so the tube bends smoothly at every joint.
  const segHeight = Y_GRIP - Y_BASE;
  const geometry = new CylinderGeometry(
    ARM_RADIUS,
    ARM_RADIUS * 1.3,
    segHeight,
    16,
    40,
    true,
  );
  // CylinderGeometry is centred on origin; shift so its base sits at Y_BASE.
  geometry.translate(0, segHeight / 2 + Y_BASE, 0);

  const jointY = [Y_BASE, Y_COLUMN, Y_SHOULDER, Y_ELBOW, Y_WRIST, Y_GRIP];
  const pos = geometry.attributes.position;
  const skinIndices: number[] = [];
  const skinWeights: number[] = [];

  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i);
    // find the segment [j, j+1] this vertex falls in.
    let j = 0;
    while (j < jointY.length - 2 && y > jointY[j + 1]) j++;
    const yA = jointY[j];
    const yB = jointY[j + 1];
    const t = Math.min(1, Math.max(0, (y - yA) / Math.max(1e-5, yB - yA)));
    // weight blends from bone j (lower) to bone j+1 (upper).
    skinIndices.push(j, j + 1, 0, 0);
    skinWeights.push(1 - t, t, 0, 0);
  }
  geometry.setAttribute("skinIndex", new Uint16BufferAttribute(skinIndices, 4));
  geometry.setAttribute("skinWeight", new Float32BufferAttribute(skinWeights, 4));

  const armMaterial = new MeshStandardMaterial({
    color: new Color("#cdd6e4"),
    metalness: 0.85,
    roughness: 0.35,
  });
  const skinnedMesh = new SkinnedMesh(geometry, armMaterial);
  skinnedMesh.name = "loader_arm_mesh";
  skinnedMesh.castShadow = true;
  // bones move the geometry well outside its rest-pose bounds; disable culling
  // so the arm never pops out when it reaches deep into a rack.
  skinnedMesh.frustumCulled = false;
  skinnedMesh.add(base);
  skinnedMesh.bind(skeleton);

  // a chunky base plinth so the arm reads as mounted, not floating.
  const plinth = new Mesh(
    new CylinderGeometry(0.55, 0.7, 0.4, 24),
    new MeshStandardMaterial({ color: new Color("#161b27"), metalness: 0.7, roughness: 0.5 }),
  );
  plinth.position.y = 0.2;
  plinth.castShadow = true;
  plinth.receiveShadow = true;

  // ── gripper claw (parented to the gripper bone, follows the skeleton) ──────
  const claw = new Group();
  claw.name = "gripper_claw";
  const clawMat = new MeshStandardMaterial({
    color: new Color("#59d6ff"),
    emissive: new Color("#1c6fa0"),
    emissiveIntensity: 0.4,
    metalness: 0.6,
    roughness: 0.4,
  });
  const fingerGeo = new BoxGeometry(0.06, 0.4, 0.12);
  const fingerL = new Mesh(fingerGeo, clawMat);
  fingerL.name = "loader_finger_l";
  fingerL.position.set(-0.16, 0.2, 0);
  const fingerR = new Mesh(fingerGeo, clawMat);
  fingerR.name = "loader_finger_r";
  fingerR.position.set(0.16, 0.2, 0);
  const palm = new Mesh(new BoxGeometry(0.4, 0.12, 0.24), clawMat);
  claw.add(fingerL, fingerR, palm);
  gripper.add(claw);

  object3D.add(plinth, skinnedMesh);

  // ── animation: programmatic clips on the bones ─────────────────────────────
  const mixer = new AnimationMixer(skinnedMesh);

  // helper: a quaternion from an Euler rotation (radians).
  const q = (x: number, y: number, z: number): number[] => {
    const e = new Euler(x, y, z);
    const quat = new Quaternion().setFromEuler(e);
    return [quat.x, quat.y, quat.z, quat.w];
  };

  // IDLE — a gentle 4s breathing sway on shoulder + elbow, looping.
  const idleClip = new AnimationClip("idle", 4, [
    new QuaternionKeyframeTrack(
      `${shoulder.name}.quaternion`,
      [0, 2, 4],
      [...q(0.05, 0, 0), ...q(-0.05, 0.06, 0), ...q(0.05, 0, 0)],
    ),
    new QuaternionKeyframeTrack(
      `${elbow.name}.quaternion`,
      [0, 2, 4],
      [...q(-0.08, 0, 0), ...q(0.06, 0, 0), ...q(-0.08, 0, 0)],
    ),
    new QuaternionKeyframeTrack(
      `${column.name}.quaternion`,
      [0, 2, 4],
      [...q(0, 0.04, 0), ...q(0, -0.04, 0), ...q(0, 0.04, 0)],
    ),
  ]);

  // REACH — bend the chain forward/down over ~1.2s (target column yaw is set at
  // runtime by re-pointing the whole rig; the clip handles the bend posture).
  const reachClip = new AnimationClip("reach", 1.2, [
    new QuaternionKeyframeTrack(
      `${shoulder.name}.quaternion`,
      [0, 1.2],
      [...q(0.05, 0, 0), ...q(0.55, 0, 0)],
    ),
    new QuaternionKeyframeTrack(
      `${elbow.name}.quaternion`,
      [0, 1.2],
      [...q(-0.08, 0, 0), ...q(-0.9, 0, 0)],
    ),
    new QuaternionKeyframeTrack(
      `${wrist.name}.quaternion`,
      [0, 1.2],
      [...q(0, 0, 0), ...q(0.45, 0, 0)],
    ),
  ]);

  // PLACE — small lower + open the claw fingers (finger scale via position).
  const placeClip = new AnimationClip("place", 1.0, [
    new QuaternionKeyframeTrack(
      `${shoulder.name}.quaternion`,
      [0, 0.5, 1.0],
      [...q(0.55, 0, 0), ...q(0.7, 0, 0), ...q(0.7, 0, 0)],
    ),
    new QuaternionKeyframeTrack(
      `${elbow.name}.quaternion`,
      [0, 0.5, 1.0],
      [...q(-0.9, 0, 0), ...q(-1.05, 0, 0), ...q(-1.05, 0, 0)],
    ),
    // open the gripper: move the two finger meshes apart (named-object tracks;
    // the mixer resolves these by object name under the skinned-mesh root).
    new VectorKeyframeTrack(
      `${fingerL.name}.position`,
      [0, 0.5, 1.0],
      [-0.16, 0.2, 0, -0.26, 0.2, 0, -0.26, 0.2, 0],
    ),
    new VectorKeyframeTrack(
      `${fingerR.name}.position`,
      [0, 0.5, 1.0],
      [0.16, 0.2, 0, 0.26, 0.2, 0, 0.26, 0.2, 0],
    ),
  ]);

  // RETRACT — unwind back to the idle posture over 1.0s, fingers closed.
  const retractClip = new AnimationClip("retract", 1.0, [
    new QuaternionKeyframeTrack(
      `${shoulder.name}.quaternion`,
      [0, 1.0],
      [...q(0.7, 0, 0), ...q(0.05, 0, 0)],
    ),
    new QuaternionKeyframeTrack(
      `${elbow.name}.quaternion`,
      [0, 1.0],
      [...q(-1.05, 0, 0), ...q(-0.08, 0, 0)],
    ),
    new QuaternionKeyframeTrack(
      `${wrist.name}.quaternion`,
      [0, 1.0],
      [...q(0.45, 0, 0), ...q(0, 0, 0)],
    ),
    new VectorKeyframeTrack(
      `${fingerL.name}.position`,
      [0, 1.0],
      [-0.26, 0.2, 0, -0.16, 0.2, 0],
    ),
    new VectorKeyframeTrack(
      `${fingerR.name}.position`,
      [0, 1.0],
      [0.26, 0.2, 0, 0.16, 0.2, 0],
    ),
  ]);

  const idle: AnimationAction = mixer.clipAction(idleClip);
  idle.setLoop(LoopRepeat, Infinity);
  idle.play();

  const reach = mixer.clipAction(reachClip);
  const place = mixer.clipAction(placeClip);
  const retract = mixer.clipAction(retractClip);
  for (const a of [reach, place, retract]) {
    a.setLoop(LoopOnce, 1);
    a.clampWhenFinished = true;
  }

  // sequence state machine driven off mixer "finished" events.
  type Phase = "idle" | "reach" | "place" | "retract";
  let phase: Phase = "idle";

  function crossTo(action: AnimationAction, from?: AnimationAction): void {
    action.reset();
    action.enabled = true;
    action.setEffectiveWeight(1);
    if (from) {
      action.crossFadeFrom(from, 0.25, false);
    }
    action.play();
  }

  mixer.addEventListener("finished", (e: { action: AnimationAction }) => {
    if (e.action === reach && phase === "reach") {
      phase = "place";
      crossTo(place, reach);
    } else if (e.action === place && phase === "place") {
      phase = "retract";
      crossTo(retract, place);
    } else if (e.action === retract && phase === "retract") {
      phase = "idle";
      idle.reset();
      idle.enabled = true;
      idle.setEffectiveWeight(1);
      idle.crossFadeFrom(retract, 0.3, false);
      idle.play();
    }
  });

  // scratch for re-pointing the rig toward a world target.
  const tmp = new Vector3();

  function aimAt(targetWorldPos: Vector3): void {
    // yaw the whole rig so it faces the target in the XZ plane.
    object3D.getWorldPosition(tmp);
    const dx = targetWorldPos.x - tmp.x;
    const dz = targetWorldPos.z - tmp.z;
    const yaw = Math.atan2(dx, dz);
    object3D.rotation.y = yaw;
  }

  return {
    object3D,
    mixer,
    update(dt: number): void {
      mixer.update(dt);
    },
    playPlace(targetWorldPos: Vector3): void {
      aimAt(targetWorldPos);
      phase = "reach";
      crossTo(reach, idle);
    },
    toIdle(): void {
      phase = "idle";
      idle.reset();
      idle.enabled = true;
      idle.setEffectiveWeight(1);
      idle.play();
    },
    dispose(): void {
      mixer.stopAllAction();
      geometry.dispose();
      armMaterial.dispose();
      object3D.clear();
    },
  };
}
