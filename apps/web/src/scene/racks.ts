// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  rack_root — the grounded physical pod. Each rack is a frame mesh + node
//  shelves + an InstancedMesh of GPU slot tiles + a halo (doc G scene graph).
//
//  Why InstancedMesh: a 4-rack demo pod has up to 128 GPU slots. three.js
//  documents InstancedMesh specifically to collapse many same-geometry objects
//  into one draw call, and the Raycaster returns `instanceId` on intersection so
//  a single GPU slot can be picked without exploding the scene into hundreds of
//  meshes. Occupancy is animated by RECOLOURING tiles, never by spawning meshes
//  (doc G "Occupancy updates").

import {
  BoxGeometry,
  Color,
  DynamicDrawUsage,
  Group,
  InstancedBufferAttribute,
  InstancedMesh,
  Matrix4,
  Mesh,
  Vector3,
} from "three";
import type { GPUDTO, NodeDTO, RackDTO } from "../api/client";
import { PALETTE, type MaterialSet } from "./materials";

const RACK_W = 3.2;
const RACK_D = 2.0;
const RACK_H = 4.2;
const RACK_GAP = 2.4;
const SLOT = 0.32;

/** Per-rack metadata used by overlays/raycasting and halo animation. */
export interface RackView {
  rackId: string;
  group: Group;
  halo: Mesh;
  center: Vector3;
  designedGpuCap: number;
}

/** Mapping from a flat GPU instanceId back to its semantic identity. */
export interface SlotIndex {
  gpuUuid: string;
  nodeUid: string;
  rackId: string;
}

/** Mapping from a flat node-shelf instanceId back to its semantic identity. */
export interface ShelfIndex {
  nodeUid: string;
  rackId: string;
}

export interface RacksResult {
  root: Group;
  racks: RackView[];
  /** the single InstancedMesh of all GPU slots across the pod. */
  slotMesh: InstancedMesh;
  /** the single InstancedMesh of all node shelves across the pod. */
  shelfMesh: InstancedMesh;
  /** instanceId -> identity, for Raycaster picking. */
  slotIndex: SlotIndex[];
  /** shelf instanceId -> node identity, for Raycaster picking. */
  shelfIndex: ShelfIndex[];
  /** gpu_uuid -> instanceId, for occupancy recolouring from rollouts. */
  slotByUuid: Map<string, number>;
}

/**
 * Build the rack_root sub-graph from a snapshot. Racks are laid out on a 2×N
 * grid keyed by block so the camera presets frame the whole pod.
 */
export function buildRacks(
  racks: RackDTO[],
  nodes: NodeDTO[],
  gpus: GPUDTO[],
  mats: MaterialSet,
): RacksResult {
  const root = new Group();
  root.name = "rack_root";

  const rackViews: RackView[] = [];
  const slotIndex: SlotIndex[] = [];
  const shelfIndex: ShelfIndex[] = [];
  const slotByUuid = new Map<string, number>();

  // Group nodes & gpus by rack for stable per-rack layout.
  const nodesByRack = new Map<string, NodeDTO[]>();
  for (const n of nodes) {
    (nodesByRack.get(n.rack_id) ?? nodesByRack.set(n.rack_id, []).get(n.rack_id)!).push(n);
  }
  const gpusByNode = new Map<string, GPUDTO[]>();
  for (const g of gpus) {
    (gpusByNode.get(g.node_uid) ?? gpusByNode.set(g.node_uid, []).get(g.node_uid)!).push(g);
  }

  const totalSlots = gpus.length;
  const slotGeo = new BoxGeometry(SLOT * 0.82, SLOT * 0.82, 0.12);
  const slotMesh = new InstancedMesh(slotGeo, mats.gpuSlot, Math.max(1, totalSlots));
  slotMesh.name = "gpu_slot_instanced_mesh";
  slotMesh.instanceMatrix.setUsage(DynamicDrawUsage);
  const slotColors = new Float32Array(Math.max(1, totalSlots) * 3);
  slotMesh.instanceColor = new InstancedBufferAttribute(slotColors, 3);
  slotMesh.instanceColor.setUsage(DynamicDrawUsage);

  // node_instanced_mesh (doc G scene_graph): ONE InstancedMesh for ALL node
  // shelves across the pod, collapsing them into a single draw call and exposing
  // an instanceId the Raycaster maps back to a node identity (like the GPU slots).
  const totalNodes = nodes.length;
  const shelfGeo = new BoxGeometry(RACK_W * 0.88, 0.12, RACK_D * 0.78);
  const shelfMesh = new InstancedMesh(shelfGeo, mats.nodeShelf, Math.max(1, totalNodes));
  shelfMesh.name = "node_instanced_mesh";
  shelfMesh.instanceMatrix.setUsage(DynamicDrawUsage);

  const m = new Matrix4();
  const tmpColor = new Color();
  let slotCursor = 0;
  let shelfCursor = 0;

  const cols = 2;
  racks.forEach((rack, idx) => {
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    const x = (col - (cols - 1) / 2) * (RACK_W + RACK_GAP);
    const z = (row - (Math.ceil(racks.length / cols) - 1) / 2) * (RACK_D + RACK_GAP);

    const rg = new Group();
    rg.name = `rack_group_${rack.rack_id}`;
    rg.position.set(x, 0, z);

    // rack_frame_mesh — a thin open shell.
    const frame = new Mesh(new BoxGeometry(RACK_W, RACK_H, RACK_D), mats.rackFrame);
    frame.position.y = RACK_H / 2;
    frame.name = "rack_frame_mesh";
    frame.userData = { kind: "rack", id: rack.rack_id };
    rg.add(frame);

    // node shelves + GPU slot tiles (instanced into the shared slotMesh).
    const rackNodes = (nodesByRack.get(rack.rack_id) ?? []).sort((a, b) =>
      a.node_uid.localeCompare(b.node_uid),
    );
    const nodeCount = Math.max(1, rackNodes.length);
    rackNodes.forEach((node, nIdx) => {
      const shelfY = 0.4 + (nIdx / nodeCount) * (RACK_H - 0.8);
      // node shelf -> one instance in the shared node_instanced_mesh (world-space
      // matrix, since the InstancedMesh lives at rack_root, not the rack group).
      m.makeTranslation(x, shelfY, z);
      shelfMesh.setMatrixAt(shelfCursor, m);
      shelfIndex.push({ nodeUid: node.node_uid, rackId: rack.rack_id });
      shelfCursor += 1;

      const slots = (gpusByNode.get(node.node_uid) ?? []).sort((a, b) =>
        a.gpu_uuid.localeCompare(b.gpu_uuid),
      );
      const perRow = Math.max(1, slots.length);
      slots.forEach((gpu, gIdx) => {
        const sx = (gIdx - (perRow - 1) / 2) * SLOT;
        m.makeTranslation(x + sx, shelfY + 0.12, z + RACK_D * 0.3);
        slotMesh.setMatrixAt(slotCursor, m);
        const filled = gpu.allocated_to != null;
        tmpColor.copy(filled ? PALETTE.slotFilled : PALETTE.slotEmpty);
        if (gpu.health_state !== "healthy") tmpColor.copy(PALETTE.blocked);
        slotMesh.setColorAt(slotCursor, tmpColor);
        slotIndex.push({
          gpuUuid: gpu.gpu_uuid,
          nodeUid: node.node_uid,
          rackId: rack.rack_id,
        });
        slotByUuid.set(gpu.gpu_uuid, slotCursor);
        slotCursor += 1;
      });
    });

    // rack_halo_mesh — a translucent box that thickens as learned hotspot risk
    // rises BEFORE any hard alert (doc G "Risk emergence"). Opacity is animated.
    const halo = new Mesh(
      new BoxGeometry(RACK_W + 0.6, RACK_H + 0.6, RACK_D + 0.6),
      mats.rackHalo.clone(),
    );
    halo.position.y = RACK_H / 2;
    halo.name = "rack_halo_mesh";
    (halo.material as { opacity: number }).opacity = (rack.hotspot_score ?? 0) * 0.3;
    rg.add(halo);

    root.add(rg);
    rackViews.push({
      rackId: rack.rack_id,
      group: rg,
      halo,
      center: new Vector3(x, RACK_H / 2, z),
      designedGpuCap: rack.designed_gpu_cap,
    });
  });

  slotMesh.count = slotCursor;
  slotMesh.instanceMatrix.needsUpdate = true;
  if (slotMesh.instanceColor) slotMesh.instanceColor.needsUpdate = true;
  root.add(slotMesh);

  shelfMesh.count = shelfCursor;
  shelfMesh.instanceMatrix.needsUpdate = true;
  root.add(shelfMesh);

  return { root, racks: rackViews, slotMesh, shelfMesh, slotIndex, shelfIndex, slotByUuid };
}

/**
 * Per-rack GROUNDED occupancy fraction computed straight from the snapshot's
 * allocated GPU slots. This is the "now" the renderer must hold the grounded
 * pod at in BOTH decision-studio and compare modes — predicted occupancy is
 * layered on top as a translucent ghost, never promoted to the solid base
 * (doc G core_views: grounded now vs ghost future).
 */
export function baselineOccupancyByRack(
  result: RacksResult,
  gpus: GPUDTO[],
): Record<string, number> {
  const allocByRack: Record<string, number> = {};
  const totalByRack: Record<string, number> = {};
  const rackByUuid = new Map(result.slotIndex.map((s) => [s.gpuUuid, s.rackId]));
  for (const g of gpus) {
    const rackId = rackByUuid.get(g.gpu_uuid);
    if (!rackId) continue;
    totalByRack[rackId] = (totalByRack[rackId] ?? 0) + 1;
    if (g.allocated_to != null) allocByRack[rackId] = (allocByRack[rackId] ?? 0) + 1;
  }
  const occ: Record<string, number> = {};
  for (const rackId of Object.keys(totalByRack)) {
    occ[rackId] = (allocByRack[rackId] ?? 0) / Math.max(1, totalByRack[rackId]);
  }
  return occ;
}

/**
 * Recolour a fraction of a rack's slots to "filled" to reflect rollout
 * occupancy (occupancy update = recolour, never spawn). `ghost` desaturates the
 * fill so predicted occupancy reads as translucent against the grounded base.
 *
 * When `ghost` is true the slots BEYOND the grounded baseline (the predicted
 * EXTRA fill the branch would add) read in the translucent predicted colour,
 * while the grounded baseline slots stay solid — so a ghost overlay never
 * promotes the predicted future to a solid grounded fill (doc G grounded-now vs
 * ghost-future). `baseline` is the grounded fraction to hold solid underneath.
 */
export function applyOccupancy(
  result: RacksResult,
  rackId: string,
  occupancy: number,
  hotspot: number,
  ghost: boolean,
  baseline = occupancy,
): void {
  const indices: number[] = [];
  result.slotIndex.forEach((s, i) => {
    if (s.rackId === rackId) indices.push(i);
  });
  const fillCount = Math.round(occupancy * indices.length);
  const baseCount = Math.round(Math.min(occupancy, baseline) * indices.length);
  const tmp = new Color();
  indices.forEach((instanceId, k) => {
    const filled = k < fillCount;
    const isPredictedExtra = ghost && k >= baseCount; // beyond grounded baseline
    if (!filled) {
      tmp.copy(PALETTE.slotEmpty);
    } else if (hotspot > 0.6) {
      tmp.copy(PALETTE.hot);
    } else if (isPredictedExtra) {
      tmp.copy(PALETTE.predicted);
    } else {
      tmp.copy(PALETTE.slotFilled);
    }
    result.slotMesh.setColorAt(instanceId, tmp);
  });
  if (result.slotMesh.instanceColor) result.slotMesh.instanceColor.needsUpdate = true;

  const view = result.racks.find((r) => r.rackId === rackId);
  if (view) {
    (view.halo.material as { opacity: number }).opacity = Math.min(0.45, hotspot * 0.45);
  }
}
