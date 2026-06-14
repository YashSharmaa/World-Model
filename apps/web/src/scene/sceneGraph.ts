// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  The scene graph orchestrator (doc G). It assembles ONE stable world_root:
//
//    world_root
//      ├─ floor_plane
//      ├─ rack_root        (instanced node shelves + GPU slot tiles + halos)
//      ├─ network_root     (ToR switches + a few inter-rack lines)
//      ├─ domain_root      (translucent topology / power / cooling layers)
//      ├─ workload_root    (queue lane, capsules, allocation paths, branch ghosts)
//      └─ interaction_root (Raycaster picking via instanceId, selection outline)
//
//  Everything semantic lives under this one root so modes never remount the
//  scene — "compare is not a separate world, it is the same scene plus branch
//  layers" (doc G).

import {
  BoxGeometry,
  Camera,
  GridHelper,
  Group,
  Mesh,
  PlaneGeometry,
  Raycaster,
  Vector2,
} from "three";
import type { StateSnapshotDTO } from "../api/client";
import { createMaterials, type MaterialSet } from "./materials";
import { buildRacks, type RacksResult } from "./racks";
import { buildWorkloads, type WorkloadResult } from "./workloads";
import { buildNetwork, buildDomains, type DomainResult, type NetworkResult } from "./overlays";
import { createBranchesRoot, type BranchesResult } from "./branches";
import type { Selection } from "../state/store";

export interface SceneGraph {
  worldRoot: Group;
  interactionRoot: Group;
  materials: MaterialSet;
  racks: RacksResult;
  workloads: WorkloadResult;
  network: NetworkResult;
  domains: DomainResult;
  branches: BranchesResult;
  selectionOutline: Mesh;
  raycaster: Raycaster;
  /** Picks the entity under normalized device coords; null on a miss. */
  pick(ndc: Vector2, camera: Camera): Selection | null;
  /** Move the selection outline onto a picked entity (or hide it). */
  highlight(sel: Selection | null): void;
  /** Rebuild the whole world_root from a fresh snapshot. */
  rebuild(snapshot: StateSnapshotDTO): void;
  dispose(): void;
}

/** Build the full scene graph from a snapshot. */
export function buildSceneGraph(snapshot: StateSnapshotDTO): SceneGraph {
  const materials = createMaterials();
  const worldRoot = new Group();
  worldRoot.name = "world_root";

  // floor_plane + a faint grid for spatial grounding.
  const floor = new Mesh(new PlaneGeometry(40, 40), materials.floor);
  floor.rotation.x = -Math.PI / 2;
  floor.name = "floor_plane";
  worldRoot.add(floor);
  const grid = new GridHelper(40, 40, 0x1b2436, 0x121826);
  (grid.material as { opacity: number; transparent: boolean }).opacity = 0.25;
  (grid.material as { transparent: boolean }).transparent = true;
  worldRoot.add(grid);

  const interactionRoot = new Group();
  interactionRoot.name = "interaction_root";

  // selection_outline — a slightly enlarged translucent box reused across picks.
  const selectionOutline = new Mesh(new BoxGeometry(1, 1, 1), materials.selectionOutline);
  selectionOutline.name = "selection_outline";
  selectionOutline.visible = false;
  interactionRoot.add(selectionOutline);

  const sg: SceneGraph = {
    worldRoot,
    interactionRoot,
    materials,
    // placeholders, filled by attach()
    racks: undefined as unknown as RacksResult,
    workloads: undefined as unknown as WorkloadResult,
    network: undefined as unknown as NetworkResult,
    domains: undefined as unknown as DomainResult,
    branches: undefined as unknown as BranchesResult,
    selectionOutline,
    raycaster: new Raycaster(),
    pick,
    highlight,
    rebuild,
    dispose,
  };

  attach(snapshot);
  worldRoot.add(interactionRoot);
  return sg;

  // ── construction ────────────────────────────────────────────────────────
  function attach(snap: StateSnapshotDTO): void {
    const { racks, nodes, gpus, workloads } = snap.entities;
    sg.racks = buildRacks(racks, nodes, gpus, materials);
    sg.network = buildNetwork(sg.racks.racks, materials, snap.relations);
    sg.domains = buildDomains(racks, sg.racks.racks, materials);
    sg.workloads = buildWorkloads(workloads, materials);
    sg.branches = createBranchesRoot();
    sg.workloads.root.add(sg.branches.root);

    worldRoot.add(sg.racks.root, sg.network.root, sg.domains.root, sg.workloads.root);
  }

  function detach(): void {
    for (const root of [
      sg.racks?.root,
      sg.network?.root,
      sg.domains?.root,
      sg.workloads?.root,
    ]) {
      if (root) worldRoot.remove(root);
    }
  }

  function rebuild(snap: StateSnapshotDTO): void {
    detach();
    attach(snap);
  }

  // ── interaction ──────────────────────────────────────────────────────────
  function pick(ndc: Vector2, camera: Camera): Selection | null {
    sg.raycaster.setFromCamera(ndc, camera);
    const hits = sg.raycaster.intersectObjects(worldRoot.children, true);
    for (const hit of hits) {
      // InstancedMesh GPU slots return an instanceId we map back to identity.
      if (hit.object === sg.racks.slotMesh && hit.instanceId != null) {
        const slot = sg.racks.slotIndex[hit.instanceId];
        if (slot) return { kind: "gpu", id: slot.gpuUuid };
      }
      // node_instanced_mesh shelves: map the instanceId back to a node identity.
      if (hit.object === sg.racks.shelfMesh && hit.instanceId != null) {
        const shelf = sg.racks.shelfIndex[hit.instanceId];
        if (shelf) return { kind: "node", id: shelf.nodeUid };
      }
      const ud = hit.object.userData as { kind?: Selection["kind"]; id?: string };
      if (ud?.kind && ud.id) return { kind: ud.kind, id: ud.id };
    }
    return null;
  }

  function highlight(sel: Selection | null): void {
    if (!sel) {
      selectionOutline.visible = false;
      return;
    }
    // Frame the outline on the selected rack (the legible focus unit). GPU/node
    // selections snap the outline onto their parent rack so the camera context
    // stays stable (doc G "Focused: outline plus panel pin").
    let rackId: string | null = null;
    if (sel.kind === "rack") rackId = sel.id;
    else if (sel.kind === "gpu") {
      rackId = sg.racks.slotIndex.find((s) => s.gpuUuid === sel.id)?.rackId ?? null;
    } else if (sel.kind === "node") {
      rackId = sg.racks.slotIndex.find((s) => s.nodeUid === sel.id)?.rackId ?? null;
    } else if (sel.kind === "domain") {
      rackId = sel.id.replace("dom-", "");
    }
    const view = rackId ? sg.racks.racks.find((r) => r.rackId === rackId) : null;
    if (!view) {
      selectionOutline.visible = false;
      return;
    }
    selectionOutline.position.copy(view.center);
    selectionOutline.scale.set(3.8, view.center.y * 2 + 1.0, 2.6);
    selectionOutline.visible = true;
  }

  function dispose(): void {
    detach();
    worldRoot.clear();
  }
}
