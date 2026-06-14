// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Public API surface for the apps/web Decision Studio module. The entrypoint is
//  src/main.ts (loaded by index.html); this barrel re-exports the typed client,
//  the DTOs that mirror the six backend contracts, the client store, the scene
//  graph builder, and the camera/mode helpers so the module presents one clean
//  surface for embedding or testing.

// API client + the six-contract DTOs (mirror of the Python serving surface).
export * from "./api/client";
export {
  MODEL_VERSION,
  POLICY_VERSION,
  CREDIT,
  buildMockSnapshot,
  buildMockCompare,
  buildMockWhyNot,
  buildMockReplay,
  buildMockReplayIndex,
} from "./api/mock";

// Client state.
export { Store } from "./state/store";
export type { StoreState, StudioMode, Selection } from "./state/store";

// Scene graph + sub-graph builders.
export { buildSceneGraph } from "./scene/sceneGraph";
export type { SceneGraph } from "./scene/sceneGraph";
export { buildRacks, applyOccupancy } from "./scene/racks";
export { buildWorkloads, commitToken, setAllocationPath } from "./scene/workloads";
export { buildNetwork, buildDomains, setDomainLayerVisibility } from "./scene/overlays";
export { createBranchesRoot, rebuildBranches, updateBranches } from "./scene/branches";
export { createMaterials, PALETTE, branchGhostColor } from "./scene/materials";
export type { MaterialSet } from "./scene/materials";

// Modes + camera.
export {
  CameraDirector,
  createModeTabs,
  CAMERA_PRESETS,
  MODE_TABS,
} from "./routes/modes";
export type { CameraPreset } from "./routes/modes";

// Professional 3D upgrade surface (environment, dressing, rig, FX, director).
export { installEnvironment } from "./scene/environment";
export type { EnvironmentHandle } from "./scene/environment";
export { buildProAssets } from "./scene/proAssets";
export type { ProAssetsResult } from "./scene/proAssets";
export { createProMaterialKit, createThermalMaterial } from "./scene/proMaterials";
export type { ProMaterialKit, ThermalMaterial } from "./scene/proMaterials";
export { createLoaderArm } from "./scene/loaderArm";
export type { LoaderArm } from "./scene/loaderArm";
export { createTextSprite, buildRackLabels, createCreditPlate } from "./scene/labels";
export { createPostFX } from "./render/postfx";
export type { PostFX } from "./render/postfx";
export { CinematicDirector, SHOTS, defaultGoldenPath } from "./anim/director";
export type { Shot, Beat } from "./anim/director";
