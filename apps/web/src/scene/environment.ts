// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  installEnvironment — the cinematic lighting + atmosphere rig.
//
//    - Image-based lighting: PMREMGenerator bakes the RoomEnvironment into a
//      pre-filtered cube so every PBR surface gets soft, believable reflections
//      and ambient bounce (the single biggest "this looks rendered" upgrade).
//    - Cool datacenter mood: FogExp2 + a dark backdrop so racks fall off into
//      depth, a 3-point rig (warm key, cool fill, cyan rim) with ONE
//      shadow-casting key, and subtle ceiling light strips for set-dressing.
//    - A reflective raised floor: low-roughness PBR plane that catches the IBL
//      and the emissive bloom from the racks — cheap, no extra render pass.
//
//  It returns a handle so main.ts can dispose the PMREM target and (optionally)
//  read the rig back for the director. It does NOT add the existing floor/grid
//  from sceneGraph.ts — those stay; this dims/upgrades the look around them.

import {
  Color,
  DirectionalLight,
  FogExp2,
  Group,
  HemisphereLight,
  Mesh,
  MeshStandardMaterial,
  PlaneGeometry,
  PMREMGenerator,
  RectAreaLight,
  Scene,
  SpotLight,
  type Texture,
  WebGLRenderer,
} from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { RectAreaLightUniformsLib } from "three/examples/jsm/lights/RectAreaLightUniformsLib.js";
import { onTheme, type SceneTheme } from "../theme";

export interface EnvironmentHandle {
  /** the rig group (lights + floor + ceiling strips) added under the scene. */
  rig: Group;
  keyLight: DirectionalLight;
  dispose(): void;
}

/**
 * Install IBL + the 3-point rig + atmosphere. Call ONCE after the renderer and
 * scene exist. Returns a handle for disposal.
 */
export function installEnvironment(
  scene: Scene,
  renderer: WebGLRenderer,
): EnvironmentHandle {
  // RectAreaLight needs its BRDF LUTs initialised once before it lights anything.
  RectAreaLightUniformsLib.init();

  // ── image-based lighting (PMREM of the RoomEnvironment) ────────────────────
  const pmrem = new PMREMGenerator(renderer);
  pmrem.compileEquirectangularShader();
  const roomScene = new RoomEnvironment();
  const envTexture: Texture = pmrem.fromScene(roomScene, 0.04).texture;
  scene.environment = envTexture;
  // theme-driven visible background; IBL is for shading. Recoloured by applyTheme.
  const background = new Color("#eef2f8");
  scene.background = background;
  roomScene.traverse((o) => {
    const mesh = o as Mesh;
    if (mesh.geometry) mesh.geometry.dispose();
  });

  // ── atmosphere: exponential fog for depth falloff (theme-driven) ───────────
  const fog = new FogExp2(0xe2e9f2, 0.012);
  scene.fog = fog;

  const rig = new Group();
  rig.name = "lighting_rig";

  // ── 3-point lighting rig ───────────────────────────────────────────────────
  // soft ambient bounce (cool top, warm floor) under the IBL.
  const hemi = new HemisphereLight(0xaac4ff, 0x10131c, 0.35);
  rig.add(hemi);

  // KEY — the only shadow caster; slightly warm, high and to the side.
  const keyLight = new DirectionalLight(0xfff2e0, 2.1);
  keyLight.position.set(10, 16, 9);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.camera.near = 1;
  keyLight.shadow.camera.far = 60;
  keyLight.shadow.camera.left = -20;
  keyLight.shadow.camera.right = 20;
  keyLight.shadow.camera.top = 20;
  keyLight.shadow.camera.bottom = -20;
  keyLight.shadow.bias = -0.0004;
  keyLight.shadow.normalBias = 0.02;
  rig.add(keyLight);
  rig.add(keyLight.target);

  // FILL — soft cool fill from the opposite side, no shadow.
  const fill = new DirectionalLight(0x88a6ff, 0.55);
  fill.position.set(-12, 8, 6);
  rig.add(fill);

  // RIM — cyan back-rim to separate the racks from the dark backdrop.
  const rim = new DirectionalLight(0x59d6ff, 0.9);
  rim.position.set(-6, 9, -14);
  rig.add(rim);

  // ── ceiling light strips (set dressing; emissive + a soft area glow) ───────
  const ceiling = new Group();
  ceiling.name = "ceiling_strips";
  const stripMat = new MeshStandardMaterial({
    color: new Color("#cfe2ff"),
    emissive: new Color("#7fb4ff"),
    emissiveIntensity: 1.6,
    roughness: 0.6,
    metalness: 0.0,
  });
  for (let i = -2; i <= 2; i++) {
    const strip = new Mesh(new PlaneGeometry(0.4, 18), stripMat);
    strip.rotation.x = Math.PI / 2;
    strip.position.set(i * 4.5, 9.2, 0);
    ceiling.add(strip);

    // a faint RectAreaLight under each strip for believable soft top light.
    const area = new RectAreaLight(0x9fc4ff, 2.2, 0.4, 18);
    area.position.set(i * 4.5, 9.0, 0);
    area.rotation.x = -Math.PI / 2;
    ceiling.add(area);

    // a wide downward spot adds a gentle pool of light on the floor.
    if (i % 2 === 0) {
      const spot = new SpotLight(0xbcd6ff, 18, 26, Math.PI / 5, 0.6, 1.2);
      spot.position.set(i * 4.5, 9.0, 0);
      spot.target.position.set(i * 4.5, 0, 0);
      ceiling.add(spot, spot.target);
    }
  }
  rig.add(ceiling);

  // ── reflective raised floor (low roughness; catches IBL + bloom) ───────────
  const floorMat = new MeshStandardMaterial({
    color: new Color("#dbe2ec"),
    metalness: 0.22,
    roughness: 0.62,
    envMapIntensity: 0.85,
  });
  const floor = new Mesh(new PlaneGeometry(60, 60), floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.02;
  floor.receiveShadow = true;
  floor.name = "reflective_floor";
  rig.add(floor);

  scene.add(rig);

  // ── theme reactivity ───────────────────────────────────────────────────────
  // One switch retunes background, fog, floor, and the whole light rig so the
  // render reads as a coherent bright studio (light) or cinematic room (dark).
  function applyTheme(t: SceneTheme): void {
    background.set(t.background);
    fog.color.set(t.fog);
    fog.density = t.fogDensity;
    hemi.color.set(t.hemiSky);
    hemi.groundColor.set(t.hemiGround);
    hemi.intensity = t.hemiIntensity;
    keyLight.color.set(t.keyColor);
    keyLight.intensity = t.keyIntensity;
    fill.color.set(t.fillColor);
    fill.intensity = t.fillIntensity;
    rim.color.set(t.rimColor);
    rim.intensity = t.rimIntensity;
    stripMat.emissiveIntensity = t.ceilingEmissive;
    floorMat.color.set(t.floorColor);
    floorMat.metalness = t.floorMetalness;
    floorMat.roughness = t.floorRoughness;
    floorMat.envMapIntensity = t.floorEnvIntensity;
    floorMat.needsUpdate = true;
  }
  const unsubscribe = onTheme(applyTheme); // fires immediately with current theme

  return {
    rig,
    keyLight,
    dispose(): void {
      unsubscribe();
      scene.remove(rig);
      envTexture.dispose();
      pmrem.dispose();
      scene.environment = null;
    },
  };
}
