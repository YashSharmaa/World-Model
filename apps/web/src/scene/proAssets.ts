// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  proAssets — the animation-grade "set dressing" layer. It reads the rack
//  positions the semantic builder already produced (racks.ts -> RackView.center)
//  and wraps each rack with genuinely modelled hardware:
//
//    - rack chassis : boxed frame + perforated side panels + a glossy front
//                     bezel, driven by a thermal heatmap material.
//    - node shelves : thin bezels with a row of tiny emissive status LEDs.
//    - GPU trays    : extruded heatsink FIN geometry + circular fan-grille rings
//                     + an emissive activity strip whose intensity tracks util.
//    - ToR switch   : 1U chassis with a row of blinking port LEDs.
//    - cable runs   : NVLink/power TubeGeometry along CatmullRomCurve3 between
//                     racks and the spine.
//
//  IMPORTANT: this group is PURELY decorative. It carries no userData identity
//  and is never raycast for selection (it is added with `raycast = () => {}`),
//  so applyOccupancy / commitToken / branch logic in the semantic layer is
//  untouched. The thermal panels are driven per frame from rollout hotspot.

import {
  BoxGeometry,
  CatmullRomCurve3,
  CylinderGeometry,
  Group,
  InstancedMesh,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Object3D,
  RingGeometry,
  TorusGeometry,
  TubeGeometry,
  Vector3,
} from "three";
import type { RacksResult, RackView } from "./racks";
import { PALETTE } from "./materials";
import {
  createProMaterialKit,
  createThermalMaterial,
  type ProMaterialKit,
  type ThermalMaterial,
} from "./proMaterials";
import { buildRackLabels } from "./labels";

// Mirror the geometry constants used by racks.ts so the dressing aligns to the
// semantic racks exactly (they share the same RackView.center anchor).
const RACK_W = 3.2;
const RACK_D = 2.0;
const RACK_H = 4.2;

/** Per-rack dressing bundle (kept so we can drive thermal + LEDs per frame). */
interface RackDressing {
  rackId: string;
  /** thermal-driven side panels (left + right). */
  thermal: ThermalMaterial;
  /** emissive activity strips on the GPU trays (intensity tracks util). */
  activity: MeshStandardMaterial;
  /** node status LEDs as a basic emissive material we pulse. */
  statusLeds: MeshBasicMaterial;
}

/** A blinking switch-port LED bank. */
interface SwitchDressing {
  ledMaterial: MeshBasicMaterial;
  /** per-port random phase so the blink looks organic. */
  phases: number[];
}

export interface ProAssetsResult {
  /** root group to add under world_root (decorative; never picked). */
  root: Group;
  kit: ProMaterialKit;
  /**
   * Drive the dressing for the current frame.
   * @param tSec   wall-clock seconds (for LED blink / activity shimmer)
   * @param heatByRack  rackId -> 0..1 hotspot (smoothed by the caller)
   * @param utilByRack  rackId -> 0..1 occupancy/util (activity-strip intensity)
   */
  update(
    tSec: number,
    heatByRack: Record<string, number>,
    utilByRack: Record<string, number>,
  ): void;
  dispose(): void;
}

/**
 * Build the full professional dressing for the pod from the semantic racks.
 */
export function buildProAssets(racks: RacksResult): ProAssetsResult {
  const root = new Group();
  root.name = "pro_set_dressing";

  const kit = createProMaterialKit();
  const dressings: RackDressing[] = [];
  const switches: SwitchDressing[] = [];

  // Shared simple geometries reused across racks (cheap, one upload each).
  const finGeo = new BoxGeometry(0.04, 0.28, RACK_D * 0.5);
  const grilleGeo = new RingGeometry(0.06, 0.13, 16);
  const ledGeo = new BoxGeometry(0.05, 0.05, 0.03);

  for (const rv of racks.racks) {
    const rg = new Group();
    rg.name = `pro_rack_${rv.rackId}`;
    rg.position.set(rv.center.x, 0, rv.center.z);

    // ── thermal side panels (the heatmap reads here) ─────────────────────────
    const thermal = createThermalMaterial(PALETTE.rackFrame.clone());
    const panelGeo = new BoxGeometry(0.08, RACK_H * 0.92, RACK_D * 0.9);
    const panelL = new Mesh(panelGeo, thermal.material);
    panelL.position.set(-RACK_W / 2 - 0.04, RACK_H / 2, 0);
    panelL.castShadow = true;
    panelL.receiveShadow = true;
    const panelR = new Mesh(panelGeo, thermal.material);
    panelR.position.set(RACK_W / 2 + 0.04, RACK_H / 2, 0);
    panelR.castShadow = true;
    panelR.receiveShadow = true;
    rg.add(panelL, panelR);

    // ── chassis frame (four vertical posts + top/bottom rails) ───────────────
    const postGeo = new BoxGeometry(0.14, RACK_H, 0.14);
    const railGeo = new BoxGeometry(RACK_W, 0.12, 0.14);
    const px = RACK_W / 2 - 0.07;
    const pz = RACK_D / 2 - 0.07;
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        const post = new Mesh(postGeo, kit.chassis);
        post.position.set(sx * px, RACK_H / 2, sz * pz);
        post.castShadow = true;
        rg.add(post);
      }
    }
    for (const sz of [-1, 1]) {
      for (const y of [0.08, RACK_H - 0.08]) {
        const rail = new Mesh(railGeo, kit.chassis);
        rail.position.set(0, y, sz * pz);
        rg.add(rail);
      }
    }

    // ── glossy front bezel / door ────────────────────────────────────────────
    const bezel = new Mesh(
      new BoxGeometry(RACK_W * 0.94, RACK_H * 0.9, 0.05),
      kit.glass,
    );
    bezel.position.set(0, RACK_H / 2, RACK_D / 2 + 0.02);
    rg.add(bezel);

    // ── node shelves with status LEDs + GPU trays with fins & fan grilles ─────
    const activity = new MeshStandardMaterial({
      color: PALETTE.grounded.clone(),
      emissive: PALETTE.grounded.clone(),
      emissiveIntensity: 0.4,
      metalness: 0.2,
      roughness: 0.5,
    });
    const statusLeds = new MeshBasicMaterial({ color: PALETTE.safe.clone() });

    const SHELVES = 6;
    for (let s = 0; s < SHELVES; s++) {
      const shelfY = 0.55 + (s / SHELVES) * (RACK_H - 1.1);

      // thin shelf bezel
      const shelf = new Mesh(
        new BoxGeometry(RACK_W * 0.84, 0.06, RACK_D * 0.74),
        kit.bezel,
      );
      shelf.position.set(0, shelfY, 0.05);
      shelf.castShadow = true;
      rg.add(shelf);

      // tiny status LED row on the shelf face
      for (let i = 0; i < 4; i++) {
        const led = new Mesh(ledGeo, statusLeds);
        led.position.set(-RACK_W * 0.32 + i * 0.18, shelfY + 0.04, RACK_D / 2 + 0.02);
        rg.add(led);
      }

      // GPU tray: heatsink fins + fan-grille rings + activity strip
      const tray = new Group();
      tray.position.set(0, shelfY + 0.12, 0);

      // a small bank of aluminium fins
      for (let f = 0; f < 7; f++) {
        const fin = new Mesh(finGeo, kit.aluminium);
        fin.position.set(-0.55 + f * 0.075, 0, -0.1);
        fin.castShadow = true;
        tray.add(fin);
      }
      // two fan-grille rings up front
      for (const gx of [0.55, 0.85]) {
        const grille = new Mesh(grilleGeo, kit.bezel);
        grille.position.set(gx, 0, RACK_D / 2 - 0.02);
        tray.add(grille);
      }
      // emissive activity strip across the tray front (util-driven)
      const strip = new Mesh(
        new BoxGeometry(RACK_W * 0.7, 0.03, 0.02),
        activity,
      );
      strip.position.set(-0.1, -0.14, RACK_D / 2 + 0.01);
      tray.add(strip);

      rg.add(tray);
    }

    root.add(rg);
    dressings.push({ rackId: rv.rackId, thermal, activity, statusLeds });
  }

  // ── ToR switches: a 1U chassis above each rack with blinking ports ─────────
  for (const rv of racks.racks) {
    const sw = new Group();
    sw.position.set(rv.center.x, 4.55, rv.center.z);

    const body = new Mesh(new BoxGeometry(1.6, 0.26, 0.7), kit.chassis);
    body.castShadow = true;
    sw.add(body);

    const ledMat = new MeshBasicMaterial({ color: PALETTE.grounded.clone() });
    const PORTS = 12;
    const portMesh = new InstancedMesh(
      new BoxGeometry(0.05, 0.05, 0.03),
      ledMat,
      PORTS,
    );
    const m = new Matrix4();
    const phases: number[] = [];
    for (let p = 0; p < PORTS; p++) {
      const lx = -0.66 + (p % 6) * 0.22;
      const ly = p < 6 ? 0.05 : -0.05;
      m.makeTranslation(lx, ly, 0.36);
      portMesh.setMatrixAt(p, m);
      phases.push(Math.random() * Math.PI * 2);
    }
    portMesh.instanceMatrix.needsUpdate = true;
    sw.add(portMesh);

    root.add(sw);
    switches.push({ ledMaterial: ledMat, phases });
  }

  // ── cable runs: NVLink (fabric) + power tubes between racks and the spine ───
  buildCableRuns(root, racks.racks, kit);

  // ── crisp in-scene rack labels ─────────────────────────────────────────────
  // The persistent HTML footer carries the single, clean author attribution, so
  // the redundant in-scene credit plate is intentionally omitted here.
  root.add(buildRackLabels(racks.racks));

  // The whole dressing is decorative: never participate in raycasting.
  root.traverse((o: Object3D) => {
    o.raycast = () => {};
  });

  return {
    root,
    kit,
    update(tSec, heatByRack, utilByRack): void {
      for (const d of dressings) {
        const heat = heatByRack[d.rackId] ?? d.thermal.heat();
        // smooth toward the target heat so transitions read as material changes.
        const cur = d.thermal.heat();
        d.thermal.setHeat(cur + (heat - cur) * 0.08);

        const util = utilByRack[d.rackId] ?? 0;
        // activity strips brighten with util + a gentle data-center shimmer.
        const shimmer = 0.85 + 0.15 * Math.sin(tSec * 3.0 + d.rackId.length);
        d.activity.emissiveIntensity = (0.25 + util * 1.4) * shimmer;
        // status LEDs settle to safe-green but pulse if the rack runs hot.
        const alarm = heat > 0.7;
        d.statusLeds.color.copy(alarm ? PALETTE.hot : PALETTE.safe);
        d.statusLeds.opacity = 0.7 + 0.3 * Math.abs(Math.sin(tSec * (alarm ? 8 : 2)));
      }
      for (const sw of switches) {
        // pick a representative blink colour; individual ports use phase offsets.
        const blink = 0.45 + 0.55 * Math.abs(Math.sin(tSec * 6 + sw.phases[0]));
        sw.ledMaterial.color.copy(PALETTE.grounded).multiplyScalar(blink);
      }
    },
    dispose(): void {
      kit.dispose();
      finGeo.dispose();
      grilleGeo.dispose();
      ledGeo.dispose();
      root.traverse((o: Object3D) => {
        const mesh = o as Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        // dispose sprite label textures/materials (CanvasTexture-backed).
        const mat = (o as unknown as { material?: { map?: { dispose(): void }; dispose?(): void } }).material;
        if (mat) {
          mat.map?.dispose?.();
          mat.dispose?.();
        }
      });
      root.clear();
    },
  };
}

/**
 * Lay NVLink/power cables as smooth tubes. We connect each rack to the spine
 * point above the pod centre (fabric) and adjacent racks to each other (power),
 * which reads as a believable structured-cabling run without packet noise.
 */
function buildCableRuns(
  root: Group,
  rackViews: RackView[],
  kit: ProMaterialKit,
): void {
  if (rackViews.length === 0) return;

  // spine anchor: above the centroid of the pod.
  const cx = rackViews.reduce((s, r) => s + r.center.x, 0) / rackViews.length;
  const cz = rackViews.reduce((s, r) => s + r.center.z, 0) / rackViews.length;
  const spine = new Vector3(cx, 6.6, cz);

  // a spine bus-bar so the fabric tubes have something to terminate into.
  const bus = new Mesh(new CylinderGeometry(0.08, 0.08, 0.4, 12), kit.aluminium);
  bus.position.copy(spine);
  bus.rotation.z = Math.PI / 2;
  root.add(bus);

  for (let i = 0; i < rackViews.length; i++) {
    const a = rackViews[i].center;
    const top = new Vector3(a.x, 4.6, a.z);

    // fabric (NVLink) tube: rack-top -> arched -> spine.
    const fabricCurve = new CatmullRomCurve3([
      top,
      new Vector3((top.x + spine.x) / 2, 5.8, (top.z + spine.z) / 2),
      spine,
    ]);
    const fabricTube = new Mesh(
      new TubeGeometry(fabricCurve, 24, 0.035, 8, false),
      kit.cableFabric,
    );
    fabricTube.castShadow = true;
    root.add(fabricTube);

    // power whip: a short tidy drop at the rack's own back-base that curls into
    // the rack — clearly terminates at the chassis instead of sagging across the
    // open floor to a non-adjacent rack (which read as a cable going nowhere).
    const back = a.z + RACK_D / 2;
    const powerCurve = new CatmullRomCurve3([
      new Vector3(a.x, 1.0, back + 0.06),
      new Vector3(a.x, 0.45, back + 0.16),
      new Vector3(a.x, 0.16, back + 0.06),
      new Vector3(a.x, 0.12, back - 0.05),
    ]);
    const powerTube = new Mesh(
      new TubeGeometry(powerCurve, 18, 0.05, 8, false),
      kit.cablePower,
    );
    powerTube.castShadow = true;
    root.add(powerTube);
  }

  // a couple of decorative fabric loops between adjacent racks at the top.
  for (let i = 0; i + 1 < rackViews.length; i++) {
    const ring = new Mesh(
      new TorusGeometry(0.12, 0.02, 8, 16),
      kit.cableFabric,
    );
    const a = rackViews[i].center;
    ring.position.set(a.x, 4.62, a.z);
    ring.rotation.x = Math.PI / 2;
    root.add(ring);
  }
}
