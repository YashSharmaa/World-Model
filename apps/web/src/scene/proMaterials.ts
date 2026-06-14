// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  proMaterials — the "professional grade" material kit that dresses the pod:
//
//    - createThermalMaterial(): a MeshStandardMaterial whose emissive is driven
//      by a `uHeat` uniform [0..1] through a calibrated blue -> cyan -> amber ->
//      red thermal ramp (the same grammar an operator reads on a DCIM heatmap).
//      Implemented via onBeforeCompile injection so it keeps full PBR shading,
//      shadows, and IBL from the RoomEnvironment — we only add emissive heat.
//
//    - PRO PBR chassis/metal/plastic/glass materials tuned for the cool
//      datacenter mood. These are shared (one instance, many meshes) so the
//      set-dressing stays cheap: it is still a semantic read model, not a game.
//
//  Everything here is additive: the existing materials.ts PALETTE + MaterialSet
//  remain authoritative for the semantic states (grounded / predicted / blocked).

import {
  Color,
  type IUniform,
  MeshStandardMaterial,
  type WebGLProgramParametersWithUniforms,
} from "three";

/** A thermal material exposes its live heat handle so callers can drive it. */
export interface ThermalMaterial {
  material: MeshStandardMaterial;
  /** Set the 0..1 heat level (clamped); maps to the blue->amber->red ramp. */
  setHeat(heat: number): void;
  /** Current heat (last value written), for smooth interpolation by callers. */
  heat(): number;
}

/** The shared professional PBR kit (created once at boot, reused everywhere). */
export interface ProMaterialKit {
  /** Brushed dark metal — rack frames, switch chassis. */
  chassis: MeshStandardMaterial;
  /** Perforated side panels (slightly lighter, more matte). */
  panel: MeshStandardMaterial;
  /** Bright machined aluminium — heatsink fins, rails. */
  aluminium: MeshStandardMaterial;
  /** Matte black plastic — bezels, fan grilles. */
  bezel: MeshStandardMaterial;
  /** Dark glossy front glass / acrylic door. */
  glass: MeshStandardMaterial;
  /** Reflective raised-floor tile. */
  floorTile: MeshStandardMaterial;
  /** Copper-ish cable jacket (power runs). */
  cablePower: MeshStandardMaterial;
  /** Teal fibre jacket (NVLink / fabric runs). */
  cableFabric: MeshStandardMaterial;
  dispose(): void;
}

/**
 * Build a thermal-driven material. We start from a PBR base so the panel still
 * reflects the environment and casts/receives shadows, then inject a small
 * fragment hook that adds emissive colour from `uHeat`. The ramp is authored to
 * read like a professional thermal map:
 *
 *     0.00  deep blue   (cold / lots of headroom)
 *     0.35  cyan
 *     0.60  amber       (warming — watch)
 *     0.85  orange
 *     1.00  red         (hotspot — at the limit)
 */
export function createThermalMaterial(baseColor: Color): ThermalMaterial {
  const uHeat: IUniform<number> = { value: 0 };

  const material = new MeshStandardMaterial({
    color: baseColor,
    metalness: 0.55,
    roughness: 0.55,
  });

  material.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms) => {
    shader.uniforms.uHeat = uHeat;

    // Declare the ramp helper + uniform in the fragment program.
    shader.fragmentShader = shader.fragmentShader.replace(
      "void main() {",
      /* glsl */ `
      uniform float uHeat;

      // Professional blue -> cyan -> amber -> orange -> red thermal ramp.
      vec3 thermalRamp(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 cold   = vec3(0.10, 0.42, 0.95); // #1a6bf2-ish deep blue
        vec3 cool   = vec3(0.16, 0.78, 0.92); // cyan
        vec3 warm   = vec3(1.00, 0.78, 0.30); // amber
        vec3 hot    = vec3(1.00, 0.46, 0.18); // orange
        vec3 alarm  = vec3(1.00, 0.18, 0.22); // red
        vec3 c;
        if (t < 0.35)      c = mix(cold, cool, t / 0.35);
        else if (t < 0.60) c = mix(cool, warm, (t - 0.35) / 0.25);
        else if (t < 0.85) c = mix(warm, hot,  (t - 0.60) / 0.25);
        else               c = mix(hot,  alarm,(t - 0.85) / 0.15);
        return c;
      }

      void main() {
      `,
    );

    // Add the heat as emissive AFTER the standard emissive block is computed, so
    // it lights up the panel without flattening the PBR lighting underneath.
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <emissivemap_fragment>",
      /* glsl */ `
      #include <emissivemap_fragment>
      // ease the low end so cold panels stay near-black, then bloom as it heats.
      float heatGlow = smoothstep(0.15, 1.0, uHeat);
      totalEmissiveRadiance += thermalRamp(uHeat) * heatGlow * 1.4;
      `,
    );
  };
  // Force a program recompile key so onBeforeCompile is honoured per material.
  material.customProgramCacheKey = () => "thermal-v1";

  return {
    material,
    setHeat(heat: number): void {
      uHeat.value = Math.max(0, Math.min(1, heat));
    },
    heat(): number {
      return uHeat.value;
    },
  };
}

/** Build the shared professional PBR kit. Created once; reused across the pod. */
export function createProMaterialKit(): ProMaterialKit {
  const chassis = new MeshStandardMaterial({
    color: new Color("#1b2230"),
    metalness: 0.85,
    roughness: 0.42,
  });

  const panel = new MeshStandardMaterial({
    color: new Color("#222b3c"),
    metalness: 0.6,
    roughness: 0.62,
  });

  const aluminium = new MeshStandardMaterial({
    color: new Color("#aeb8c6"),
    metalness: 0.95,
    roughness: 0.32,
  });

  const bezel = new MeshStandardMaterial({
    color: new Color("#0c1018"),
    metalness: 0.3,
    roughness: 0.78,
  });

  const glass = new MeshStandardMaterial({
    color: new Color("#0a121f"),
    metalness: 0.2,
    roughness: 0.12,
    transparent: true,
    opacity: 0.34,
  });

  const floorTile = new MeshStandardMaterial({
    color: new Color("#0a0f18"),
    metalness: 0.6,
    roughness: 0.38,
  });

  const cablePower = new MeshStandardMaterial({
    color: new Color("#b5642f"),
    metalness: 0.4,
    roughness: 0.6,
  });

  const cableFabric = new MeshStandardMaterial({
    color: new Color("#2f8f96"),
    metalness: 0.35,
    roughness: 0.55,
  });

  return {
    chassis,
    panel,
    aluminium,
    bezel,
    glass,
    floorTile,
    cablePower,
    cableFabric,
    dispose(): void {
      for (const m of [
        chassis,
        panel,
        aluminium,
        bezel,
        glass,
        floorTile,
        cablePower,
        cableFabric,
      ]) {
        m.dispose();
      }
    },
  };
}
