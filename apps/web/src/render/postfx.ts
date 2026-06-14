// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  createPostFX — the professional render pipeline (pmndrs/postprocessing).
//
//  Pass order:
//     RenderPass  -> draws the scene into the composer buffer
//     EffectPass  -> a single merged shader combining, in order:
//         BloomEffect        (mipmap blur; threshold tuned so ONLY emissive
//                             GPU/LED/active elements bloom — not the whole pod)
//         ToneMappingEffect  (ACES_FILMIC — the renderer itself does NoToneMapping;
//                             postprocessing owns the curve so HDR bloom maps
//                             into LDR cleanly)
//         VignetteEffect     (subtle darkening at the frame edges)
//         SMAAEffect         (clean sub-pixel edges; cheaper than MSAA here)
//
//  Merging these into one EffectPass keeps it to a single extra fullscreen draw,
//  so the bundle stays light and the recorded demo runs smoothly.

import { HalfFloatType, type Camera, type Scene, type WebGLRenderer } from "three";
import {
  BloomEffect,
  EdgeDetectionMode,
  EffectComposer,
  EffectPass,
  RenderPass,
  SMAAEffect,
  SMAAPreset,
  ToneMappingEffect,
  ToneMappingMode,
  VignetteEffect,
} from "postprocessing";
import { onTheme } from "../theme";

export interface PostFX {
  composer: EffectComposer;
  /** Render one frame; pass the frame delta (seconds) for animated effects. */
  render(deltaTime: number): void;
  setSize(width: number, height: number): void;
  dispose(): void;
}

/**
 * Build the EffectComposer pipeline over an existing renderer/scene/camera.
 * The renderer should be configured for NoToneMapping + SRGB output by the
 * caller (see main.ts) so this pass owns the tone curve.
 */
export function createPostFX(
  renderer: WebGLRenderer,
  scene: Scene,
  camera: Camera,
): PostFX {
  // HalfFloat frame buffers preserve HDR range for bloom + tone mapping (so
  // emissive values above 1.0 bloom correctly instead of clamping).
  const composer = new EffectComposer(renderer, {
    multisampling: 0,
    frameBufferType: HalfFloatType,
  });
  composer.addPass(new RenderPass(scene, camera));

  // Bloom: only bright/emissive things (LEDs, activity strips, hot panels) glow.
  const bloom = new BloomEffect({
    mipmapBlur: true,
    intensity: 0.9,
    luminanceThreshold: 0.72,
    luminanceSmoothing: 0.22,
    radius: 0.72,
    levels: 7,
  });

  const toneMapping = new ToneMappingEffect({
    mode: ToneMappingMode.ACES_FILMIC,
  });

  const vignette = new VignetteEffect({
    offset: 0.32,
    darkness: 0.55,
  });

  const smaa = new SMAAEffect({
    preset: SMAAPreset.HIGH,
    edgeDetectionMode: EdgeDetectionMode.COLOR,
  });

  // Order matters: bloom (HDR) -> tone map to LDR -> vignette -> AA last.
  const effectPass = new EffectPass(camera, bloom, toneMapping, vignette, smaa);
  composer.addPass(effectPass);

  // Theme reactivity: a bright light background needs far less bloom and a much
  // softer vignette than the cinematic dark room, or accents wash out / glare.
  // BloomEffect exposes a typed public `luminanceMaterial` accessor whose
  // `threshold` setter drives the bloom cutoff, so we set it directly — no cast
  // and no catch-and-ignore.
  const unsubscribe = onTheme((t) => {
    bloom.intensity = t.bloomIntensity;
    bloom.luminanceMaterial.threshold = t.bloomThreshold;
    vignette.darkness = t.vignetteDarkness;
  });

  return {
    composer,
    render(deltaTime: number): void {
      composer.render(deltaTime);
    },
    setSize(width: number, height: number): void {
      composer.setSize(width, height);
    },
    dispose(): void {
      unsubscribe();
      composer.dispose();
    },
  };
}
