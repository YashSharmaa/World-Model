// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  labels — crisp in-scene text. We use canvas-texture sprites rather than a
//  heavyweight SDF text dependency: troika-three-text is great but adds bundle
//  weight and an async font load, and the prompt explicitly allows a canvas
//  sprite fallback that ALWAYS builds. Sprites are billboarded (always face the
//  camera), high-DPI (devicePixelRatio-aware), and pooled per text string.
//
//  Provides:
//    - createTextSprite(text, opts) : one billboarded label
//    - buildRackLabels(rackViews)   : a label group anchored over each rack
//    - createCreditPlate()          : a 3D "Built by Yash Sharma" plate that
//      keeps the author attribution visible IN the scene (in addition to the
//      persistent HTML footer credit).

import {
  CanvasTexture,
  Color,
  Group,
  LinearFilter,
  Sprite,
  SpriteMaterial,
  SRGBColorSpace,
  Vector3,
} from "three";
import type { RackView } from "./racks";

export interface TextSpriteOptions {
  /** CSS colour for the glyphs. */
  color?: string;
  /** background pill colour (rgba ok); empty string = transparent. */
  background?: string;
  /** font px size on the source canvas (drives crispness). */
  fontPx?: number;
  /** world height of the sprite; width derives from the text aspect. */
  worldHeight?: number;
  /** font weight. */
  weight?: number;
}

/** Build one billboarded text sprite backed by a high-DPI canvas texture. */
export function createTextSprite(
  text: string,
  opts: TextSpriteOptions = {},
): Sprite {
  const color = opts.color ?? "#e8eef7";
  const background = opts.background ?? "rgba(10,16,26,0.72)";
  const fontPx = opts.fontPx ?? 64;
  const worldHeight = opts.worldHeight ?? 0.5;
  const weight = opts.weight ?? 600;

  const dpr = Math.min(typeof window !== "undefined" ? window.devicePixelRatio : 1, 2);
  const pad = fontPx * 0.5;

  // measure on a scratch canvas first.
  const measure = document.createElement("canvas").getContext("2d");
  const font = `${weight} ${fontPx}px "Inter", system-ui, sans-serif`;
  if (measure) measure.font = font;
  const textW = measure ? measure.measureText(text).width : text.length * fontPx * 0.6;

  const canvas = document.createElement("canvas");
  const w = Math.ceil((textW + pad * 2) * dpr);
  const h = Math.ceil((fontPx + pad) * dpr);
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.scale(dpr, dpr);
    const cw = w / dpr;
    const ch = h / dpr;
    if (background) {
      ctx.fillStyle = background;
      roundRect(ctx, 0, 0, cw, ch, ch * 0.32);
      ctx.fill();
      ctx.strokeStyle = "rgba(120,150,200,0.35)";
      ctx.lineWidth = 1.5;
      roundRect(ctx, 0.75, 0.75, cw - 1.5, ch - 1.5, ch * 0.32);
      ctx.stroke();
    }
    ctx.font = font;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = color;
    ctx.fillText(text, cw / 2, ch / 2 + 1);
  }

  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  texture.minFilter = LinearFilter;
  texture.magFilter = LinearFilter;
  texture.anisotropy = 4;

  const material = new SpriteMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
    depthTest: true,
  });
  const sprite = new Sprite(material);
  const aspect = w / h;
  sprite.scale.set(worldHeight * aspect, worldHeight, 1);
  sprite.name = `label_${text}`;
  return sprite;
}

/** Anchor a crisp label above each rack (and over the ToR switch). */
export function buildRackLabels(rackViews: RackView[]): Group {
  const group = new Group();
  group.name = "rack_labels";
  for (const rv of rackViews) {
    const label = createTextSprite(rv.rackId.toUpperCase(), {
      color: "#cfe2ff",
      background: "rgba(10,16,26,0.78)",
      worldHeight: 0.46,
    });
    label.position.copy(rv.center).add(new Vector3(0, rv.center.y + 1.4, 0));
    // labels are non-interactive.
    label.raycast = () => {};
    group.add(label);
  }
  return group;
}

/** A 3D credit plate carrying the author attribution inside the scene. */
export function createCreditPlate(): Sprite {
  const plate = createTextSprite("Built by Yash Sharma · linkedin.com/in/yashsharmaa", {
    color: "#9fb0c8",
    background: "rgba(12,18,28,0.66)",
    fontPx: 44,
    worldHeight: 0.4,
    weight: 500,
  });
  plate.name = "credit_plate";
  plate.raycast = () => {};
  // tint slightly toward the brand accent.
  (plate.material as SpriteMaterial).color = new Color("#ffffff");
  return plate;
}

/** Canvas rounded-rect helper (kept local; no DOM-lib polyfills needed). */
function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}
