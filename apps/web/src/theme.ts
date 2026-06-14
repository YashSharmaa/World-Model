// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Theme system. ONE source of truth for the visual theme, driving BOTH the HTML
//  overlay (via the `data-theme` attribute + CSS variables) AND the 3D scene
//  (background, fog, floor, lighting, bloom) so light/dark is a single coherent
//  switch — never a darkened-then-lightened retrofit.
//
//  Default is a professional LIGHT theme (operator-dashboard grade). The choice
//  persists to localStorage. Scene consumers register via `onTheme`, which fires
//  immediately with the current tokens and again on every change.

export type ThemeName = "light" | "dark";

/** 3D scene tokens for one theme (colours are hex ints / strings for three). */
export interface SceneTheme {
  name: ThemeName;
  background: string;
  fog: string;
  fogDensity: number;
  floorColor: string;
  floorMetalness: number;
  floorRoughness: number;
  floorEnvIntensity: number;
  gridColor: string;
  gridOpacity: number;
  hemiSky: number;
  hemiGround: number;
  hemiIntensity: number;
  keyColor: number;
  keyIntensity: number;
  fillColor: number;
  fillIntensity: number;
  rimColor: number;
  rimIntensity: number;
  ceilingEmissive: number;
  bloomIntensity: number;
  bloomThreshold: number;
  vignetteDarkness: number;
  labelColor: string;
  labelBg: string;
}

const LIGHT: SceneTheme = {
  name: "light",
  background: "#eef2f8", // bright neutral studio
  fog: "#e2e9f2",
  fogDensity: 0.012,
  floorColor: "#dbe2ec",
  floorMetalness: 0.22,
  floorRoughness: 0.62,
  floorEnvIntensity: 0.85,
  gridColor: "#c2ccdb",
  gridOpacity: 0.5,
  hemiSky: 0xffffff,
  hemiGround: 0xcdd6e4,
  hemiIntensity: 0.95,
  keyColor: 0xfff4e6,
  keyIntensity: 2.0,
  fillColor: 0xdce6ff,
  fillIntensity: 0.7,
  rimColor: 0x9fc0ff,
  rimIntensity: 0.45,
  ceilingEmissive: 0.5,
  bloomIntensity: 0.42, // restrained on a bright background; accents still pop
  bloomThreshold: 0.85,
  vignetteDarkness: 0.22,
  labelColor: "#1f2733",
  labelBg: "rgba(255,255,255,0.86)",
};

const DARK: SceneTheme = {
  name: "dark",
  background: "#080b12",
  fog: "#070a11",
  fogDensity: 0.022,
  floorColor: "#0a0f18",
  floorMetalness: 0.7,
  floorRoughness: 0.35,
  floorEnvIntensity: 1.0,
  gridColor: "#1b2436",
  gridOpacity: 1.0,
  hemiSky: 0xaac4ff,
  hemiGround: 0x10131c,
  hemiIntensity: 0.35,
  keyColor: 0xfff2e0,
  keyIntensity: 2.1,
  fillColor: 0x88a6ff,
  fillIntensity: 0.55,
  rimColor: 0x59d6ff,
  rimIntensity: 0.9,
  ceilingEmissive: 1.6,
  bloomIntensity: 0.9,
  bloomThreshold: 0.72,
  vignetteDarkness: 0.55,
  labelColor: "#e8eef7",
  labelBg: "rgba(10,14,22,0.7)",
};

const TOKENS: Record<ThemeName, SceneTheme> = { light: LIGHT, dark: DARK };
const STORAGE_KEY = "gcwm-theme";

type Listener = (t: SceneTheme) => void;
const listeners = new Set<Listener>();

function readInitial(): ThemeName {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* localStorage may be unavailable (e.g. sandboxed) */
  }
  return "light"; // professional light theme is the default
}

let current: ThemeName = readInitial();

/** Apply the active theme to the document root (sets `data-theme`). */
function applyToDocument(): void {
  if (typeof document !== "undefined") {
    document.documentElement.dataset.theme = current;
  }
}
applyToDocument();

export function getTheme(): ThemeName {
  return current;
}

export function getSceneTheme(): SceneTheme {
  return TOKENS[current];
}

/** Set the active theme, persist it, update the DOM, and notify consumers. */
export function setTheme(name: ThemeName): void {
  current = name;
  applyToDocument();
  try {
    localStorage.setItem(STORAGE_KEY, name);
  } catch {
    /* ignore */
  }
  const tokens = TOKENS[current];
  for (const cb of listeners) cb(tokens);
}

export function toggleTheme(): ThemeName {
  setTheme(current === "light" ? "dark" : "light");
  return current;
}

/**
 * Register a scene consumer. Fires immediately with the current tokens and on
 * every subsequent change. Returns an unsubscribe function.
 */
export function onTheme(cb: Listener): () => void {
  listeners.add(cb);
  cb(TOKENS[current]);
  return () => listeners.delete(cb);
}
