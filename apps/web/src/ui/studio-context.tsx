// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio (React overlay)
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  The React overlay reads the SAME vanilla observable Store the renderer drives,
//  so the 2D UI and the 3D scene never diverge. `useStore` bridges the store into
//  React via useSyncExternalStore; `useThemeName` bridges the shared theme. The
//  StudioProvider also carries the imperative scene callbacks main.ts owns
//  (run the golden-path demo, toggle a domain context layer).

import * as React from "react";

import type { WorldModelClient } from "../api/client";
import type { Store, StoreState } from "../state/store";
import { getTheme, onTheme, type ThemeName } from "../theme";

/** Read-only domain context layers the scene can show/hide (doc G context layers). */
export type DomainLayer = "topology" | "power" | "cooling";
/** Which replay trace the predicted-vs-realized overlay shows. */
export type ReplayTrace = "predicted" | "realized" | "both";

/** A tiny external signal (the imperative render loop owns the value). */
export interface Signal<T> {
  get: () => T;
  subscribe: (cb: () => void) => () => void;
}

export interface StudioBridge {
  store: Store;
  api: WorldModelClient;
  /** Run the 8-beat golden-path cinematic (owned by main.ts). */
  playDemo: () => void;
  /** Toggle a domain context layer on the live scene graph (owned by main.ts). */
  setLayer: (layer: DomainLayer, visible: boolean) => void;
  /** Initial domain-layer visibility so the React control seeds correctly. */
  initialLayers: Record<DomainLayer, boolean>;
  /** Tell the render loop which replay trace to overlay (predicted/realized/both). */
  setTrace: (trace: ReplayTrace) => void;
  /** Self-captioning beat text — written by both the cinematic + the store. */
  beat: Signal<string | null>;
}

const StudioContext = React.createContext<StudioBridge | null>(null);

export function StudioProvider(
  props: StudioBridge & { children: React.ReactNode },
): React.ReactElement {
  const { children, ...bridge } = props;
  return (
    <StudioContext.Provider value={bridge}>{children}</StudioContext.Provider>
  );
}

export function useStudio(): StudioBridge {
  const ctx = React.useContext(StudioContext);
  if (!ctx) throw new Error("useStudio must be used within a StudioProvider");
  return ctx;
}

/** Subscribe a component to the whole store snapshot (immutable, replaced on set). */
export function useStore(): StoreState {
  const { store } = useStudio();
  return React.useSyncExternalStore(
    React.useCallback((cb) => store.subscribe(() => cb()), [store]),
    () => store.get(),
  );
}

/** Subscribe to the shared light/dark theme name (kept in sync with the scene). */
export function useThemeName(): ThemeName {
  return React.useSyncExternalStore(
    (cb) => onTheme(() => cb()),
    () => getTheme(),
  );
}

/** Subscribe to the self-captioning beat text signal driven by main.ts. */
export function useBeat(): string | null {
  const { beat } = useStudio();
  return React.useSyncExternalStore(beat.subscribe, beat.get);
}
