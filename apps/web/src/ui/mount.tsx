// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio (React overlay)
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//  Mount the React 2D overlay into the dom_overlay layer. main.ts owns the three
//  .js scene, the store, and the imperative scene callbacks passed via `bridge`.

import { createRoot } from "react-dom/client";

import { App } from "./App";
import { StudioProvider, type StudioBridge } from "./studio-context";

export function mountOverlay(container: HTMLElement, bridge: StudioBridge): void {
  // No StrictMode: the store/SSE side-effects must run exactly once (StrictMode
  // double-invokes effects in dev, which would double-open the stream).
  createRoot(container).render(
    <StudioProvider {...bridge}>
      <App />
    </StudioProvider>,
  );
}
