// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Vite build config. The dev server proxies /api/* to the local Python serving
//  layer (doc G backend contract) so the browser can talk to GET state, POST
//  compare / why-not, GET replays, and the GET /api/v1/stream SSE channel
//  without CORS friction. When the backend is absent the typed client falls
//  back to deterministic mock data (see src/api/client.ts), so `npm run dev`
//  and `npm run build` work standalone.

import { defineConfig } from "vite";

/**
 * The backend serving layer (FastAPI, optional) defaults to localhost:8000.
 * Override with `VITE_API_TARGET` for a remote shadow-mode service.
 */
const API_TARGET = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  root: ".",
  base: "./",
  build: {
    outDir: "dist",
    sourcemap: true,
    target: "es2022",
    // The scene is intentionally tiny (instanced primitives); keep a single
    // chunk so the recorded demo loads fast and deterministically.
    chunkSizeWarningLimit: 1500,
  },
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
        // SSE needs the connection kept open and un-buffered.
        ws: false,
        configure: (proxy) => {
          proxy.on("error", () => {
            // Swallow proxy errors: the client degrades to mock data when the
            // backend is offline. This keeps the demo recordable anywhere.
          });
        },
      },
    },
  },
  preview: {
    port: 4173,
  },
});
