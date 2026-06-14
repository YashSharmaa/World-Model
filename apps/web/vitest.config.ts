// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Vitest + jsdom config for the Decision Studio unit tests (design doc H test
//  pyramid). The store / client / scene math run against a jsdom DOM so the
//  three.js CPU-side buffers (instanced colours, geometry) and the store's
//  EventSource / fetch fallbacks resolve without a real browser or backend.

import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // jsdom gives the client a DOM + (mockable) fetch/EventSource globals and the
    // scene code an HTMLCanvasElement, so the studio's read-model logic is
    // testable headless.
    environment: "jsdom",
    globals: true,
    // Co-located *.test.ts files plus the pre-existing __tests__ harnesses.
    include: ["src/**/*.test.ts"],
    // The legacy self-running harnesses in src/__tests__ call process.exit() and
    // are run by the e2e/tsc lane, not Vitest; keep them out of the unit run.
    exclude: ["src/__tests__/**", "node_modules/**", "dist/**"],
    clearMocks: true,
    restoreMocks: true,
  },
});
