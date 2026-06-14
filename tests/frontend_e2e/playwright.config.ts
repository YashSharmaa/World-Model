// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio · end-to-end (Playwright)
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Doc H "Test pyramid": a small Playwright suite over the MAIN decision flow,
//  run headless in CI. This config boots the real Vite dev server for the web app
//  (apps/web) and points the suite at it. No backend is required — the typed API
//  client (apps/web/src/api/client.ts) degrades to deterministic mock data when
//  the FastAPI service is offline, so the three-action compare flow is fully
//  exercisable headless and hermetically.
//
//  The webServer command runs from the repo root; `cwd` switches it into
//  apps/web so Vite resolves that package's vite.config.ts. The spec opens the
//  page with `?autostart=0` so the 8-beat cinematic auto-demo does NOT race the
//  assertions — the test drives the compare flow itself.

import { defineConfig, devices } from "@playwright/test";

const PORT = Number(process.env.PW_WEB_PORT ?? 5180);
const BASE_URL = `http://127.0.0.1:${PORT}`;

export default defineConfig({
  testDir: ".",
  testMatch: "**/*.spec.ts",
  // Headless, deterministic, CI-safe.
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  timeout: 60_000,
  expect: { timeout: 15_000 },

  use: {
    baseURL: BASE_URL,
    headless: true,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],

  // Boot the real web app. `vite` serves apps/web with no build step; the app
  // runs standalone against mock data when the backend is absent.
  webServer: {
    command: `npx vite --host 127.0.0.1 --port ${PORT} --strictPort`,
    cwd: "../../apps/web",
    url: BASE_URL,
    timeout: 120_000,
    reuseExistingServer: !process.env.CI,
    stdout: "pipe",
    stderr: "pipe",
  },
});
