// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio · end-to-end (Playwright)
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Doc H "Test pyramid": one Playwright spec covering the MAIN decision flow,
//  end-to-end, headless:
//
//      load the scene  →  pick an action (Compare actions)
//                       →  a ranked recommendation appears
//                       →  the blocked branch explains itself (why-not)
//
//  This is the doc-G three-action compare (hold + admit→A + admit→B): the core
//  promise of the studio — "same state, different action, different future" with
//  an explicit safety GATE (a blocked branch can never win, and it must say why).
//
//  It runs against the REAL web app booted by playwright.config.ts. No backend is
//  needed: the typed client degrades to deterministic mock data (mock.ts), whose
//  golden path is the same one the unit harness (compareBlockedGoldenPath) pins —
//  admit→A (dom-rack-a1) is the safe rank-#1 recommendation and admit→B
//  (dom-rack-b2, drained) is BLOCKED on gang-atomicity. We assert the rendered
//  decision UI, not the internal store, so this is a true end-to-end check.

import { test, expect, type Page } from "@playwright/test";

/** Open the studio with the cinematic auto-demo disabled so the test drives. */
async function openStudio(page: Page): Promise<void> {
  await page.goto("/?autostart=0");
  // The HTML overlay shell mounts after main() boots the renderer + store.
  await expect(page.locator("#dom-overlay")).toBeAttached();
  // Boot focuses the pending workload and renders the queue strip.
  await expect(page.locator(".queue-strip .panel-title")).toHaveText(
    "Pending Workloads",
  );
}

test.describe("Decision Studio — three-action compare flow", () => {
  test("load scene → pick action → ranked recommendation + why-not", async ({
    page,
  }) => {
    await openStudio(page);

    // ── load scene ──────────────────────────────────────────────────────────
    // The WebGL canvas exists and the pending-workload card is on the strip.
    await expect(page.locator("#webgl-canvas")).toBeVisible();
    const focusedCard = page.locator(".queue-strip .wl-card").first();
    await expect(focusedCard).toBeVisible();

    // Before comparing, the recommendation rail prompts for an action.
    const rec = page.locator(".panel.recommendation");
    await expect(rec).toContainText(/compare safe actions/i);

    // ── pick an action: run the three-branch compare ────────────────────────
    // The "Compare actions" chip focuses the workload and calls store.runCompare().
    await focusedCard.getByRole("button", { name: /compare actions/i }).click();

    // ── ranked recommendation appears ───────────────────────────────────────
    // The best summary names a recommended SAFE action (admit→Rack Group A).
    await expect(rec.locator(".rec-best")).toContainText(/Recommended:/i);
    await expect(rec.locator(".rec-best")).toContainText(/Rack Group A/i);

    // Exactly three ranked rows (the shape-stable three-action compare).
    const rows = rec.locator(".rec-row");
    await expect(rows).toHaveCount(3);

    // The top row is the recommended, SAFE, rank-#1 feasible admit. The status
    // badge is `.badge.safe` WITHOUT `.conf` (the confidence Epi/OOD badges reuse
    // `.badge.safe.conf` for their green tint), so exclude `.conf` to target it.
    const topRow = rows.first();
    await expect(topRow).toHaveClass(/recommended/);
    await expect(topRow.locator(".badge.safe:not(.conf)")).toHaveText("SAFE");
    await expect(topRow.locator(".rec-rank")).toHaveText("#1");

    // At least one row is BLOCKED by the exact kernel — safety is a GATE, so a
    // blocked branch is shown but can NEVER be the recommendation.
    const blockedRow = rec.locator(".rec-row.blocked");
    await expect(blockedRow).toHaveCount(1);
    await expect(blockedRow.locator(".badge.blocked:not(.conf)")).toHaveText(
      "BLOCKED",
    );
    await expect(blockedRow.locator(".rec-rank")).toHaveText("🔒");
    // A blocked branch is NOT the recommended row.
    await expect(blockedRow).not.toHaveClass(/recommended/);

    // ── why-not: the blocked branch explains itself ─────────────────────────
    // The why-not card is hidden until requested.
    const whyNot = page.locator(".panel.why-not");
    await expect(whyNot).toHaveClass(/hidden/);

    await blockedRow.getByRole("button", { name: /why not/i }).click();

    // The card opens and keeps the two reason families SEPARATE (doc G/S): the
    // hard exact-kernel gate vs. learned predicted risk.
    await expect(whyNot).not.toHaveClass(/hidden/);
    await expect(whyNot.locator(".panel-title")).toContainText(/Why not/i);
    await expect(whyNot).toContainText(/Exact blockers \(kernel gate\)/i);
    // The drained target rack-b2 is rejected on a gang-atomicity capacity deficit.
    // The kernel may emit more than one blocker group (e.g. a degraded node also
    // trips health_schedulability), so match the group list non-strictly.
    await expect(whyNot.locator(".blocker-group").first()).toContainText(
      /gang atomicity/i,
    );
    await expect(whyNot).toContainText(/Rejected domains:/i);
    await expect(whyNot).toContainText(/dom-rack-b2/i);

    // Closing the card hides it again (the × control).
    await whyNot.getByRole("button", { name: "×" }).click();
    await expect(whyNot).toHaveClass(/hidden/);
  });
});
