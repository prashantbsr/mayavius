import { test, expect } from "@playwright/test";
import { debugNumber, openLoadedViewer } from "./helpers";

// T-401 — example.reconstructs (spec/10 §4).
//
// Click the seeded preloaded example → the viewer mounts and a THREE.Points
// cloud appears: assert window.__mayaviusDebug.staticPointCount > 0 (the
// test-observability contract, spec/07 §4.4). The static background is visible.
// A seeded example is a terminal `done` job (spec/06 §6), so the unified loader
// fetches+decodes immediately — no upload, no GPU.

test.describe("T-401 example.reconstructs", () => {
  test("opening the seeded example mounts the viewer and renders a static cloud", async ({
    page,
  }) => {
    // Start on the landing and CLICK the example card (the real user path).
    await page.goto("/");
    await page.getByTestId("example-card-example").click();
    await expect(page).toHaveURL(/\/view\/example$/);

    // The WebGL canvas mounts only after hydration (ssr:false boundary, T-407).
    await page.waitForSelector("canvas", { timeout: 30_000 });

    // The reveal gate: a real THREE.Points static layer exists → count > 0.
    await expect
      .poll(() => debugNumber(page, "staticPointCount", -1), {
        timeout: 30_000,
        intervals: [100, 200, 300],
      })
      .toBeGreaterThan(0);
