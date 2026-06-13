import { test, expect } from "@playwright/test";
import {
  openLoadedViewer,
  storeField,
  debugNumber,
  cameraQuaternion,
} from "./helpers";

// T-405 — bullet_time.orbit (spec/10 §4).
//
// Enter bullet-time (freeze) → frozen=true, playback halts; orbit drag rotates
// the camera around the frozen frame — assert window.__mayaviusDebug.
// cameraQuaternion changes while window.__mayaviusDebug.frameIndex is constant
// (spec/07 §4.4/§5). OrbitControls stay enabled in bullet-time; it only STOPS
// time. We assert: frozen flips true + isPlaying false; the quaternion moves
// under an orbit drag; frameIndex is unchanged across the whole drag.

function quatDelta(
  a: [number, number, number, number],
  b: [number, number, number, number],
): number {
  return Math.max(...a.map((v, i) => Math.abs(v - b[i])));
}

test.describe("T-405 bullet_time.orbit", () => {
  test("bullet-time freezes time and an orbit drag rotates the camera", async ({
    page,
  }) => {
    await openLoadedViewer(page, "example");

    // Park time mid-clip so frozen frameIndex is a stable, non-edge value.
    await page.$eval('input[aria-label="Timeline"]', (el) => {
      const input = el as HTMLInputElement;
      const setter = Object.getOwnPropertyDescriptor(
        HTMLInputElement.prototype,
        "value",
      )?.set;
      setter?.call(input, "0.5");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await page.waitForTimeout(60);

    // ── Enter bullet-time → frozen true, playback halted (spec/07 §5). ──────────
    await page.getByRole("button", { name: /bullet.?time/i }).click({
      force: true,
    });
    await expect
      .poll(() => storeField<boolean>(page, "frozen"), { timeout: 5_000 })
      .toBe(true);
    expect(await storeField<boolean>(page, "isPlaying")).toBe(false);
    expect(await storeField<string>(page, "cameraMode")).toBe("bulletTime");

    // The render debug surface's `frameIndex` is published by the R3F loop, so it
    // can lag the (synchronous) store write by a frame. Derive the expected frozen
    // frame from the store (time × frameCount) and wait until the debug surface has
    // SETTLED to it before treating it as the baseline — otherwise we might capture
    // a stale value and see it "change" after the drag (which would be the surface
    // catching up, not time advancing). Once frozen, time cannot advance, so this
    // settle is a one-time convergence, not a moving target.
    const expectedFrame = await page.evaluate(() => {
      const s = window.__mayaviusStore?.();
      if (!s || s.frameCount <= 1) return 0;
      return Math.round(s.time * (s.frameCount - 1));
    });
    await expect
      .poll(() => debugNumber(page, "frameIndex", -999), {
        timeout: 5_000,
        intervals: [50, 100],
      })
      .toBe(expectedFrame);
    const frozenFrame = expectedFrame;
    const qBefore = await cameraQuaternion(page);
    expect(qBefore).not.toBeNull();

    // ── Orbit drag over the canvas (OrbitControls stay enabled — spec/07 §5). ────
    const canvas = page.locator("canvas").first();
    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();
    const cx = box!.x + box!.width / 2;
    const cy = box!.y + box!.height / 2;

    await page.mouse.move(cx, cy);
    await page.mouse.down();
    // Several incremental moves so OrbitControls integrates a real rotation.
    for (let i = 1; i <= 8; i++) {
      await page.mouse.move(cx + i * 18, cy + i * 6, { steps: 2 });
    }
    await page.mouse.up();
    // Let the loop publish the post-drag camera quaternion.
    await page.waitForTimeout(120);
