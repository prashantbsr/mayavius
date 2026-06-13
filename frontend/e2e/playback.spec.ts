import { test, expect } from "@playwright/test";
import { openLoadedViewer, storeField, scrubTimeline } from "./helpers";

// T-404 — playback.toggle (spec/10 §4).
//
// Click play → isPlaying=true, time advances on the R3F loop; click again →
// pauses; toggle loop → playback wraps at time=1→0 when loop=true. We observe the
// store (isPlaying/time/loop) via the always-on store surface and assert state
// transitions + that time advanced over the run — never a single mid-flight DOM
// frame.

test.describe("T-404 playback.toggle", () => {
  test("play advances time, pause halts it, and loop wraps 1→0", async ({
    page,
  }) => {
    await openLoadedViewer(page, "example");

    const playBtn = page.getByRole("button", { name: /play/i });
    const pauseBtn = page.getByRole("button", { name: /pause/i });
    const loopBtn = page.getByRole("button", { name: /^loop$/i });

    // Loop defaults to true (spec/07 §4.1) — confirm via the store.
    expect(await storeField<boolean>(page, "loop")).toBe(true);

    // ── Play → isPlaying flips true and time advances on the loop. ──────────────
    await scrubTimeline(page, 'input[aria-label="Timeline"]', 0, 0, 1); // reset to 0
    await playBtn.click({ force: true });
    await expect
      .poll(() => storeField<boolean>(page, "isPlaying"), { timeout: 5_000 })
      .toBe(true);
    // time advanced past 0 (the R3F loop writes setTime while playing).
    await expect
      .poll(() => storeField<number>(page, "time"), {
        timeout: 5_000,
        intervals: [50, 100],
      })
      .toBeGreaterThan(0);

    // ── Pause → isPlaying false and time stops advancing. ───────────────────────
    await pauseBtn.click({ force: true });
    await expect
      .poll(() => storeField<boolean>(page, "isPlaying"), { timeout: 5_000 })
      .toBe(false);
    const tA = (await storeField<number>(page, "time")) ?? -1;
    await page.waitForTimeout(300); // would advance if still playing
    const tB = (await storeField<number>(page, "time")) ?? -1;
    expect(tB).toBeCloseTo(tA, 5); // halted

    // ── Loop wrap: with loop=true, playing past time=1 wraps back into [0,1). ────
    // Seed near the end, then play; the loop's `next % 1` wraps it (spec/07 §5).
    await scrubTimeline(page, 'input[aria-label="Timeline"]', 0.97, 0.97, 1);
    expect(await storeField<boolean>(page, "loop")).toBe(true);
    await playBtn.click({ force: true });
    // Observe the wrap: time returns to a small value (< the 0.97 we seeded) while
    // still playing — i.e. it crossed 1 and wrapped, not clamped at 1.
    await expect
      .poll(() => storeField<number>(page, "time"), {
        timeout: 6_000,
        intervals: [30, 60, 100],
      })
