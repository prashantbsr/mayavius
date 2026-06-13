import { test, expect } from "@playwright/test";
import {
  openLoadedViewer,
  debugNumber,
  storeField,
  scrubTimeline,
} from "./helpers";

// T-403 — timeline.scrub (spec/10 §4).
//
// Drag the timeline scrubber → viewerStore.time changes across [0,1]; the
// rendered dynamic frame changes — assert window.__mayaviusDebug.frameIndex
// updates (spec/07 §4.4). The Timeline is a range <input> bound to `time`; its
// `input` handler is the same single-field transient write the R3F loop performs
// (spec/07 §4.2). We drive that contract over the run and assert the observed SET
// of frameIndex values spans more than one frame — race-free (no mid-flight DOM
// frame dependency).

test.describe("T-403 timeline.scrub", () => {
  test("scrubbing the timeline advances time and updates the rendered frame", async ({
    page,
  }) => {
    await openLoadedViewer(page, "example");

    // Pause first so the R3F loop is not also writing `time` (deterministic).
    await page.getByRole("button", { name: /play/i }).isVisible();

    const scrubber = 'input[aria-label="Timeline"]';
    await expect(page.locator(scrubber)).toBeVisible();

    // Start at 0.
    await scrubTimeline(page, scrubber, 0, 0, 1);
    const timeAtStart = (await storeField<number>(page, "time")) ?? -1;
    expect(timeAtStart).toBeCloseTo(0, 5);

    // Collect frameIndex across a full scrub 0→1, capturing the SET of frames the
    // renderer showed (mirror T-303: assert the observed set, not one snapshot).
    const frames = new Set<number>();
    const times = new Set<number>();
    const N = 10;
    for (let i = 0; i <= N; i++) {
      const v = i / N;
      await scrubTimeline(page, scrubber, v, v, 1);
      // Let one R3F frame publish the debug surface for this time.
      await page.waitForTimeout(60);
      frames.add(await debugNumber(page, "frameIndex", -1));
      times.add((await storeField<number>(page, "time")) ?? -1);
    }

    // time moved across [0,1].
    const maxTime = Math.max(...times);
    const minTime = Math.min(...times);
    expect(minTime).toBeLessThanOrEqual(0.01);
    expect(maxTime).toBeGreaterThanOrEqual(0.99);

    // The rendered frame index changed (the dynamic cluster moved over the static
    // background). With a multi-frame scene this set spans ≥2 distinct frames.
    expect(
      frames.size,
      `frameIndex should span >1 frame across a full scrub, saw ${JSON.stringify([...frames])}`,
    ).toBeGreaterThan(1);

    // And specifically: frame at time=1 differs from frame at time=0.
    await scrubTimeline(page, scrubber, 0, 0, 1);
    await page.waitForTimeout(60);
    const f0 = await debugNumber(page, "frameIndex", -1);
    await scrubTimeline(page, scrubber, 1, 1, 1);
    await page.waitForTimeout(60);
    const f1 = await debugNumber(page, "frameIndex", -1);
    expect(f1).not.toBe(f0);
  });
});
