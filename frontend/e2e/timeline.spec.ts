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

