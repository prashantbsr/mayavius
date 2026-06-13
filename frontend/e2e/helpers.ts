import { expect, type Page } from "@playwright/test";

// Shared e2e helpers (spec/10 §4). These centralise how the suite reads the two
// always-on observation surfaces so every T-id reads them the SAME way:
//
//   window.__mayaviusDebug   — the pinned RENDER surface (spec/07 §4.4):
//       staticPointCount / dynamicPointCount / frameIndex / cameraQuaternion.
//   window.__mayaviusStore() — the test-observability STORE surface
//       (src/lib/state/testObservability.ts): the live viewerStore snapshot
//       (time / isPlaying / loop / progress / loadState / frozen …).
//
// Race-free framing (mirror of T-303): tests assert observed SETS / monotonicity
// captured OVER THE RUN, polled via expect.poll — never a single guaranteed
// mid-flight DOM frame.

export interface MayaviusDebug {
  staticPointCount: number;
  dynamicPointCount: number;
  frameIndex: number;
  cameraQuaternion: [number, number, number, number];
}

/** Read the pinned render debug surface (spec/07 §4.4). */
export async function readDebug(page: Page): Promise<MayaviusDebug | null> {
  return page.evaluate(() => window.__mayaviusDebug ?? null);
}

/** Read a single numeric field from the render debug surface, or `fallback`. */
export async function debugNumber(
  page: Page,
  field: "staticPointCount" | "dynamicPointCount" | "frameIndex",
  fallback = -1,
): Promise<number> {
  return page.evaluate(
    ({ f, fb }) => window.__mayaviusDebug?.[f] ?? fb,
    { f: field, fb: fallback },
  );
}

/** Read the camera quaternion from the render debug surface. */
export async function cameraQuaternion(
  page: Page,
): Promise<[number, number, number, number] | null> {
  return page.evaluate(() => window.__mayaviusDebug?.cameraQuaternion ?? null);
}

/** Read a single field from the live viewerStore snapshot. */
export async function storeField<T = unknown>(
  page: Page,
  field: string,
): Promise<T | null> {
  return page.evaluate((f) => {
    const s = window.__mayaviusStore?.();
    return s ? ((s as Record<string, unknown>)[f] as T) : null;
  }, field);
}

/**
 * Open `/view/<id>` and wait until the scene is rendered: the WebGL `<canvas>`
 * is present AND `__mayaviusDebug.staticPointCount > 0` (the T-401 reveal gate).
 * Used by every viewer-flow test so they start from a deterministic "scene
 * loaded" state. Polls — no wall-clock sleep.
 */
export async function openLoadedViewer(page: Page, id: string): Promise<void> {
  await page.goto(`/view/${id}`);
  await page.waitForSelector("canvas", { timeout: 30_000 });
  await expect
    .poll(() => debugNumber(page, "staticPointCount", -1), {
      timeout: 30_000,
      intervals: [100, 200, 300],
    })
    .toBeGreaterThan(0);
}

/**
 * Return the always-on progress history (`window.__mayaviusProgressLog`) — the
 * ordered, de-duplicated set of every `progress` value the store has held since
 * the viewer mounted (src/lib/state/testObservability.ts). The recorder is
 * installed BEFORE the loader effect, so the buffered SSE progression
 * (0.25 → 0.75 → 1) is captured deterministically — T-402 reads this rather than
 * catching a mid-flight DOM frame (mirror of T-303's observed-set framing).
 */
export async function progressSeen(page: Page): Promise<number[]> {
  return page.evaluate(() => window.__mayaviusProgressLog ?? []);
}

/**
 * Drag a range `<input>` thumb across [0,1] by dispatching the native `input`
 * sequence the React handler listens to. Real pointer drags on a styled range
 * input are flaky across engines; the scrubber's contract (spec/07 §4.2) is "an
