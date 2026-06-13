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
