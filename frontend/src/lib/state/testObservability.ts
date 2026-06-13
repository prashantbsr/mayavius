"use client";

import { useEffect } from "react";
import { useViewerStore } from "@/lib/state/viewerStore";

// Test-observability glue (spec/10 §4 e2e). The canonical RENDER surface is
// `window.__mayaviusDebug` (spec/07 §4.4: staticPointCount / dynamicPointCount /
// frameIndex / cameraQuaternion) — written by <PlaybackDriver> from the R3F loop.
// That surface is pinned to exactly those four render fields, so it deliberately
// does NOT expose the playback/job STORE state the Playwright suite must observe
// for T-402/T-403/T-404 (progress, time, isPlaying, loop).
//
// This module adds a SEPARATE, always-on observation surface so the e2e specs can
// read the live store without weakening the §4.4 contract or adding ad-hoc DOM
// testids. It is plain test-observability glue (the W2 gate brief explicitly
// permits this): it never changes app behaviour and is a no-op on the server.
//
//   window.__mayaviusStore()      → a snapshot of the current viewerStore state
//   window.__mayaviusOnStore(cb)  → subscribe; returns an unsubscribe fn
//   window.__mayaviusProgressLog  → the ordered, de-duplicated history of every
//                                   `progress` value the store has held since the
//                                   viewer mounted (T-402 reads this).
//
// T-402 asserts an observed-SET property over the run (≥1 value with 0<p<1, then
// 1), mirroring T-303 — never a single mid-flight DOM frame. Because this hook is
// mounted BEFORE the loader effect (ViewerCanvas calls it first), its store
// subscription is active before the first progress write, so the buffered SSE
// progression (0.25 → 0.75 → 1, spec/06 §4.6) is captured deterministically.

declare global {
  interface Window {
    /** Snapshot of the live viewerStore state (test-only observation). */
    __mayaviusStore?: () => ReturnType<typeof useViewerStore.getState>;
    /** Subscribe to store changes; returns an unsubscribe fn (test-only). */
    __mayaviusOnStore?: (
      cb: (state: ReturnType<typeof useViewerStore.getState>) => void,
    ) => () => void;
    /** Ordered, de-duplicated history of `progress` values (test-only). */
    __mayaviusProgressLog?: number[];
  }
}

/** Attach the test-observability store surface to `window` and start recording
 * the progress history. Mount once where the viewer mounts, BEFORE the loader, so
 * recording is active before the first progress write. No-op during SSR. */
export function useTestObservability(): void {
  useEffect(() => {
    if (typeof window === "undefined") return;

    window.__mayaviusStore = () => useViewerStore.getState();
    window.__mayaviusOnStore = (cb) => useViewerStore.subscribe(cb);

    // Progress history: seed with the current value, then record every distinct
    // value the store takes. De-duped + order-preserving so the e2e can assert
    // both "saw an intermediate 0<p<1" and monotonicity over the run.
    const log: number[] = [];
    const push = (p: number) => {
      if (typeof p === "number" && log[log.length - 1] !== p) log.push(p);
    };
    window.__mayaviusProgressLog = log;
