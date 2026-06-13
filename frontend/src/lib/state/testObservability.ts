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
