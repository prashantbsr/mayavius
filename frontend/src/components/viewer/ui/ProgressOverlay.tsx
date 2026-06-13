"use client";

import { useViewerStore } from "@/lib/state/viewerStore";
import type { LoadState } from "@/lib/state/viewerStore";

// ProgressOverlay — PLAIN DOM, talks ONLY to the Zustand store (spec/07 §1
// render-path decoupling rule): NO three / Path-1 import. Mirrors the job
// lifecycle (spec/07 §6 step 2): shows the current `loadState`, a 0..1 progress
// bar, the active weights-license label (`weights_license` surfaced via the
// store), and any error message. The reveal is "a cloud, not a spinner"
// (handover §4.4) — once `loadState==='ready'` we render nothing so the WebGL
// canvas is unobstructed.
//
// Narrow selectors (spec/07 §4.2): the per-frame `time` write never touches any
// of these fields, so this overlay does not re-render during playback.

const STATE_LABEL: Record<LoadState, string> = {
  idle: "Preparing…",
  submitting: "Uploading clip…",
  processing: "Reconstructing…",
  loading: "Decoding scene…",
  ready: "Ready",
  error: "Failed",
};

export function ProgressOverlay() {
  const loadState = useViewerStore((s) => s.loadState);
  const progress = useViewerStore((s) => s.progress);
  const error = useViewerStore((s) => s.error);
  const weightsLicense = useViewerStore((s) => s.weightsLicense);
