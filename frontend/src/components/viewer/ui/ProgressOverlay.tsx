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

  // Once the scene is ready, get out of the way — the point cloud is the reveal.
  if (loadState === "ready") return null;

  const isError = loadState === "error";
  // Clamp the bar to [0,1] defensively (the store value comes from the backend).
  const pct = Math.round(Math.min(1, Math.max(0, progress)) * 100);

  return (
    <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
      <div className="pointer-events-auto w-72 rounded-lg bg-black/70 p-5 text-white backdrop-blur">
        <div className="mb-2 text-sm font-medium">{STATE_LABEL[loadState]}</div>

        {isError ? (
          <p className="text-xs text-red-300">
            {error ?? "Something went wrong."}
          </p>
        ) : (
          <>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/15">
              <div
                className="h-full rounded-full bg-white transition-[width] duration-200"
                style={{ width: `${pct}%` }}
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={pct}
              />
            </div>
            <div className="mt-1 text-right text-[10px] tabular-nums text-white/60">
              {pct}%
            </div>
          </>
        )}

        {weightsLicense ? (
          <div className="mt-3 text-[10px] text-white/50">{weightsLicense}</div>
        ) : null}
      </div>
    </div>
  );
}
