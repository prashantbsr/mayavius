"use client";

import { useEffect, useState } from "react";
import { useViewerStore } from "@/lib/state/viewerStore";
import type { LoadState } from "@/lib/state/viewerStore";

// ProgressOverlay — PLAIN DOM, talks ONLY to the Zustand store (spec/07 §1
// render-path decoupling rule): NO three / Path-1 import. Mirrors the job
// lifecycle (spec/07 §6 step 2): shows the current `loadState` (refined by the
// backend `stage` token), a 0..1 progress bar, the active weights-license label
// (`weights_license` surfaced via the store), and any error message. The reveal
// is "a cloud, not a spinner" (handover §4.4) — once `loadState==='ready'` we
// render nothing so the WebGL canvas is unobstructed.
//
// Feeling alive (spec/07 §6 step 2): the long VGGT geometry pass pins `progress`
// at one value, so a static bar reads as crashed. Three cheap fixes, all PLAIN
// DOM: (a) a friendly per-`stage` label, (b) an elapsed mm:ss timer, and (c) an
// indeterminate shimmer overlaid on the bar so it visibly moves even when `pct`
// is frozen. The shimmer keyframe is defined inline (self-contained — no global
// CSS dependency) and gated on prefers-reduced-motion (WCAG 2.3.3): the media
// query inside the <style> drops the animation when reduced motion is requested,
// and `motion-reduce:hidden` removes the band entirely as a belt-and-braces.
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

// Friendly labels for the backend `stage` tokens (spec/06 §6 `job_to_json`).
// When the stage is empty or unknown we fall back to STATE_LABEL[loadState].
const STAGE_LABEL: Record<string, string> = {
  queued: "Waiting…",
  decode: "Reading your video…",
  "loading models": "Loading the model…",
  geometry: "Reconstructing geometry…",
  tracking: "Tracking motion…",
  assembling: "Assembling the scene…",
  assembled: "Assembling the scene…",
  running: "Reconstructing…",
  done: "Ready",
};

/** Format whole seconds as mm:ss for the elapsed timer. */
function formatElapsed(totalSeconds: number): string {
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function ProgressOverlay() {
  const loadState = useViewerStore((s) => s.loadState);
  const progress = useViewerStore((s) => s.progress);
  const error = useViewerStore((s) => s.error);
  const weightsLicense = useViewerStore((s) => s.weightsLicense);
  const stage = useViewerStore((s) => s.stage);

  // Elapsed timer: ticks once a second while the job is in flight so the user
  // sees time passing during the long, pct-static geometry pass. Starts when we
  // enter processing/loading and is torn down on unmount or once we leave those
  // states (ready/idle/error) — the interval is the only side effect here.
  const isActive = loadState === "processing" || loadState === "loading";
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!isActive) {
      setElapsed(0);
      return;
    }
    setElapsed(0);
    const started = Date.now();
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - started) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, [isActive]);

  // Once the scene is ready, get out of the way — the point cloud is the reveal.
  if (loadState === "ready") return null;

  const isError = loadState === "error";
  // Clamp the bar to [0,1] defensively (the store value comes from the backend).
  const pct = Math.round(Math.min(1, Math.max(0, progress)) * 100);

  // Prefer the friendly stage label; fall back to the loadState label when the
  // stage is empty/unknown (spec/07 §6 step 2).
  const label =
    (stage && STAGE_LABEL[stage]) || STATE_LABEL[loadState];

  return (
    <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
      {/* Self-contained shimmer keyframe (no globals.css dependency). The
          reduced-motion media query holds the band still — paired with the
          `motion-reduce:hidden` utility on the band itself (WCAG 2.3.3). */}
      <style>{`
        @keyframes mv-shimmer {
          0%   { transform: translateX(0); }
          100% { transform: translateX(400%); }
        }
        .mv-shimmer { animation: mv-shimmer 1.4s ease-in-out infinite; }
        @media (prefers-reduced-motion: reduce) {
          .mv-shimmer { animation: none; }
        }
      `}</style>
      <div className="pointer-events-auto w-72 rounded-lg bg-black/70 p-5 text-white backdrop-blur">
        <div className="mb-2 flex items-baseline justify-between gap-2">
          <span className="text-sm font-medium">{label}</span>
          {/* Elapsed timer — only while a job is in flight (not on error). */}
          {isActive ? (
            <span className="text-[10px] tabular-nums text-white/50">
              {formatElapsed(elapsed)}
            </span>
          ) : null}
        </div>

        {isError ? (
          <p className="text-xs text-red-300">
            {error ?? "Something went wrong."}
          </p>
        ) : (
          <>
            <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-white/15">
              <div
                className="h-full rounded-full bg-white transition-[width] duration-200"
                style={{ width: `${pct}%` }}
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={pct}
              />
              {/* Indeterminate shimmer: a translucent band that sweeps across the
                  track so the bar visibly moves even when `pct` is frozen on the
                  long geometry pass. Decorative only (aria-hidden); the band is
                  hidden under prefers-reduced-motion via `motion-reduce:hidden`
                  and the keyframe itself is suppressed by the media query in the
                  inline <style> below (WCAG 2.3.3). */}
              {isActive ? (
                <div
                  aria-hidden
                  className="mv-shimmer pointer-events-none absolute inset-y-0 -left-1/3 w-1/3 rounded-full bg-gradient-to-r from-transparent via-white/40 to-transparent motion-reduce:hidden"
                />
              ) : null}
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
