"use client";

import { useEffect } from "react";
import { streamJob } from "@/lib/api/client";
import { useViewerStore } from "@/lib/state/viewerStore";

// On-mount loader (spec/07 §6 "On-mount loader" + step 2/3). ONE unified path for
// BOTH seeded examples AND live upload jobs: the viewer always opens
// `streamJob(resultId)` (SSE, poll fallback). A seeded example is a terminal
// `done` job (spec/06 §6), so its stream emits the terminal state once and the
// loader immediately fetches+decodes; a live job streams progress → done →
// fetch. The client NEVER branches on EXAMPLE_SLUGS — that list is server-only
// (spec/07 §6, §8). `/view/[id]` only loads an existing id; it never submits
// (UploadDropzone is the only producer of submitClip).
//
// Store lifecycle written here:
//   onProgress → setLoadState('processing') + setProgress(p) (+ license + stage)
//   onDone     → setScene(scene) (flips loadState='ready' in the store action),
//                then setTime(0) + play() unless prefers-reduced-motion (auto-play)
//   onError    → setLoadState('error') + setError(msg)
//
// All writes go through store actions (the HUD reads them via narrow selectors);
// this hook never imports three or a Path-1 component.

export function useLoadScene(resultId: string): void {
  // Read action references once (stable in Zustand) — not inside the effect body,
  // so the effect's only reactive input is `resultId`.
  const setLoadState = useViewerStore((s) => s.setLoadState);
  const setProgress = useViewerStore((s) => s.setProgress);
  const setScene = useViewerStore((s) => s.setScene);
  const setError = useViewerStore((s) => s.setError);
  const setWeightsLicense = useViewerStore((s) => s.setWeightsLicense);
  const setStage = useViewerStore((s) => s.setStage);
  const setTime = useViewerStore((s) => s.setTime);
  const play = useViewerStore((s) => s.play);

  useEffect(() => {
    // Fresh load: clear any prior error and enter the processing phase. The
    // stream resolves the rest (a seeded example jumps straight to `done`).
    setError(null);
    setProgress(0);
    setLoadState("processing");

    const stop = streamJob(resultId, {
      onProgress: (handle, json) => {
        setLoadState("processing");
        setProgress(handle.progress ?? json.progress ?? 0);
        // Surface the active weights-license label as soon as it's known
        // (spec/07 §6 step 2 — ProgressOverlay renders it).
        if (json.weights_license) setWeightsLicense(json.weights_license);
        // Surface the backend `stage` so the overlay can label the current pass
        // (keeps the pct-static VGGT geometry step feeling alive, §6 step 2).
        if (json.stage) setStage(json.stage);
      },
      onDone: (scene, json) => {
        // The blob is in hand → decoding to a scene; setScene flips to 'ready'.
        setLoadState("loading");
        if (json.weights_license) setWeightsLicense(json.weights_license);
        if (json.stage) setStage(json.stage);
        setProgress(1);
        setScene(scene);
        // Auto-play the 4D motion on reveal so the scene isn't frozen at frame 0
        // — the "wow" is immediacy (handover §intro). Rewind to 0 first, then
        // play, both via store actions (the §4.2 single-surface rule).
        //
        // WCAG 2.3.3 (Animation from Interactions): respect a user's
        // prefers-reduced-motion setting — if reduced motion is requested, leave
        // playback paused at frame 0 so the reveal doesn't auto-animate.
        const prefersReducedMotion =
          typeof window !== "undefined" &&
          window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        setTime(0);
        if (!prefersReducedMotion) play();
      },
      onError: (message) => {
        setLoadState("error");
        setError(message);
      },
    });

    // Tear down the stream/poll loop on unmount or when the id changes.
    return stop;
  }, [
    resultId,
    setLoadState,
    setProgress,
    setScene,
    setError,
    setWeightsLicense,
    setStage,
    setTime,
    play,
  ]);
}
