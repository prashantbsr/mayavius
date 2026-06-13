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
//   onProgress → setLoadState('processing') + setProgress(p) (+ license label)
//   onDone     → setScene(scene) (flips loadState='ready' in the store action)
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
      },
      onDone: (scene, json) => {
        // The blob is in hand → decoding to a scene; setScene flips to 'ready'.
        setLoadState("loading");
        if (json.weights_license) setWeightsLicense(json.weights_license);
        setProgress(1);
        setScene(scene);
      },
      onError: (message) => {
        setLoadState("error");
        setError(message);
      },
    });

