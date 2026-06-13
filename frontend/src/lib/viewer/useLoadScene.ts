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
