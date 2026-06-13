"use client";

import { useCallback, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { submitClip } from "@/lib/api/client";
import { useViewerStore } from "@/lib/state/viewerStore";
import { MAX_UPLOAD_MB } from "@/config";

// UploadDropzone — the ONLY producer of a new `submitClip` (spec/07 §6 detail).
// A `'use client'` island on the otherwise-static (Server Component) landing
// page (spec/07 §8): drag/drop or file-pick → validate type+size → submitClip →
// router.push('/view/'+jobId). The `/view/[id]` route never submits; it only
// loads an existing id (spec/07 §6 step 1).
//
// Client-side validation mirrors the backend's `MAYAVIUS_MAX_UPLOAD_MB`
// (spec/08 §6): reject `> MAX_UPLOAD_MB` and non-`video/*` BEFORE the POST so the
// user gets an instant message (the backend still re-validates → 413/415,
// spec/06 §2.2 — this is a UX shortcut, not the authority).

const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;

/** Validate a picked/dropped file. Returns an error string, or null if OK. */
function validateClip(file: File): string | null {
  // type is `video/*` per spec/07 §6 step 1. Some browsers report an empty
  // `type` for unknown files — treat that as a rejection (not a video).
  if (!file.type.startsWith("video/")) {
    return `Please choose a video file (got "${file.type || "unknown"}").`;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    const mb = (file.size / (1024 * 1024)).toFixed(1);
    return `Clip is ${mb} MB — the limit is ${MAX_UPLOAD_MB} MB.`;
  }
  return null;
}

export function UploadDropzone() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  // loadState lives in the store so the (future) ProgressOverlay and any other
  // surface observe the same upload lifecycle (spec/07 §4.1). We flip it to
  // 'submitting' for the POST, and to 'error' on failure.
  const loadState = useViewerStore((s) => s.loadState);
  const setLoadState = useViewerStore((s) => s.setLoadState);
  const setError = useViewerStore((s) => s.setError);
  const submitting = loadState === "submitting";

  const handleFile = useCallback(
    async (file: File) => {
      if (submitting) return;
      const validationError = validateClip(file);
      if (validationError) {
        setLocalError(validationError);
        setError(validationError);
        setLoadState("error");
        return;
      }
      setLocalError(null);
