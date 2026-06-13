// Shared frontend types.
//
// The AUTHORITATIVE data shape — exact byte layout, quantization ranges, how
// tracks reference points, per-frame visibility encoding — is defined by
// spec/05-data-contract.md and MUST match the backend encoder exactly. The
// decoded type below (`Mv4dScene`) is spec/05 §5.2 verbatim; it replaces the
// old placeholder `ReconstructionResult`.

/** Lifecycle of an async reconstruction job (backend: spec/06-backend-spec.md
 * §6 — the terminal value the backend emits is `"done"`, NOT `"succeeded"`). */
export type JobStatus = "queued" | "running" | "done" | "failed";

export interface JobHandle {
  id: string;
  status: JobStatus;
  /** 0..1 progress while running. */
  progress?: number;
}

/** Decoded reconstruction payload. TODO(spec/05): real fields. */
export interface ReconstructionResult {
  id: string;
  frameCount: number;
  // frames, per-point color, 3D tracks, camera poses, confidence/visibility …
  // — all defined by the binary wire format in spec/05-data-contract.md.
}
