// Shared frontend types.
//
// The AUTHORITATIVE data shapes — exact byte layout, quantization ranges, how
// tracks reference points, per-frame visibility encoding — are defined by
// spec/05-data-contract.md and MUST match the backend encoder exactly. The
// types below are minimal placeholders so the scaffold type-checks; they are
// NOT the final contract.

/** Lifecycle of an async reconstruction job (backend: spec/06-backend-spec.md). */
export type JobStatus = "queued" | "running" | "succeeded" | "failed";

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
