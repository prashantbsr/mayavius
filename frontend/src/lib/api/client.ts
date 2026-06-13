import type { JobHandle, ReconstructionResult } from "@/types";
import { decodeReconstruction } from "@/lib/wire/decoder";

// Thin client for the FastAPI reconstruction backend (async job model,
// handover §4.4): submit a clip → poll/stream status → fetch the binary result.
// Endpoint paths, payloads and the error contract are defined in
// spec/06-backend-spec.md. Base URL: API_BASE_URL in config.ts.

export async function submitClip(file: File): Promise<JobHandle> {
  void file;
  // TODO(spec/06): POST multipart/form-data to /jobs; return the job handle.
  throw new Error("submitClip: not implemented (see spec/06-backend-spec.md)");
}

export async function getJobStatus(jobId: string): Promise<JobHandle> {
  void jobId;
  // TODO(spec/06): GET /jobs/{id} (or subscribe to an SSE progress stream).
  throw new Error("getJobStatus: not implemented (see spec/06-backend-spec.md)");
}

export async function fetchResult(jobId: string): Promise<ReconstructionResult> {
  void jobId;
  // TODO(spec/06): GET /jobs/{id}/result as an ArrayBuffer, then decode.
  return decodeReconstruction(new ArrayBuffer(0));
}
