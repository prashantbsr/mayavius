import type { ReconstructionResult } from "@/types";

// Decoder for the compact BINARY wire format.
//
// JSON is forbidden for point payloads (handover §4.5) — it is the difference
// between a ~2s load and a ~40s one and it gates shareable result links. The
// exact header, version byte, dtypes, quantization ranges and track indexing
// are specified in spec/05-data-contract.md and are the SINGLE source of truth
// shared with the backend encoder (spec/06-backend-spec.md).
export function decodeReconstruction(buffer: ArrayBuffer): ReconstructionResult {
  void buffer;
  // TODO(spec/05): parse header → version → typed-array views (zero-copy).
  throw new Error(
    "decodeReconstruction: not implemented (see spec/05-data-contract.md)",
  );
}
