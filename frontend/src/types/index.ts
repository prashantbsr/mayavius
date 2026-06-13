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

/**
 * Decoded MV4D v1 reconstruction (spec/05 §5.2). All array fields are
 * **zero-copy views** over the fetched `ArrayBuffer`; Path-1 shaders dequantize
 * `positionsQ` on the GPU (spec/07 §2.1). Track positions (small) may be
 * dequantized to `Float32Array` for `Line2` ribbons on the CPU.
 */
export interface Mv4dScene {
  version: 1;
  /** T — number of timesteps. */
  frameCount: number;
  fps: number;
  aabbMin: [number, number, number];
  aabbMax: [number, number, number];
  static?: {
    count: number;
    /** length count*3, quantized (dequant in shader). */
    positionsQ: Uint16Array;
    /** length count*3. */
    colors: Uint8Array;
    /** length count; present iff HAS_STATIC_CONF. */
    conf?: Uint8Array;
  };
  dynamic?: {
    frames: Array<{ count: number; positionsQ: Uint16Array; colors: Uint8Array }>;
  };
  tracks?: {
    /** M — number of tracks. */
    count: number;
    /** length M*T*3. */
    positionsQ: Uint16Array;
    /** packed LSB-first bitmask, length ceil(M*T/8). */
    visibility: Uint8Array;
    /** length M*3; present iff HAS_TRACK_COLOR. */
    colors?: Uint8Array;
    isVisible(m: number, t: number): boolean;
  };
  cameras?: {
    /** length T*7 — {qx,qy,qz,qw,tx,ty,tz} per frame. */
    poses: Float32Array;
    /** length T*4 — {fx,fy,cx,cy} normalized per frame. */
    intrinsics: Float32Array;
  };
}
