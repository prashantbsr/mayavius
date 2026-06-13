import type { Mv4dScene } from "@/types";

// Decoder for the compact BINARY wire format (MV4D v1).
//
// JSON is forbidden for point payloads (handover §4.5) — it is the difference
// between a ~2s load and a ~40s one and it gates shareable result links. The
// exact header, version byte, dtypes, quantization ranges and track indexing are
// specified in spec/05-data-contract.md §3 and are the SINGLE source of truth
// shared with the backend encoder (backend/app/wire/encoder.py). This decoder is
// the exact inverse of that encoder; if the three ever disagree, spec/05 wins.
//
// Zero-copy contract (spec/05 §1): every section is decoded as a TypedArray
// *view* over the same fetched ArrayBuffer — no per-point copying. Quantized
// positions stay Uint16Array and are dequantized in the vertex shader (Path 1,
// spec/07 §2.1); the CPU touches nothing. `dequantize()` below is the off-GPU
// mirror of that shader math (spec/05 §2), used for CPU-side track ribbons and
// for conformance tests.

/** MV4D major version this decoder understands (spec/05 §7). Mirrors the
 * encoder's `MV4D_VERSION`; parity is asserted by spec/10 T-201. */
export const MV4D_VERSION = 1;

// Header / block sizes (spec/05 §3.1–§3.3).
const HEADER_BYTES = 24;
const AABB_BYTES = 24;
const DIR_ENTRY_BYTES = 16;
const POS_BITS = 16;
const QMAX = 65535;

// `"MV4D"` magic as little-endian bytes (0x4D 0x56 0x34 0x44), spec/05 §3.1.
const MAGIC_0 = 0x4d; // 'M'
const MAGIC_1 = 0x56; // 'V'
const MAGIC_2 = 0x34; // '4'
const MAGIC_3 = 0x44; // 'D'

// flags bits (spec/05 §3.1). Bits 0–3 are presence hints (directory is
// authoritative); bits 4–5 gate optional sub-arrays and MUST be honored.
const FLAG_HAS_STATIC_CONF = 1 << 4;
const FLAG_HAS_TRACK_COLOR = 1 << 5;

// section kinds (spec/05 §3.3).
const KIND_STATIC = 1;
const KIND_DYNAMIC = 2;
const KIND_TRACKS = 3;
const KIND_CAMERAS = 4;

/** Typed error thrown by {@link decodeReconstruction} on every malformed-buffer
 * case in spec/05 §8: bad magic, unsupported major version, `posBits ≠ 16`, a
 * section whose `byteOffset+byteLength` exceeds the buffer, or a misaligned
 * section offset. The decoder never returns a partially-filled scene. */
export class Mv4dDecodeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "Mv4dDecodeError";
    // Restore the prototype chain so `instanceof Mv4dDecodeError` holds even
    // after TS downlevels `extends Error` (ES2017 target).
    Object.setPrototypeOf(this, Mv4dDecodeError.prototype);
  }
}

/** Inverse of the encoder quantization (spec/05 §2): the off-GPU mirror of the
 * Path-1 vertex shader's dequant. `p = min + q/65535 * (max - min)`. A
 * degenerate axis (`max == min`) yields `min` because every `q` there is 0. */
export function dequantize(q: number, min: number, max: number): number {
  return min + (q / QMAX) * (max - min);
}

interface DirEntry {
  kind: number;
  byteOffset: number;
  byteLength: number;
  count: number;
}

/**
 * Decode an MV4D v1 buffer into a {@link Mv4dScene} of zero-copy typed-array
 * views over the SAME `ArrayBuffer` (spec/05 §3, §5.2). Throws
 * {@link Mv4dDecodeError} on any malformed buffer (spec/05 §8) and never returns
 * a partial scene. Uses the section directory for offsets (order-agnostic) and
 * skips unknown kinds (forward compatibility, spec/05 §1).
 */
export function decodeReconstruction(buffer: ArrayBuffer): Mv4dScene {
  if (buffer.byteLength < HEADER_BYTES + AABB_BYTES) {
    throw new Mv4dDecodeError(
      `MV4D buffer too short (${buffer.byteLength} bytes) for header + AABB`,
    );
  }

  // All header/AABB/directory scalars are read little-endian via DataView
  // (spec/05 §2). Bulk arrays are host-endian (LE assumption documented there).
  const view = new DataView(buffer);

  // ---- header (spec/05 §3.1) ----
  if (
    view.getUint8(0) !== MAGIC_0 ||
    view.getUint8(1) !== MAGIC_1 ||
    view.getUint8(2) !== MAGIC_2 ||
    view.getUint8(3) !== MAGIC_3
  ) {
    const got = [0, 1, 2, 3].map((i) => view.getUint8(i));
    throw new Mv4dDecodeError(
      `bad MV4D magic: [${got.join(",")}] (expected "MV4D")`,
    );
  }

  const version = view.getUint8(4);
  if (version !== MV4D_VERSION) {
    throw new Mv4dDecodeError(
      `unsupported MV4D version: ${version} (expected ${MV4D_VERSION})`,
    );
  }

  const flags = view.getUint8(5);
  const posBits = view.getUint8(6);
  if (posBits !== POS_BITS) {
    throw new Mv4dDecodeError(
      `unsupported posBits: ${posBits} (expected ${POS_BITS})`,
    );
  }

  const sectionCount = view.getUint8(7);
  const frameCount = view.getUint16(8, true);
  const fps = view.getFloat32(12, true);

  // ---- AABB block (spec/05 §3.2) ----
  const aabbMin: [number, number, number] = [
    view.getFloat32(HEADER_BYTES + 0, true),
    view.getFloat32(HEADER_BYTES + 4, true),
    view.getFloat32(HEADER_BYTES + 8, true),
  ];
  const aabbMax: [number, number, number] = [
    view.getFloat32(HEADER_BYTES + 12, true),
    view.getFloat32(HEADER_BYTES + 16, true),
    view.getFloat32(HEADER_BYTES + 20, true),
  ];

  // ---- section directory (spec/05 §3.3) ----
  const dirOffset = HEADER_BYTES + AABB_BYTES;
  if (dirOffset + sectionCount * DIR_ENTRY_BYTES > buffer.byteLength) {
    throw new Mv4dDecodeError(
      `section directory (${sectionCount} entries) exceeds buffer`,
    );
  }

  const entries: DirEntry[] = [];
  for (let i = 0; i < sectionCount; i++) {
    const base = dirOffset + i * DIR_ENTRY_BYTES;
    const kind = view.getUint32(base + 0, true);
    const byteOffset = view.getUint32(base + 4, true);
    const byteLength = view.getUint32(base + 8, true);
    const count = view.getUint32(base + 12, true);
    if (byteOffset + byteLength > buffer.byteLength) {
      throw new Mv4dDecodeError(
        `section kind=${kind} exceeds buffer: offset=${byteOffset} ` +
          `length=${byteLength} bufLen=${buffer.byteLength}`,
      );
    }
    // Every section payload begins at an 8-byte-aligned absolute offset
    // (spec/05 §2); a typed view at a misaligned offset is also invalid.
    if (byteOffset % 8 !== 0) {
      throw new Mv4dDecodeError(
        `section kind=${kind} offset ${byteOffset} is not 8-byte aligned`,
      );
    }
    entries.push({ kind, byteOffset, byteLength, count });
  }

  const hasStaticConf = (flags & FLAG_HAS_STATIC_CONF) !== 0;
  const hasTrackColor = (flags & FLAG_HAS_TRACK_COLOR) !== 0;

  const scene: Mv4dScene = {
    version: 1,
    frameCount,
    fps,
    aabbMin,
    aabbMax,
  };

  for (const { kind, byteOffset, count } of entries) {
    if (kind === KIND_STATIC) {
      const n = count;
      let cur = byteOffset;
      const positionsQ = new Uint16Array(buffer, cur, n * 3);
      cur += n * 3 * 2;
      const colors = new Uint8Array(buffer, cur, n * 3);
      cur += n * 3;
      let conf: Uint8Array | undefined;
      if (hasStaticConf) {
        conf = new Uint8Array(buffer, cur, n);
      }
      scene.static = { count: n, positionsQ, colors, ...(conf ? { conf } : {}) };
    } else if (kind === KIND_DYNAMIC) {
      const t = count; // == frameCount
      let cur = byteOffset;
      // frameDir: u32[T*2] {startPoint, pointCount} cumulative (spec/05 §3.5).
      const frameDir = new Uint32Array(buffer, cur, t * 2);
      cur += t * 2 * 4;
      let total = 0;
      for (let i = 0; i < t; i++) total += frameDir[i * 2 + 1];
      const positionsQ = new Uint16Array(buffer, cur, total * 3);
      cur += total * 3 * 2;
      const colors = new Uint8Array(buffer, cur, total * 3);
      const frames: Array<{
        count: number;
        positionsQ: Uint16Array;
        colors: Uint8Array;
      }> = [];
      for (let i = 0; i < t; i++) {
        const start = frameDir[i * 2 + 0];
        const cnt = frameDir[i * 2 + 1];
        frames.push({
          count: cnt,
          // subarray keeps the same backing ArrayBuffer (zero-copy view).
          positionsQ: positionsQ.subarray(start * 3, (start + cnt) * 3),
          colors: colors.subarray(start * 3, (start + cnt) * 3),
        });
      }
      scene.dynamic = { frames };
    } else if (kind === KIND_TRACKS) {
      const m = count;
      const t = frameCount;
      const mt = m * t;
      let cur = byteOffset;
      const positionsQ = new Uint16Array(buffer, cur, mt * 3);
      cur += mt * 3 * 2;
      const visBytes = (mt + 7) >> 3;
      const visibility = new Uint8Array(buffer, cur, visBytes);
      cur += visBytes;
      let colors: Uint8Array | undefined;
      if (hasTrackColor) {
        colors = new Uint8Array(buffer, cur, m * 3);
      }
      // LSB-first bitmask: bit i = m*T + t, byte[i>>3] & (1 << (i & 7)).
      const isVisible = (mm: number, tt: number): boolean => {
        const i = mm * t + tt;
        return (visibility[i >> 3] & (1 << (i & 7))) !== 0;
      };
      scene.tracks = {
        count: m,
        positionsQ,
        visibility,
        ...(colors ? { colors } : {}),
        isVisible,
      };
    } else if (kind === KIND_CAMERAS) {
      const t = count;
      let cur = byteOffset;
      const poses = new Float32Array(buffer, cur, t * 7);
      cur += t * 7 * 4;
      const intrinsics = new Float32Array(buffer, cur, t * 4);
      scene.cameras = { poses, intrinsics };
    }
    // Unknown kind — skip (forward compatibility, spec/05 §1).
  }

  return scene;
}
