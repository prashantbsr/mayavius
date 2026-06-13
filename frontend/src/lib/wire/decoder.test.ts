import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  MV4D_VERSION,
  Mv4dDecodeError,
  decodeReconstruction,
  dequantize,
} from "./decoder";

// Frontend decoder unit + conformance suite (spec/10 §1.3 + §2). The committed
// golden fixture is the SAME `backend/tests/fixtures/golden_scene.mv4d` the
// Python encoder produced (T-200) — Python-encoded → TS-decoded proven correct.
// We read it (and the shared expectation table) from disk via node `fs` with
// absolute paths (spec/10 / W0.T4 fixture wiring).

// Vitest's cwd is `frontend/`. Fixtures are read from disk with absolute paths
// (spec/10 / W0.T4 fixture wiring) — robust under the jsdom environment, which
// does not expose a file-scheme `import.meta.url`.
const FRONTEND_ROOT = process.cwd();
const GOLDEN_PATH = resolve(
  FRONTEND_ROOT,
  "../backend/tests/fixtures/golden_scene.mv4d",
);
const EXPECTED_PATH = resolve(
  FRONTEND_ROOT,
  "src/lib/wire/__fixtures__/golden_expected.json",
);

/** Read a file as a standalone `ArrayBuffer` (a fresh copy, so `.buffer`
 * identity checks for zero-copy views are meaningful — the views must point at
 * THIS buffer). */
function readArrayBuffer(path: string): ArrayBuffer {
  const buf = readFileSync(path);
  const ab = new ArrayBuffer(buf.byteLength);
  new Uint8Array(ab).set(buf);
  return ab;
}

interface GoldenExpected {
  version: number;
  frameCount: number;
  fps: number;
  aabbMin: [number, number, number];
  aabbMax: [number, number, number];
  posTolerance: [number, number, number];
  static: {
    count: number;
    positions: number[][];
    colors: number[][];
    conf: number[];
  };
  dynamic: { frames: { count: number; positions: number[][]; colors: number[][] }[] };
  tracks: {
    count: number;
    positions: number[][][];
    visibility: boolean[][];
    colors: number[][];
  };
  cameras: { poses: number[][]; intrinsics: number[][] };
}

const expected: GoldenExpected = JSON.parse(
  readFileSync(EXPECTED_PATH, "utf-8"),
);

describe("MV4D decoder — golden fixture (T-150)", () => {
  it("decodes header/AABB and exposes zero-copy static views", () => {
    const buffer = readArrayBuffer(GOLDEN_PATH);
    const scene = decodeReconstruction(buffer);

    expect(scene.version).toBe(1);
    expect(scene.frameCount).toBe(expected.frameCount);
    expect(scene.fps).toBeCloseTo(expected.fps, 5);
    for (let i = 0; i < 3; i++) {
      expect(scene.aabbMin[i]).toBeCloseTo(expected.aabbMin[i], 5);
      expect(scene.aabbMax[i]).toBeCloseTo(expected.aabbMax[i], 5);
    }

    expect(scene.static).toBeDefined();
    expect(scene.static!.count).toBe(4);
    // Zero-copy: positionsQ is a Uint16Array VIEW over the decoded buffer.
    expect(scene.static!.positionsQ).toBeInstanceOf(Uint16Array);
    expect(scene.static!.positionsQ.buffer).toBe(buffer);
    expect(scene.static!.positionsQ.length).toBe(4 * 3);
    expect(scene.static!.colors).toBeInstanceOf(Uint8Array);
    expect(scene.static!.colors.buffer).toBe(buffer);
  });
});

describe("MV4D decoder — dynamic slicing (T-151)", () => {
  it("slices each frame per frameDir incl. the empty frame", () => {
    const buffer = readArrayBuffer(GOLDEN_PATH);
    const scene = decodeReconstruction(buffer);

    expect(scene.dynamic).toBeDefined();
    const frames = scene.dynamic!.frames;
    expect(frames.length).toBe(expected.frameCount);

    expected.dynamic.frames.forEach((ef, t) => {
      const f = frames[t];
      expect(f.count).toBe(ef.count);
      expect(f.positionsQ.length).toBe(ef.count * 3);
      expect(f.colors.length).toBe(ef.count * 3);
      // sub-views still point at the SAME backing buffer (zero-copy).
      expect(f.positionsQ.buffer).toBe(buffer);
      // exact colors (u8, no quantization).
      ef.colors.forEach((c, k) => {
        expect(f.colors[k * 3 + 0]).toBe(c[0]);
        expect(f.colors[k * 3 + 1]).toBe(c[1]);
        expect(f.colors[k * 3 + 2]).toBe(c[2]);
      });
    });

    // The middle frame (t=1) is the empty frame.
    expect(frames[1].count).toBe(0);
    expect(frames[1].positionsQ.length).toBe(0);
  });
});

describe("MV4D decoder — tracks visibility (T-152)", () => {
  it("isVisible(m,t) matches the packed LSB-first bitmask; colors present", () => {
    const buffer = readArrayBuffer(GOLDEN_PATH);
    const scene = decodeReconstruction(buffer);

    expect(scene.tracks).toBeDefined();
    const tr = scene.tracks!;
    expect(tr.count).toBe(expected.tracks.count);

    expected.tracks.visibility.forEach((row, m) => {
      row.forEach((vis, t) => {
        expect(tr.isVisible(m, t)).toBe(vis);
      });
    });

    // HAS_TRACK_COLOR is set in the golden fixture → colors present + exact.
    expect(tr.colors).toBeDefined();
    expect(tr.colors!.buffer).toBe(buffer);
    expected.tracks.colors.forEach((c, m) => {
      expect(tr.colors![m * 3 + 0]).toBe(c[0]);
      expect(tr.colors![m * 3 + 1]).toBe(c[1]);
      expect(tr.colors![m * 3 + 2]).toBe(c[2]);
    });
  });
});

describe("MV4D decoder — error contract (T-153, T-154)", () => {
  it("bad magic throws typed Mv4dDecodeError (not a generic Error) — T-153", () => {
    const buffer = readArrayBuffer(GOLDEN_PATH);
    new Uint8Array(buffer)[0] = 0x00; // corrupt the magic
    let caught: unknown;
    try {
      decodeReconstruction(buffer);
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(Mv4dDecodeError);
    expect(() => decodeReconstruction(buffer)).toThrow(Mv4dDecodeError);
  });

  it("version=2 throws Mv4dDecodeError — T-154", () => {
    const buffer = readArrayBuffer(GOLDEN_PATH);
    new Uint8Array(buffer)[4] = 2; // version byte
    expect(() => decodeReconstruction(buffer)).toThrow(Mv4dDecodeError);
  });

  it("posBits != 16 throws Mv4dDecodeError — T-154", () => {
    const buffer = readArrayBuffer(GOLDEN_PATH);
    new Uint8Array(buffer)[6] = 8; // posBits byte
    expect(() => decodeReconstruction(buffer)).toThrow(Mv4dDecodeError);
  });

  it("section bounds overflow throws Mv4dDecodeError — T-154", () => {
    const buffer = readArrayBuffer(GOLDEN_PATH);
    const view = new DataView(buffer);
    // First directory entry's byteLength (dir@48, +8) → absurdly large.
    view.setUint32(48 + 8, 0xffffffff, true);
    expect(() => decodeReconstruction(buffer)).toThrow(Mv4dDecodeError);
  });

  it("misaligned section offset throws Mv4dDecodeError — T-154", () => {
    const buffer = readArrayBuffer(GOLDEN_PATH);
    const view = new DataView(buffer);
    // First directory entry's byteOffset (dir@48, +4) → not 8-aligned (and in
    // bounds so the overflow check passes first).
    view.setUint32(48 + 4, 113, true); // 113 % 8 != 0, < buffer length
    view.setUint32(48 + 8, 4, true); // small length so it stays in bounds
    expect(() => decodeReconstruction(buffer)).toThrow(Mv4dDecodeError);
  });
});

describe("MV4D decoder — version constant (T-155)", () => {
  it("exports MV4D_VERSION === 1", () => {
    expect(MV4D_VERSION).toBe(1);
  });
});

describe("MV4D dequantize helper (T-160)", () => {
  it("matches the encoder inverse p = min + q/65535*(max-min)", () => {
    expect(dequantize(0, 0, 1)).toBeCloseTo(0, 9);
    expect(dequantize(65535, 0, 1)).toBeCloseTo(1, 9);
    expect(dequantize(32768, 0, 1)).toBeCloseTo(32768 / 65535, 9);
    // arbitrary AABB axis
    const min = -2.5;
    const max = 4.0;
    for (const q of [0, 1, 1000, 30000, 65535]) {
      expect(dequantize(q, min, max)).toBeCloseTo(
        min + (q / 65535) * (max - min),
        9,
      );
    }
    // degenerate axis (max == min) → always min (q must be 0 there).
    expect(dequantize(0, 7.0, 7.0)).toBe(7.0);
  });
});

describe("MV4D decoder — golden conformance (T-202)", () => {
  it("every decoded value equals golden_expected.json", () => {
    const buffer = readArrayBuffer(GOLDEN_PATH);
    const scene = decodeReconstruction(buffer);

    // header / AABB — exact (f32 round-trip).
    expect(scene.version).toBe(expected.version);
    expect(scene.frameCount).toBe(expected.frameCount);
    expect(scene.fps).toBeCloseTo(expected.fps, 5);
    for (let i = 0; i < 3; i++) {
      expect(scene.aabbMin[i]).toBeCloseTo(expected.aabbMin[i], 5);
      expect(scene.aabbMax[i]).toBeCloseTo(expected.aabbMax[i], 5);
    }

    const deq = (q: number, axis: number): number =>
      dequantize(q, expected.aabbMin[axis], expected.aabbMax[axis]);

    // ---- static ----
    expect(scene.static!.count).toBe(expected.static.count);
    expected.static.positions.forEach((p, n) => {
      for (let a = 0; a < 3; a++) {
        const got = deq(scene.static!.positionsQ[n * 3 + a], a);
        expect(Math.abs(got - p[a])).toBeLessThanOrEqual(
          expected.posTolerance[a] + 1e-9,
