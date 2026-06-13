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

