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
