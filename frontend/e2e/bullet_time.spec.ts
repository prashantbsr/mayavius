import { test, expect } from "@playwright/test";
import {
  openLoadedViewer,
  storeField,
  debugNumber,
  cameraQuaternion,
} from "./helpers";

// T-405 — bullet_time.orbit (spec/10 §4).
//
// Enter bullet-time (freeze) → frozen=true, playback halts; orbit drag rotates
// the camera around the frozen frame — assert window.__mayaviusDebug.
// cameraQuaternion changes while window.__mayaviusDebug.frameIndex is constant
// (spec/07 §4.4/§5). OrbitControls stay enabled in bullet-time; it only STOPS
// time. We assert: frozen flips true + isPlaying false; the quaternion moves
// under an orbit drag; frameIndex is unchanged across the whole drag.

function quatDelta(
  a: [number, number, number, number],
  b: [number, number, number, number],
): number {
  return Math.max(...a.map((v, i) => Math.abs(v - b[i])));
}

test.describe("T-405 bullet_time.orbit", () => {
  test("bullet-time freezes time and an orbit drag rotates the camera", async ({
    page,
  }) => {
    await openLoadedViewer(page, "example");

