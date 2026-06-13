import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { Mv4dScene } from "@/types";
import { useViewerStore } from "./viewerStore";

// Zustand viewerStore unit suite (spec/07 §4 / §4.2; T-170 actions, T-171
// transient-update discipline). The store is a module singleton, so we snapshot
// its initial state once and reset it between tests to keep cases independent.
//
// We drive the store through `getState()` exactly the way the R3F loop and the
// DOM HUD do (outside React renders, spec/07 §1 / §4.2) — no component mount is
// needed to exercise the action surface.

const INITIAL_STATE = useViewerStore.getState();

beforeEach(() => {
  // Replace (not merge) so every field returns to its scaffold/spec default.
  useViewerStore.setState(INITIAL_STATE, true);
});

afterEach(() => {
  useViewerStore.setState(INITIAL_STATE, true);
});

/** Minimal hand-authored `Mv4dScene` — only the fields `setScene` reads
 * (`frameCount`) matter here; the rest satisfy the type (spec/05 §5.2). */
function makeScene(frameCount: number): Mv4dScene {
  return {
    version: 1,
    frameCount,
    fps: 24,
    aabbMin: [0, 0, 0],
    aabbMax: [1, 1, 1],
  };
}

describe("viewerStore defaults", () => {
  it("starts at the spec defaults", () => {
    const s = useViewerStore.getState();
    expect(s.time).toBe(0);
    expect(s.isPlaying).toBe(false);
    expect(s.loop).toBe(true);
    expect(s.frozen).toBe(false);
    // Added state (spec/07 §4.1).
    expect(s.scene).toBeNull();
    expect(s.loadState).toBe("idle");
    expect(s.progress).toBe(0);
    expect(s.error).toBeNull();
    expect(s.cameraMode).toBe("orbit");
    expect(s.frameCount).toBe(0);
  });
});

// ── T-170: actions ────────────────────────────────────────────────────────────
describe("viewerStore actions (T-170)", () => {
  it("play / pause set isPlaying", () => {
    useViewerStore.getState().play();
    expect(useViewerStore.getState().isPlaying).toBe(true);
    useViewerStore.getState().pause();
    expect(useViewerStore.getState().isPlaying).toBe(false);
  });
