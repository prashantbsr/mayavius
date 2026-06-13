"use client";

import { useFrame, useThree } from "@react-three/fiber";
import { useViewerStore } from "@/lib/state/viewerStore";
import { timeToFrame } from "@/lib/viewer/buildScene";
import { DEFAULT_FPS_FALLBACK } from "@/config";
// Pulls in the `Window.__mayaviusDebug` global augmentation (spec/07 §4.4).
import "@/types/debug";

// PlaybackDriver — rendered INSIDE <Canvas> (it needs the R3F render loop). It
// owns two jobs, both per-frame and BOTH outside React renders (spec/07 §5):
//
//   1. Advance `time` while playing (spec/07 §5 loop): read the store via
//      getState() (NO subscription → no per-frame re-render), advance `time` by
//      `delta*fps/max(T-1,1)`, wrap if `loop` else pause at 1. The early-return
//      on !isPlaying || frozen || !scene is what makes bullet-time freeze time
//      (frozen===true halts advancement; the camera still free-orbits).
//
//   2. Write the test-observability surface `window.__mayaviusDebug` EVERY frame
//      (spec/07 §4.4). This is the ONE canonical surface Playwright reads
//      (T-401/T-403/T-405) — EXACT field names, always-on (no build flag), so the
//      e2e build and the shipped build expose the same contract. We write it
//      every frame (not only while playing) so the camera quaternion stays live
//      during bullet-time orbit while `frameIndex` holds constant (T-405).
//
// Render-path note: this component never returns DOM/geometry — it is a pure
// driver. The HUD reads the same store; it does NOT read __mayaviusDebug.

export function PlaybackDriver() {
  const camera = useThree((state) => state.camera);

  useFrame((_, delta) => {
    const s = useViewerStore.getState();
    const scene = s.scene;

    // ── 1. Advance time (only when playing, not frozen, scene loaded). ──────────
    if (s.isPlaying && !s.frozen && scene) {
      const T = s.frameCount;
      const fps = scene.fps > 0 ? scene.fps : DEFAULT_FPS_FALLBACK;
      const dt = (delta * fps) / Math.max(T - 1, 1); // normalized advance / sec
      let next = s.time + dt;
      if (next >= 1) {
        if (s.loop) {
          next = next % 1;
        } else {
          next = 1;
          s.pause();
        }
      }
      s.setTime(next); // cheap single-field transient write (spec/07 §4.2)
    }

    // ── 2. Publish the debug surface (EXACT field names, spec/07 §4.4). ─────────
    // Re-read time: the advance above may have changed it this same frame.
    const time = useViewerStore.getState().time;
    const frameCount = s.frameCount;
    const t = timeToFrame(time, frameCount);

    const staticPointCount = scene?.static?.count ?? 0;
    const dynamicFrame = scene?.dynamic?.frames[t];
