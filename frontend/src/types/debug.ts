// Test-observability contract (spec/07 §4.4): the ONE canonical surface the
// Playwright e2e suite reads via `page.evaluate(() => window.__mayaviusDebug)`.
// <PlaybackDriver> writes it from the R3F useFrame loop EVERY frame. The field
// names below are load-bearing — spec/10 T-401/T-403/T-405 read them verbatim:
//
//   T-401  staticPointCount > 0 once a scene is loaded (cloud present)
//   T-403  frameIndex changes when the timeline is scrubbed
//   T-405  cameraQuaternion changes while frameIndex stays constant (bullet-time)
//
// This is the ONLY debug surface (no ad-hoc data-testids for these values), and
// it is always-on with no build flag, so the e2e build and the shipped build
// expose the identical contract.

export interface MayaviusDebug {
  /** Points in the static layer (0 until a scene loads). */
  staticPointCount: number;
  /** Points in the current dynamic frame t. */
  dynamicPointCount: number;
  /** Current t = round(time*(frameCount-1)). */
  frameIndex: number;
  /** Camera world quaternion x,y,z,w. */
  cameraQuaternion: [number, number, number, number];
}

declare global {
  interface Window {
    __mayaviusDebug?: MayaviusDebug;
  }
}

