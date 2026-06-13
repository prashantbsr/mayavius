"use client";

import { useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { Scene } from "./Scene";
import { PlaybackDriver } from "./PlaybackDriver";
import { CameraRig } from "./CameraRig";
import { ViewerOverlay } from "./ViewerOverlay";
import { useLoadScene } from "@/lib/viewer/useLoadScene";
import { useTestObservability } from "@/lib/state/testObservability";

// Host for the Three.js scene (spec/07 §1, §5, §6). Inside <Canvas>: the Path-1
// <Scene> layers, the <PlaybackDriver> (R3F loop: advances time + writes the
// __mayaviusDebug surface, §4.4/§5), the <CameraRig> (pinned initial fit + camera
// modes, §5), and <OrbitControls makeDefault> (enabled in every mode — §5). The
// DOM HUD <ViewerOverlay> is a SIBLING of the canvas (NOT in WebGL): it talks to
// the viewer ONLY through the Zustand store (the §1 render-path decoupling rule),
// so this file is the only place the WebGL and DOM halves meet — and they meet
// through the store, not through props.
//
// On-mount loader (spec/07 §6): `useLoadScene(resultId)` runs the SINGLE unified
// load path (SSE + poll fallback) for both seeded examples and live jobs — it
// never branches on EXAMPLE_SLUGS (server-only). The ssr:false boundary stays in
// ViewerClient; this file is already client-only.
export function ViewerCanvas({ resultId }: { resultId: string }) {
  // OrbitControls instance, shared with CameraRig so it can re-target on the
  // initial fit and toggle `enabled` for the asShot camera mode (spec/07 §5).
  const controlsRef = useRef<OrbitControlsImpl | null>(null);

  // Always-on test-observability store surface (spec/10 §4 e2e glue) — mounts
  // with the viewer so Playwright can observe progress/time/isPlaying/loop OVER
  // THE RUN (T-402/T-403/T-404). Separate from the pinned __mayaviusDebug render
  // surface (spec/07 §4.4); changes no app behaviour. Installed BEFORE the loader
  // so its store subscription (which records the progress history) is active
  // before the first progress write — making T-402's intermediate-progress
  // observation deterministic, not racy.
  useTestObservability();

  // Kick off the one unified load path for this /view/[id] (spec/07 §6).
  useLoadScene(resultId);

  return (
    <div className="relative flex-1">
      <Canvas camera={{ position: [0, 0, 4], fov: 50 }} dpr={[1, 2]}>
        <color attach="background" args={["#0a0a0a"]} />
        <ambientLight intensity={0.6} />
        <Scene />
        <PlaybackDriver />
        <CameraRig controlsRef={controlsRef} />
        <OrbitControls ref={controlsRef} makeDefault />
      </Canvas>
      <ViewerOverlay />
    </div>
  );
}
