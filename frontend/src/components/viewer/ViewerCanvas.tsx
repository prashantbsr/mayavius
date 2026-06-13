"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Scene } from "./Scene";

// Host for the Three.js scene. The timeline scrubber, play/pause/loop and
// bullet-time freeze controls (spec/07-frontend-spec.md) overlay this canvas
// and drive playback via the Zustand store in lib/state/viewerStore.ts.
export function ViewerCanvas({ resultId }: { resultId: string }) {
  return (
    <div className="relative flex-1">
      <Canvas camera={{ position: [0, 0, 4], fov: 50 }} dpr={[1, 2]}>
        <color attach="background" args={["#0a0a0a"]} />
        <ambientLight intensity={0.6} />
        <Scene />
        <OrbitControls makeDefault />
      </Canvas>
      <div className="pointer-events-none absolute left-3 top-3 text-xs opacity-50">
        viewer scaffold · result: {resultId}
      </div>
    </div>
  );
}
