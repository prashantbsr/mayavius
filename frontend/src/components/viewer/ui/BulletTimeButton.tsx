"use client";

import { useViewerStore } from "@/lib/state/viewerStore";

// BulletTimeButton — PLAIN DOM, talks ONLY to the Zustand store (spec/07 §1
// render-path decoupling rule): NO three / Path-1 import. The signature
// interaction (spec/07 §5): toggle between `enterBulletTime()` (pause + freeze +
// cameraMode='bulletTime') and `exitBulletTime()` (unfreeze + cameraMode='orbit').
// OrbitControls stay enabled in every mode — bullet-time only STOPS time, so the
// user free-orbits the frozen frame (e2e T-405).
//
// Narrow selector (spec/07 §4.2): subscribe to `frozen` only. `frozen===true` ⇔
// cameraMode==='bulletTime' (kept consistent by the store actions, spec/07 §4.1).

export function BulletTimeButton() {
  const frozen = useViewerStore((s) => s.frozen);
  const enterBulletTime = useViewerStore((s) => s.enterBulletTime);
  const exitBulletTime = useViewerStore((s) => s.exitBulletTime);

  return (
    <button
      type="button"
      aria-label="Bullet time"
      aria-pressed={frozen}
      onClick={() => (frozen ? exitBulletTime() : enterBulletTime())}
      className={`rounded px-3 py-1 text-xs font-medium text-white hover:bg-white/20 ${
        frozen ? "bg-amber-500/40" : "bg-white/10"
      }`}
    >
      {frozen ? "Exit bullet-time" : "Bullet-time"}
    </button>
  );
}
