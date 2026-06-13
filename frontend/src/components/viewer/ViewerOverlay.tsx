"use client";

import { Timeline } from "./ui/Timeline";
import { PlaybackControls } from "./ui/PlaybackControls";
import { BulletTimeButton } from "./ui/BulletTimeButton";
import { ProgressOverlay } from "./ui/ProgressOverlay";

// ViewerOverlay — the DOM HUD positioned OVER the WebGL canvas (spec/07 §1
// component tree: a DOM layer, NOT in WebGL). It imports NEITHER three NOR any
// Path-1 component (the §1 render-path decoupling rule): every child talks to
// the viewer ONLY through the Zustand store. This is what lets a Path-2
// <SplatMesh> mount at the Scene.tsx seam without touching a single control.
//
// Layout: the bottom control bar (timeline + playback + bullet-time) plus the
// centered ProgressOverlay that covers the canvas until the scene is `ready`.
// The container is pointer-events:none so orbit/scrub on the canvas isn't
// blocked; only the interactive widgets re-enable pointer events.
export function ViewerOverlay() {
  return (
    <div className="pointer-events-none absolute inset-0">
      {/* Centered load/progress card (clears itself once loadState==='ready'). */}
      <ProgressOverlay />

      {/* Bottom control bar. */}
      <div className="pointer-events-auto absolute inset-x-0 bottom-0 flex flex-col gap-2 bg-gradient-to-t from-black/70 to-transparent p-4">
        <Timeline />
        <div className="flex items-center justify-between">
          <PlaybackControls />
          <BulletTimeButton />
        </div>
      </div>
    </div>
  );
}
