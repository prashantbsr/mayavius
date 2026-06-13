"use client";

import { useViewerStore } from "@/lib/state/viewerStore";

// PlaybackControls — PLAIN DOM, talks ONLY to the Zustand store (spec/07 §1
// render-path decoupling rule): NO three / Path-1 import. A play/pause toggle
// and a loop toggle, each a single store action; the R3F <PlaybackDriver> loop
// reads `isPlaying`/`loop` via getState() and advances `time` (spec/07 §5).
//
// Narrow selectors (spec/07 §4.2): subscribe to `isPlaying` and `loop` only, so
// the scrubber's per-frame `time` writes never re-render these buttons.

export function PlaybackControls() {
  const isPlaying = useViewerStore((s) => s.isPlaying);
  const loop = useViewerStore((s) => s.loop);
  const play = useViewerStore((s) => s.play);
  const pause = useViewerStore((s) => s.pause);
  const toggleLoop = useViewerStore((s) => s.toggleLoop);

  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        aria-label={isPlaying ? "Pause" : "Play"}
        aria-pressed={isPlaying}
        onClick={() => (isPlaying ? pause() : play())}
        className="rounded bg-white/10 px-3 py-1 text-xs font-medium text-white hover:bg-white/20"
      >
        {isPlaying ? "Pause" : "Play"}
      </button>
      <button
        type="button"
        aria-label="Loop"
        aria-pressed={loop}
        onClick={() => toggleLoop()}
        className={`rounded px-3 py-1 text-xs font-medium text-white hover:bg-white/20 ${
          loop ? "bg-white/25" : "bg-white/10"
        }`}
      >
        Loop
      </button>
    </div>
  );
}
