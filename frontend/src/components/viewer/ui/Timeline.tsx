"use client";

import { useViewerStore } from "@/lib/state/viewerStore";

// Timeline scrubber — PLAIN DOM, talks ONLY to the Zustand store (spec/07 §1
// render-path decoupling rule): it imports NEITHER three NOR any Path-1
// component, only the store. A <input type="range"> bound to `time` (0..1);
// dragging it writes `setTime`, the same single-field transient write the R3F
// loop performs while playing (spec/07 §4.2).
//
// Narrow selector (spec/07 §4.2 transient-update discipline): this component
// subscribes to `time` ALONE, so a scrub (or a playing loop's per-frame
// `setTime`) re-renders ONLY the scrubber thumb — not the rest of the HUD.
// `setTime` is read once (it is a stable action reference in Zustand).

// 1000 steps across [0,1] → the thumb tracks `frameIndex` finely enough for any
// clip the MVP caps allow, while staying a cheap native range input.
const STEPS = 1000;

export function Timeline() {
  const time = useViewerStore((s) => s.time);
  const setTime = useViewerStore((s) => s.setTime);

  return (
    <label className="flex items-center gap-3 text-xs text-white/70">
      <span className="tabular-nums select-none">{time.toFixed(2)}</span>
      <input
        type="range"
        aria-label="Timeline"
        min={0}
        max={1}
        step={1 / STEPS}
        value={time}
        onChange={(e) => setTime(Number(e.target.value))}
        className="h-1 flex-1 cursor-pointer accent-white"
      />
    </label>
  );
}
