import { create } from "zustand";
import type { Mv4dScene } from "@/types";

// Playback + interaction state for the viewer.
//
// State architecture decision (handover §6): Zustand. The R3F render loop reads
// and writes this store outside React's render cycle (no re-render per frame),
// it has minimal boilerplate, and transient high-frequency updates (the
// timeline scrubber) stay cheap. Rationale recorded in the root CLAUDE.md.
type ViewerState = {
  /** Current playback time, normalized 0..1 across the clip. */
  time: number;
  isPlaying: boolean;
  loop: boolean;
  /** Bullet-time: freeze playback and free-orbit the frozen frame. */
  frozen: boolean;
  setTime: (t: number) => void;
  play: () => void;
  pause: () => void;
  toggleLoop: () => void;
  setFrozen: (frozen: boolean) => void;
};

export const useViewerStore = create<ViewerState>((set) => ({
  time: 0,
  isPlaying: false,
  loop: true,
  frozen: false,
  setTime: (time) => set({ time }),
  play: () => set({ isPlaying: true }),
  pause: () => set({ isPlaying: false }),
  toggleLoop: () => set((s) => ({ loop: !s.loop })),
  setFrozen: (frozen) => set({ frozen }),
}));
