import { create } from "zustand";
import type { Mv4dScene } from "@/types";

// Playback + interaction state for the viewer.
//
// State architecture decision (handover §6): Zustand. The R3F render loop reads
// and writes this store outside React's render cycle (no re-render per frame),
// it has minimal boilerplate, and transient high-frequency updates (the
// timeline scrubber) stay cheap. Rationale recorded in the root CLAUDE.md.
//
// Render-path decoupling (spec/07 §1 rule / §4): this store is the ONLY surface
// the DOM HUD and the R3F loop share. The HUD never imports three or Path-1
// components; the loop reads/writes via `useViewerStore.getState()/setState()`
// outside React renders. UI selectors subscribe narrowly (e.g. only `time`) so
// scrubbing/playback does not re-render the rest of the HUD (D6, spec/07 §4.2).

/** Job lifecycle + decode phase that drives `ProgressOverlay` (spec/07 §4.1).
 * Mirrors the backend job lifecycle (spec/06) plus the client-side decode step:
 * `submitting` (POST /jobs) → `processing` (poll/SSE) → `loading` (decode) →
 * `ready`; `error` on any failure; `idle` before a load starts. */
export type LoadState =
  | "idle"
  | "submitting"
  | "processing"
  | "loading"
  | "ready"
  | "error";

/** Camera behaviour (spec/07 §5). Drives `OrbitControls` + camera, NOT the
 * render path. `frozen===true` ⇔ `cameraMode==='bulletTime'` (kept consistent
 * by the actions below; `frozen` retained for back-compat with the scaffold). */
export type CameraMode = "orbit" | "asShot" | "bulletTime";

type ViewerState = {
  // ── Kept (scaffold) — DO NOT rename/remove; the R3F loop already uses these.
  /** Current playback time, normalized 0..1 across the clip. */
  time: number;
  isPlaying: boolean;
  loop: boolean;
  /** Bullet-time: freeze playback and free-orbit the frozen frame. */
  frozen: boolean;

  // ── Added (spec/07 §4.1).
  /** Decoded result; `null` until loaded. */
  scene: Mv4dScene | null;
  /** Drives `ProgressOverlay`; mirrors job lifecycle (spec/06) + decode. */
  loadState: LoadState;
  /** 0..1 backend job progress (from poll/SSE). */
  progress: number;
  /** User-facing message (decode or job failure). */
  error: string | null;
  /** Camera behaviour (spec/07 §5). */
  cameraMode: CameraMode;
  /** Cached `scene.frameCount` (T) so the loop avoids re-reading `scene` each tick. */
  frameCount: number;
  /** Active weights-license label surfaced from job metadata
   * (`AdapterInfo.weights_license`, e.g. "VGGT-1B · CC-BY-NC-4.0" — D2/spec/08
   * §7). `ProgressOverlay` renders it (spec/07 §6 step 2); `null` until the
   * loader sees a poll/SSE payload carrying it. */
  weightsLicense: string | null;

  // ── Actions (kept).
  setTime: (t: number) => void;
  play: () => void;
  pause: () => void;
  toggleLoop: () => void;
  setFrozen: (frozen: boolean) => void;

  // ── Actions (added, spec/07 §4.2).
  setScene: (scene: Mv4dScene) => void;
  setLoadState: (s: LoadState) => void;
  setProgress: (p: number) => void;
  setError: (msg: string | null) => void;
  setCameraMode: (mode: CameraMode) => void;
  /** Surface the active weights-license label from job metadata (spec/07 §6
   * step 2 / §4.4 sibling). */
  setWeightsLicense: (label: string | null) => void;
  /** Bullet-time enter: pause + freeze + `cameraMode='bulletTime'` (spec/07 §5). */
  enterBulletTime: () => void;
  /** Bullet-time exit: unfreeze + `cameraMode='orbit'`. */
  exitBulletTime: () => void;
};

export const useViewerStore = create<ViewerState>((set) => ({
  // Defaults (spec/07 §4.1): scaffold fields keep their scaffold defaults.
  time: 0,
  isPlaying: false,
  loop: true,
  frozen: false,
  scene: null,
  loadState: "idle",
  progress: 0,
  error: null,
  cameraMode: "orbit",
  frameCount: 0,
  weightsLicense: null,

  // Clamp to [0,1] — the loop and the scrubber both write here, so guard once
  // (spec/07 §4.2: a deliberately cheap, single-field transient write).
  setTime: (t) => set({ time: t < 0 ? 0 : t > 1 ? 1 : t }),
  play: () => set({ isPlaying: true }),
  pause: () => set({ isPlaying: false }),
  toggleLoop: () => set((s) => ({ loop: !s.loop })),
  setFrozen: (frozen) => set({ frozen }),

  // setScene: cache frameCount and flip to ready in one write (spec/07 §4.2).
  setScene: (scene) =>
    set({ scene, frameCount: scene.frameCount, loadState: "ready" }),
  setLoadState: (loadState) => set({ loadState }),
  setProgress: (progress) => set({ progress }),
  setError: (error) => set({ error }),
  setCameraMode: (cameraMode) => set({ cameraMode }),
  setWeightsLicense: (weightsLicense) => set({ weightsLicense }),
  // enterBulletTime ≡ pause()+setFrozen(true)+cameraMode='bulletTime' (§5),
  // applied as a single set so the three stay consistent.
  enterBulletTime: () =>
    set({ isPlaying: false, frozen: true, cameraMode: "bulletTime" }),
  // exitBulletTime ≡ setFrozen(false)+cameraMode='orbit' (§5).
  exitBulletTime: () => set({ frozen: false, cameraMode: "orbit" }),
}));
