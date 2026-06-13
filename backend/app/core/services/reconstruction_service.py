"""Pure orchestration over the ReconstructionPort + model-agnostic post-processing.

No FastAPI, no torch, no concrete-adapter imports — the hexagonal boundary
(handover §3, §6 / T-130). The adapter is injected; tunables arrive as PLAIN
float/int constructor args (the worker/deps reads them from Settings) — core never
imports ``app.config``.

`ReconstructionService.run` stays pure orchestration: validate → adapter
(already-split Scene4D) → smooth/cull → caps → emptiness guard → stamp provenance.
The two helpers (`smooth_and_cull`, `enforce_caps`) are pure-numpy operations on
the canonical `Scene4D` fields (spec/06 §5 steps 6-7) — testable on CI without ML.

NOTE (logged): these helpers are pulled forward from W1.T1 because the W0 gate's
T-120 requires them; the rigorous over-cap test T-104 stays in W1. They do NOT
import ``app.config`` — conf_thresh / window are plain args.
"""

from __future__ import annotations

import logging

import numpy as np

from app.core.domain.errors import EmptyReconstructionError
from app.core.domain.models import CameraTrack, Scene4D, Tracks
from app.core.ports.reconstruction_port import ProgressSink, ReconstructionPort

logger = logging.getLogger(__name__)

# MV4D caps — the contract constants (spec/05 §4). Not tunables.
_MAX_STATIC = 150_000
_MAX_DYNAMIC_PER_FRAME = 20_000
_MAX_TRACKS = 4_096
_MAX_FRAMES = 64

# Deterministic subsample seed (spec/06 §5 step 7).
_DYNAMIC_SUBSAMPLE_SEED = 0


class ReconstructionService:
    """Validate the request, invoke the injected adapter, post-process the scene."""

    def __init__(
        self,
        adapter: ReconstructionPort,
        *,
        conf_thresh: float = 0.5,
        smooth_window: int = 3,
    ) -> None:
        self._adapter = adapter          # tunables are PLAIN floats — core never imports app.config.
        self._conf_thresh = conf_thresh  # the worker/deps builds the service from Settings (spec/06 §6)
        self._smooth_window = smooth_window

    def run(
        self,
        request,
        progress: ProgressSink | None = None,
    ) -> Scene4D:
        request.validate()                                    # caps (raises ClipTooLongError)
        scene = self._adapter.reconstruct(request, progress)  # ALREADY-split Scene4D (split is adapter-side, §4.6)
        scene = smooth_and_cull(
            scene, conf_thresh=self._conf_thresh, window=self._smooth_window
        )                                                     # step 6
        scene = enforce_caps(scene)                           # step 7 (MV4D caps are constants, spec/05 §4)
        if scene.static_positions.size == 0 and not scene.tracks:
            raise EmptyReconstructionError("no usable points after culling")
        scene.adapter_id = self._adapter.info.name
        scene.weights_license = self._adapter.info.weights_license
        return scene


def smooth_and_cull(scene: Scene4D, *, conf_thresh: float, window: int = 3) -> Scene4D:
    """Step 6 — temporal smoothing of TRACK positions + confidence cull of statics.

    Smoothing: a centered moving-average (default window 3) over each track's
    positions to kill jitter. The window shrinks at the t=0 / t=T-1 edges; invisible
    samples are skipped. If no visible sample falls within the shrunk window for an
    ``(m, t)``, that sample keeps its OWN position (no smoothing) and its visibility
    bit is left untouched.

    Confidence cull: drop static points whose ``static_conf/255 < conf_thresh``.
    Skipped entirely when ``static_conf is None``.

    Returns a new ``Scene4D`` (input is not mutated).
    """
    # --- temporal smoothing of track positions ---
    tracks = scene.tracks
    if tracks is not None:
        positions = np.asarray(tracks.positions, dtype=np.float32)  # (M, T, 3)
        visibility = np.asarray(tracks.visibility, dtype=bool)      # (M, T)
        smoothed = positions.copy()
        M, T = visibility.shape[0], visibility.shape[1]
        half = max(int(window), 1) // 2
        for m in range(M):
            for t in range(T):
                lo = max(0, t - half)
                hi = min(T, t + half + 1)
                vis_slice = visibility[m, lo:hi]
                if vis_slice.any():
                    win = positions[m, lo:hi][vis_slice]
                    smoothed[m, t] = win.mean(axis=0)
                else:
                    # No visible sample in the shrunk window → keep own position,
                    # leave visibility untouched.
                    smoothed[m, t] = positions[m, t]
        tracks = Tracks(
            positions=smoothed,
            visibility=visibility.copy(),
            colors=None if tracks.colors is None else np.asarray(tracks.colors).copy(),
        )

    # --- confidence cull of static points ---
    static_positions = np.asarray(scene.static_positions, dtype=np.float32)
    static_colors = np.asarray(scene.static_colors)
    static_conf = scene.static_conf
    if static_conf is not None:
        static_conf = np.asarray(static_conf)
        keep = (static_conf.astype(np.float32) / 255.0) >= conf_thresh
        static_positions = static_positions[keep]
        static_colors = static_colors[keep]
        static_conf = static_conf[keep]

    return Scene4D(
        frame_count=scene.frame_count,
        fps=scene.fps,
        aabb_min=np.asarray(scene.aabb_min, dtype=np.float32).copy(),
        aabb_max=np.asarray(scene.aabb_max, dtype=np.float32).copy(),
        static_positions=static_positions,
        static_colors=static_colors,
        static_conf=None if static_conf is None else static_conf,
        dynamic_positions=[np.asarray(p, dtype=np.float32) for p in scene.dynamic_positions],
        dynamic_colors=[np.asarray(c) for c in scene.dynamic_colors],
        tracks=tracks,
        cameras=scene.cameras,
        adapter_id=scene.adapter_id,
        weights_license=scene.weights_license,
    )


def enforce_caps(scene: Scene4D) -> Scene4D:
    """Step 7 — enforce the MV4D caps (spec/05 §4). No-op on a within-cap scene.

    - static -> 150 000 (drop lowest ``static_conf`` first; if conf is None keep the first N)
    - dynamic per-frame -> 20 000 (deterministic fixed-seed uniform random subsample)
    - tracks -> 4 096 (drop lowest mean-visibility first)
    - frames T -> 64 (uniform temporal subsample)
    """
    # --- frames T -> 64 (uniform temporal subsample). Do this first so the per-frame
    #     dynamic / track / camera arrays are all sliced on the SAME frame indices. ---
    frame_count = scene.frame_count
    fps = scene.fps
    dynamic_positions = list(scene.dynamic_positions)
    dynamic_colors = list(scene.dynamic_colors)
    tracks = scene.tracks
    cameras = scene.cameras

    if frame_count > _MAX_FRAMES:
        frame_idx = np.linspace(0, frame_count - 1, _MAX_FRAMES).round().astype(np.int64)
        frame_idx = np.unique(frame_idx)
        dynamic_positions = [dynamic_positions[i] for i in frame_idx]
        dynamic_colors = [dynamic_colors[i] for i in frame_idx]
        if tracks is not None:
            tracks = Tracks(
                positions=np.asarray(tracks.positions)[:, frame_idx, :],
                visibility=np.asarray(tracks.visibility)[:, frame_idx],
                colors=tracks.colors,
            )
        if cameras is not None:
            cameras = CameraTrack(
                poses=np.asarray(cameras.poses)[frame_idx],
                intrinsics=np.asarray(cameras.intrinsics)[frame_idx],
            )
        frame_count = int(len(frame_idx))

    # --- static -> 150 000 (drop lowest static_conf first; conf None -> keep first N) ---
    static_positions = np.asarray(scene.static_positions, dtype=np.float32)
    static_colors = np.asarray(scene.static_colors)
    static_conf = scene.static_conf
    if static_conf is not None:
        static_conf = np.asarray(static_conf)
    n_static = static_positions.shape[0]
    if n_static > _MAX_STATIC:
        if static_conf is not None:
            # Highest confidence kept: take the top-_MAX_STATIC by conf, then restore order.
            order = np.argsort(static_conf, kind="stable")  # ascending
            keep_idx = np.sort(order[-_MAX_STATIC:])
            static_positions = static_positions[keep_idx]
            static_colors = static_colors[keep_idx]
            static_conf = static_conf[keep_idx]
        else:
            static_positions = static_positions[:_MAX_STATIC]
            static_colors = static_colors[:_MAX_STATIC]

    # --- dynamic per-frame -> 20 000 (deterministic fixed-seed uniform random subsample) ---
    rng = np.random.default_rng(_DYNAMIC_SUBSAMPLE_SEED)
    capped_dyn_pos: list[np.ndarray] = []
    capped_dyn_col: list[np.ndarray] = []
    for pos, col in zip(dynamic_positions, dynamic_colors):
        pos = np.asarray(pos, dtype=np.float32)
        col = np.asarray(col)
        n = pos.shape[0]
        if n > _MAX_DYNAMIC_PER_FRAME:
            sel = rng.choice(n, size=_MAX_DYNAMIC_PER_FRAME, replace=False)
            sel.sort()  # deterministic, preserves original ordering
            pos = pos[sel]
            col = col[sel]
        capped_dyn_pos.append(pos)
        capped_dyn_col.append(col)

    # --- tracks -> 4 096 (drop lowest mean-visibility first) ---
    if tracks is not None:
        positions = np.asarray(tracks.positions)
        visibility = np.asarray(tracks.visibility, dtype=bool)
        colors = tracks.colors
        M = positions.shape[0]
        if M > _MAX_TRACKS:
            mean_vis = visibility.mean(axis=1)
            order = np.argsort(mean_vis, kind="stable")  # ascending mean-visibility
            keep_idx = np.sort(order[-_MAX_TRACKS:])
            positions = positions[keep_idx]
            visibility = visibility[keep_idx]
            if colors is not None:
                colors = np.asarray(colors)[keep_idx]
            tracks = Tracks(positions=positions, visibility=visibility, colors=colors)

    result = Scene4D(
        frame_count=frame_count,
        fps=fps,
        aabb_min=np.asarray(scene.aabb_min, dtype=np.float32),
        aabb_max=np.asarray(scene.aabb_max, dtype=np.float32),
        static_positions=static_positions,
        static_colors=static_colors,
        static_conf=static_conf,
        dynamic_positions=capped_dyn_pos,
        dynamic_colors=capped_dyn_col,
        tracks=tracks,
        cameras=cameras,
        adapter_id=scene.adapter_id,
        weights_license=scene.weights_license,
    )

    n_dyn = sum(int(p.shape[0]) for p in result.dynamic_positions)
    n_trk = 0 if result.tracks is None else int(np.asarray(result.tracks.positions).shape[0])
    logger.info(
        "enforce_caps: T=%d static=%d dynamic_total=%d tracks=%d",
        result.frame_count,
        int(result.static_positions.shape[0]),
        n_dyn,
        n_trk,
    )
    return result
