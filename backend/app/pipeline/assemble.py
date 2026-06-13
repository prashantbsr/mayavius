"""Static/dynamic split + assemble a RAW ``Scene4D`` (spec/06 §5 step 5).

This is the **adapter-side** assembly (spec/06 §4.6 ownership): it consumes the raw
per-frame VGGT world-point maps in ``GeometryResult`` — which therefore NEVER enter
``Scene4D`` and never cross the port — and produces an already-split ``Scene4D``:

  - A frame-``t`` VGGT point is **dynamic** if it lies within radius ``r`` (default
    2% of the AABB diagonal) of ANY CoTracker track sample whose inter-frame
    displacement exceeds the **motion threshold** (95th-pct of inter-frame track
    motion, with an absolute floor of 1% of the AABB diagonal).
  - ``dynamic_positions[t]`` = the moving subset of frame ``t``'s VGGT points (a
    dense colored cluster) + ``dynamic_colors[t]``.
  - ``static_positions`` = the low-motion union across frames, **deduped by a
    voxel-grid downsample** (voxel = 0.5% AABB diag; keep the highest-conf point +
    its color per voxel — NO averaging).
  - ``tracks`` from the ``TrackResult``; ``cameras`` from ``geo.camera``; the AABB
    spans static ∪ dynamic ∪ tracks.
  - ``static_conf`` from VGGT ``world_points_conf`` as
    ``clip(round(per-scene min-max-normalized conf * 255), 0, 255)`` (u8).

Spatial query = **numpy brute-force** chunked ``(N, M)`` distances — no scipy.
**Fallback (logged):** if per-frame VGGT points are too noisy / absent, set
``dynamic_positions[t]`` = the lifted MOVING track points only (sparse).

``assemble_scene4d`` returns a **RAW** ``Scene4D`` — NO smoothing/culling/caps
(those are the core service, spec/06 §5 steps 6-7).

May import ``app.core.domain.models`` (adapters/pipeline MAY import core domain).
Pure numpy — NO torch, NO opencv, NO fastapi.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.core.domain.models import CameraTrack, Scene4D, Tracks

logger = logging.getLogger(__name__)

# Defaults expressed as fractions of the AABB diagonal (spec/06 §5 step 5).
_DYNAMIC_RADIUS_FRAC = 0.02      # "within radius r" of a moving track sample
_VOXEL_FRAC = 0.005             # static dedup voxel size (0.5% AABB diag)
_MOTION_ABS_FLOOR_FRAC = 0.01   # min absolute motion threshold (1% AABB diag)
# Net-excursion "moving" floor (8% AABB diag). W4.T3 / risk #4 / decision-log §J:
# real VGGT camera/depth noise is ~zero-mean, so a static-background track JITTERS in
# place (net excursion ~ σ·√T, small) while a moving subject TRAVELS (net ~ T·v,
# large). The spec's literal per-FRAME displacement signal can't separate them (a
# walking subject's ~3%/frame sits inside the camera-noise tail), so the split floods
# noisy real reconstructions (static-scene control → static=3). Classifying by NET
# excursion separates them; this floor keeps zero-mean jitter out (measured: 8% leaves
# ~0.5% of control tracks "moving" vs ~16% per-frame). On a clean fixed-camera
# reconstruction net excursion is ~0 so the split behaves as before.
_MOTION_NET_FLOOR_FRAC = 0.08
_DIST_CHUNK = 4096              # row-chunk for the brute-force (N,M) distance query


@dataclass
class GeometryResult:
    """VGGT geometry, ALREADY in mayavius world space (spec/06 §4.5a)."""

    world_points: np.ndarray       # (S, H, W, 3) f32, mayavius world space
    world_points_conf: np.ndarray  # (S, H, W)    f32
    depth: np.ndarray              # (S, H, W)    f32, z-along-axis
    depth_conf: np.ndarray         # (S, H, W)    f32
    camera: CameraTrack            # (T == S) per-frame pose + intrinsics (mayavius)


@dataclass
class TrackResult:
    """Lifted 3D tracks (after the §5 step 4 lift) (spec/06 §4.5a)."""

    positions: np.ndarray   # (M, T, 3) f32 world space
    visibility: np.ndarray  # (M, T)    bool
    colors: np.ndarray      # (M, 3)    u8


def _aabb_over(*arrays: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """AABB (f32 min, max) + diagonal length over all given ``(*,3)`` arrays."""
    chunks = [np.asarray(a, dtype=np.float32).reshape(-1, 3) for a in arrays if a is not None]
    chunks = [c for c in chunks if c.size]
    if not chunks:
        amin = np.zeros(3, dtype=np.float32)
        amax = np.ones(3, dtype=np.float32)
    else:
        allp = np.concatenate(chunks, axis=0)
        amin = allp.min(axis=0).astype(np.float32)
        amax = allp.max(axis=0).astype(np.float32)
    diag = float(np.linalg.norm((amax - amin).astype(np.float32)))
    return amin, amax, diag


def _min_dist_to_set(points: np.ndarray, query: np.ndarray, chunk: int = _DIST_CHUNK) -> np.ndarray:
    """Min Euclidean distance from each ``points[i]`` to the nearest ``query`` point.

    Brute-force chunked ``(N, M)`` distances (no scipy). ``points`` is ``(N,3)``,
    ``query`` is ``(M,3)``; returns ``(N,)`` float32. Empty ``query`` → all ``inf``.
    """
    n = points.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.float32)
    if query.shape[0] == 0:
        return np.full((n,), np.inf, dtype=np.float32)
    q = query.astype(np.float32)
    out = np.empty((n,), dtype=np.float32)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        blk = points[lo:hi].astype(np.float32)            # (b,3)
        d2 = ((blk[:, None, :] - q[None, :, :]) ** 2).sum(axis=2)  # (b,M)
        out[lo:hi] = np.sqrt(d2.min(axis=1)).astype(np.float32)
    return out


def _moving_track_samples(
    tr: TrackResult, motion_thresh_pct: float, net_floor: float
) -> np.ndarray:
    """World positions of the samples of tracks that genuinely MOVE (dynamic seeds).

    A track is "moving" if its NET excursion over the clip — the max distance of its
    visible world positions from their mean — exceeds the threshold. Net excursion is
    used instead of per-FRAME displacement (spec/06 §5 step 5's literal signal)
    because real VGGT camera/depth noise is ~zero-mean: a static-background track
    jitters in place (net ~ σ·√T, small) while a moving subject travels (net ~ T·v,
    large), so they separate even when their per-frame motions overlap the noise — the
    per-frame threshold floods noisy real reconstructions (W4.T3 / risk #4 /
    decision-log §J). Threshold = max(``motion_thresh_pct``-percentile of net
    excursions, ``net_floor``). Returns the visible samples of the moving tracks as
    ``(K, 3)`` float32 (possibly empty).
    """
    pos = np.asarray(tr.positions, dtype=np.float32)     # (M,T,3)
    vis = np.asarray(tr.visibility, dtype=bool)          # (M,T)
    M, T = pos.shape[0], pos.shape[1]
    if M == 0 or T < 2:
        return np.empty((0, 3), dtype=np.float32)

    # Per-track net excursion (max distance of visible samples from their centroid).
    excursion = np.zeros(M, dtype=np.float64)
    for m in range(M):
        vm = vis[m]
        if int(vm.sum()) >= 2:
            p = pos[m][vm]
            excursion[m] = float(np.linalg.norm(p - p.mean(axis=0), axis=1).max())

    has = excursion > 0.0
    if not has.any():
        return np.empty((0, 3), dtype=np.float32)
    # Threshold = max(percentile of net excursions, net floor). >= so that on small
    # track sets (percentile can equal the max excursion) the genuinely moving tracks
    # are still selected; the net floor keeps zero-mean camera jitter out.
    pct = float(np.percentile(excursion[has], motion_thresh_pct * 100.0))
    thresh = max(pct, float(net_floor))

    moving_track = excursion >= thresh                    # (M,)
    moving_node = vis & moving_track[:, None]             # (M,T)
    if not moving_node.any():
        return np.empty((0, 3), dtype=np.float32)
    return pos[moving_node].reshape(-1, 3).astype(np.float32)


def _voxel_dedup(
    points: np.ndarray, colors: np.ndarray, conf: np.ndarray, voxel: float, origin: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxel-grid downsample keeping the HIGHEST-conf point+color per voxel (no averaging).

    ``voxel`` is the edge length; ``origin`` anchors the grid. Returns
    ``(points_kept, colors_kept, conf_kept)`` (conf left in its input dtype).
    """
    n = points.shape[0]
    if n == 0:
        return points, colors, conf
    if voxel <= 0:
        return points, colors, conf
    keys = np.floor((points.astype(np.float32) - origin[None, :]) / np.float32(voxel)).astype(np.int64)
    # Sort by voxel key then DESCENDING conf so the first row per key is the max-conf.
    order = np.lexsort((-conf.astype(np.float64), keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_s = keys[order]
    first = np.ones(keys_s.shape[0], dtype=bool)
    first[1:] = np.any(keys_s[1:] != keys_s[:-1], axis=1)
    sel = order[first]
    return points[sel], colors[sel], conf[sel]


def assemble_scene4d(
    geo: GeometryResult,
    tr: TrackResult,
    request,
    *,
    motion_thresh: float = 0.95,
) -> Scene4D:
    """Static/dynamic split → RAW ``Scene4D`` (spec/06 §5 step 5). No caps/smoothing.

    ``motion_thresh`` is the inter-frame-motion percentile (default 0.95). The
    radius / voxel / motion-floor are fractions of the AABB diagonal (constants
    above). Returns a RAW scene; the core service applies smoothing/culling/caps.
    """
    world_points = np.asarray(geo.world_points, dtype=np.float32)      # (S,H,W,3)
    wp_conf = np.asarray(geo.world_points_conf, dtype=np.float32)      # (S,H,W)
    S = world_points.shape[0]

    tr_pos = np.asarray(tr.positions, dtype=np.float32)               # (M,T,3)
    tr_vis = np.asarray(tr.visibility, dtype=bool)                     # (M,T)
    tr_col = np.asarray(tr.colors, dtype=np.uint8).reshape(-1, 3)

    fps = float(getattr(request, "target_fps", 12.0))

    # ---- AABB over all VGGT points + lifted tracks → radius / voxel / motion floor.
    #      ``amin`` anchors the static voxel-dedup grid (just a grid origin).
    flat_wp = world_points.reshape(-1, 3)
    amin, _amax, diag = _aabb_over(flat_wp, tr_pos.reshape(-1, 3))
    radius = _DYNAMIC_RADIUS_FRAC * diag
    voxel = _VOXEL_FRAC * diag
    net_floor = _MOTION_NET_FLOOR_FRAC * diag        # main split (net excursion)
    motion_floor = _MOTION_ABS_FLOOR_FRAC * diag     # per-frame floor (sparse fallback path)

    # ---- moving track samples (the dynamic "seeds"): tracks with large NET excursion.
    moving_samples = _moving_track_samples(tr, motion_thresh, net_floor)

    # If VGGT per-frame maps are absent/degenerate, fall back to sparse moving tracks.
    vggt_usable = world_points.size > 0 and np.isfinite(world_points).any()

    dynamic_positions: list[np.ndarray] = []
    dynamic_colors: list[np.ndarray] = []
    static_chunks_p: list[np.ndarray] = []
    static_chunks_c: list[np.ndarray] = []
    static_chunks_conf: list[np.ndarray] = []

    fallback_used = False

    if vggt_usable and moving_samples.shape[0] > 0:
        for t in range(S):
            pts = world_points[t].reshape(-1, 3)             # (H*W,3)
            cnf = wp_conf[t].reshape(-1) if wp_conf.size else np.zeros(pts.shape[0], np.float32)
            finite = np.isfinite(pts).all(axis=1)
            pts_f = pts[finite]
            cnf_f = cnf[finite]
            # frame-t color: VGGT world map has no color; derive a grayscale-ish
            # color from confidence is wrong — instead colors come from the source
            # frame in a real run. Here the per-point color is supplied by the
