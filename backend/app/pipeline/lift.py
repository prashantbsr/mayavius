"""Lift 2D CoTracker3 tracks → 3D world-space track positions (spec/06 §5 step 4).

Given per-frame 2D pixel tracks ``(u_px, v_px)`` (on the **processed 518-width
grid** — grid-consistency, spec/06 §5 step 4), VGGT per-frame depth, and **pixel**
intrinsics, unproject each sample into the OpenCV camera frame and transform it into
**mayavius world space** (spec/05 §2 / spec/06 §4.1a):

    OpenCV unprojection (z-along-axis depth ``D`` at pixel ``(u_px, v_px)``):
        x_cam = (u_px - cx) * D / fx
        y_cam = (v_px - cy) * D / fy
        z_cam = D
    camera->world:   p_world_opencv = R_c2w · p_cam + t_c2w
    OpenCV->mayavius axis flip (F = diag(1, -1, -1)):
        p_may = F · p_world_opencv = (x, -y, -z)

The flip and the cam->world transform MUST be applied identically to camera poses
(done in the adapter) so points and cameras stay consistent. Depth is sampled at
``(round(v_px), round(u_px))``; samples landing on an invalid/hole depth (≤0 or
non-finite) or out of bounds are marked **invisible** (a ribbon gap).

Pure numpy — NO torch, NO opencv, NO fastapi.
"""

from __future__ import annotations

import numpy as np

# OpenCV -> mayavius axis flip (spec/06 §4.1a). F == F^-1.
_F = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def _normalize_intrinsics(intrinsics: np.ndarray, T: int) -> np.ndarray:
    """Coerce per-frame intrinsics to ``(T, 4)`` float32 ``(fx, fy, cx, cy)`` (pixels).

    Accepts either ``(T, 4)`` already, a single ``(4,)`` broadcast across frames, a
    ``(T, 3, 3)`` pixel K-matrix stack, or a single ``(3, 3)`` K broadcast.
    """
    arr = np.asarray(intrinsics, dtype=np.float32)
    if arr.ndim == 1 and arr.shape == (4,):
        return np.broadcast_to(arr, (T, 4)).astype(np.float32).copy()
    if arr.ndim == 2 and arr.shape == (T, 4):
        return arr.copy()
    if arr.ndim == 2 and arr.shape == (3, 3):
        fx, fy, cx, cy = arr[0, 0], arr[1, 1], arr[0, 2], arr[1, 2]
        row = np.array([fx, fy, cx, cy], dtype=np.float32)
        return np.broadcast_to(row, (T, 4)).astype(np.float32).copy()
    if arr.ndim == 3 and arr.shape == (T, 3, 3):
        out = np.empty((T, 4), dtype=np.float32)
        out[:, 0] = arr[:, 0, 0]
        out[:, 1] = arr[:, 1, 1]
        out[:, 2] = arr[:, 0, 2]
        out[:, 3] = arr[:, 1, 2]
        return out
    raise ValueError(
        f"intrinsics must be (T,4), (4,), (T,3,3) or (3,3); got shape {arr.shape} for T={T}"
    )


def _normalize_c2w(c2w: np.ndarray, T: int) -> np.ndarray:
    """Coerce camera->world transforms to ``(T, 4, 4)`` float32.

    Accepts ``(T, 4, 4)``, ``(T, 3, 4)`` (homogeneous bottom row appended), a single
    ``(4, 4)`` / ``(3, 4)`` broadcast across frames.
    """
    arr = np.asarray(c2w, dtype=np.float32)

    def _pad34(m: np.ndarray) -> np.ndarray:
        bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return np.concatenate([m, bottom[None, :]], axis=0)

    if arr.ndim == 2 and arr.shape == (4, 4):
        return np.broadcast_to(arr, (T, 4, 4)).astype(np.float32).copy()
    if arr.ndim == 2 and arr.shape == (3, 4):
        return np.broadcast_to(_pad34(arr), (T, 4, 4)).astype(np.float32).copy()
    if arr.ndim == 3 and arr.shape == (T, 4, 4):
        return arr.copy()
    if arr.ndim == 3 and arr.shape == (T, 3, 4):
        out = np.empty((T, 4, 4), dtype=np.float32)
        for t in range(T):
            out[t] = _pad34(arr[t])
        return out
    raise ValueError(
        f"c2w must be (T,4,4), (T,3,4), (4,4) or (3,4); got shape {arr.shape} for T={T}"
    )


def lift_tracks_to_3d(
    tracks_2d: np.ndarray,
    visibility: np.ndarray,
    depth: np.ndarray,
    intrinsics_px,
    c2w,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject 2D pixel tracks to mayavius world space (spec/06 §5 step 4 / §4.1a).

    Args:
      tracks_2d:    ``(M, T, 2)`` pixel coords ``(u_px, v_px)`` on the processed grid.
      visibility:   ``(M, T)`` bool — CoTracker ``pred_visibility``.
      depth:        ``(T, H, W)`` float32 — VGGT z-along-axis depth (pixel grid).
      intrinsics_px: per-frame ``(fx, fy, cx, cy)`` pixels — ``(T,4)`` / ``(4,)`` /
                    ``(T,3,3)`` / ``(3,3)`` (the divisor is in the SAME processed
                    ``(W, H)`` as ``tracks_2d``, grid-consistency).
      c2w:          camera->world transform per frame — ``(T,4,4)`` / ``(T,3,4)`` /
                    ``(4,4)`` / ``(3,4)``.

    Returns ``(positions, out_visibility)``:
      positions:      ``(M, T, 3)`` float32 world space (mayavius convention).
      out_visibility: ``(M, T)`` bool — input visibility AND a valid (in-bounds,
                      finite, > 0) depth sample. Holes/occlusion -> invisible.
    """
    uv = np.asarray(tracks_2d, dtype=np.float32)
    if uv.ndim != 3 or uv.shape[2] != 2:
        raise ValueError(f"tracks_2d must be (M,T,2); got {uv.shape}")
    M, T = uv.shape[0], uv.shape[1]

    vis_in = np.asarray(visibility, dtype=bool).reshape(M, T)
    dep = np.asarray(depth, dtype=np.float32)
    if dep.ndim != 3 or dep.shape[0] != T:
        raise ValueError(f"depth must be (T,H,W) with T={T}; got {dep.shape}")
    H, W = dep.shape[1], dep.shape[2]

    K = _normalize_intrinsics(intrinsics_px, T)   # (T,4)
    C = _normalize_c2w(c2w, T)                     # (T,4,4)

    positions = np.zeros((M, T, 3), dtype=np.float32)
    out_vis = np.zeros((M, T), dtype=bool)

    for t in range(T):
        fx, fy, cx, cy = (float(K[t, 0]), float(K[t, 1]), float(K[t, 2]), float(K[t, 3]))
        R = C[t, :3, :3]          # (3,3) c2w rotation
        tr = C[t, :3, 3]          # (3,) c2w translation
        u = uv[:, t, 0]           # (M,)
        v = uv[:, t, 1]

        # Depth sampled at (round(v), round(u)); out-of-bounds -> invalid.
        ui = np.rint(u).astype(np.int64)
        vi = np.rint(v).astype(np.int64)
        in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        ui_c = np.clip(ui, 0, W - 1)
        vi_c = np.clip(vi, 0, H - 1)
        D = dep[t, vi_c, ui_c]    # (M,)

        valid = in_bounds & np.isfinite(D) & (D > 0) & vis_in[:, t]

        # OpenCV unprojection (z-along-axis depth).
        x_cam = (u - cx) * D / np.float32(fx if fx != 0 else 1.0)
        y_cam = (v - cy) * D / np.float32(fy if fy != 0 else 1.0)
        z_cam = D
        p_cam = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)  # (M,3)

