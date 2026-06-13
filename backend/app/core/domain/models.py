"""Canonical, model-agnostic domain types for the reconstruction core.

Authoritative shapes/dtypes are owned by spec/05-data-contract.md §5.1 (reproduced
in spec/06 §3 for convenience; if they disagree, 05 wins). These types are what the
encoder (`app/wire/encoder.py`) consumes and what crosses the `ReconstructionPort`.

NumPy only — no torch, no FastAPI. This module is part of the pure core
(hexagonal boundary, handover §3 / T-130).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.domain.errors import ClipTooLongError


@dataclass
class CameraTrack:
    """Per-frame camera, mayavius world space (spec/05 §3.7)."""

    poses: np.ndarray        # (T, 7) f32  quaternion(xyzw) + translation, cam->world
    intrinsics: np.ndarray   # (T, 4) f32  normalized fx, fy, cx, cy


@dataclass
class Tracks:
    """M 3D trajectories (the ribbons), one polyline per tracked point."""

    positions: np.ndarray        # (M, T, 3) f32  world space
    visibility: np.ndarray       # (M, T)    bool
    colors: np.ndarray | None    # (M, 3)    u8, optional


@dataclass
class Scene4D:
    """The canonical reconstruction the encoder serializes into MV4D v1.

    Float positions (the encoder quantizes); colors are u8 RGB. The four sections
    (static / dynamic / tracks / cameras) mirror spec/05 §1.
    """

    frame_count: int                     # T
    fps: float
    aabb_min: np.ndarray                 # (3,) f32
    aabb_max: np.ndarray                 # (3,) f32
    static_positions: np.ndarray         # (N_s, 3) f32
    static_colors: np.ndarray            # (N_s, 3) u8
    static_conf: np.ndarray | None       # (N_s,)  u8, optional
    dynamic_positions: list[np.ndarray]  # len T, each (N_d_t, 3) f32
    dynamic_colors: list[np.ndarray]     # len T, each (N_d_t, 3) u8
    tracks: Tracks | None
    cameras: CameraTrack | None
    # Provenance (NOT serialized into MV4D; returned via job metadata, D2):
    adapter_id: str = ""
    weights_license: str = ""


@dataclass(frozen=True)
class ReconstructionRequest:
    """Input to a reconstruction job (device/clip-only; thresholds do not travel here)."""

    video_path: str           # local path to the uploaded clip (worker-resolved)
    max_frames: int = 24      # post-subsample cap; MUST be <= 64 (MV4D T<=64, spec/05 §4)
    target_fps: float = 12.0  # subsample target; playback fps written to the MV4D header
    device: str = "mps"       # "mps" | "cpu" | "cuda"; default from MAYAVIUS_DEVICE

    def validate(self) -> None:
        """Assert the clip-cap invariants. Called by the service (spec/06 §5).

        Frozen instance → no mutation; the clamp happens at the construction site
        (the POST handler builds with ``max_frames=min(settings.max_clip_frames,
        64)``), so this raises rather than clamps.
        """
        if not (1 <= self.max_frames <= 64):
            raise ClipTooLongError(f"max_frames must be 1..64, got {self.max_frames}")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be > 0")
