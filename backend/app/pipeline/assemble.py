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
