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
