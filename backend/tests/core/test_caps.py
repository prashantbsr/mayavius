"""T-104 — rigorous MV4D cap enforcement (`enforce_caps`) + caps semantics.

W1.T1 acceptance gate (spec/10 §1.2 T-104). `enforce_caps` lives in
``app/core/services/reconstruction_service.py`` (the core service, NOT the encoder —
spec/06 §5 step 7). These tests build an OVER-CAP ``Scene4D`` and assert it is
culled/subsampled to within EVERY MV4D cap (spec/05 §4), in the documented order:

  - frames ``T`` -> 64 (uniform temporal subsample), run FIRST so the dynamic list,
    track T-axis, and camera arrays stay aligned to the new ``frame_count``;
  - static -> 150 000 (drop lowest ``static_conf`` first);
  - dynamic per-frame -> 20 000 (DETERMINISTIC fixed-seed uniform random subsample,
    confidence-free — dynamic frames carry no per-point conf in ``Scene4D``);
  - tracks ``M`` -> 4 096 (drop lowest mean-visibility first).

There is NO separate ">24 MB" exception path — escalating cull keeps the payload
under the ceiling (spec/05 §4, spec/06 §5 step 7); the ONLY encode-path raise is
``EmptyReconstructionError`` (when culling removes everything), and that fires from
the SERVICE ``run()`` emptiness guard, not from ``enforce_caps`` itself.

Pure numpy, no torch — keeps the hexagonal boundary (T-130) green.
"""

from __future__ import annotations

import inspect

import numpy as np

import app.core.services.reconstruction_service as svc_mod
from app.core.domain.errors import EmptyReconstructionError
from app.core.domain.models import (
    CameraTrack,
    ReconstructionRequest,
    Scene4D,
    Tracks,
)
from app.core.ports.reconstruction_port import (
    AdapterInfo,
    ProgressSink,
    ReconstructionPort,
)
from app.core.services.reconstruction_service import (
    ReconstructionService,
    enforce_caps,
)

# MV4D caps (spec/05 §4) — the contract constants. Asserted, not imported-as-truth,
# so a silent loosening of the module constants would still fail this test.
MAX_STATIC = 150_000
MAX_DYNAMIC_PER_FRAME = 20_000
MAX_TRACKS = 4_096
MAX_FRAMES = 64

# Over-cap dimensions for the canonical T-104 scene.
T_OVER = 80          # > 64 frames
N_STATIC_OVER = 200_000  # > 150k static points
N_DYN_OVER = 25_000  # one frame with > 20k dynamic points
M_OVER = 5_000       # > 4096 tracks


def _over_cap_scene(*, seed: int = 1234, with_cameras: bool = True) -> Scene4D:
    """Build a deterministic OVER-CAP ``Scene4D`` (every cap exceeded).

    Design choices that make the cull-order assertions rigorous:
      - the over-cap dynamic frame (25 000 points) sits at frame index 0, which the
        uniform temporal subsample ALWAYS keeps (``linspace`` includes the endpoints)
        — so the per-frame dynamic cap is genuinely exercised after the frame cull;
      - ``static_conf`` is a known array so we can assert "every dropped conf <= every
        kept conf" by multiset against the top-N of the sorted full conf;
      - track visibility has a known mean per track so the mean-visibility cull order
        is checkable the same way.
    """
    rng = np.random.default_rng(seed)

    # --- static: 200k points, each with a confidence we can audit ---
    static_positions = rng.random((N_STATIC_OVER, 3)).astype(np.float32)
    static_colors = (rng.random((N_STATIC_OVER, 3)) * 255).astype(np.uint8)
    static_conf = rng.integers(0, 256, size=N_STATIC_OVER).astype(np.uint8)

    # --- dynamic: frame 0 is over-cap (25k), the rest are tiny (incl. an empty one) ---
    dynamic_positions: list[np.ndarray] = []
    dynamic_colors: list[np.ndarray] = []
    for t in range(T_OVER):
        if t == 0:
            n = N_DYN_OVER
        elif t == 1:
            n = 0  # a valid empty frame (spec/05 §3.5)
        else:
            n = 50
        dynamic_positions.append(rng.random((n, 3)).astype(np.float32))
        dynamic_colors.append((rng.random((n, 3)) * 255).astype(np.uint8))

    # --- tracks: 5000, each with a distinct mean visibility so the order is auditable ---
    # Give track m a visibility fraction that increases with m (so the lowest-mean
    # tracks are a well-defined prefix), then shuffle so kept != "first 4096".
    visibility = np.zeros((M_OVER, T_OVER), dtype=bool)
    for m in range(M_OVER):
        k = int(round((m / (M_OVER - 1)) * T_OVER))  # 0..T_OVER visible samples
        visibility[m, :k] = True
    perm = rng.permutation(M_OVER)
    visibility = visibility[perm]
    track_positions = rng.random((M_OVER, T_OVER, 3)).astype(np.float32)
    track_colors = (rng.random((M_OVER, 3)) * 255).astype(np.uint8)
    tracks = Tracks(positions=track_positions, visibility=visibility, colors=track_colors)

    cameras = (
        CameraTrack(
            poses=rng.random((T_OVER, 7)).astype(np.float32),
            intrinsics=rng.random((T_OVER, 4)).astype(np.float32),
        )
        if with_cameras
        else None
    )

    return Scene4D(
        frame_count=T_OVER,
        fps=24.0,
        aabb_min=np.zeros(3, dtype=np.float32),
        aabb_max=np.ones(3, dtype=np.float32),
        static_positions=static_positions,
