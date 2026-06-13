"""Default combo adapter ``vggt+cotracker3`` (spec/06 §4.6) — the MAYAVIUS_ADAPTER default.

A small COMPOSING adapter (not a sixth model file) that orchestrates ``VggtAdapter``
(static + depth/camera) and ``CoTracker3Adapter`` (tracks), running VGGT ONCE and
feeding its depth/intrinsics/pose to the CoTracker3 lift. ``reconstruct`` returns a
**RAW** ``Scene4D`` (the static/dynamic split is done by ``assemble_scene4d``); the
core ``ReconstructionService`` then smooths/culls/caps it (spec/06 §5 steps 6-7).

LAZY IMPORTS (hexagonal / T-130, spec/06 §4): no torch import lives here — the
sub-adapters import their SDKs lazily INSIDE their own ``run_geometry`` / ``run_tracks``
(spec/06 §4). Importing this MODULE (for the registry / ``info``) never imports torch,
so the API can advertise the combo's capabilities with zero ML deps installed. The
sub-adapters (``VggtAdapter`` / ``CoTracker3Adapter``) are constructed LAZILY on first
use so even constructing the combo stays torch-free.

MPS DISCIPLINE (spec/08 §5, C3): the sub-adapters set
``PYTORCH_ENABLE_MPS_FALLBACK=1`` before importing torch, take the device from
``request.device`` (default "mps"), force FP32 (no autocast), and run under
``torch.no_grad()``. No torch tensor crosses the port — the combo returns numpy /
Python only.
"""

from __future__ import annotations

from app.core.domain.models import Scene4D
from app.core.ports.reconstruction_port import (
    AdapterInfo,
    ProgressSink,
    ReconstructionPort,
)
from app.pipeline.assemble import assemble_scene4d
from app.pipeline.decode import cap_frames_to_token_budget, decode_and_subsample


class VggtCoTracker3Adapter(ReconstructionPort):
    name = "vggt+cotracker3"

    def __init__(self, settings=None) -> None:
        self._settings = settings
        # Sub-adapters constructed LAZILY (first use) so importing/constructing the
        # combo never imports torch/vggt/cotracker (T-130, spec/06 §4).
        self._vggt = None
        self._cot = None

    @property
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="vggt+cotracker3",
            produces_tracks=True,
            dynamic=True,
            mps_capable=True,
            weights_license="cc-by-nc-4.0",
            default_weights="facebook/VGGT-1B + facebook/cotracker3",
        )

    # ----------------------------------------------------------- lazy sub-adapters
    def _vggt_adapter(self):
        """Lazily construct + cache the VGGT half (no torch at construction)."""
        if self._vggt is None:
            from app.adapters.vggt_adapter import VggtAdapter  # lazy

            self._vggt = VggtAdapter(self._settings)
        return self._vggt

    def _cot_adapter(self):
        """Lazily construct + cache the CoTracker3 half (no torch at construction)."""
        if self._cot is None:
            from app.adapters.cotracker3_adapter import CoTracker3Adapter  # lazy

            self._cot = CoTracker3Adapter(self._settings)
        return self._cot

    def _motion_thresh(self) -> float:
        """Static/dynamic split percentile (spec/06 §5 step 5); default 0.95."""
        if self._settings is not None:
            mt = getattr(self._settings, "motion_thresh", None)
            if mt is not None:
                return float(mt)
        return 0.95

    # ---------------------------------------------------------------- reconstruct
    def reconstruct(
        self, request, progress: ProgressSink | None = None
    ) -> Scene4D:
        """Orchestrate VGGT (once) + CoTracker3 + assemble → RAW ``Scene4D`` (spec/06 §4.6).

        Flow (spec/06 §4.6 / §5):
          1. decode + subsample the clip ONCE → the shared width-518 frame set
             (grid-consistency: VGGT and CoTracker3 consume the SAME frames,
             spec/06 §5 step 4);
          2. ``VggtAdapter.run_geometry`` → ``GeometryResult`` (world points + conf,
             depth + conf, per-frame camera, ALL in mayavius world space) — VGGT runs
             ONCE here and its depth/camera/intrinsics feed the lift;
          3. ``CoTracker3Adapter.run_tracks`` lifts the 2D tracks to 3D using VGGT's
             ``depth`` / ``camera`` / pixel intrinsics → ``TrackResult``;
          4. ``assemble_scene4d`` does the static/dynamic split → a RAW ``Scene4D``.

        Returns the RAW scene; the core ``ReconstructionService`` applies
        smoothing/culling/caps (spec/06 §5 steps 6-7) — NOT done here.
        """
        vggt = self._vggt_adapter()
        cot = self._cot_adapter()

        frames = decode_and_subsample(request)  # numpy uint8 [S,3,H,W] @ width 518
        # MPS self-attention OOM guard: cap S so S·tokens-per-frame fits memory (a
        # square/high-res clip OOMs the default max_clip_frames on a 36 GB Mac;
        # decision-log §J.1). Both VGGT and CoTracker3 consume these SAME frames, so
        # the cap preserves grid-consistency. CUDA/cloud lifts the cap.
        if str(getattr(request, "device", "mps")) == "mps":
            frames = cap_frames_to_token_budget(frames)
        if progress is not None:
            progress(0.10, "decode")

        # VGGT runs ONCE; geo carries depth + camera (+ pixel intrinsics on the SAME
        # processed grid, attached by run_geometry as geo.intrinsics_px) for the lift.
        geo = vggt.run_geometry(frames, request)
        if progress is not None:
            progress(0.55, "vggt")

        # Feed VGGT's depth/camera/pixel-intrinsics to the CoTracker3 lift (grid
