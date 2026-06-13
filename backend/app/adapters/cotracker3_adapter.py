"""CoTracker3 adapter — 2D point tracking, lifted to 3D with VGGT depth.

Pairs with a static reconstructor (VGGT) to produce the D4RT-style 'cloud +
coloured tracks' look — the pragmatic MVP recipe (handover §5). Repo IDs / weights
are pinned in spec/08 §4.4.

W3 (this file): the real tracking pass. ``run_tracks`` seeds a regular grid of query
points over frame 0, runs ``cotracker3_offline`` over the SAME width-518 frames VGGT
consumed (grid consistency, spec/06 §5 step 4), lifts the 2D pixel tracks to mayavius
world space with the VGGT per-frame depth + camera (``pipeline.lift``), samples
frame-0 colors, and returns a ``TrackResult``. ``reconstruct`` alone returns a
``Scene4D`` with ``tracks`` populated + empty static/dynamic (it needs a depth
source; the combo, spec/06 §4.6, supplies VGGT's depth/camera).

LAZY IMPORTS (hexagonal / T-130, spec/06 §4): ``torch`` and the CoTracker hub model
are loaded INSIDE ``run_tracks`` / ``_load_model`` only — importing this MODULE (for
the registry / ``info``) never imports torch. The numpy lift in ``pipeline/lift.py``
carries no torch.

MPS DISCIPLINE (spec/08 §5, C3): ``PYTORCH_ENABLE_MPS_FALLBACK=1`` is set BEFORE the
torch import; device from ``request.device`` (default "mps"); FP32 (no autocast);
``torch.no_grad()`` for inference. CoTracker auto-selects ``cuda > mps > cpu`` but we
pin the device explicitly to match VGGT. No torch tensor crosses the port.
"""

from __future__ import annotations

import logging

import numpy as np

from app.core.domain.errors import InferenceError, ModelLoadError
from app.core.domain.models import CameraTrack, Scene4D
from app.core.ports.reconstruction_port import (
    AdapterInfo,
    ProgressSink,
    ReconstructionPort,
)
from app.pipeline.assemble import TrackResult
from app.pipeline.lift import lift_tracks_to_3d

logger = logging.getLogger(__name__)

# Query-point seeding (spec/06 §4.2): a regular grid over frame 0, capped to M<=4096
# (the MV4D track cap, spec/05 §4). 32x32 = 1024 <= 4096.
_GRID_SIZE = 32
_MAX_TRACKS = 4_096


class CoTracker3Adapter(ReconstructionPort):
    name = "cotracker3"

    def __init__(self, settings=None) -> None:
        self._settings = settings
        # Lazily-loaded torch.hub CoTracker3 model, cached on the instance.
        self._model = None
        self._model_device: str | None = None

    @property
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="cotracker3",
            produces_tracks=True,
            dynamic=True,
            mps_capable=True,
            weights_license="cc-by-nc-4.0",
            default_weights="facebook/cotracker3",
        )

    # ------------------------------------------------------------------ model load
    def _load_model(self, device: str):
        """Load + cache the ``cotracker3_offline`` hub model once (LAZY torch import).

        Sets ``PYTORCH_ENABLE_MPS_FALLBACK=1`` BEFORE importing torch (spec/08 §5).
        ``torch.hub.load`` pulls ``facebookresearch/co-tracker`` (needs network on the
        first run / a warmed ~/.cache/torch/hub). Forces eval; fp32 (no autocast).
        """
        if self._model is not None and self._model_device == device:
            return self._model

        import os

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        try:
            import torch  # noqa: F401  (lazy; never imported in core)
        except Exception as exc:
            raise ModelLoadError(
                f"torch import failed ({exc!r}); install requirements-ml.txt (spec/08 §4.1)"
            ) from exc

        try:
            # offline = the whole short clip at once (cotracker3_online is for streaming).
            model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
            model = model.to(device).eval()
        except Exception as exc:
            raise ModelLoadError(
                f"torch.hub.load(cotracker3_offline) failed on device={device!r}: {exc}"
            ) from exc

        self._model = model
        self._model_device = device
        logger.info("CoTracker3 loaded: cotracker3_offline device=%s", device)
        return model

    # ------------------------------------------------------------------- track pass
    def run_tracks(
        self,
        frames_u8: np.ndarray,
        depth: np.ndarray,
        camera: CameraTrack,
        intrinsics_px,
    ) -> TrackResult:
        """Track a grid over frame 0 → lift to mayavius world space → ``TrackResult``.

        Args:
          frames_u8:    ``[S,3,H,W]`` uint8 RGB — the SAME width-518 processed frames
                        VGGT consumed (grid consistency, spec/06 §5 step 4).
          depth:        ``(T,H,W)`` f32 VGGT z-along-axis depth (same processed grid).
          camera:       mayavius ``CameraTrack`` (cam->world poses) — its ``poses`` are
                        converted to a ``(T,4,4)`` c2w stack for the lift.
          intrinsics_px: per-frame PIXEL intrinsics ``(fx,fy,cx,cy)`` on the SAME
                        processed ``(W,H)`` as ``frames_u8`` / ``depth`` (the divisor
                        and CAMERAS normalization share that grid — silent-failure
                        guard, spec/06 §5 step 4).

        Returns a ``TrackResult`` (positions ``(M,T,3)`` world, visibility ``(M,T)``
        bool, colors ``(M,3)`` u8). Numpy only — no torch tensor escapes.
        """
        import os

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        import torch

        frames = np.asarray(frames_u8)
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise InferenceError(
                f"CoTracker3 expects frames [S,3,H,W] uint8; got shape {frames.shape}"
            )
        S, _, H, W = frames.shape
        device = self._device_from_camera_or_default()
        model = self._load_model(device)

        # --- seed a regular grid of query points on frame 0 (capped to M<=4096). ---
        queries_np = _grid_queries(H, W, _GRID_SIZE, _MAX_TRACKS)  # (M,3) = (t=0, x, y)
        M = queries_np.shape[0]

        try:
            # [S,3,H,W] uint8 -> torch video [1,S,3,H,W] fp32 (CoTracker takes 0..255).
            video = (
                torch.from_numpy(np.ascontiguousarray(frames))
                .to(device)
                .float()
                .unsqueeze(0)
            )
            queries = torch.from_numpy(queries_np).to(device).float().unsqueeze(0)  # [1,M,3]
            with torch.no_grad():  # fp32; NO autocast on MPS (C3)
                pred_tracks, pred_visibility = model(video, queries=queries)
        except Exception as exc:  # MPS op gap / OOM / shape mismatch
            raise InferenceError(f"CoTracker3 forward failed: {exc}") from exc

        # pred_tracks (B,T,N,2) px, pred_visibility (B,T,N,1)|(B,T,N) -> numpy (M,T,2)/(M,T).
        tracks_2d, vis_2d = _unpack_cotracker(pred_tracks, pred_visibility, M, S)

        # --- lift 2D->3D via the shared pipeline util (SAME processed (W,H)). ---
        c2w = _camera_to_opencv_c2w_stack(camera, S)  # OpenCV c2w; lift applies the single F flip
        positions, visibility = lift_tracks_to_3d(
            tracks_2d, vis_2d, np.asarray(depth, dtype=np.float32), intrinsics_px, c2w
        )

        # --- frame-0 color per track (sample the source frame at the query pixel). ---
        colors = _sample_frame0_colors(frames, queries_np)

        logger.info(
            "run_tracks: S=%d M=%d grid=%d -> positions=%s visible_total=%d",
            S,
            M,
            _GRID_SIZE,
            tuple(positions.shape),
            int(visibility.sum()),
        )
        return TrackResult(
            positions=positions.astype(np.float32),
            visibility=visibility.astype(bool),
            colors=colors.astype(np.uint8),
        )

    def reconstruct(
        self, request, progress: ProgressSink | None = None
    ) -> Scene4D:
        """Tracks-only reconstruct (spec/06 §4.2): needs an external depth source.

        CoTracker3 alone produces only 2D tracks; the 2D->3D lift requires a depth +
        intrinsics source, which this adapter does NOT own. Per spec/06 §4.2 it
        "depends on a depth+intrinsics source for the lift" — in the default combo
        (spec/06 §4.6) VGGT supplies them via ``run_tracks``. Standalone there is no
        depth provider, so this honestly raises (stub convention: throw with a message
        pointing at the combo) rather than fabricate depth and emit wrong-depth tracks.
        Call ``run_tracks(frames, depth, camera, intrinsics_px)`` directly when a depth
        source is available.
        """
        del request, progress
        raise InferenceError(
            "CoTracker3Adapter.reconstruct needs a depth/intrinsics source for the "
            "2D->3D lift; run the 'vggt+cotracker3' combo (spec/06 §4.2/§4.6) which "
            "supplies VGGT depth+camera, or call run_tracks(...) directly."
        )

    # ------------------------------------------------------------------------ glue
    def _device_from_camera_or_default(self) -> str:
        """Resolve the device from settings (default "mps"); CoTracker also auto-selects."""
        if self._settings is not None:
            d = getattr(self._settings, "device", None)
            if d:
                return str(d)
        return "mps"


# --------------------------------------------------------------------------- helpers


def _grid_queries(H: int, W: int, grid: int, max_tracks: int) -> np.ndarray:
    """A regular grid of query points on frame 0 → ``(M,3)`` float32 ``(t=0, x, y)``.

    ``grid x grid`` interior points (a small inset margin keeps queries off the very
    edge so depth sampling stays in-bounds). Capped to ``max_tracks`` by an even
    stride. CoTracker query format is ``(t, x, y)`` in PIXELS on the processed grid.
    """
    g = max(1, int(grid))
    # Inset by half a cell so points sit at cell centers (off the border).
    xs = np.linspace(0.5 * W / g, W - 0.5 * W / g, g, dtype=np.float32)
    ys = np.linspace(0.5 * H / g, H - 0.5 * H / g, g, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)  # (g,g)
    pts = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1).astype(np.float32)  # (g*g,2)

    if pts.shape[0] > max_tracks:
        sel = np.linspace(0, pts.shape[0] - 1, max_tracks).round().astype(np.int64)
        sel = np.unique(sel)
        pts = pts[sel]

    queries = np.zeros((pts.shape[0], 3), dtype=np.float32)
    queries[:, 0] = 0.0          # query frame index t == 0
    queries[:, 1] = pts[:, 0]    # x (u_px)
    queries[:, 2] = pts[:, 1]    # y (v_px)
    return queries


def _unpack_cotracker(pred_tracks, pred_visibility, M: int, S: int):
    """CoTracker outputs → ``tracks_2d (M,T,2)`` f32 + ``vis (M,T)`` bool (numpy).

    Accepts torch tensors or arrays; ``pred_tracks`` is ``(B,T,N,2)`` and
    ``pred_visibility`` is ``(B,T,N,1)`` or ``(B,T,N)``. Squeezes the leading batch,
    transposes ``(T,N,*) -> (N,T,*)`` so M (tracks) leads, and thresholds visibility
    (float occlusion logits/probs -> bool at 0.5; bool passes through).
    """
    tr = _np(pred_tracks)   # (B,T,N,2) or (T,N,2)
    vs = _np(pred_visibility)

    if tr.ndim == 4:  # (B,T,N,2) -> drop batch
        tr = tr[0]
    # now (T,N,2) -> (N,T,2)
    tracks_2d = np.transpose(tr, (1, 0, 2)).astype(np.float32)

    if vs.ndim == 4:        # (B,T,N,1) -> (T,N,1)
        vs = vs[0]
    if vs.ndim == 3 and vs.shape[-1] == 1:  # (T,N,1) -> (T,N)
        vs = vs[..., 0]
    if vs.ndim == 3:        # leftover (B,T,N) -> (T,N)
        vs = vs[0]
    # (T,N) -> (N,T)
    vis = np.transpose(vs, (1, 0))
    if vis.dtype != np.bool_:
        vis = vis >= 0.5
    return tracks_2d, vis.astype(bool)


# OpenCV<->mayavius axis flip (spec/06 §4.1a). F = diag(1,-1,-1) is a 180-degree
# rotation about +X (det=+1, same handedness) and is its OWN INVERSE (F·F = I).
_FLIP = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)


def _camera_to_opencv_c2w_stack(camera: CameraTrack, S: int) -> np.ndarray:
    """mayavius ``CameraTrack`` (xyzw+t cam->world) → ``(T,4,4)`` **OpenCV** c2w.

    ``pipeline.lift`` is contractually fed the OPENCV camera->world transform and
    applies the single OpenCV->mayavius flip ``F`` ITSELF (spec/06 §5 step 4 / §4.1a).
    But ``camera`` is already in MAYAVIUS space — VggtAdapter built it as
    ``R_may = F·R_c2w·F`` and ``t_may = F·t_c2w`` (§4.1a). So we INVERT the flip here
    to recover the native OpenCV c2w the lift expects (``F`` is an involution):

        ``R_c2w = F·R_may·F``      ``t_c2w = F·t_may``

    Feeding the lift the mayavius c2w instead would apply ``F`` twice and detach the
    track ribbons from the VGGT cloud/cameras — the exact silent-failure spec/06 §5
    step 4 warns about (passes "≥1 track" while the geometry is wrong). The
    round-trip guard is ``tests/adapters/test_track_lift_roundtrip.py``. Identity if
    ``camera`` is absent (then the lift's lone flip is the only transform).
    """
    if camera is None:
        return np.broadcast_to(np.eye(4, dtype=np.float32), (S, 4, 4)).copy()
    poses = np.asarray(camera.poses, dtype=np.float32)
    T = poses.shape[0]
    out = np.zeros((T, 4, 4), dtype=np.float32)
    for i in range(T):
        r_may = _quat_xyzw_to_rotmat(poses[i, :4])   # mayavius c2w rotation
        t_may = poses[i, 4:7]
        out[i, :3, :3] = (_FLIP @ r_may @ _FLIP).astype(np.float32)  # -> OpenCV c2w R
        out[i, :3, 3] = (_FLIP @ t_may).astype(np.float32)           # -> OpenCV c2w t
        out[i, 3, 3] = 1.0
    return out


def _sample_frame0_colors(frames_u8: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Sample frame-0 RGB at each query pixel → ``(M,3)`` u8.

    ``frames_u8`` is ``[S,3,H,W]``; ``queries`` is ``(M,3)`` = ``(t, x, y)``. Reads the
    source pixel at ``(round(y), round(x))`` on frame 0 (clamped in-bounds).
    """
    frames = np.asarray(frames_u8)
    _, _, H, W = frames.shape
    f0 = np.transpose(frames[0], (1, 2, 0))  # (H,W,3) RGB u8
    xi = np.clip(np.rint(queries[:, 1]).astype(np.int64), 0, W - 1)
    yi = np.clip(np.rint(queries[:, 2]).astype(np.int64), 0, H - 1)
    return f0[yi, xi, :3].astype(np.uint8)


def _np(x) -> np.ndarray:
    """torch.Tensor / array-like -> contiguous CPU numpy (torch-free downstream)."""
    detach = getattr(x, "detach", None)
    if callable(detach):
        return x.detach().to("cpu").float().numpy()
    return np.asarray(x)


def _quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (x,y,z,w) -> 3x3 rotation matrix (normalizes defensively)."""
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n == 0.0:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = (q / n)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return R
