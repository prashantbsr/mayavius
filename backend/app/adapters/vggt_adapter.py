"""VGGT adapter — facebookresearch/vggt (CVPR 2025). Fast feedforward geometry,
STATIC (no native dynamics/tracks); pair with a tracker for the D4RT look.

MPS: a community float32 path runs on Apple Silicon (no fp16 autocast on MPS).
License: prefer the VGGT-1B-Commercial checkpoint for an open release. Repo IDs,
weights and tensor shapes are RE-VERIFIED and pinned in spec/08 §4.3. Candidate
default for the Mac MPS demo path (handover §5).

W3 (this file): the real geometry pass. ``run_geometry`` runs VGGT once on a frame
set ``[S,3,H,W]`` and returns a ``GeometryResult`` (world points + conf, depth +
conf, per-frame camera) ALREADY in mayavius world space (spec/06 §4.1a). The combo
(spec/06 §4.6) reuses that result for the CoTracker3 lift, so the forward pass runs
ONCE. ``reconstruct`` handles the static-only case (a ``Scene4D`` with the static
cloud + cameras and no tracks).

LAZY IMPORTS (hexagonal / T-130, spec/06 §4): ``torch`` and the ``vggt`` SDK are
imported INSIDE ``run_geometry`` / ``_load_model`` only — importing this MODULE (for
the registry / ``info``) never imports torch. The numpy/opencv pipeline utils
(decode/lift/assemble/quantize) carry no torch.

MPS DISCIPLINE (spec/08 §5, C3): ``PYTORCH_ENABLE_MPS_FALLBACK=1`` is set BEFORE the
torch import; device comes from ``request.device`` (default "mps"); dtype is FP32 —
NO ``torch.cuda.amp.autocast`` / NO fp16 (the VGGT community-port pattern); inference
runs under ``torch.no_grad()``. No torch tensor crosses the port — the adapter
returns numpy / Python only.
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
from app.pipeline.assemble import GeometryResult
from app.pipeline.decode import decode_and_subsample

logger = logging.getLogger(__name__)

# OpenCV -> mayavius axis flip, F = diag(1,-1,-1) (spec/06 §4.1a). F == F^-1.
_F = np.array([1.0, -1.0, -1.0], dtype=np.float32)
# Default static confidence floor (fraction in [0,1] after per-scene normalize) used
# only by the static-only reconstruct() union; the dense split lives in assemble.
_STATIC_CONF_KEEP_FRAC = 0.1


class VggtAdapter(ReconstructionPort):
    name = "vggt"

    def __init__(self, settings=None) -> None:
        self._settings = settings
        # Lazily-loaded VGGT model, cached on the instance (loaded once).
        self._model = None
        self._model_device: str | None = None

    @property
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="vggt",
            produces_tracks=False,
            dynamic=False,
            mps_capable=True,
            weights_license="cc-by-nc-4.0",
            default_weights="facebook/VGGT-1B",
        )

    # ------------------------------------------------------------------ model load
    def _weights_id(self) -> str:
        """The VGGT checkpoint id — ``request``/``settings`` override the NC default."""
        if self._settings is not None:
            w = getattr(self._settings, "vggt_weights", None)
            if w:
                return str(w)
        return "facebook/VGGT-1B"

    def _load_model(self, device: str):
        """Load + cache ``VGGT.from_pretrained`` once (LAZY torch / vggt import).

        Sets ``PYTORCH_ENABLE_MPS_FALLBACK=1`` BEFORE importing torch (spec/08 §5).
        Forces fp32 (no autocast) and moves the model to ``device`` in eval mode.
        """
        if self._model is not None and self._model_device == device:
            return self._model

        import os

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        try:
            import torch  # noqa: F401  (lazy; never imported in core)
            from vggt.models.vggt import VGGT
        except Exception as exc:  # SDK / torch missing or broken
            raise ModelLoadError(
                f"VGGT/torch import failed ({exc!r}); install requirements-ml.txt "
                "(spec/08 §4.1/§4.3)"
            ) from exc

        weights = self._weights_id()
        try:
            model = VGGT.from_pretrained(weights)
            model = model.to(device).eval().float()  # fp32 (C3) — no fp16/autocast
        except Exception as exc:
            raise ModelLoadError(
                f"VGGT.from_pretrained({weights!r}) failed on device={device!r}: {exc}"
            ) from exc

        self._model = model
        self._model_device = device
        logger.info("VGGT loaded: weights=%s device=%s (fp32)", weights, device)
        return model

    # --------------------------------------------------------------- geometry pass
    def run_geometry(self, frames_u8: np.ndarray, request) -> GeometryResult:
        """Run VGGT ONCE on ``frames_u8`` ``[S,3,H,W]`` → ``GeometryResult`` (spec/06 §4.5a).

        ``frames_u8`` are the width-518 processed frames from
        ``decode_and_subsample`` (the SAME frames CoTracker3 consumes — grid
        consistency, spec/06 §5 step 4). Returns world points (+conf), depth (+conf)
        and the per-frame camera, ALL transformed into mayavius world space
        (spec/06 §4.1a). Numpy only — no torch tensor escapes.
        """
        import os

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        import torch

        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        device = getattr(request, "device", "mps")
        model = self._load_model(device)

        frames = np.asarray(frames_u8)
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise InferenceError(
                f"VGGT expects frames [S,3,H,W] uint8; got shape {frames.shape}"
            )
        S, _, H, W = frames.shape

        # numpy uint8 [S,3,H,W] -> torch fp32 [1,S,3,H,W] in [0,1]. VGGT consumes a
        # batched image set (the model runs over ALL S frames at once). The leading
        # batch dim of 1 matches the VGGT forward / pose-decode contract.
        try:
            t = torch.from_numpy(np.ascontiguousarray(frames)).to(device).float() / 255.0
            t = t.unsqueeze(0)  # [1,S,3,H,W]
            with torch.no_grad():  # fp32; NO autocast on MPS (C3)
                preds = model(t)
        except Exception as exc:  # MPS op gap / OOM / shape mismatch
            raise InferenceError(f"VGGT forward failed: {exc}") from exc

        # --- decode predictions. Symbol names are confirmed on-device at T-510; the
        #     access is centralized in _extract_* so the orchestrator reconciles in one
        #     place against the installed vggt package (spec/08 §4.3). ---
        try:
            world_points, wp_conf = self._extract_world_points(preds, S, H, W)
            depth, depth_conf = self._extract_depth(preds, S, H, W)
            w2c, K_px = self._decode_pose(
                preds, pose_encoding_to_extri_intri, t, S, H, W
            )
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"VGGT output decode failed: {exc}") from exc

        # --- transform world points + camera into mayavius world space (spec/06 §4.1a).
        world_may = self._world_points_to_mayavius(world_points, w2c)
        camera = self._cameras_to_mayavius(w2c, K_px, W, H)

        geo = GeometryResult(
            world_points=world_may.astype(np.float32),
            world_points_conf=np.asarray(wp_conf, dtype=np.float32),
            depth=np.asarray(depth, dtype=np.float32),
            depth_conf=np.asarray(depth_conf, dtype=np.float32),
            camera=camera,
        )
        # Attach the source-frame colors (HWC per frame) so the assembler can color
        # the per-frame VGGT points (assemble._frame_colors reads geo.colors).
        geo.colors = self._frames_to_hwc_rgb(frames)  # type: ignore[attr-defined]
        # Attach the PIXEL intrinsics (T,4)=(fx,fy,cx,cy) on the SAME processed (W,H)
        # so the combo (spec/06 §4.6) can feed them to run_tracks' lift directly (the
        # CameraTrack.intrinsics are NORMALIZED; the lift needs pixel intrinsics on the
        # processed grid — grid-consistency, spec/06 §5 step 4).
        geo.intrinsics_px = self._pixel_intrinsics_t4(K_px)  # type: ignore[attr-defined]

        logger.info(
            "run_geometry: S=%d H=%d W=%d -> world_points=%s depth=%s",
            S,
            H,
            W,
            tuple(geo.world_points.shape),
            tuple(geo.depth.shape),
        )
        return geo

    def reconstruct(
        self, request, progress: ProgressSink | None = None
    ) -> Scene4D:
        """Static-only reconstruct: ``Scene4D`` = static cloud + cameras, NO tracks.

        Decodes the clip itself (the combo, spec/06 §4.6, passes frames in via
        ``run_geometry`` instead). Builds ``static_positions``/``static_colors``
        (+``static_conf``) from a simple confidence-thresholded union of the per-frame
        world map — the dense static/dynamic split lives in ``assemble`` for the combo.
        """
        if progress is not None:
            progress(0.10, "decode")
        frames = decode_and_subsample(request)  # numpy uint8 [S,3,H,W] @ width 518

        if progress is not None:
            progress(0.30, "vggt")
        geo = self.run_geometry(frames, request)

        if progress is not None:
            progress(0.85, "assemble")

        scene = self._static_scene_from_geo(geo, request)

        if progress is not None:
            progress(0.95, "assembled")
        return scene

    # ---------------------------------------------------------- mayavius transforms
    def _world_points_to_mayavius(
        self, world_points: np.ndarray, w2c: np.ndarray
    ) -> np.ndarray:
        """Flip VGGT world points (already in VGGT world frame) into mayavius space.

        VGGT ``world_points`` are in VGGT's own world frame (OpenCV-handed). The
        per-point transform is the OpenCV->mayavius axis flip only — ``p_may = F·p =
        (x,-y,-z)`` (spec/06 §4.1a). (The cam->world step in §4.1a applies to points
        expressed in CAMERA space, e.g. the depth-lifted track points; VGGT's
        world_points are already world-framed, so only the flip remains.) Applied
        identically to the camera (see ``_cameras_to_mayavius``) so they stay
        consistent. ``w2c`` is accepted for signature symmetry / future use.
        """
        del w2c
        wp = np.asarray(world_points, dtype=np.float32)
        return (wp * _F[(None,) * (wp.ndim - 1)]).astype(np.float32)

    def _cameras_to_mayavius(
        self, w2c: np.ndarray, K_px: np.ndarray, W: int, H: int
    ) -> CameraTrack:
        """world->camera extrinsics + pixel intrinsics → mayavius ``CameraTrack``.

        Per spec/06 §4.1a, for every frame:
          (1) c2w = inv(w2c);
          (2) axis flip F = diag(1,-1,-1): t_may = F·t_c2w; the camera rotation is the
              SIMILARITY R_may = F · R_c2w · F (a left-only F·R is WRONG);
          (3) R_may -> xyzw unit quaternion;
          (4) intrinsics -> normalized (fx/W, fy/H, cx/W, cy/H).
        Returns ``CameraTrack(poses (S,7) xyzw+t, intrinsics (S,4) normalized)``.
        """
        w2c = np.asarray(w2c, dtype=np.float32)
        K = np.asarray(K_px, dtype=np.float32)
        S = w2c.shape[0]
        Fm = np.diag(_F).astype(np.float32)  # (3,3)

        poses = np.zeros((S, 7), dtype=np.float32)
        intr = np.zeros((S, 4), dtype=np.float32)
        for i in range(S):
            ext = w2c[i]
            if ext.shape == (3, 4):
                ext44 = np.eye(4, dtype=np.float32)
                ext44[:3, :] = ext
            else:
                ext44 = ext.astype(np.float32)
            c2w = np.linalg.inv(ext44)
            R_c2w = c2w[:3, :3]
            t_c2w = c2w[:3, 3]

            # (2) similarity transform of the rotation + flip of the translation.
            R_may = (Fm @ R_c2w @ Fm).astype(np.float32)
            t_may = (_F * t_c2w).astype(np.float32)

            # (3) rotation -> xyzw quaternion.
            poses[i, :4] = _rotmat_to_quat_xyzw(R_may)
            poses[i, 4:] = t_may

            # (4) normalized intrinsics from the pixel K (S,3,3).
            fx, fy = float(K[i, 0, 0]), float(K[i, 1, 1])
            cx, cy = float(K[i, 0, 2]), float(K[i, 1, 2])
            intr[i] = np.array(
                [fx / float(W), fy / float(H), cx / float(W), cy / float(H)],
                dtype=np.float32,
            )
        return CameraTrack(poses=poses, intrinsics=intr)

    def _static_scene_from_geo(self, geo: GeometryResult, request) -> Scene4D:
        """Build a static-only ``Scene4D`` (no tracks) from the per-frame world map.

        Confidence-thresholded union: keep finite per-frame VGGT points whose
        per-scene-normalized confidence is above a small floor; color them from the
        source frame; AABB over the kept static cloud. Cameras come from ``geo``.
        (The dense static/dynamic split is the combo's ``assemble`` job — here we just
        need a usable static cloud + cameras.)
        """
        from app.pipeline.assemble import _frame_colors, _normalize_conf_u8

        world = np.asarray(geo.world_points, dtype=np.float32)  # (S,H,W,3)
        conf = np.asarray(geo.world_points_conf, dtype=np.float32)  # (S,H,W)
        S = world.shape[0]

        pts_chunks: list[np.ndarray] = []
        col_chunks: list[np.ndarray] = []
        conf_chunks: list[np.ndarray] = []
        for t in range(S):
            p = world[t].reshape(-1, 3)
            c = conf[t].reshape(-1) if conf.size else np.zeros(p.shape[0], np.float32)
            finite = np.isfinite(p).all(axis=1)
            pts_chunks.append(p[finite])
            col_chunks.append(_frame_colors(geo, t, finite))
            conf_chunks.append(c[finite])

        if any(c.size for c in pts_chunks):
            pts = np.concatenate([c for c in pts_chunks if c.size], axis=0).astype(np.float32)
            cols = np.concatenate([c for c in col_chunks if c.size], axis=0).astype(np.uint8)
            cnf = np.concatenate([c for c in conf_chunks if c.size], axis=0).astype(np.float32)
        else:
            pts = np.empty((0, 3), np.float32)
            cols = np.empty((0, 3), np.uint8)
            cnf = np.empty((0,), np.float32)

        conf_u8 = _normalize_conf_u8(cnf)
        # Confidence-threshold the union (keep above a small per-scene floor).
        if conf_u8.size:
            keep = conf_u8.astype(np.float32) / 255.0 >= _STATIC_CONF_KEEP_FRAC
            if keep.any():
                pts, cols, conf_u8 = pts[keep], cols[keep], conf_u8[keep]

        if pts.size:
            amin = pts.min(axis=0).astype(np.float32)
            amax = pts.max(axis=0).astype(np.float32)
        else:
            amin = np.zeros(3, np.float32)
            amax = np.ones(3, np.float32)

        return Scene4D(
            frame_count=int(S),
            fps=float(getattr(request, "target_fps", 12.0)),
            aabb_min=amin,
            aabb_max=amax,
            static_positions=pts,
            static_colors=cols,
            static_conf=conf_u8 if conf_u8.size else None,
            dynamic_positions=[np.empty((0, 3), np.float32) for _ in range(S)],
            dynamic_colors=[np.empty((0, 3), np.uint8) for _ in range(S)],
            tracks=None,
            cameras=geo.camera,
        )

    # --------------------------------------------------- prediction extraction glue
    # NOTE (T-510): the exact VGGT output container keys can only be confirmed against
    # the installed `vggt` package on-device. These helpers centralize every
    # model-specific access so the orchestrator reconciles names in ONE place. They
    # accept either a dict-like or an attribute object and the documented key set
    # (decision-log §D + the VGGT README): world_points (+conf), depth (+conf), and a
    # packed pose_enc. Shapes are normalized to the (S,...) the pipeline expects.
    @staticmethod
    def _get(preds, *keys):
        """Fetch the first present key (dict-like or attribute), else None."""
        for k in keys:
            if isinstance(preds, dict):
                if k in preds:
                    return preds[k]
            elif hasattr(preds, k):
                return getattr(preds, k)
        return None

    def _extract_world_points(self, preds, S: int, H: int, W: int):
        """world_points (S,H,W,3) f32 + conf (S,H,W) f32 (squeeze a leading batch)."""
        wp = self._get(preds, "world_points", "world_points_map", "points", "point_map")
        if wp is None:
            raise InferenceError("VGGT preds missing world_points (T-510: reconcile key)")
        wp = self._to_numpy(wp)
        wp = self._squeeze_batch(wp, expect_trailing=3)
        conf = self._get(preds, "world_points_conf", "points_conf", "conf", "world_conf")
        conf = self._to_numpy(conf) if conf is not None else np.ones((S, H, W), np.float32)
        conf = self._squeeze_batch(conf, expect_trailing=None)
        return wp.astype(np.float32), conf.astype(np.float32)

    def _extract_depth(self, preds, S: int, H: int, W: int):
        """depth (S,H,W) f32 + depth_conf (S,H,W) f32 (squeeze batch / trailing-1)."""
        dep = self._get(preds, "depth", "depth_map", "depths")
        if dep is None:
            raise InferenceError("VGGT preds missing depth (T-510: reconcile key)")
        dep = self._to_numpy(dep)
        dep = self._squeeze_to_shw(dep, S, H, W)
        dconf = self._get(preds, "depth_conf", "depth_confidence", "depth_conf_map")
        dconf = self._to_numpy(dconf) if dconf is not None else np.ones((S, H, W), np.float32)
        dconf = self._squeeze_to_shw(dconf, S, H, W)
        return dep.astype(np.float32), dconf.astype(np.float32)

    def _decode_pose(self, preds, pose_encoding_to_extri_intri, images_t, S, H, W):
        """Decode the packed pose encoding → extrinsics (S,3,4) w2c + intrinsics (S,3,3) px.

        ``pose_encoding_to_extri_intri(pose_enc, image_hw)`` is the documented VGGT
        helper (spec/06 §4.1a). It returns world->camera extrinsics and PIXEL
        intrinsics; we squeeze the leading batch to (S,...).
        """
        pose_enc = self._get(preds, "pose_enc", "pose_encoding", "camera_pose_enc", "camera")
        if pose_enc is None:
            raise InferenceError("VGGT preds missing pose_enc (T-510: reconcile key)")
        try:
            extri, intri = pose_encoding_to_extri_intri(pose_enc, images_t.shape[-2:])
        except TypeError:
            # Some versions take (pose_enc, (H, W)) positionally / differently.
            extri, intri = pose_encoding_to_extri_intri(pose_enc, (H, W))
        w2c = self._squeeze_batch(self._to_numpy(extri), expect_trailing=4)  # (S,3,4)
        K = self._squeeze_batch(self._to_numpy(intri), expect_trailing=3)    # (S,3,3)
        return w2c.astype(np.float32), K.astype(np.float32)

    # ------------------------------------------------------------- numpy/shape glue
    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        """torch.Tensor / array-like -> contiguous numpy (detached, CPU). Torch-free path."""
        if x is None:
            return None  # type: ignore[return-value]
        detach = getattr(x, "detach", None)
        if callable(detach):  # a torch tensor
            return x.detach().to("cpu").float().numpy()
        return np.asarray(x)

    @staticmethod
    def _squeeze_batch(arr: np.ndarray, *, expect_trailing: int | None) -> np.ndarray:
        """Drop a leading singleton batch dim if present.

        VGGT may return (B,S,...) or (S,...). If ``arr.shape[0] == 1`` and dropping it
        still leaves a sensible rank, squeeze it. ``expect_trailing`` (when given) is
        the size the last axis should have (e.g. 3 for points / K rows) — used only as
        a sanity hint; the squeeze is purely about the leading batch.
        """
        a = np.asarray(arr)
        if a.ndim >= 4 and a.shape[0] == 1:
            a = a[0]
        return a

    def _squeeze_to_shw(self, arr: np.ndarray, S: int, H: int, W: int) -> np.ndarray:
        """Coerce a depth-like array to (S,H,W): squeeze a leading batch and a trailing 1."""
        a = self._squeeze_batch(arr, expect_trailing=None)
        if a.ndim == 4 and a.shape[-1] == 1:  # (S,H,W,1) -> (S,H,W)
            a = a[..., 0]
        if a.ndim == 4 and a.shape[0] == 1:   # (1,S,H,W) leftover
            a = a[0]
        return a

    @staticmethod
    def _frames_to_hwc_rgb(frames_u8: np.ndarray) -> np.ndarray:
        """[S,3,H,W] uint8 -> [S,H,W,3] uint8 (for assemble._frame_colors)."""
        return np.ascontiguousarray(np.transpose(np.asarray(frames_u8), (0, 2, 3, 1)))

    @staticmethod
    def _pixel_intrinsics_t4(K_px: np.ndarray) -> np.ndarray:
        """Pixel K stack (S,3,3) -> (S,4) ``(fx,fy,cx,cy)`` for the CoTracker3 lift."""
        K = np.asarray(K_px, dtype=np.float32)
        out = np.empty((K.shape[0], 4), dtype=np.float32)
        out[:, 0] = K[:, 0, 0]
        out[:, 1] = K[:, 1, 1]
        out[:, 2] = K[:, 0, 2]
        out[:, 3] = K[:, 1, 2]
        return out


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion (x, y, z, w), mayavius / Three.js order.

    Numerically-stable trace method; the returned quaternion is normalized.
    """
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0  # s = 4*w
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0  # s = 4*x
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0  # s = 4*y
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0  # s = 4*z
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n > 0.0:
        q = (q / n).astype(np.float32)
    else:
        q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return q
