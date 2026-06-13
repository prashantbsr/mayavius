"""Video decode + temporal subsample → frames ``[S, 3, H, W]`` RGB uint8.

This is pipeline step 1 (spec/06 §5 step 1). It is **model-agnostic** and
deliberately **torch-free**: it returns a numpy ``uint8`` array and every adapter
does its own ``torch.from_numpy(...).to(device).float()/255`` (spec/06 §4.1, W3.T1
purity). Each frame is RGB, width rescaled to **518 px** (the VGGT/CoTracker3
processed grid — spec/06 §5 step 4 grid-consistency: CoTracker3 MUST consume the
EXACT same frames VGGT does, so this single decode feeds both layers).

Decoding uses ``opencv-python`` (`cv2.VideoCapture`) as the primary path with an
``imageio[ffmpeg]`` fallback for awkward containers (spec/06 §5 step 1, spec/08
§4.2). Both imports are **LAZY** (inside the function) so importing this module
(for registry/info) never imports cv2/imageio — and the module stays importable
with neither installed (the call then raises a clear ``UnsupportedMediaError``).

numpy only at module import time — NO torch, NO fastapi. cv2/imageio imported
lazily inside ``decode_and_subsample``.
"""

from __future__ import annotations

import logging

import numpy as np

from app.core.domain.errors import UnsupportedMediaError

logger = logging.getLogger(__name__)

# The processed grid width all models consume (spec/06 §4.1 / §5 step 4).
_TARGET_WIDTH = 518
# VGGT/CoTracker3 ViT patch size — BOTH processed dims MUST be multiples of this or
# the VGGT forward raises "height/width is not a multiple of patch" (on-device, W3).
# 518 == 14*37; the height is rounded to a 14-multiple to match VGGT's own crop-mode
# preprocessing (vggt.utils.load_fn `load_and_preprocess_images`).
_PATCH = 14
# Hard MV4D frame ceiling (spec/05 §4); requests clamp max_frames <= this.
_MAX_FRAMES_HARD = 64
# VGGT runs GLOBAL self-attention over all frames at once; MPS has no flash-attention
# kernel, so the scores buffer scales as ``(S · tokens_per_frame)²`` and a 518×518 clip
# OOMs at S=24 (64.8 GiB) on a 36 GB Mac. Cap S so ``S·tokens`` stays under this budget
# (16:9 → ~16 frames, square → ~8) — an on-device finding (decision-log §J.1). Used by
# the MPS combo only; cloud GPUs lift the cap.
_VGGT_TOKEN_BUDGET = 12000


def cap_frames_to_token_budget(
    frames: np.ndarray, budget: int = _VGGT_TOKEN_BUDGET, patch: int = _PATCH
) -> np.ndarray:
    """Uniformly subsample ``frames`` ``[S,3,H,W]`` so ``S·(H//patch)·(W//patch) ≤ budget``.

    Guards the VGGT MPS self-attention OOM (decision-log §J.1). Returns ``frames``
    unchanged when already within budget; otherwise keeps an endpoints-inclusive
    uniform subset. Pure numpy.
    """
    arr = np.asarray(frames)
    if arr.ndim != 4 or arr.shape[0] < 2 or budget <= 0:
        return arr
    s, _, h, w = arr.shape
    tok = max(1, (h // patch) * (w // patch))
    max_s = max(2, int(budget) // tok)
    if s <= max_s:
        return arr
    idx = np.unique(np.linspace(0, s - 1, max_s).round().astype(np.int64))
    logger.info(
        "VGGT frame-budget: S=%d tokens/frame=%d (S·tok=%d > %d) -> %d frames",
        s, tok, s * tok, budget, idx.size,
    )
    return arr[idx]


def _subsample_indices(n_src: int, src_fps: float, target_fps: float, max_frames: int) -> np.ndarray:
    """Frame indices to keep: uniform subsample to ``target_fps``, capped to ``max_frames``.

    First pick every ``stride``-th source frame where ``stride = round(src_fps /
    target_fps)`` (≥1) to hit the target playback rate, then if that still exceeds
    ``max_frames`` uniformly subsample (``np.linspace`` endpoints-inclusive) down to
    the cap. Returns a strictly-increasing int index array (length ≤ max_frames).
    """
    if n_src <= 0:
        return np.empty((0,), dtype=np.int64)

    if src_fps and src_fps > 0 and target_fps > 0 and src_fps > target_fps:
        stride = max(1, int(round(src_fps / target_fps)))
    else:
        stride = 1
    idx = np.arange(0, n_src, stride, dtype=np.int64)

    cap = min(int(max_frames), _MAX_FRAMES_HARD)
    if cap >= 1 and idx.shape[0] > cap:
        pick = np.linspace(0, idx.shape[0] - 1, cap).round().astype(np.int64)
        pick = np.unique(pick)
        idx = idx[pick]
    return idx


def _to_chw_rgb_518(frames_hwc_rgb: list[np.ndarray]) -> np.ndarray:
    """Stack RGB HWC frames → ``[S, 3, H, W]`` uint8, width rescaled to 518.

    Width is set to 518; height is scaled to preserve aspect ratio (rounded, ≥1).
    Resize uses cv2 if available (area interpolation), else a numpy nearest-neighbour
    fallback so a missing cv2 cannot crash an already-decoded (imageio) path.
    """
    if not frames_hwc_rgb:
        return np.empty((0, 3, 0, _TARGET_WIDTH), dtype=np.uint8)

    h0, w0 = frames_hwc_rgb[0].shape[:2]
    # Width -> 518 (== 14*37); height -> aspect-preserving and rounded to a multiple
    # of the ViT patch size (14), matching VGGT crop-mode preprocessing
    # (vggt.utils.load_fn): new_h = round(h0 * (518/w0) / 14) * 14. Capped at 518 so
    # the grid never exceeds VGGT's processed extent (portrait clips are center-fit;
    # the landscape MVP corpus stays well under). BOTH dims divisible by 14 — the
    # VGGT forward rejects any non-patch-multiple side (spec/06 §5 step 4).
    new_w = _TARGET_WIDTH
    aspect_h = h0 * (new_w / float(w0))
    new_h = int(round(aspect_h / _PATCH)) * _PATCH
    new_h = min(max(_PATCH, new_h), _TARGET_WIDTH)

    out = np.empty((len(frames_hwc_rgb), 3, new_h, new_w), dtype=np.uint8)
    try:
        import cv2  # lazy; resize only

        _resize = lambda im: cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)  # noqa: E731
    except Exception:  # pragma: no cover - exercised only without cv2
        def _resize(im: np.ndarray) -> np.ndarray:
            ys = (np.linspace(0, im.shape[0] - 1, new_h)).round().astype(np.int64)
            xs = (np.linspace(0, im.shape[1] - 1, new_w)).round().astype(np.int64)
            return im[ys][:, xs]

    for i, im in enumerate(frames_hwc_rgb):
        rgb = np.ascontiguousarray(im[..., :3]).astype(np.uint8)
        if (rgb.shape[1], rgb.shape[0]) != (new_w, new_h):
            rgb = _resize(rgb)
        out[i] = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW
    return out


def _decode_cv2(video_path: str) -> tuple[list[np.ndarray], float]:
    """Decode all frames via cv2 → (list of HWC RGB uint8 frames, source fps).

    Returns an empty list (not a raise) if the container cannot be opened, so the
    caller can fall back to imageio. cv2 yields BGR — converted to RGB here.
    """
    import cv2  # lazy

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return [], 0.0
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    ok, frame = cap.read()
    while ok:
        if frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ok, frame = cap.read()
    cap.release()
    return frames, src_fps


def _decode_imageio(video_path: str) -> tuple[list[np.ndarray], float]:
    """Fallback decode via imageio[ffmpeg] → (list of HWC RGB uint8 frames, fps)."""
    import imageio.v3 as iio  # lazy

    frames: list[np.ndarray] = []
    src_fps = 0.0
    try:
        meta = iio.immeta(video_path, plugin="pyav")
        src_fps = float(meta.get("fps", 0.0) or 0.0)
    except Exception:  # pragma: no cover - meta is best-effort
        src_fps = 0.0
    for frame in iio.imiter(video_path, plugin="pyav"):
        arr = np.asarray(frame)
        if arr.ndim == 2:  # grayscale -> RGB
            arr = np.repeat(arr[..., None], 3, axis=2)
        frames.append(arr[..., :3].astype(np.uint8))
    return frames, src_fps


def decode_and_subsample(request) -> np.ndarray:
    """Decode ``request.video_path`` → frames ``[S, 3, H, W]`` RGB uint8 (spec/06 §5 step 1).

    Uniformly subsamples to ``request.target_fps`` and caps to
    ``request.max_frames`` (hard ceiling 64, spec/05 §4); width is rescaled to 518.
    Tries cv2 first, then imageio. Raises ``UnsupportedMediaError`` if neither
    backend is installed or the clip yields zero frames.

    Torch-free: returns numpy ``uint8``; the adapter converts to a tensor itself.
    """
    video_path = request.video_path
    target_fps = float(getattr(request, "target_fps", 12.0))
    max_frames = int(getattr(request, "max_frames", 24))

    frames: list[np.ndarray] = []
    src_fps = 0.0
    cv2_err: Exception | None = None
    try:
        frames, src_fps = _decode_cv2(video_path)
    except ImportError as exc:  # cv2 absent — fall through to imageio
        cv2_err = exc
    except Exception as exc:  # noqa: BLE001 - cv2 raised on a bad container; try imageio
        cv2_err = exc
        logger.warning("cv2 decode failed for %s (%s); trying imageio", video_path, exc)

    if not frames:
        try:
            frames, src_fps = _decode_imageio(video_path)
        except ImportError as exc:
            if cv2_err is not None:
                raise UnsupportedMediaError(
