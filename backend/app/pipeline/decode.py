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
