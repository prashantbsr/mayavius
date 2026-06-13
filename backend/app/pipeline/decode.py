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
