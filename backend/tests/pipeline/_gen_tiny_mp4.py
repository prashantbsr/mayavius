"""Generate a small REAL decodable ``tiny.mp4`` for the decode test (cv2-gated).

The committed ``frontend/e2e/fixtures/tiny.mp4`` is a 164-byte header-only stub
(synthetic, no real frame data) that ``cv2.VideoCapture`` cannot decode. The decode
test needs an actually-decodable clip, so it uses a **backend copy** generated here
on demand via ``cv2.VideoWriter`` (the spec allows "tiny.mp4 (or backend copy)").

Importing this module imports nothing heavy; ``ensure_tiny_mp4`` imports cv2 lazily
and returns ``None`` if cv2 cannot write mp4 (the test then ``importorskip``s / skips).
No torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Source dims chosen so width != 518 (exercises the rescale) and S is small.
_W, _H, _FPS, _N = 96, 72, 10, 12


def ensure_tiny_mp4(out_path: str | Path) -> Path | None:
    """Write a small real mp4 at ``out_path`` (idempotent). Returns the path or None.

    Returns ``None`` if cv2 is unavailable or cannot open an mp4 writer / read the
    result back (so the caller skips cleanly rather than failing).
    """
    out = Path(out_path)
    if out.exists() and out.stat().st_size > 256:
