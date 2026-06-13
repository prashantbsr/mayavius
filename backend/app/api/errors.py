"""The SINGLE error -> HTTP mapping (spec/06 §2.2). DRIVING side.

The pure core defines the ``ReconstructionError`` hierarchy (``core/domain/errors.py``)
but MUST NOT map it to HTTP. This module owns that mapping table, used by the POST
handler's synchronous validation. (Async job failures keep their HTTP 200 + a
``status:"failed"`` body — only sync validation at submit returns these 4xx/5xx.)
"""

from __future__ import annotations

from app.core.domain.errors import (
    ClipTooLongError,
    UnsupportedDeviceError,
    UnsupportedMediaError,
)


def http_status_for(err: Exception) -> int:
    """Map a ``ReconstructionError`` to its synchronous-validation HTTP status (06 §2.2).

    | UnsupportedMediaError  -> 415 |
    | ClipTooLongError       -> 413 |
    | UnsupportedDeviceError -> 501 |
    | anything else          -> 500 |
    """
    if isinstance(err, UnsupportedMediaError):
        return 415
    if isinstance(err, ClipTooLongError):
        return 413
    if isinstance(err, UnsupportedDeviceError):
