"""Typed error hierarchy for the reconstruction core (spec/06 §2.2).

`core` DEFINES these errors but MUST NOT map them to HTTP — that mapping lives on
the driving side (`api/errors.py`). Every adapter/pipeline failure raises a
`ReconstructionError` subclass; each carries a stable `code` the API surfaces in
job metadata / the HTTP status table (spec/06 §2.2).

Pure stdlib — no FastAPI, no torch, no numpy.
"""

from __future__ import annotations


class ReconstructionError(Exception):
    """Base. Carries a human message + a stable ``code`` for the API."""

    code: str = "reconstruction_error"


class UnsupportedDeviceError(ReconstructionError):
    """e.g. a CUDA-only adapter asked to run on MPS/CPU."""

    code = "unsupported_device"


class ClipTooLongError(ReconstructionError):
    """Frames/duration exceed the MV4D caps (spec/05 §4)."""

    code = "clip_too_long"

