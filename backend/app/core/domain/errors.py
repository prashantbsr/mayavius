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


class UnsupportedMediaError(ReconstructionError):
    """Undecodable upload / not a video."""

    code = "unsupported_media"


class ModelLoadError(ReconstructionError):
    """Weights download / model init failed."""

    code = "model_load_failed"


class InferenceError(ReconstructionError):
    """Runtime failure (e.g. an MPS op gap / OOM mid-run)."""

    code = "inference_failed"


class EmptyReconstructionError(ReconstructionError):
    """Reconstruction produced 0 usable points (culling removed everything)."""

    code = "empty_reconstruction"
