"""API job-lifecycle integration tests (spec/10 §3, T-300..T-310).

FastAPI ``TestClient`` (sync, via ``httpx``) drives the real ASGI app with the no-ML
``FixtureAdapter`` (``MAYAVIUS_ADAPTER=fake``, spec/06 §4.6) — no torch. Covers the
async job model end to end: POST → poll/stream → result, plus upload rejections and
the parametrized adapter-contract suite.

T-300 (health membership) lives in ``tests/test_health.py``.
"""

from __future__ import annotations

import time

import pytest

from app.config import settings
from app.core.domain.models import ReconstructionRequest
from app.wire.decoder import decode

# A tiny in-memory "video" — bytes are ignored in fixture mode, but POST /jobs
# validates content-type + size FIRST, so a real-ish body is provided.
_TINY_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


def _multipart(content_type: str = "video/mp4", body: bytes = _TINY_MP4):
    return {"clip": ("clip.mp4", body, content_type)}


def _poll_until_terminal(client, job_id: str, *, timeout_s: float = 10.0):
