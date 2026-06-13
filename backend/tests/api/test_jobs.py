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
    """Poll ``GET /jobs/{id}`` until DONE/FAILED, collecting every observed body."""
    deadline = time.time() + timeout_s
    observed: list[dict] = []
    while time.time() < deadline:
        r = client.get(f"/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        observed.append(body)
        if body["status"] in ("done", "failed"):
            return observed
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached a terminal state: {observed[-3:]}")


# --- T-302 ---------------------------------------------------------------------
def test_job_submit_returns_id(client) -> None:
    r = client.post("/jobs", files=_multipart())
    assert r.status_code == 202
    body = r.json()
    assert "job_id" in body and body["job_id"]
    assert body["status"] == "queued"
    assert body["poll"] == f"/jobs/{body['job_id']}"
    assert body["stream"] == f"/jobs/{body['job_id']}/stream"
    assert body["result"] == f"/jobs/{body['job_id']}/result"


# --- T-303 ---------------------------------------------------------------------
def test_job_lifecycle_poll(client) -> None:
    job_id = client.post("/jobs", files=_multipart()).json()["job_id"]
    observed = _poll_until_terminal(client, job_id)
