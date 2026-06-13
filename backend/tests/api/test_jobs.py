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

    statuses = {b["status"] for b in observed}
    assert statuses <= {"queued", "running", "done"}, statuses

    # progress monotonically non-decreasing in [0, 1], ending at 1.0.
    progresses = [b["progress"] for b in observed]
    for p in progresses:
        assert 0.0 <= p <= 1.0
    for a, b in zip(progresses, progresses[1:]):
        assert b >= a, progresses
    assert progresses[-1] == 1.0

    terminal = observed[-1]
    assert terminal["status"] == "done"
    assert terminal["weights_license"], terminal
    assert terminal["adapter_id"], terminal


# --- T-304 ---------------------------------------------------------------------
def test_job_result_is_mv4d(client) -> None:
    job_id = client.post("/jobs", files=_multipart()).json()["job_id"]
    _poll_until_terminal(client, job_id)

    r = client.get(f"/jobs/{job_id}/result")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    assert r.content[:4] == b"MV4D"
    assert "immutable" in r.headers["cache-control"]

    scene = decode(r.content)  # chains T-100
    assert scene.frame_count >= 1


# --- T-305 ---------------------------------------------------------------------
def test_unknown_job_404(client) -> None:
    assert client.get("/jobs/does-not-exist").status_code == 404


# --- T-306 ---------------------------------------------------------------------
def test_clip_frame_cap(client) -> None:
    job_id = client.post("/jobs", files=_multipart()).json()["job_id"]
    _poll_until_terminal(client, job_id)
    scene = decode(client.get(f"/jobs/{job_id}/result").content)
    assert scene.frame_count <= 64


# --- T-307 ---------------------------------------------------------------------
def test_upload_rejections(client) -> None:
    # Oversize upload -> 413 (set a tiny cap, POST a slightly larger video body).
    original = settings.max_upload_mb
    settings.max_upload_mb = 1  # 1 MB cap
    try:
        big = b"\x00" * (1 * 1024 * 1024 + 1024)  # just over 1 MB
        r = client.post("/jobs", files=_multipart(body=big))
        assert r.status_code == 413
    finally:
        settings.max_upload_mb = original

    # Non-video content-type -> 415.
    r = client.post("/jobs", files={"clip": ("x.txt", b"not a video", "text/plain")})
    assert r.status_code == 415

    # Missing clip field -> 422 (FastAPI validation).
    r = client.post("/jobs", files={})
    assert r.status_code == 422


# --- T-308 ---------------------------------------------------------------------
def test_sse_progress_stream(client) -> None:
    import json

    job_id = client.post("/jobs", files=_multipart()).json()["job_id"]

    with client.stream("GET", f"/jobs/{job_id}/stream") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        # NOT gzip (SSE safety, C7 — no GZipMiddleware).
        assert r.headers.get("content-encoding", "").lower() != "gzip"

        events: list[dict] = []
        for line in r.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:"):].strip()))

    assert len(events) >= 1
    progresses = [e["progress"] for e in events]
    for a, b in zip(progresses, progresses[1:]):
        assert b >= a, progresses
    # Last event is terminal done (tolerate a fast job emitting only the terminal).
    assert events[-1]["status"] == "done"
    assert events[-1]["progress"] == 1.0


# --- T-310 — adapter-contract suite (parametrized) -----------------------------
# FixtureAdapter always; real model adapters are added under @pytest.mark.mps when
# their reconstruct() lands (W3) — skipped now.
def _fixture_adapter():
    from app.adapters.fixture_adapter import FixtureAdapter

    return FixtureAdapter(settings)


@pytest.mark.parametrize("adapter_factory", [_fixture_adapter], ids=["fixture"])
def test_adapter_contract(adapter_factory) -> None:
    from app.core.domain.models import Scene4D

    adapter = adapter_factory()
    info = adapter.info
    assert info.weights_license, "weights_license must be populated (D2)"

    req = ReconstructionRequest(video_path="(ignored)", max_frames=8, target_fps=12.0)
    scene = adapter.reconstruct(req)
    assert isinstance(scene, Scene4D)

    # Honors caps (shapes within spec/05 §4).
    assert 1 <= scene.frame_count <= 64
    assert scene.static_positions.shape[1] == 3
    assert scene.static_positions.shape[0] <= 150_000
    assert len(scene.dynamic_positions) == scene.frame_count
    for frame in scene.dynamic_positions:
        assert frame.shape[0] <= 20_000
        assert frame.shape[1] == 3 if frame.shape[0] else True
    if scene.tracks is not None:
        assert scene.tracks.positions.shape[0] <= 4_096
        assert scene.tracks.positions.shape[1] == scene.frame_count
        assert scene.tracks.positions.shape[2] == 3
    # Frame cap is honored relative to the request (FixtureAdapter takes min).
    assert scene.frame_count <= req.max_frames


# The OPTIONAL model adapters are HONEST STUBS (W3.T5, spec/06 §4.3/§4.4/§4.5/§4.7):
# on the Mac's local devices ("mps"/"cpu") their reconstruct() raises
# UnsupportedDeviceError (NOT a silent fall-back), naming the CUDA/no-MPS/unverified
# constraint + pointing at the cloud-GPU deploy (spec/11). This is the W1→W3.T5
# cutover the suite header (spec/10 T-310) anticipated — the W1 NotImplementedError
# placeholder is replaced by the device-refusing honest stub. The full per-adapter
# contract (message content, the @pytest.mark.gpu on-device test) lives in
# tests/adapters/test_optional_adapters.py.
#
# The default-combo halves — VggtAdapter + CoTracker3Adapter — landed their real
# reconstruct() in W3.T2/T3, so they MOVED OUT of this "optional" list (their module
# import stays torch-free — asserted by test_default_adapters_module_import_is_torch_free
# below — and their on-device reconstruct() is exercised by the gated T-310 / T-510
# mps suite).
@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("app.adapters.spatialtracker_adapter", "SpatialTrackerV2Adapter"),
        ("app.adapters.pi3_adapter", "Pi3Adapter"),
        ("app.adapters.open_d4rt_adapter", "OpenD4RTAdapter"),
    ],
)
def test_optional_adapters_refuse_mps(module_name, class_name) -> None:
    import importlib

    from app.core.domain.errors import UnsupportedDeviceError

    cls = getattr(importlib.import_module(module_name), class_name)
    adapter = cls(settings)
    # info is cheap + license-tagged even though reconstruct refuses the Mac device.
    assert adapter.info.weights_license
    assert adapter.info.mps_capable is False
    # Default device is "mps" (the Mac local path) — the honest stub refuses it.
    req = ReconstructionRequest(video_path="(ignored)", max_frames=4)
    with pytest.raises(UnsupportedDeviceError):
        adapter.reconstruct(req)


# The two default-combo adapters (W3.T2/T3) MUST stay importable WITHOUT torch — the
# module import (registry/info path) never imports torch/vggt/cotracker; only
# reconstruct()/run_geometry()/run_tracks() do, lazily (T-130, spec/06 §4). This is
# the no-ML guarantee that the API can advertise capabilities with zero ML deps.
@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("app.adapters.vggt_adapter", "VggtAdapter"),
        ("app.adapters.cotracker3_adapter", "CoTracker3Adapter"),
    ],
)
def test_default_adapters_module_import_is_torch_free(module_name, class_name) -> None:
    import subprocess
    import sys

    # Clean subprocess so torch imported by another test cannot mask a leak here.
    code = (
        f"import importlib, sys; m = importlib.import_module({module_name!r}); "
        f"cls = getattr(m, {class_name!r}); "
        "info = cls(None).info; "
        "leaked = {k for k in sys.modules "
        "if k.split('.')[0] in {'torch', 'vggt', 'cotracker_utils'} "
        "or k == 'cotracker'}; "
        "assert not leaked, leaked; "
        "assert info.weights_license"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, (
        f"{module_name} import pulled in torch/vggt/cotracker (lazy-import violation, "
        f"T-130).\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )


# --- T-310 / T-510 (gated) — the REAL default-combo adapters on MPS --------------
# Skipped unless MAYAVIUS_RUN_MPS_SMOKE=1 AND torch+mps are available (spec/10 §5).
# torch / vggt / cotracker are imported INSIDE the test bodies so collection works
# with zero ML deps. The combo runs VGGT once and feeds depth to the CoTracker3 lift;
# the full smoke (wall-time / peak-mem record, ≥1 static point, ≥1 track, encodes
# within caps) lives in T-510 (backend/tests/mps/test_mps_smoke.py, on-device).
def _mps_smoke_enabled() -> bool:
    import os

    if os.environ.get("MAYAVIUS_RUN_MPS_SMOKE") != "1":
        return False
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


@pytest.mark.mps
@pytest.mark.skipif(
    not _mps_smoke_enabled(),
    reason="needs MAYAVIUS_RUN_MPS_SMOKE=1 + torch MPS (spec/10 §5); on-device T-510",
)
def test_default_combo_adapter_contract_mps() -> None:
    """T-310 (gated): the real combo returns a caps-honoring mayavius-space Scene4D."""
    from pathlib import Path

    from app.adapters.combo import VggtCoTracker3Adapter
    from app.core.domain.models import Scene4D

    # A bundled ≤3s sample clip (spec/10 §6) decoded by the combo itself.
    sample = (
        Path(__file__).resolve().parents[3] / "assets" / "samples" / "static-scene.mp4"
    )
    if not sample.exists():
        pytest.skip(f"no on-device sample clip at {sample} (corpus is W4)")
