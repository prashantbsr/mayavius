"""T-300 — `/health` liveness (membership/superset, not exact-dict).

W1.T3 owns the equality->membership rewrite (spec/06 §7 / spec/10 §3): the build
adds ``adapter`` / ``device`` / ``weights_license`` from the resolved adapter's
``info``, which the scaffold's exact-dict assertion would break. This is a logged
rewrite, not a forbidden weakening.

The ``TestClient`` is used as a context manager so the FastAPI lifespan runs (it
populates ``app.state.adapter_info`` that ``/health`` reads). Fixture mode (``fake``)
is forced so the lifespan resolves with no torch on any machine.
"""

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

# Fixture mode (no torch) — set before the lifespan runs inside the context manager.
settings.adapter = "fake"


def test_health_ok() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert {"adapter", "device", "weights_license"} <= body.keys()
