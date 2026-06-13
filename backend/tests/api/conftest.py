"""Shared fixtures for the API job-lifecycle tests (spec/10 §3, T-300..T-310).

All these tests run in FIXTURE MODE (``MAYAVIUS_ADAPTER=fake`` → the no-ML
``FixtureAdapter``, spec/06 §4.6) so they need no torch and no GPU. The
``TestClient`` is context-managed so the lifespan runs (it builds the queue, wires
the adapter, seeds examples).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings


@pytest.fixture()
def client():
    """A context-managed ``TestClient`` against the real app in fixture mode.

    Forces ``settings.adapter = "fake"`` BEFORE the lifespan runs so the adapter
    resolves to ``FixtureAdapter`` regardless of the ambient ``MAYAVIUS_ADAPTER``.
    """
    settings.adapter = "fake"
    from app.main import app

    with TestClient(app) as c:
        yield c
