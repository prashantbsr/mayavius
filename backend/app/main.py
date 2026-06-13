"""FastAPI application entry point (driving-side adapter).

Local run (from backend/, venv active):
    uvicorn app.main:app --reload --port 8000

The lifespan (spec/06 §6/§7) resolves the adapter from config ONCE, builds the
``ReconstructionService`` + ``JobQueue``, stashes them on ``app.state``, and seeds any
committed example MV4D blobs from the repo-root corpus. No ``GZipMiddleware`` is added
(SSE safety, C7).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.adapters.registry import build_adapter
from app.api.routes import jobs
from app.config import RESULT_DIR, SAMPLES_DIR, UPLOAD_DIR, settings
from app.core.services.reconstruction_service import ReconstructionService
from app.jobs.queue import JobQueue


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Resolve the adapter, wire the service + queue, seed example blobs (spec/06 §6)."""
    adapter = build_adapter(settings)  # unknown id -> RuntimeError (500 at startup, §2.2)
    service = ReconstructionService(adapter, conf_thresh=settings.conf_thresh)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    queue = JobQueue(service, str(RESULT_DIR))
    app.state.queue = queue
    app.state.adapter_info = adapter.info

    # Seed pre-baked example results (slug = filename stem). W1: no committed *.mv4d
    # under assets/samples → seeds nothing, no error (spec/06 §6).
    if SAMPLES_DIR.is_dir():
        for path in sorted(SAMPLES_DIR.glob("*.mv4d")):
            queue.seed_example(path.stem, str(path))

    yield


app = FastAPI(title="mayavius-backend", version="0.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    """Liveness probe — used by the smoke test and local dev."""
    return {"status": "ok"}
