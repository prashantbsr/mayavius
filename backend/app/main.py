"""FastAPI application entry point (driving-side adapter).

Local run (from backend/, venv active):
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import jobs
from app.config import settings

app = FastAPI(title="mayavius-backend", version="0.0.0")

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
