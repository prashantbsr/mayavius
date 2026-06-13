"""Backend settings (env-driven). All vars are MAYAVIUS_-prefixed; see .env.example.

Spec: 06-backend-spec.md §8 (config fields + path anchoring). The runtime dirs
(RESULT_DIR / UPLOAD_DIR / SAMPLES_DIR) are resolved ABSOLUTELY so a `cd backend`
run/test command cannot misplace them, and created at import time.

Path anchoring (06 §6/§8): ``parents[1]`` from ``app/config.py`` is ``backend/``;
``parents[2]`` is the repo root. ``RESULT_DIR`` / ``UPLOAD_DIR`` anchor to
``backend/``; ``SAMPLES_DIR`` (the example corpus) anchors to the repo root.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MAYAVIUS_", env_file=".env", extra="ignore"
    )

    #: Allowed CORS origins for the local frontend.
    cors_origins: list[str] = ["http://localhost:3000"]
    #: Short-clip cap (frames). Long video is the #1 scope risk (handover §4.6, §7).
    max_clip_frames: int = 24
    #: Upload size guard (MB).
    max_upload_mb: int = 64


settings = Settings()
