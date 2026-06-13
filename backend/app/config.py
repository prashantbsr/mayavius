"""Backend settings (env-driven). All vars are MAYAVIUS_-prefixed; see .env.example."""

from __future__ import annotations

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
