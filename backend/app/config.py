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
    #: Active adapter id resolved by ``build_adapter`` at startup (06 §4.6). "fake"
    #: selects the no-ML FixtureAdapter (fixture mode, Waves 1-2 + e2e).
    adapter: str = "vggt+cotracker3"
    #: Inference device passed into ``ReconstructionRequest.device``.
    device: str = "mps"
    #: Subsample target + MV4D playback fps (06 §5 step 1 / spec/05 header).
    target_fps: float = 12.0
    #: Static/dynamic split percentile threshold (06 §5 step 5).
    motion_thresh: float = 0.95
    #: Confidence-cull floor for static points (06 §5 step 6).
    conf_thresh: float = 0.5
    #: Per-job inference deadline (s). A stuck MPS op or a cold multi-GB weight
    #: download is FAILED (not left hanging) after this many seconds, so an
    #: awaiting SSE client always sees a terminal event (spec/06 §6 reliability).
    job_timeout_s: int = 180
    #: VGGT checkpoint id (commercial swap = facebook/VGGT-1B-Commercial, 06 §4.1).
    vggt_weights: str = "facebook/VGGT-1B"
    #: Result blob store, resolved ABSOLUTE to backend/outputs (06 §8); served by /result.
    result_dir: str = "outputs"


settings = Settings()

# --- Absolute runtime dirs (06 §6/§8). parents[1] = backend/, parents[2] = repo root. ---
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Worker blob store (backend/outputs by default); served immutable by /result.
RESULT_DIR: Path = _BACKEND_ROOT / settings.result_dir
#: Streamed-upload landing dir (backend/uploads); gitignored.
UPLOAD_DIR: Path = _BACKEND_ROOT / "uploads"
#: Repo-root example corpus the lifespan seeds from (assets/samples/*.mv4d, 06 §6).
SAMPLES_DIR: Path = _REPO_ROOT / "assets" / "samples"

# Create the writable dirs at import/startup. SAMPLES_DIR is read-only corpus —
# create it too so a fresh clone with no committed samples still globs cleanly.
RESULT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
