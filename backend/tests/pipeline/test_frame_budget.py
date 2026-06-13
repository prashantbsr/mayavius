"""VGGT MPS self-attention frame-budget guard (decision-log §J.1).

A 518×518 clip OOMs VGGT's global self-attention at S=24 on a 36 GB Mac (no MPS
flash-attention). ``cap_frames_to_token_budget`` uniformly subsamples so
``S·(H//14)·(W//14)`` stays under budget. Pure numpy — runs in the no-ML CI.
"""

from __future__ import annotations

import numpy as np

from app.pipeline.decode import cap_frames_to_token_budget


def _frames(s: int, h: int, w: int) -> np.ndarray:
    return np.zeros((s, 3, h, w), dtype=np.uint8)


def test_square_clip_capped_under_budget():
    # 518×518 => 37×37 = 1369 tokens/frame; budget 12000 => max ~8 frames.
    out = cap_frames_to_token_budget(_frames(24, 518, 518), budget=12000)
    s, _, h, w = out.shape
    tok = (h // 14) * (w // 14)
    assert s < 24
    assert s * tok <= 12000
    assert (h, w) == (518, 518)  # only frames dropped, never resized


def test_landscape_clip_allows_more_frames():
    # 518×280 (16:9) => 37×20 = 740 tokens/frame; 16 frames = 11840 <= 12000 -> kept.
    out = cap_frames_to_token_budget(_frames(16, 280, 518), budget=12000)
    assert out.shape[0] == 16


def test_within_budget_unchanged_and_endpoints_kept():
    fr = _frames(6, 280, 518)
    out = cap_frames_to_token_budget(fr, budget=12000)
    assert out.shape[0] == 6  # already under budget
    # A tiny S is never dropped below 2.
    assert cap_frames_to_token_budget(_frames(2, 518, 518), budget=1).shape[0] == 2
