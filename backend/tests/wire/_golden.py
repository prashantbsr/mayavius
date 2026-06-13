"""The single canonical golden ``Scene4D`` literal (W0.T4 / spec/10 §2).

This is THE golden scene: a tiny (<4 KB encoded) reconstruction that exercises
all four MV4D v1 sections (static + dynamic incl. an empty frame + tracks with
mixed visibility + cameras). ``encode_reconstruction(golden_scene())`` is the
committed `backend/tests/fixtures/golden_scene.mv4d` (T-200), and the SAME
literal values are mirrored as ground truth in
`frontend/src/lib/wire/__fixtures__/golden_expected.json` for the TS decoder
(T-202). The byte layout it serializes to is owned by spec/05-data-contract.md
§3 — this module only provides the in-memory literal, never redefines bytes.

Do NOT change these values without regenerating the committed fixture AND the
shared expectation JSON in the same commit (spec/05 §7 versioning rules).
"""

from __future__ import annotations

import numpy as np

from app.core.domain.models import CameraTrack, Scene4D, Tracks


def golden_scene() -> Scene4D:
    """Return THE canonical golden ``Scene4D`` (spec/10 §2 — exact literal)."""
    # static: N_s = 4 (positions f32, colors u8, conf u8 -> HAS_STATIC_CONF)
    static_positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.5]],
        dtype=np.float32,
    )
    static_colors = np.array(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]], dtype=np.uint8
    )
    static_conf = np.array([200, 180, 160, 255], dtype=np.uint8)

    # dynamic: T = 3, counts [2, 0, 1] (frame 1 empty -> pointCount==0)
    dynamic_positions = [
        np.array([[0.2, 0.2, 0.2], [0.8, 0.2, 0.2]], dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
        np.array([[0.5, 0.9, 0.1]], dtype=np.float32),
    ]
    dynamic_colors = [
        np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
        np.empty((0, 3), dtype=np.uint8),
        np.array([[70, 80, 90]], dtype=np.uint8),
    ]

    # tracks: M = 2, T = 3 (colors u8 -> HAS_TRACK_COLOR)
    track_positions = np.array(
        [
            [[0.1, 0.1, 0.1], [0.15, 0.12, 0.1], [0.2, 0.14, 0.1]],
            [[0.9, 0.9, 0.9], [0.85, 0.88, 0.9], [0.8, 0.86, 0.9]],
        ],
        dtype=np.float32,
    )
    track_visibility = np.array(
        [[True, True, False], [False, True, True]], dtype=bool
    )
    track_colors = np.array([[200, 10, 10], [10, 200, 10]], dtype=np.uint8)
    tracks = Tracks(
        positions=track_positions,
