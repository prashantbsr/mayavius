#!/usr/bin/env python3
"""Deterministically generate frontend/public/og.png — the branded share card.

No third-party deps (no Pillow): a hand-rolled 1200x630 RGB PNG via zlib only,
drawn with a tiny built-in 5x7 pixel font. Dark background, "mayavius" wordmark,
and the one-line tagline. Re-run to regenerate byte-stably:

    python3 frontend/scripts/gen_og.py

Output: a valid PNG (8-bit RGB, color type 2) at frontend/public/og.png.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

W, H = 1200, 630
BG = (0x0A, 0x0A, 0x12)          # near-black, slight blue
FG = (0xE8, 0xEA, 0xF6)          # off-white wordmark
SUB = (0x8A, 0x90, 0xB8)         # muted lavender-grey tagline
ACCENT = (0x6E, 0x8B, 0xFF)      # accent rule

# 5x7 uppercase/lowercase-agnostic pixel font. Each glyph = 7 rows of 5 bits.
FONT: dict[str, list[str]] = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "11110", "10001", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "11110", "10000", "10000", "10000", "11111"],
    "F": ["11111", "10000", "11110", "10000", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "11111", "10001", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ",": ["00000", "00000", "00000", "00000", "00100", "00100", "01000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00110", "00110"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


def new_canvas() -> list[list[tuple[int, int, int]]]:
    return [[BG for _ in range(W)] for _ in range(H)]


def put(px, x: int, y: int, c) -> None:
    if 0 <= x < W and 0 <= y < H:
        px[y][x] = c


def rect(px, x0: int, y0: int, w: int, h: int, c) -> None:
    for yy in range(y0, y0 + h):
        for xx in range(x0, x0 + w):
            put(px, xx, yy, c)


def text(px, s: str, x: int, y: int, scale: int, c) -> int:
    """Draw `s` (uppercased into the font) at (x,y); return the x cursor end."""
    cx = x
    for ch in s.upper():
        glyph = FONT.get(ch, FONT[" "])
        for ry, row in enumerate(glyph):
            for rxi, bit in enumerate(row):
                if bit == "1":
                    rect(px, cx + rxi * scale, y + ry * scale, scale, scale, c)
        cx += (5 + 1) * scale  # 1-cell letter spacing
    return cx


def text_width(s: str, scale: int) -> int:
    return len(s) * (5 + 1) * scale - scale


def png_bytes(px) -> bytes:
    raw = bytearray()
    for row in px:
        raw.append(0)  # filter type 0
