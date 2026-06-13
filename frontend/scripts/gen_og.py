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
        for (r, g, b) in row:
            raw += bytes((r, g, b))
    comp = zlib.compress(bytes(raw), 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)  # 8-bit, RGB
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", comp) + chunk(b"IEND", b"")


def main() -> None:
    px = new_canvas()

    # Wordmark, centered.
    word = "mayavius"
    wm_scale = 18
    wm_w = text_width(word, wm_scale)
    wm_x = (W - wm_w) // 2
    wm_y = 210
    text(px, word, wm_x, wm_y, wm_scale, FG)

    # Accent rule under the wordmark.
    rule_w = 360
    rect(px, (W - rule_w) // 2, wm_y + 7 * wm_scale + 34, rule_w, 6, ACCENT)

    # Tagline, centered, two lines (kept within ~80px side margins at scale 6).
    line1 = "DROP IN A VIDEO - ORBIT A LIVE"
    line2 = "4D RECONSTRUCTION IN YOUR BROWSER"
    tl_scale = 5
    ty = wm_y + 7 * wm_scale + 84
    text(px, line1, (W - text_width(line1, tl_scale)) // 2, ty, tl_scale, SUB)
    text(
        px,
        line2,
        (W - text_width(line2, tl_scale)) // 2,
        ty + 7 * tl_scale + 18,
        tl_scale,
        SUB,
    )

    out = Path(__file__).resolve().parents[1] / "public" / "og.png"
    out.write_bytes(png_bytes(px))
    print(f"wrote {out} ({out.stat().st_size} bytes, {W}x{H})")


if __name__ == "__main__":
    main()
