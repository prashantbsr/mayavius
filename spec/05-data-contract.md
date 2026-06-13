# 05 — Data Contract: the `MV4D` v1 binary wire format

**This file is the single source of truth for the backend↔frontend point
payload.** Two implementations, one spec:
- backend encoder: [`backend/app/wire/encoder.py`](../backend/app/wire/encoder.py)
- frontend decoder: [`frontend/src/lib/wire/decoder.ts`](../frontend/src/lib/wire/decoder.ts)

They MUST stay **byte-for-byte compatible**. Change the format → change both →
bump the version byte → update this file in the same commit. JSON for point
payloads is forbidden (handover §4.5). The same layout is referenced by
[06-backend-spec.md](06-backend-spec.md) (encoder + domain model) and
[07-frontend-spec.md](07-frontend-spec.md) (decoder + render attributes); if any
of the three disagree, **this file wins**.

---

## 1. Why this shape

The signature visual is a **stable static background** with **moving objects
animating as point clouds trailing motion ribbons** (handover §2; risk #3). The
format mirrors that directly with four independent sections:

1. **STATIC_POINTS** — the background cloud, sent **once**, rendered every frame.
2. **DYNAMIC_FRAMES** — only the moving foreground points, one variable-size set
   per timestep.
3. **TRACKS** — `M` 3D trajectories (one polyline per tracked point) → the ribbons.
4. **CAMERAS** — per-frame pose + intrinsics (drives the "as-shot" camera and
   bullet-time orbit reference).

Design constraints baked in:
- **Compactness:** positions are **16-bit quantized** against a global AABB
  (the `.splat` trick); colors are `u8` RGB. JSON is banned.
- **Zero-copy decode:** every section is a contiguous, 8-byte-aligned,
  structure-of-arrays blob so the decoder builds `TypedArray` **views over the
  original `ArrayBuffer`** with no per-point copying. Quantized positions stay
  `Uint16Array` and are **dequantized in the vertex shader** (Path 1), keeping
  CPU work and memory minimal — this is what makes a ~2 s load instead of ~40 s.
- **Static-first rendering:** a section directory lets the viewer locate and draw
  STATIC_POINTS before finishing the rest.
- **Forward compatibility:** unknown section `kind`s are skipped; a `flags` byte
  and reserved fields allow additive growth without a version bump.

---

## 2. Conventions

- **Endianness:** little-endian for all scalars and arrays. All target platforms
  are LE; the decoder reads header scalars with `DataView(..., littleEndian=true)`
  and asserts the magic; bulk `TypedArray` views are host-endian (LE assumption
  documented and asserted).
- **Alignment:** every section payload begins at an **8-byte-aligned** absolute
  offset (zero-pad the preceding bytes). **Within** a section, sub-arrays are packed
  **tightly — no intra-section padding** (each starts where the previous ends). This
  is sufficient because the fixed sub-array order in §3.4–§3.7 places every typed
  (`u16`/`u32`/`f32`) sub-array either **first** in its section (so the 8-aligned
  section start covers it) or **after an even-byte-length** array, and only `u8`
  sub-arrays (which need no alignment) ever follow an odd-length array. Therefore
  every typed sub-array begins at a dtype-aligned absolute offset and a zero-copy
  `new Uint16Array(buffer, off, len)` / `Float32Array` / `Uint32Array` view is always
  valid. **Encoders MUST NOT insert intra-section padding** (it would break the golden
  fixture's byte-stability and the decoder's fixed offsets).
- **Coordinate system (mayavius world space):** right-handed, **+X right, +Y up,
  −Z forward** (Three.js camera default looks down −Z). The scene is normalized to
  fit inside the AABB. The **backend adapter is responsible** for transforming each
  model's native output into this convention *before* encoding (spec/06). The
  frontend assumes this convention with no further transform.
- **Quantization (positions):** per axis, `q = round((p − aabbMin) / (aabbMax −
  aabbMin) * 65535)`, clamped to `[0, 65535]`; dequant `p = aabbMin + q/65535 *
  (aabbMax − aabbMin)`. **`round` is round-half-to-even** (Python `round` /
  `numpy.rint` semantics) — any non-Python encoder (e.g. a future TS encoder for
  the reverse conformance vector) MUST use the same mode, since JS `Math.round`
  (half-up) would break byte parity on exact half-grid points (spec/10 T-203).
  Degenerate axis (`aabbMax==aabbMin`) → all `q=0`, dequant yields `aabbMin`.
  **Working precision = `float32` end-to-end** (same byte-parity rationale as the
  rounding mode): cast positions **and** the AABB to `float32` first, and use the
  **same f32 `aabbMin`/`aabbMax` written to the 24-byte header** as the divisor (not
  an f64 pre-rounding value), so the decoder reconstructs from the identical f32
  range — i.e. `q = numpy.rint(((p_f32 − aabbMin_f32)/(aabbMax_f32 − aabbMin_f32) *
  65535)).astype(uint16)` clamped to `[0,65535]`. Any independent encoder (the TS
  reverse vector, a future client encoder) MUST use f32 working precision, or it can
  diverge by 1 ULP at AABB-boundary points and break byte parity.
- **Color:** linear-ish sRGB `u8` per channel `[0,255]`; the shader treats it as
  sRGB and converts as needed.
- **Time:** frame index `t ∈ [0, T)`. Playback maps store `time ∈ [0,1]` →
  `t = round(time*(T−1))`.

---

