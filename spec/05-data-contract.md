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

## 3. Byte layout

### 3.1 Header — 24 bytes, offset 0
| off | type | field | value / meaning |
|----:|------|-------|-----------------|
| 0 | `char[4]` | `magic` | `"MV4D"` (0x4D 0x56 0x34 0x44) |
| 4 | `u8` | `version` | `1` |
| 5 | `u8` | `flags` | bit0 `HAS_STATIC`, bit1 `HAS_DYNAMIC`, bit2 `HAS_TRACKS`, bit3 `HAS_CAMERAS`, bit4 `HAS_STATIC_CONF`, bit5 `HAS_TRACK_COLOR`; bits 6–7 reserved=0 |
| 6 | `u8` | `posBits` | `16` (only value supported in v1) |
| 7 | `u8` | `sectionCount` | number of section-directory entries |
| 8 | `u16` | `frameCount` (`T`) | timesteps; MVP cap **T ≤ 64** |
| 10 | `u16` | `reserved0` | `0` |
| 12 | `f32` | `fps` | playback frame rate of the reconstructed sequence |
| 16 | `u32` | `reserved1` | `0` |
| 20 | `u32` | `reserved2` | `0` |

### 3.2 AABB block — 24 bytes, offset 24
| off | type | field |
|----:|------|-------|
| 24 | `f32[3]` | `aabbMin` (x,y,z) |
| 36 | `f32[3]` | `aabbMax` (x,y,z) |

### 3.3 Section directory — `sectionCount × 16` bytes, offset 48
Each entry (16 bytes):
| off | type | field | meaning |
|----:|------|-------|---------|
| +0 | `u32` | `kind` | `1`=STATIC_POINTS, `2`=DYNAMIC_FRAMES, `3`=TRACKS, `4`=CAMERAS |
| +4 | `u32` | `byteOffset` | absolute offset of this section's payload (8-aligned) |
| +8 | `u32` | `byteLength` | payload length in bytes |
| +12 | `u32` | `count` | semantic count (see each section) |

Payloads follow the directory, each at its `byteOffset`. The decoder MUST use the
directory offsets (not assume packing order) and MUST skip unknown `kind`s.
**Canonical encoder order (for golden-fixture byte-stability):** the encoder MUST
write directory entries **and** their payload blocks in **ascending `kind` order**
(STATIC=1, DYNAMIC=2, TRACKS=3, CAMERAS=4), skipping absent sections; the first
payload begins at the 8-aligned offset immediately after the directory. (The decoder
stays order-agnostic; this rule only makes the committed `golden_scene.mv4d`
reproducible across independent builders — matching the §1 static-first rationale and
the §3.4–§3.7 listing order.)

**Flags vs directory (presence):** the **section directory is authoritative** for
which sections exist. The encoder MUST set `flags` bits 0–3 (`HAS_STATIC`/
`HAS_DYNAMIC`/`HAS_TRACKS`/`HAS_CAMERAS`) iff the corresponding section is present
in the directory; the decoder MAY treat bits 0–3 as a fast-path hint and MUST fall
back to the directory. Bits 4–5 (`HAS_STATIC_CONF`/`HAS_TRACK_COLOR`) are different
— they gate **optional sub-arrays** within a present section and MUST be honored by
both sides. (spec/10 T-102 asserts bits 0–3 match the present sections.)

### 3.4 STATIC_POINTS payload (`kind=1`, `count = N_s`)
Sub-arrays in order, each starting where the previous ends:
1. `positions` : `u16[N_s * 3]` — quantized (x,y,z) per point.
2. `colors` : `u8[N_s * 3]` — RGB per point.
3. `conf` : `u8[N_s]` — **present iff** `HAS_STATIC_CONF`; per-point confidence 0–255.

### 3.5 DYNAMIC_FRAMES payload (`kind=2`, `count = T`)
1. `frameDir` : `u32[T * 2]` — per frame `{ startPoint, pointCount }`, where
   `startPoint` is the cumulative point index into this section's `positions`.
2. `positions` : `u16[ (Σ pointCount) * 3 ]` — all frames concatenated.
3. `colors` : `u8[ (Σ pointCount) * 3 ]`.
Frame `t`'s points = `positions` view sliced `[startPoint*3, (startPoint+pointCount)*3)`.
A frame with `pointCount=0` is valid (a frame with no moving points).

### 3.6 TRACKS payload (`kind=3`, `count = M`)
Each track spans all `T` frames (occlusion handled by the visibility bit).
1. `positions` : `u16[M * T * 3]` — track `m`, frame `t` at index `((m*T)+t)*3`.
2. `visibility`: `u8[ ceil(M*T / 8) ]` — packed bitmask; bit `(m*T + t)` set ⇒
   point visible at that frame (LSB-first within each byte:
   `byte[i>>3] & (1 << (i & 7))`). Invisible = a gap in the ribbon.
3. `colors` : `u8[M * 3]` — **present iff** `HAS_TRACK_COLOR`; per-track RGB.

### 3.7 CAMERAS payload (`kind=4`, `count = T`)
1. `poses` : `f32[T * 7]` — per frame **camera-to-world** `{ qx, qy, qz, qw,
   tx, ty, tz }` (unit quaternion + translation, in mayavius world space).
2. `intrinsics` : `f32[T * 4]` — per frame `{ fx, fy, cx, cy }` **normalized**
   (focal lengths in units of image width/height; principal point in `[0,1]`),
   resolution-independent.

---

## 4. MVP caps (part of the contract)

Bound payload size so loads stay fast and shareable links stay small:

| Quantity | Symbol | MVP cap | Note |
|----------|--------|---------|------|
| Frames (timesteps) | `T` | **≤ 64** (default subsample target 32–48) | handover §4.6 |
| Static points | `N_s` | **≤ 150 000** | confidence-culled |
| Dynamic points / frame | — | **≤ 20 000** | moving foreground only |
| Tracks | `M` | **≤ 4 096** | the ribbons |
| Total **uncompressed** payload | — | **target ≤ 12 MB** (hard ceiling 24 MB) | encoder logs actual size |

These caps are enforced by the **core service** `enforce_caps()` (cull by confidence /
subsample) **before** encode — see [06 §5 step 7](06-backend-spec.md). The
**`encode_reconstruction` encoder assumes an already-capped `Scene4D`** and does **not**
cull; it only **logs** the final counts + actual payload size (and clamps `q` to
`[0,65535]` defensively). The backend serves the immutable blob with a long-lived
cache header; compression is the serving layer's job ([06 §7](06-backend-spec.md)).

---

## 5. Canonical in-memory structures

### 5.1 Backend domain model (`backend/app/core/domain/models.py`)
The encoder consumes this `Scene4D` (float positions; it quantizes). NumPy arrays;
no torch types cross the core boundary.

```python
# Authoritative shape — replaces the placeholder ReconstructionResult.
from dataclasses import dataclass
import numpy as np

@dataclass
class CameraTrack:
    poses: np.ndarray        # (T, 7) f32  quaternion(xyzw)+translation, cam→world
    intrinsics: np.ndarray   # (T, 4) f32  normalized fx,fy,cx,cy

@dataclass
class Tracks:
    positions: np.ndarray    # (M, T, 3) f32  world space
    visibility: np.ndarray   # (M, T)    bool
    colors: np.ndarray | None  # (M, 3)  u8, optional

@dataclass
class Scene4D:
    frame_count: int                 # T
    fps: float
    aabb_min: np.ndarray             # (3,) f32
    aabb_max: np.ndarray             # (3,) f32
    static_positions: np.ndarray     # (N_s, 3) f32
    static_colors: np.ndarray        # (N_s, 3) u8
    static_conf: np.ndarray | None   # (N_s,)  u8, optional
    dynamic_positions: list[np.ndarray]  # len T, each (N_d_t, 3) f32
    dynamic_colors: list[np.ndarray]     # len T, each (N_d_t, 3) u8
    tracks: Tracks | None
    cameras: CameraTrack | None
    # Provenance (not serialized into MV4D; returned via job metadata):
    adapter_id: str = ""
    weights_license: str = ""
```

Encoder signature (replaces the stub):
```python
def encode_reconstruction(scene: Scene4D) -> bytes: ...   # returns an MV4D v1 buffer
```
The AABB is computed over **all** positions (static ∪ dynamic ∪ tracks) so every
section shares one quantization range.

### 5.2 Frontend decoded type (`frontend/src/types/index.ts`)
The decoder returns **zero-copy views** plus the AABB; Path-1 shaders dequantize
positions on the GPU. Track positions (small) may be dequantized to `Float32Array`
for `Line2` ribbons.

```ts
export interface Mv4dScene {
  version: 1;
  frameCount: number;          // T
  fps: number;
  aabbMin: [number, number, number];
  aabbMax: [number, number, number];
  static?: {
    count: number;
    positionsQ: Uint16Array;   // length count*3, view (dequant in shader)
    colors: Uint8Array;        // length count*3
    conf?: Uint8Array;         // length count
  };
  dynamic?: {
    frames: Array<{ count: number; positionsQ: Uint16Array; colors: Uint8Array }>;
  };
  tracks?: {
    count: number;             // M
    positionsQ: Uint16Array;   // length M*T*3
    visibility: Uint8Array;    // packed bitmask, length ceil(M*T/8)
    colors?: Uint8Array;       // length M*3
    isVisible(m: number, t: number): boolean;
  };
  cameras?: {
    poses: Float32Array;       // length T*7
    intrinsics: Float32Array;  // length T*4
  };
}

export function decodeReconstruction(buffer: ArrayBuffer): Mv4dScene;
```
(The placeholder `ReconstructionResult` interface is replaced by `Mv4dScene`;
update `decoder.ts`, `api/client.ts`, and `types/index.ts` together.)

---

## 6. Worked micro-example

A minimal scene: `T=2`, no static, one dynamic point per frame, no tracks, no
cameras (`flags = HAS_DYNAMIC = 0b000010 = 0x02`), `fps=24`,
`aabbMin=(0,0,0)`, `aabbMax=(1,1,1)`, `sectionCount=1`.

- Header(24) + AABB(24) + directory(16) = 64 bytes prefix.
- Directory entry: `kind=2, byteOffset=64, byteLength=?, count=2`.
- DYNAMIC payload at 64: `frameDir = [ (0,1), (1,1) ]` (`u32[4]` = 16 B);
  `positions = u16[2*3]` for the two points (12 B); `colors = u8[2*3]` (6 B).
- A point at world (0.5,0.5,0.5) quantizes to `q=32768` per axis.
- The decoder reads frame 0 = `positionsQ.subarray(0,3)`, frame 1 =
  `positionsQ.subarray(3,6)`.

---

## 7. Versioning rules
- `version` byte = `1`. Any **breaking** change (field reorder, dtype change,
  quantization change) **increments** it; the decoder rejects unknown major
  versions with a clear error.
- **Additive** changes (new section `kind`, new `flags` bit, use of a reserved
  field) do **not** bump the version: old decoders skip unknown sections/flags.
- The encoder and decoder both export `MV4D_VERSION = 1`; tests assert they match
  (spec/10 cross-format round-trip test).

## 8. Decoder error contract
`decodeReconstruction` throws a typed `Mv4dDecodeError` on: bad magic, unsupported
major version, `posBits ≠ 16`, a section whose `byteOffset+byteLength` exceeds the
buffer, or a misaligned section offset. It never returns a partially-filled scene.
