# 02 — Research Dossier

The landscape mayavius lives in, **re-verified 2026-06-13** (Phase 0 sweep). Every
external fact here traces to [decisions/decision-log.md](decisions/decision-log.md)
§D–§G (cited inline as `[log §X]`); this file does **not** re-run web searches and
must not contradict the log. Where the original handover (`SPEC_BUILD_HANDOVER.md`)
was factually wrong, the corrected fact is used and tagged with the correction id
(C1–C7) from [03-decisions-locked.md](03-decisions-locked.md) Part 3.

Purpose: (a) explain D4RT, the idea mayavius reproduces without depending on it;
(b) show the niche is open; (c) contrast competitors and the closest research
analogues; (d) record the backend model candidates with their MPS/license reality;
(e) place the rendering substrate. Locked outcomes from this analysis live in
[03-decisions-locked.md](03-decisions-locked.md) (D1, D2) — this file is the *why*.

---

## (a) D4RT — the idea, not a dependency

**D4RT** (arXiv:2512.08924, Google DeepMind, Dec 2025) is a single **feedforward**
network that unifies what used to be four separate pipelines — point tracking,
point-cloud reconstruction, monocular depth, and camera estimation — into one
model. Shape of it:

- An **encoder** ingests a casual video and produces a fixed **Global Scene
  Representation (GSR)** — the entire spatiotemporal scene baked into one latent,
  computed **once**.
- A **lightweight cross-attention decoder** then answers arbitrary **point
  queries** `q = (u, v, t_src, t_tgt, t_cam)` → a 3D position: pixel `(u,v)` at
  source time `t_src`, re-expressed at target time `t_tgt`, viewed from camera time
  `t_cam`. Tracking, depth, point clouds, and camera all fall out as special cases
  of this one query API.

The "wow" mayavius copies is the *output*: a per-timestep colored 3D point cloud
plus 3D point **tracks**, from a feedforward pass (no per-scene optimization).

**Release status.** Official D4RT **code and weights remain UNRELEASED** as of
2026-06-13 (no Google-org repo; the DeepMind blog announces no code) [log §F].
An **unofficial open reimplementation now exists** — `Lijiaxin0111/Open-d4rt`
(Apache-2.0, training code shipped 2026-06-04, weights `Lijiaxin0111/OpenD4RT` on
HF) — GPU/PyTorch-oriented, **MPS unverified** [log §F, C5].

> **mayavius does NOT depend on D4RT.** It reproduces the D4RT *look* with
> *released* models behind the swappable `ReconstructionPort`. The unofficial open
> reimpl gives `OpenD4RTAdapter` a concrete (optional, GPU) thing to wrap; the
> default combo (D1) is **VGGT + CoTracker3**, which run on the 36GB Apple-Silicon
> Mac today. The architecture is built so a future official OpenD4RT-style decoder
> drops into the adapter layer without touching `app/core`.

---

## (b) Why the niche is open

Feedforward 4D reconstruction is a hot research area, but the **delivery layer is
missing**. How that research actually ships today:

| How research ships | What it is | Why it is not this product |
|---|---|---|
| **Viser** (nerfstudio's Python web viewer) | localhost server-rendered 3D viewer | dev-tool, runs from a Python process; no upload→reconstruct flow, no shareable permalink, not SEO/landing-page shaped |
| **Gradio Spaces** (VGGT / Vdpm / Any4D, etc.) | HF-hosted demo, **server-side GPU** | transient session, **no shareable result permalink**, viewer re-renders on the server every interaction; several were broken at check time [log §F] |
| **matplotlib / notebook stills** | static plots in a paper repo | not interactive, not a product at all |

The gap mayavius fills is a **polished open web app**: drop a video → feedforward
reconstruct on the backend → stream a **compact binary** scene
([05-data-contract.md](05-data-contract.md)) to a **client-only WebGL viewer** that
plays back with orbit / scrub / loop / bullet-time, behind a **shareable
permalink** with rich share cards (`app/view/[id]`), and the **viewing is
GPU-free** (only backend inference is heavy; compute is asymmetric).

**Defensive-novelty framing (do not overclaim).** Across GitHub / arXiv / HF
Spaces / Papers-with-Code, **no direct competitor surfaced** for the exact
combination *(open + browser upload + feedforward 4D cloud + track ribbons +
shareable link + GPU-free viewing)* [log §F]. Absence of a competitor is
unfalsifiable, so state it as **"none surfaced"**, not "none exists" — re-run the
scan near launch. The defensible wedge is the **combination**, not any single
component.

---

## (c) Competitor & analogue contrast

**Direct competitors / adjacent products (different thing, stated first so the
distinction is unmissable):**

| Name | What it is | Why it is NOT mayavius |
|---|---|---|
| **Vista4D** (Eyeline-Labs, CVPR 2026 Highlight, arXiv:2604.21915, Apache-2.0) | **CUDA-12.8-only** video-reshooting / **novel-view SYNTHESIS** on a Wan2.1 diffusion backbone (uses Pi3X / Depth-Anything-3 upstream) | It **generates** new video, it is not a viewer; not feedforward-light; **not Mac-runnable**. (A blog's CC-BY-4.0 claim is wrong — repo is authoritative.) [log §F] |
| **Luma / Polycam / Splat Labs Cloud** | proprietary cloud splat **capture** apps | closed source; static (or splat-based) capture, not interactive feedforward 4D point clouds + tracks; not open / not star-able |

**Closest research analogues (future adapter candidates, NOT competitors — they
are models, not products):**

| Name | Why it is close | Relation to mayavius |
|---|---|---|
| **TracksTo4D** (NVlabs, NeurIPS 2024) | **feedforward, casual-video, point-track-driven** 4D — closest on the feedforward axis | strongest future adapter candidate behind `ReconstructionPort` (verify NVIDIA license before wiring) [log §F] |
| **Shape-of-Motion** | dynamic 4D from casual video | **optimization-based** (per-scene), not a single feedforward pass → wrong cost profile for an interactive MVP |
| **MoRe** (CVPR 2026) | feedforward dynamic reconstruction | another future adapter candidate to track |

These are inputs we could *wrap*, not products competing for the same users. The
adapter port exists precisely so they can be added without re-architecting.

---

## (d) Backend model candidates

Re-verified per-model [log §D, §E]. **Corrections C1–C3, C6 from the handover are
applied.** MPS column is the binding constraint for the 36GB Apple-Silicon Mac.

| Model | Role / output | MPS on Mac (fp32) | Weights license | mayavius use |
|---|---|---|---|---|
| **VGGT** `facebookresearch/vggt`; `facebook/VGGT-1B` | static reconstruction: camera (extrinsics+intrinsics), depth(+conf), world point map(+conf), **+ a 2D track head for STATIC scenes (C1)** — input is a **set of frames** `[S,3,H,W]` rescaled to width **518px**, *not* a video file | **No official MPS**; runs via the community-port pattern (`jmanhype/vggt-mps`, MIT, reference-only): `device="mps"`, **forced fp32**, no autocast. 8GB+ RAM Mac, macOS 14+ | `VGGT-1B` = **cc-by-nc-4.0** (NC). `VGGT-1B-Commercial` = custom `vggt-aup-license` (commercial OK except military, **gated** form; non-OSI) | **`VggtAdapter` — DEFAULT static layer** (D1). Static cloud + depth + camera; depth+intrinsics also lift CoTracker3 tracks to 3D |
| **CoTracker3** `facebookresearch/co-tracker`; `facebook/cotracker3` | dense/queried **2D point tracks** `(B,T,N,2)` + visibility `(B,T,N,1)` | **Best MPS story** — first-class in-tree auto-select `cuda > mps > cpu` (merged PR #14) | **cc-by-nc-4.0** (code+ckpts; sub-parts MIT/Apache) | **`CoTracker3Adapter` — DEFAULT dynamic layer** (D1). 2D tracks **lifted to 3D via VGGT depth+camera** → the track ribbons |
| **SpatialTrackerV2** `henry123-boy/SpaTrackerV2` (slug casing); `Yuxihenry/SpatialTrackerV2_*` | single-model **dynamic 3D tracks + geometry** | 🚫 **CUDA-only** — upstream pins `torch==2.4.1+cu124`; not Mac-installable as-is | **CC-BY-NC-4.0 on the CODE (C6)** — GitHub shows `NOASSERTION`, a CC-detection gap, *not* permissive; weights assume NC | **`SpatialTrackerV2Adapter` — cloud/optional** (single-pass dynamic alternative; spec/11). Never in the local MVP install |
| **Pi3 / π³** `yyfz/Pi3`; `yyfz233/Pi3` | feedforward visual geometry (point map + camera) | 🚫 **No official MPS** (PR #153 open/unmerged; `demo_gradio.py` hard-fails without CUDA) | **code = BSD-3-Clause, commercial OK (C2)**; **weights = CC-BY-NC-4.0** (HF inconsistently tags `bsd-2-clause` → treat as NC) | **`Pi3Adapter` — optional, GPU**. 🚫 Not the Mac default (no MPS) |
| **OpenD4RT** `Lijiaxin0111/Open-d4rt`; `Lijiaxin0111/OpenD4RT` | unofficial open D4RT reimpl (unified query API) | GPU/PyTorch-oriented; **MPS unverified** — do not assume it runs on the Mac | **Apache-2.0** | **`OpenD4RTAdapter` — optional, GPU** (C5). The "future open D4RT" seam, now with a real repo to wrap |
| **VGGT-Omega** `facebookresearch/vggt-omega`; `facebook/VGGT-Omega` | VGGT + native static **and dynamic** scenes (CVPR 2026 Oral) | gated weights, GPU-oriented | **cc-by-nc-4.0 + gated** | research-only **future** adapter candidate; **not** a commercial default (the "20×/100×" blog figures are unverified — do not cite) |
| **MegaSaM** `mega-sam/mega-sam` | dynamic SLAM (depth + camera) | 🚫 CUDA-only | code **Apache-2.0** (3rd-party weights vary) | 🚫 **Rejected on PERFORMANCE, not license** — optimization-based SLAM (~0.7 FPS), not a single feedforward pass → wrong for an interactive MVP |

**Negative knowledge (first-class — do not rediscover):**

- 🚫 **π³ has no official MPS path** → not the Mac default.
- 🚫 **SpatialTrackerV2 is CUDA-only** (`cu124` wheel pin) → cloud/optional only.
- ⚠️ **"MPS = fp32 only / no fp16 autocast" is FALSE at the framework level (C3).**
  PyTorch *does* support fp16/bf16 autocast on MPS (merged, beta). mayavius runs
  **fp32 by choice** because the working VGGT MPS port does and half-precision is
  incomplete *for these models* — not because MPS cannot do fp16.
- 🚫 **MegaSaM is optimization-based** (per-video SLAM), so it loses on cost, not
  licensing.
- ⚠️ **Memory-per-frame is not asserted here** — the only repo-confirmed Mac fact
  is "8GB+ RAM Mac"; the "7GB/8GB" figures are unverified blogspam. Measure on the
  actual 36GB Mac in spec/10 [log §E].

**License synthesis** (full table in [08-dependencies-and-env.md](08-dependencies-and-env.md)
§7): viewer/render layer is MIT-clean; **model weights are overwhelmingly
non-commercial**, and **no commercial-friendly point tracker exists** in the set —
so the signature track-ribbon feature is research/NC. License-cleanest research
combo = `VGGT-1B` + `CoTracker3` + Three.js → **D1/D2**. The adapter layer must
license-tag each model (`AdapterInfo.weights_license`, surfaced via `/jobs`
metadata — spec/06).

---

## (e) Rendering substrate

Two paths; only **Path 1** is in the MVP. See the seam comment in
`frontend/src/components/viewer/Scene.tsx`.

| | Path 1 — **MVP** | Path 2 — **OUT of MVP (design the seam, don't build)** |
|---|---|---|
| Technique | `THREE.Points` + custom vertex/fragment shader for the colored cloud; `Line2` / `LineSegments2` (from `three/addons/lines/…`, ships inside `three`) for track ribbons | **4D Gaussian Splatting** via **Spark** `@sparkjsdev/spark` `<SplatMesh>` |
| Library | Three.js 0.184 (MIT), react-three-fiber 9.6 | Spark 2.1.0 (MIT; canonical org `sparkjsdev`, npm `repository` typo `sparkjs-dev` [log §D]) |
| Why / why not | GPU-cheap, mobile-capable, exactly the D4RT-page aesthetic; quantized positions **dequantized in the shader** → ~2s loads | Turning a feedforward reconstruction into renderable 4D gaussians needs **per-scene optimization** (GPU-heavy, not feedforward) — wrong cost profile for the MVP |
| mayavius wiring | Reads the `MV4D` scene's static cloud / dynamic frames / tracks / cameras directly | A `<SplatMesh>` layer mounts **alongside Path 1 at the `Scene.tsx` seam**, without rearchitecting [handover §4.2] |

Spark 2.0 (2026-04-14) is real (streaming LOD, `.RAD` format, 100M+ splats) [log
§D] — capable, but explicitly deferred. The MVP renders Path 1 only.

---

## Cross-references

- Locked outcomes of this analysis: [03-decisions-locked.md](03-decisions-locked.md)
  (D1 model combo, D2 licensing, Part 3 corrections C1–C7).
- Verified pins / repo IDs / install commands: [08-dependencies-and-env.md](08-dependencies-and-env.md).
- Wire format the viewer consumes: [05-data-contract.md](05-data-contract.md).
- Full evidence + sources (dated 2026-06-13): [decisions/decision-log.md](decisions/decision-log.md) §D–§G.
