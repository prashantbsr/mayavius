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

