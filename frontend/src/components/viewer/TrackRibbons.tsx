"use client";

import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import type { Mv4dScene } from "@/types";
import { useViewerStore } from "@/lib/state/viewerStore";
import { dequantize } from "@/lib/wire/decoder";
import { timeToFrame } from "@/lib/viewer/buildScene";

// The M tracks become the motion ribbons (spec/07 §2.2): one polyline per track
// across T frames, with GAPS where the visibility bit is 0. Built with
// Line2/LineGeometry/LineMaterial (three/addons/lines/…, ships inside three —
// NO new dep): plain THREE.Line ignores `linewidth` on most platforms (negative
// knowledge, spec/07 §2.2), so screen-space-width ribbons need Line2.
//
// CPU dequant (the ONLY one — spec/05 §1, spec/07 §2.2): LineGeometry needs
// Float32, and M*T ≤ 64*4096 ≈ 262k verts is small, so we dequant
// `tracks.positionsQ` to world-space Float32 on the CPU here. Static/dynamic
// layers stay GPU-dequant.
//
// Growth with time (spec/07 §2.2): each run's `geometry.instanceCount` is set to
// the number of segments whose far endpoint frame ≤ current t, so trails GROW
// during playback and show fully when frozen.
//
// Render-path decoupling (spec/07 §1): reads playback `time` from the store via
// `getState()` inside useFrame (no per-frame re-render); `scene` is a prop. The
