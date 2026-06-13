"use client";

import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import type { Mv4dScene } from "@/types";
import { useViewerStore } from "@/lib/state/viewerStore";
import { buildStatic, buildDynamic, timeToFrame } from "@/lib/viewer/buildScene";
import { POINT_VERT, POINT_FRAG } from "@/lib/viewer/pointShaders";

// One <PointCloud> component, two instances (spec/07 §2.1): a `static`
// background built once + drawn every frame, and a `dynamic` foreground whose
// active point set is `scene.dynamic.frames[t]` (t = round(time*(T-1))). Both
// dequantize quantized u16 positions ON THE GPU via the shared ShaderMaterial
// (lib/viewer/pointShaders.ts) — positions never become Float32 on the CPU.
//
// Render-path decoupling (spec/07 §1): this component reads playback `time`
// from the Zustand store via `getState()` INSIDE the useFrame loop (no React
// subscription → no per-frame re-render). `scene` arrives as a prop from
// <Scene> (the seam) per spec/07 §7.
//
// Three.js objects are CONSTRUCTED in useMemo (a pure derivation from the scene)
// and handed to <primitive>; every per-frame MUTATION is done through the
// `<primitive ref>` (`pointsRef.current`, assigned outside render), so render
// stays pure and no hook result is mutated.

const DEFAULT_POINT_SIZE = 2.2;

/** Build the shared dequant ShaderMaterial (spec/07 §2.1). `uOpacity` is lower
 * for the dynamic layer so the foreground reads as motion against the static
