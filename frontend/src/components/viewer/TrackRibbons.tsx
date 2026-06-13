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
// group is CONSTRUCTED in useMemo and handed to <primitive>; every mutation goes
// through the `<primitive ref>` (`groupRef.current`), so render stays pure.

const DEFAULT_LINE_WIDTH_PX = 2.5;
const GOLDEN_ANGLE_DEG = 137.508;

/** Per-run growth metadata stashed on each Line2's `userData` so it travels
 * with the captured group object (mutated only through the ref). */
interface RunMeta {
  startFrame: number;
  segmentCount: number; // points-1 → max drawable segments
}

/** Stable per-track color: `tracks.colors` iff present, else a golden-angle hue
 * from the track index so ribbons read as distinct trajectories (spec/07 §2.2). */
function trackColor(scene: Mv4dScene, m: number, out: THREE.Color): THREE.Color {
  const c = scene.tracks?.colors;
  if (c) {
    return out.setRGB(
      c[m * 3 + 0] / 255,
      c[m * 3 + 1] / 255,
      c[m * 3 + 2] / 255,
      THREE.SRGBColorSpace,
    );
  }
  const hue = ((m * GOLDEN_ANGLE_DEG) % 360) / 360;
  return out.setHSL(hue, 0.7, 0.6);
}

/** Split every track into contiguous visible runs and build one Line2 per run
 * (spec/07 §2.2), collected under a group. CPU-dequants track positions to
 * Float32 (the only CPU dequant). Returns `null` when the scene has no tracks. */
function buildRibbonGroup(scene: Mv4dScene): THREE.Group | null {
  const tracks = scene.tracks;
  if (!tracks) return null;

  const T = scene.frameCount;
  const M = tracks.count;
  const q = tracks.positionsQ; // u16, length M*T*3
  const [minX, minY, minZ] = scene.aabbMin;
  const [maxX, maxY, maxZ] = scene.aabbMax;
  const color = new THREE.Color();
  const group = new THREE.Group();

  for (let m = 0; m < M; m++) {
    trackColor(scene, m, color);
    // One material per track (each run of a track shares its color/width).
    let material: LineMaterial | null = null;

    let t = 0;
    while (t < T) {
      if (!tracks.isVisible(m, t)) {
        t++;
        continue;
      }
      // Extend a maximal contiguous visible run [runStart .. t-1].
      const runStart = t;
      while (t < T && tracks.isVisible(m, t)) t++;
      const runEnd = t - 1; // inclusive
      const pointCount = runEnd - runStart + 1;
      if (pointCount < 2) continue; // a single visible point = no segment, no ribbon

      // CPU dequant this run's points to world-space Float32 (spec/07 §2.2).
      const positions = new Float32Array(pointCount * 3);
      for (let k = 0; k < pointCount; k++) {
        const base = (m * T + (runStart + k)) * 3;
        positions[k * 3 + 0] = dequantize(q[base + 0], minX, maxX);
        positions[k * 3 + 1] = dequantize(q[base + 1], minY, maxY);
        positions[k * 3 + 2] = dequantize(q[base + 2], minZ, maxZ);
      }

      const geometry = new LineGeometry();
      geometry.setPositions(positions);
      // Start with the full trail hidden; the loop grows it from t.
      geometry.instanceCount = 0;

      if (!material) {
        material = new LineMaterial({
          color: color.getHex(THREE.SRGBColorSpace),
          linewidth: DEFAULT_LINE_WIDTH_PX,
          worldUnits: false, // screen-space px width (spec/07 §2.2)
          transparent: true,
          opacity: 0.9,
        });
      }

      const line = new Line2(geometry, material);
      line.frustumCulled = false;
      const meta: RunMeta = { startFrame: runStart, segmentCount: pointCount - 1 };
      line.userData = meta;
      group.add(line);
    }
  }

  return group;
}

export function TrackRibbons({ scene }: { scene: Mv4dScene }) {
  // Construct once per scene — a pure derivation, no mutation here.
  const group = useMemo(() => buildRibbonGroup(scene), [scene]);
  // The mounted group, captured outside render; ALL mutation goes here.
  const groupRef = useRef<THREE.Group | null>(null);
  // Remember the last resolution so we re-set LineMaterial.resolution on resize
  // (required for px-accurate widths, spec/07 §2.2) without per-frame churn.
  const lastW = useRef<number>(0);
  const lastH = useRef<number>(0);
  const lastFrame = useRef<number>(-1);

  // Dispose geometries + materials when the scene changes / on unmount.
  useEffect(
    () => () => {
      if (!group) return;
      const seen = new Set<LineMaterial>();
      for (const child of group.children) {
        const line = child as Line2;
        line.geometry.dispose();
        const mat = line.material as LineMaterial;
        if (!seen.has(mat)) {
          seen.add(mat);
          mat.dispose();
        }
      }
    },
    [group],
  );

  useFrame((state) => {
    const g = groupRef.current;
    if (!g) return;

    // Keep LineMaterial.resolution in sync with the canvas (px widths). Only
    // re-set on an actual size change — the material list is small but shared.
    const w = state.size.width;
    const h = state.size.height;
    if (w !== lastW.current || h !== lastH.current) {
      lastW.current = w;
      lastH.current = h;
      for (const child of g.children) {
        (child as Line2).material.resolution.set(w, h);
      }
    }

    const { time, frameCount } = useViewerStore.getState();
    const t = timeToFrame(time, frameCount);
    if (t === lastFrame.current) return;
    lastFrame.current = t;
    for (const child of g.children) {
      const line = child as Line2;
      const meta = line.userData as RunMeta;
      // Segment i connects frame (startFrame+i) → (startFrame+i+1); it is drawn
      // once t reaches its far endpoint. Visible segments = clamp(t-startFrame).
      const visible = t - meta.startFrame;
      line.geometry.instanceCount =
        visible <= 0
          ? 0
          : visible >= meta.segmentCount
            ? meta.segmentCount
            : visible;
    }
  });

  if (!group) return null;
  return <primitive object={group} ref={groupRef} />;
}
