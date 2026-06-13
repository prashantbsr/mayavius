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
 * cloud; ribbons (§2.2) carry the actual motion history. */
function makePointMaterial(
  scene: Mv4dScene,
  opacity: number,
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: POINT_VERT,
    fragmentShader: POINT_FRAG,
    // `vertexColors` makes the built-in `color` attribute available to the
    // shader; we still convert sRGB→linear in the fragment shader to keep one
    // color path (spec/07 §2.1).
    vertexColors: true,
    transparent: opacity < 1,
    depthWrite: opacity >= 1,
    uniforms: {
      uAabbMin: {
        value: new THREE.Vector3(
          scene.aabbMin[0],
          scene.aabbMin[1],
          scene.aabbMin[2],
        ),
      },
      uAabbMax: {
        value: new THREE.Vector3(
          scene.aabbMax[0],
          scene.aabbMax[1],
          scene.aabbMax[2],
        ),
      },
      uPointSize: { value: DEFAULT_POINT_SIZE },
      uViewportHeight: { value: 1 },
      uOpacity: { value: opacity },
    },
  });
}

/** Construct the layer's THREE.Points (geometry + dequant material). Returns
 * `null` when the scene lacks that section (nothing to render). The dynamic
 * layer's geometry is pre-sized to the busiest frame (buildDynamic) so playback
 * mutates a fixed buffer instead of allocating per frame. */
function buildPoints(
  layer: "static" | "dynamic",
  scene: Mv4dScene,
): THREE.Points | null {
  if (layer === "static") {
    const geometry = buildStatic(scene);
    if (!geometry) return null;
    const points = new THREE.Points(geometry, makePointMaterial(scene, 1.0));
    // AABB-bounded already → never frustum-cull the static cloud (spec/07 §2.1).
    points.frustumCulled = false;
    return points;
  }
  const dynamic = buildDynamic(scene);
  if (!dynamic) return null;
  const points = new THREE.Points(dynamic.geometry, makePointMaterial(scene, 0.9));
  points.frustumCulled = false;
  return points;
}

export function PointCloud({
  layer,
  scene,
}: {
  layer: "static" | "dynamic";
  scene: Mv4dScene;
}) {
  // Construct once per (layer, scene) — a pure derivation, no mutation here.
  const points = useMemo(() => buildPoints(layer, scene), [layer, scene]);
  // The mounted THREE.Points, captured outside render; ALL mutation goes here.
  const pointsRef = useRef<THREE.Points | null>(null);

  // Dispose GPU resources when (layer, scene) changes or on unmount.
  useEffect(
    () => () => {
      if (!points) return;
      points.geometry.dispose();
      (points.material as THREE.Material).dispose();
    },
    [points],
  );

  // Track the last applied frame so we only re-copy the dynamic buffers when t
  // actually changes (the loop runs every frame; the active frame does not).
  const lastFrame = useRef<number>(-1);

  useFrame((state) => {
    const obj = pointsRef.current;
    if (!obj) return;
    const material = obj.material as THREE.ShaderMaterial;
    // Keep perspective sizing in sync with the live canvas height (cheap write).
    material.uniforms.uViewportHeight.value = state.size.height;

    if (layer !== "dynamic") return;
    const { time, frameCount } = useViewerStore.getState();
    const t = timeToFrame(time, frameCount);
    if (t === lastFrame.current) return;
    lastFrame.current = t;

    const frame = scene.dynamic?.frames[t];
    const count = frame ? frame.count : 0;
    const geom = obj.geometry;
    if (count === 0) {
      geom.setDrawRange(0, 0); // a frame with no moving points draws nothing
      return;
    }
    // Copy this frame's decoder views into the fixed-capacity prefix and flag
    // the changed range for re-upload — no per-frame BufferGeometry allocation.
    const cnt3 = count * 3;
    const posAttr = geom.getAttribute("position") as THREE.BufferAttribute;
    const colAttr = geom.getAttribute("color") as THREE.BufferAttribute;
    (posAttr.array as Uint16Array).set(frame!.positionsQ.subarray(0, cnt3));
    (colAttr.array as Uint8Array).set(frame!.colors.subarray(0, cnt3));
    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
    geom.setDrawRange(0, count);
  });

  if (!points) return null;
  return <primitive object={points} ref={pointsRef} />;
}
