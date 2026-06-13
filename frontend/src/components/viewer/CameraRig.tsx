"use client";

import { useEffect, useRef, type RefObject } from "react";
import * as THREE from "three";
import { useThree, useFrame } from "@react-three/fiber";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { useViewerStore } from "@/lib/state/viewerStore";
import { timeToFrame } from "@/lib/viewer/buildScene";

// CameraRig — drives the camera + OrbitControls, NOT the render path (spec/07
// §5). Two jobs:
//
//   1. Pinned initial camera fit (spec/07 §5): applied ONCE per loaded scene so
//      "the cloud fills the view on load" is reproducible. Math is fixed:
//        center   = (aabbMin+aabbMax)/2
//        radius   = 0.5·length(aabbMax−aabbMin)
//        fovY     = 50°
//        distance = radius / sin(0.5·fovYrad) · PADDING   (PADDING = 1.3)
//        dir      = normalize(0.3, 0.2, 1)   (a slight 3/4 view down −Z)
//        position = center + distance·dir ;  lookAt(center)
//
//   2. Camera modes (spec/07 §5): `orbit` (default — OrbitControls own the
//      camera), `asShot` (follow the reconstructed per-frame pose; OrbitControls
//      disabled while active; only available when scene.cameras present),
//      `bulletTime` (time frozen by PlaybackDriver; OrbitControls free-orbit the
//      frozen frame). OrbitControls stay ENABLED in orbit + bulletTime; disabled
//      only in asShot.
//
// The camera is read through the R3F store getter (`useThree((s) => s.get)`),
// not the `useThree(camera)` hook value — so the per-frame/effect camera
// mutations here are not flagged as modifying a hook return (and we always touch
// the live default camera).

const FIT_FOV_Y_DEG = 50;
const FIT_PADDING = 1.3;
const FIT_DIR = new THREE.Vector3(0.3, 0.2, 1).normalize();

export function CameraRig({
  controlsRef,
}: {
  controlsRef: RefObject<OrbitControlsImpl | null>;
}) {
  // R3F store getter → fresh RootState (with the live default camera) on demand.
  const getR3f = useThree((s) => s.get);
  // Subscribe narrowly: the fit effect must re-run when a NEW scene arrives.
  const scene = useViewerStore((s) => s.scene);

  // Scratch vectors/quaternions reused across frames (no per-frame allocation).
  const centerRef = useRef(new THREE.Vector3());
  const tmpVec = useRef(new THREE.Vector3());
  const tmpQuat = useRef(new THREE.Quaternion());

  // ── 1. Pinned initial fit — runs once per loaded scene. ──────────────────────
  useEffect(() => {
    if (!scene) return;
    const camera = getR3f().camera;
    const [minX, minY, minZ] = scene.aabbMin;
    const [maxX, maxY, maxZ] = scene.aabbMax;

    const center = centerRef.current.set(
      (minX + maxX) / 2,
      (minY + maxY) / 2,
      (minZ + maxZ) / 2,
    );
    const diag = tmpVec.current.set(maxX - minX, maxY - minY, maxZ - minZ);
    const radius = 0.5 * diag.length();

    const fovYRad = THREE.MathUtils.degToRad(FIT_FOV_Y_DEG);
    // Degenerate AABB (radius 0) → keep a sane non-zero distance so the camera
    // doesn't collapse onto the center.
    const distance =
      radius > 0 ? (radius / Math.sin(0.5 * fovYRad)) * FIT_PADDING : 1;

    if (camera instanceof THREE.PerspectiveCamera) {
      camera.fov = FIT_FOV_Y_DEG;
      camera.updateProjectionMatrix();
    }
    camera.position.copy(center).addScaledVector(FIT_DIR, distance);
    camera.lookAt(center);

    // Point OrbitControls at the same target so the first drag orbits the cloud.
    const controls = controlsRef.current;
    if (controls) {
      controls.target.copy(center);
      controls.update();
    }
    // Re-run only when a new scene is loaded. `scene` is the meaningful trigger.
  }, [scene, getR3f, controlsRef]);

  // ── 2. Per-frame camera-mode handling. ───────────────────────────────────────
