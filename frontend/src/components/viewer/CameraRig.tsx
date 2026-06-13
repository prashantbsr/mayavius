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
