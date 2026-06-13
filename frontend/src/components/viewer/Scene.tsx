"use client";

import { useViewerStore } from "@/lib/state/viewerStore";
import { PointCloud } from "./PointCloud";
import { TrackRibbons } from "./TrackRibbons";

// Scene graph for the 4D viewer.
//
// MVP render path (Path 1 — LOCKED, handover §4.1): animated colored point
// cloud (THREE.Points + custom shader) plus 3D track ribbons, driven by the
// playback store and the decoded wire format (spec/05-data-contract.md).
// Built out as <PointCloud /> + <TrackRibbons /> per spec/07-frontend-spec.md.
//
// Extension seam (Path 2 — OUT of MVP, handover §4.2): a Spark
// (@sparkjsdev/spark) <SplatMesh /> layer mounts *here*, alongside Path 1,
// without rearchitecting. Keep controls/timeline decoupled from Path 1 so the
// renderer can be swapped or composed later.
export function Scene() {
  // Read the decoded scene from the store (spec/07 §7). Until a scene loads the
  // layers have nothing to render — return null (the canvas stays empty, the
  // DOM HUD's ProgressOverlay covers loading).
  const scene = useViewerStore((s) => s.scene);
  if (!scene) return null;

  return (
    <>
      <PointCloud layer="static" scene={scene} />
      <PointCloud layer="dynamic" scene={scene} />
      <TrackRibbons scene={scene} />
      {/* PATH 2 SEAM (spec/03 Part 1 §2; handover §4.2): a Spark <SplatMesh>
          layer mounts HERE, alongside Path 1, reading the same store. Do NOT
          import @sparkjsdev/spark in the MVP. */}
    </>
  );
}
