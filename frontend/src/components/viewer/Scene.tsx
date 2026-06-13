"use client";

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
  // Placeholder geometry — proves the R3F pipeline renders end to end.
  // Replace with the point-cloud + track-ribbon layers.
  return (
    <mesh>
      <icosahedronGeometry args={[1, 1]} />
      <meshStandardMaterial color="#6ea8fe" wireframe />
    </mesh>
  );
}
