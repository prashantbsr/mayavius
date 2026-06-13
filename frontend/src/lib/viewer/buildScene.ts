import * as THREE from "three";
import type { Mv4dScene } from "@/types";

// Decoder (MV4D) → Three.js BufferGeometry, the view→attribute mapping
// (spec/07-frontend-spec.md §3). The decoder (lib/wire/decoder.ts, spec/05 §5.2)
// returns zero-copy typed-array VIEWS over the fetched ArrayBuffer; this module
// wraps those views in BufferAttributes WITHOUT copying or expanding them.
//
// Zero-copy / no CPU dequant (spec/05 §1, spec/07 §3): static and dynamic
// positions stay `Uint16Array` end-to-end and are dequantized in the vertex
// shader (PointCloud.tsx §2.1). They are uploaded with `normalized:true` so
// WebGL feeds the shader `q/65535 ∈ [0,1]`; the shader maps that to world space
// with `aabbMin/aabbMax`. The CPU never expands positions to Float32 — that is
// the ~2s-vs-~40s load win. (Track ribbons are the one CPU-dequant exception,
// handled in TrackRibbons.tsx, spec/07 §2.2.)

/**
 * Build the **static** background geometry once from `scene.static`
 * (spec/07 §3). Quantized `Uint16` positions + `Uint8` colors are wrapped as
 * `normalized` BufferAttributes (dequant happens in the vertex shader, §2.1).
 * `boundingBox` is set from the scene AABB so the renderer can size/cull without
 * touching positions. Returns `null` when the scene has no static section.
 */
export function buildStatic(scene: Mv4dScene): THREE.BufferGeometry | null {
  const s = scene.static;
  if (!s) return null;

  const g = new THREE.BufferGeometry();
  // u16 positions, normalized → shader reads q/65535 ∈ [0,1] (spec/07 §2.1).
  g.setAttribute(
    "position",
    new THREE.BufferAttribute(s.positionsQ, 3, /* normalized */ true),
  );
  // u8 sRGB color, normalized → shader reads byte/255 ∈ [0,1].
  g.setAttribute(
    "color",
    new THREE.BufferAttribute(s.colors, 3, /* normalized */ true),
  );

  // AABB from the scene (one quantization range for all sections, spec/05 §5.1)
  // — no per-point min/max pass, and positions stay untouched.
  g.boundingBox = new THREE.Box3(
    new THREE.Vector3(scene.aabbMin[0], scene.aabbMin[1], scene.aabbMin[2]),
    new THREE.Vector3(scene.aabbMax[0], scene.aabbMax[1], scene.aabbMax[2]),
  );
  g.boundingSphere = g.boundingBox.getBoundingSphere(new THREE.Sphere());
  return g; // dequant happens in the vertex shader (§2.1)
}

/** Result of {@link buildDynamic}: one reusable geometry sized to the busiest
 * frame, plus the backing typed arrays we mutate in place each frame. */
export interface DynamicGeometry {
  geometry: THREE.BufferGeometry;
  /** Max per-frame dynamic point count across all T frames (capacity). */
  maxCount: number;
  /** Writable position attribute (u16, normalized) over `positions`. */
  positionAttr: THREE.BufferAttribute;
  /** Writable color attribute (u8, normalized) over `colors`. */
  colorAttr: THREE.BufferAttribute;
}
