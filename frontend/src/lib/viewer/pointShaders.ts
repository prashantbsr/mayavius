// GLSL for the Path-1 point-cloud ShaderMaterial (spec/07-frontend-spec.md
// §2.1). Both PointCloud layers (static + dynamic) share this exact source:
// positions arrive as `q/65535 ∈ [0,1]` (normalized u16 attribute) and are
// dequantized to world space ON THE GPU with the AABB uniforms — the CPU never
// expands positions to Float32 (spec/05 §1, the load-time win).
//
// Aesthetics levers (risk #4, all uniform-driven, no rebuild): round sprites
// via `discard`, depth-based clamped `gl_PointSize`, and an in-shader sRGB→linear
// conversion so colors aren't washed out. `uOpacity` drives the dynamic layer's
// optional motion fade.

/** Vertex shader: dequant (`world = uAabbMin + position*(uAabbMax-uAabbMin)`)
 * + perspective point sizing (`uPointSize * uViewportHeight / -mv.z`, clamped
 * to [1,12]). `position` is `q/65535` because the attribute is `normalized`. */
export const POINT_VERT = /* glsl */ `
uniform vec3 uAabbMin, uAabbMax;
uniform float uPointSize, uViewportHeight;
varying vec3 vColor;
void main() {
  vColor = color;                                  // attribute (sRGB, see frag)
  vec3 world = uAabbMin + position * (uAabbMax - uAabbMin); // position = q/65535
  vec4 mv = modelViewMatrix * vec4(world, 1.0);
  gl_Position = projectionMatrix * mv;
  // perspective point sizing: nearer points larger, clamped (risk #4 — aesthetics)
  gl_PointSize = clamp(uPointSize * (uViewportHeight / -mv.z), 1.0, 12.0);
}
`;

/** Fragment shader: circular sprite (`discard` outside the unit disc) + sRGB→
 * linear (`pow(vColor, 2.2)`) so colors aren't washed out; `uOpacity` alpha. */
