import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Hide the dev-only on-screen route indicator. In `next dev` it renders a fixed
  // badge at bottom-left — exactly over the viewer's bottom control bar (Play /
  // Loop, ViewerOverlay.tsx) — and its <nextjs-portal> intercepts pointer events,
  // blocking e2e clicks (T-404). This is a dev-only overlay; hiding it changes no
  // app behaviour and Next still surfaces compile/runtime errors (Next 16 docs:
  // devIndicators.md). The e2e gate runs against `next dev` (spec/10 §4 webServer).
  devIndicators: false,
};

export default nextConfig;
