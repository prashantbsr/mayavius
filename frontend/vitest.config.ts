import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

// Vitest config for the frontend unit suite (spec/10 §1.3, D8 / spec/08 §2).
// Covers the MV4D decoder, the Zustand store, and pure shader-math helpers — no
// WebGL context is required (the GLSL itself is covered by e2e T-4xx). Fixtures
// (`*.mv4d` binaries + `golden_expected.json`) are read from disk via node `fs`
// with absolute paths inside the tests — the simplest, most robust path (it
// avoids bundler import assertions for binary assets).
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Mirror the `@/*` → `./src/*` path mapping from tsconfig.json so test
      // imports resolve the same way the app does.
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
});
