import { ImageResponse } from "next/og";
import { SITE_NAME } from "@/config";

// Per-result dynamic Open Graph image (spec/07 §8): a 1200x630 share card with a
// dark background + the "mayavius" wordmark + the result id. This OVERRIDES the
// site-wide /og.png default for /view/[id] only (layout.tsx still supplies the
// static fallback elsewhere). Until the representative-frame thumbnail pipeline
// exists (spec/11), this is a branded text card — no model frame yet.
//
// next/og's ImageResponse satisfies the opengraph-image return contract; it ships
// with Next and needs no new dependency. If it ever fails to build, the static
// fallback path is layout.tsx's /og.png (still resolved by the per-route OG tags
// in generateMetadata). Twitter reuses this image (no separate twitter-image.tsx
// → Next maps og:image to twitter:image via the summary_large_image card).

export const alt = `${SITE_NAME} — interactive 4D reconstruction`;
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

type Props = { params: Promise<{ id: string }> };

export default async function Image({ params }: Props) {
  const { id } = await params; // params is async in Next 16 — must be awaited

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-start",
          justifyContent: "center",
          gap: 28,
          padding: "80px 96px",
          background:
            "radial-gradient(120% 120% at 0% 0%, #161827 0%, #07080d 60%)",
          color: "#f5f6fb",
          fontFamily: "sans-serif",
        }}
      >
        <div
          style={{
            display: "flex",
            fontSize: 96,
            fontWeight: 700,
            letterSpacing: "-0.04em",
          }}
        >
          {SITE_NAME}
        </div>
        <div
          style={{
            display: "flex",
            fontSize: 36,
            opacity: 0.78,
          }}
        >
          interactive 4D scene reconstruction
