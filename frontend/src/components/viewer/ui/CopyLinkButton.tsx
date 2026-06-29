"use client";

import { useState } from "react";
import { SITE_URL } from "@/config";

// CopyLinkButton — PLAIN DOM, the named virality surface (handover §4.4: the
// shareable result link IS the growth loop). NO three / Path-1 import (spec/07
// §1 render-path decoupling rule). It needs no viewer state, only the result id,
// so it takes `resultId` as a prop rather than reading the store.
//
// Share semantics (spec/07 §8 — the `/view/[id]` route is the share target):
// build `${SITE_URL}/view/${resultId}` and, on mobile, prefer the native share
// sheet (`navigator.share`) so the link drops straight into Messages/WhatsApp;
// elsewhere fall back to `navigator.clipboard.writeText`. Either way show a
// transient "Copied!" confirmation (2s) via local state — no global store write.
//
// Styling matches the other HUD buttons (PlaybackControls / BulletTimeButton):
// the same `rounded … px-3 py-1 text-xs font-medium` vocabulary. The overlay
// root is pointer-events:none, so this re-enables pointer events on itself.

export function CopyLinkButton({ resultId }: { resultId: string }) {
  const [copied, setCopied] = useState(false);

  const onShare = async () => {
    const url = `${SITE_URL}/view/${resultId}`;

    // Prefer the native share sheet on mobile (it owns its own UI, so no local
    // confirmation is needed for that path); fall back to the clipboard.
    if (typeof navigator !== "undefined" && typeof navigator.share === "function") {
      try {
        await navigator.share({ url });
        return;
      } catch {
        // User dismissed the sheet, or share is unavailable here — fall through
        // to the clipboard copy below.
      }
    }

    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      // Clear the confirmation after 2s (transient — no global state).
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard blocked (e.g. insecure context) — leave the button unchanged;
      // the user can still copy the address bar.
    }
  };

  return (
    <button
      type="button"
      aria-label="Copy share link"
      onClick={() => void onShare()}
      className="pointer-events-auto rounded bg-white/10 px-3 py-1 text-xs font-medium text-white hover:bg-white/20"
    >
      {copied ? "Copied!" : "Copy link"}
    </button>
  );
}
