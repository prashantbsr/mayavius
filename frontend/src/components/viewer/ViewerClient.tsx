"use client";

import dynamic from "next/dynamic";

// The WebGL canvas is client-only. In Next 16 a dynamic import with
// `ssr: false` MUST live inside a Client Component (it is forbidden in Server
// Components), so this thin wrapper is the boundary the server pages import.
const ViewerCanvas = dynamic(
  () => import("./ViewerCanvas").then((m) => m.ViewerCanvas),
  {
    ssr: false,
    loading: () => (
      <div className="flex flex-1 items-center justify-center text-sm opacity-60">
        Loading viewer…
      </div>
    ),
  },
);

export function ViewerClient({ resultId }: { resultId: string }) {
  return <ViewerCanvas resultId={resultId} />;
}
