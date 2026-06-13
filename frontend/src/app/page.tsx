import Link from "next/link";
import { SITE_NAME, SITE_TAGLINE } from "@/config";

// Landing page — a Server Component, so it is fully static and SEO-indexable.
// SCAFFOLDING PLACEHOLDER: the real hero, preloaded example gallery and upload
// widget are built later per spec/07-frontend-spec.md and the launch assets in
// spec/11-deployment-and-launch.md.
export default function Home() {
  return (
    <main className="flex flex-1 flex-col items-center justify-center gap-6 p-8 text-center">
      <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">{SITE_NAME}</h1>
      <p className="max-w-xl text-lg text-balance opacity-80">{SITE_TAGLINE}</p>
      <p className="max-w-md text-sm opacity-60">
        Open-source · lightweight · interactive · no GPU required to view —
        running real frontier 4D research models. Drop in your own clip.
      </p>
      <Link
        href="/view/demo"
        className="rounded-full bg-foreground px-6 py-3 text-sm font-medium text-background"
      >
        Open the viewer
      </Link>
      <p className="text-xs opacity-40">Scaffolding only — not yet functional.</p>
    </main>
  );
}
