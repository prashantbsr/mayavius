import { SITE_NAME, SITE_TAGLINE } from "@/config";

// Hero — a Server Component (no client JS), so the landing copy is fully static
// and SEO-indexable (spec/07 §1, §8 — Server Components by default). The headline
// and tagline come from config.ts (SITE_NAME / SITE_TAGLINE) so launch copy is
// edited in one place (spec/11).
export function Hero() {
  return (
    <section className="flex flex-col items-center gap-4 text-center">
      <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
        {SITE_NAME}
      </h1>
      <p className="max-w-xl text-lg text-balance opacity-80">{SITE_TAGLINE}</p>
      <p className="max-w-md text-sm opacity-60">
        Open-source · lightweight · interactive · no GPU required to view —
        running real frontier 4D research models. Drop in your own clip.
      </p>
    </section>
  );
}
