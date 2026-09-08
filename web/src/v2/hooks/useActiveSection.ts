import { useEffect, useState } from "react";

/**
 * Which section owns the middle of the viewport — the report header's scroll-spy.
 *
 * Forked from the root edition's rail, where the same logic is module-private
 * (`components/StepRail.tsx`). The behaviour is deliberately identical, because
 * both editions answer the same question about the same seven sections:
 *
 *   · the observer's root is squeezed to a thin horizontal band across the
 *     centre of the viewport, so at most one section is intersecting at a time;
 *   · ties — a short section under a tall neighbour — resolve to whichever comes
 *     first in the story, not to whichever the observer happened to report last;
 *   · while nothing is in the band (the cover, or a gap) the LAST answer stands,
 *     so the readout never blinks off mid-scroll. Before the first answer the
 *     hook returns null, and the header says so rather than claiming section one.
 *
 * Where `IntersectionObserver` does not exist the hook simply never reports: the
 * header keeps its inert readout and every anchor still works, because they are
 * real `<a href="#id">` links.
 */
export function useActiveSection(ids: readonly string[]): string | null {
  const [active, setActive] = useState<string | null>(null);
  /* The dependency is the ids' CONTENT, not the array identity — callers build
     the list inline on every render. */
  const key = ids.join("|");

  useEffect(() => {
    if (typeof IntersectionObserver !== "function") return;

    const order = key.split("|");
    const nodes = order
      .map((id) => document.getElementById(id))
      .filter((node): node is HTMLElement => node !== null);
    if (nodes.length === 0) return;

    const inBand = new Set<string>();
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) inBand.add(entry.target.id);
          else inBand.delete(entry.target.id);
        }
        const first = order.find((id) => inBand.has(id));
        if (first !== undefined) setActive(first);
      },
      { rootMargin: "-45% 0px -45% 0px", threshold: 0 },
    );
    for (const node of nodes) observer.observe(node);
    return () => observer.disconnect();
  }, [key]);

  return active;
}
