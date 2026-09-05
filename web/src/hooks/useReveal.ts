import { useEffect, useRef, type RefObject } from "react";

/**
 * Scroll reveal — pure enhancement, never a gate on content.
 *
 * The element is in the DOM and fully readable at all times; `.reveal` only opts
 * it into a fade/rise that this hook completes by adding `.is-visible`. Three
 * rules keep that honest:
 *
 *   1. Under `prefers-reduced-motion: reduce` the hook registers NOTHING — no
 *      observer, no class. base.css only hides `.reveal` inside a
 *      `(prefers-reduced-motion: no-preference)` block, so on that machine the
 *      section was never hidden in the first place.
 *   2. Where `IntersectionObserver` does not exist, the class is added
 *      immediately rather than never.
 *   3. Once revealed the element is unobserved: a section fades in once, and
 *      scrolling back up never re-animates it.
 *
 * Threshold: the spec asks for "a quarter in", but a ratio threshold of 0.25 can
 * never be reached by a section TALLER than the viewport (its ratio peaks below
 * it), which would leave the longest sections permanently hidden. The equivalent
 * that holds at any height is a root shrunk 25 % at the bottom with threshold 0:
 * the element reveals as it crosses into the top three quarters of the viewport.
 */

const VISIBLE = "is-visible";
const REDUCED_MOTION = "(prefers-reduced-motion: reduce)";

function prefersReducedMotion(): boolean {
  if (typeof window.matchMedia !== "function") return false;
  return window.matchMedia(REDUCED_MOTION).matches;
}

export function useReveal<T extends HTMLElement>(): RefObject<T> {
  const ref = useRef<T>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (prefersReducedMotion()) return;

    if (typeof IntersectionObserver !== "function") {
      node.classList.add(VISIBLE);
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          entry.target.classList.add(VISIBLE);
          observer.unobserve(entry.target);
        }
      },
      { threshold: 0, rootMargin: "0px 0px -25% 0px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return ref;
}
