import { useEffect, useState } from "react";

/**
 * The seven-step rail: a fixed right-edge index on wide screens (dot + mono step
 * title), a thin progress bar pinned to the top under ~900 px. One component,
 * one markup — the layout switch is entirely CSS, so the labels stay in the
 * accessibility tree at every width.
 *
 * Every entry is a real `<a href="#id">`: it works before React hydrates, it is
 * reachable by keyboard, and the smooth scroll is the browser's (CSS
 * `scroll-behavior`, already switched off under reduced motion in base.css).
 */

export interface RailStep {
  id: string;
  step: number;
  title: string;
}

/**
 * Which section owns the middle of the viewport. The observer's root is squeezed
 * to a thin horizontal band across the centre, so at most one section is
 * intersecting at a time; ties (a short section, a tall neighbour) resolve to
 * whichever comes first in the story. On the landing nothing is in the band and
 * the rail shows no active step — the progress bar reads empty, which is true.
 */
function useActiveSection(ids: readonly string[]): string | null {
  const [active, setActive] = useState<string | null>(null);
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
        // Nothing in the band (the landing, or a gap): keep the last answer
        // rather than blinking the rail off mid-scroll.
        if (first !== undefined) setActive(first);
      },
      { rootMargin: "-45% 0px -45% 0px", threshold: 0 },
    );
    for (const node of nodes) observer.observe(node);
    return () => observer.disconnect();
  }, [key]);

  return active;
}

export function StepRail({ steps }: { steps: readonly RailStep[] }): JSX.Element {
  const active = useActiveSection(steps.map((entry) => entry.id));
  const activeIndex = steps.findIndex((entry) => entry.id === active);

  return (
    <nav className="step-rail" aria-label="Story steps">
      <ol className="step-rail-list">
        {steps.map((entry, index) => {
          const isActive = entry.id === active;
          // Under ~900 px the rail is a progress bar: everything up to and
          // including the active step reads as covered ground.
          const isPassed = activeIndex >= 0 && index <= activeIndex;
          const classes = ["step-rail-link"];
          if (isActive) classes.push("is-active");
          if (isPassed) classes.push("is-passed");
          return (
            <li key={entry.id}>
              <a
                className={classes.join(" ")}
                href={`#${entry.id}`}
                aria-current={isActive ? "true" : undefined}
              >
                <span className="step-rail-dot" aria-hidden="true" />
                <span className="step-rail-label">
                  <span className="step-rail-number" aria-hidden="true">
                    {entry.step}
                  </span>
                  {entry.title}
                </span>
              </a>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
