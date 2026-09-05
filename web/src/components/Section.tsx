import type { ReactNode } from "react";

import { useReveal } from "../hooks/useReveal";
import type { Provenance } from "../data/types";

/**
 * The frame every story section is poured into: the deck's two-part eyebrow, the
 * headline the exporter derived, the section's own content, and the provenance
 * row that says where its numbers came from.
 *
 * `headline` is nullable because the bundle's are: a headline the exporter
 * declined to write (its guard did not hold on this package) must not be
 * replaced by an invented one. The fallback is the step's fixed title, which
 * claims nothing.
 *
 * `tone` is the one visual choice the frame offers: "panel" gives the section a
 * full-bleed surface band of its own. Exactly one step gets it — the centrepiece
 * (story contract §1.2) — so the band reads as "this is the turn", not as
 * decoration.
 */

/** The tour's provenance vocabulary, as glyphs — the sentence carries the meaning. */
const PROVENANCE_ICON: Record<Provenance["kind"], string> = {
  recomputed: "∑",
  recorded: "↺",
  reproduced: "✓",
};

export interface SectionProps {
  id: string;
  /** 1-based position in the seven-step story. */
  step: number;
  /** The loop stage this step lights (Diagnose / Mine / Train / Evaluate). */
  stage: string;
  /** The deck's act this step belongs to, right-aligned in the eyebrow. */
  act: string;
  /** The step's fixed title — also its label in the rail. */
  title: string;
  /** The derived headline, or null where the exporter declined to write one. */
  headline: string | null;
  /** "panel": a full-bleed surface band behind the whole section. */
  tone?: "panel" | undefined;
  provenance: Provenance[];
  children?: ReactNode;
}

export function Section({
  id,
  step,
  stage,
  act,
  title,
  headline,
  tone,
  provenance,
  children,
}: SectionProps): JSX.Element {
  const reveal = useReveal<HTMLDivElement>();

  return (
    <section
      id={id}
      className={tone === "panel" ? "section section-feature" : "section"}
      aria-labelledby={`${id}-headline`}
    >
      <div className="page reveal" ref={reveal}>
        <p className="eyebrow section-eyebrow">
          <span>
            Step {step} · {stage}
          </span>
          <span className="muted">{act}</span>
        </p>

        <h2 id={`${id}-headline`} className="section-headline">
          {headline ?? title}
        </h2>

        <div className="section-body">{children}</div>

        <ul className="provenance">
          {provenance.map((entry) => (
            <li key={`${entry.kind}-${entry.detail}`}>
              <span className="provenance-icon" aria-hidden="true">
                {PROVENANCE_ICON[entry.kind]}
              </span>
              <span>
                {entry.sentence}
                {entry.detail ? ` · ${entry.detail}` : ""}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
