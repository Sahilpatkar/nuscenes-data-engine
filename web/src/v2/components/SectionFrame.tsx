import type { ReactNode } from "react";

import { useReveal } from "../../hooks/useReveal";
import type { Provenance } from "../../data/types";

/**
 * The numbered frame every section of the report is poured into.
 *
 * A section opens on a full-measure hairline, carries a two-digit index beside
 * the step's PINNED title ("01 / We found a blind spot" — the numbering is the
 * report's frame, the title string is the live tour's own), states its loop
 * stage and act in small caps on the right, and closes on a footnote block: a
 * short rule, then the bundle's provenance entries set small and muted, the way
 * a printed report sources its figures.
 *
 * Everything above is FRAME. The section's own content — figures, charts,
 * evidence tiers — arrives as `children`; the frame renders no copy of its own
 * beyond the numbering, and paraphrases no provenance sentence.
 *
 * The reveal is the shared hook, so this edition fades in exactly like the root
 * one and is equally inert under `prefers-reduced-motion`.
 */

/**
 * The report's index convention: two digits, so "01 / 07" sets in a mono column
 * without the seven jumping a character to the left. Lives here because the
 * frame owns the numbering; the header and the cover import it so all three
 * readouts cannot drift apart.
 */
export function reportIndex(step: number): string {
  return String(step).padStart(2, "0");
}

export interface SectionFrameProps {
  /** The anchor — the contents link's target and the scroll-spy's key. */
  id: string;
  /** 1-based position in the seven-step story. */
  step: number;
  /** The step's fixed title, as the live tour writes it. */
  title: string;
  /** The closed-improvement-loop stage this step lights. */
  stage: string;
  /** The deck's act, so both tellings are structured the same way. */
  act: string;
  provenance: Provenance[];
  children?: ReactNode;
}

export function SectionFrame({
  id,
  step,
  title,
  stage,
  act,
  provenance,
  children,
}: SectionFrameProps): JSX.Element {
  const reveal = useReveal<HTMLDivElement>();

  return (
    <section id={id} className="section-frame" aria-labelledby={`${id}-title`}>
      <div className="page reveal" ref={reveal}>
        <hr className="rule" />

        <div className="section-head">
          {/* The index is decoration on a heading whose accessible name must stay
              the pinned title exactly — hence aria-hidden on the numeral. */}
          <h2 id={`${id}-title`} className="section-title">
            <span className="section-index" aria-hidden="true">
              {reportIndex(step)} /{" "}
            </span>
            {title}
          </h2>
          <p className="eyebrow section-stage">
            {stage}
            <span aria-hidden="true"> · </span>
            {act}
          </p>
        </div>

        {children ? <div className="section-body">{children}</div> : null}

        {provenance.length > 0 ? (
          <div className="section-notes">
            <hr className="rule-short" />
            {provenance.map((entry, index) => (
              <p className="footnote" key={`${entry.kind}-${entry.detail}`}>
                <sup className="footnote-mark">{index + 1}</sup>
                {entry.sentence}
                {entry.detail ? ` — ${entry.detail}` : ""}
              </p>
            ))}
          </div>
        ) : null}
      </div>
    </section>
  );
}
