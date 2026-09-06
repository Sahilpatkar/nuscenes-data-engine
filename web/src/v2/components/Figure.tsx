import type { ReactNode } from "react";

/**
 * A numbered figure — how a report points at a picture.
 *
 * The NUMBER is frame: it comes from the document's fixed layout (the sequence
 * below), not from the package, and it exists so the prose could say "Figure 3"
 * and mean one thing. The CAPTION is never frame — every caption passed in is a
 * bundle string (a `caption`, a `train_pool_note`, an image's own `alt`), so a
 * re-export moves what the figure claims without anyone editing a component.
 *
 * `wide` breaks the figure out of the text measure to the column's full width
 * (`.figure-wide`); the caption stays narrow underneath it, because a caption is
 * prose and prose keeps its measure.
 */

/**
 * The document's figure sequence, in reading order — one place to look, so the
 * numbers cannot silently double up or skip. They are hard-coded because the
 * report's layout is: seven sections in a fixed order, six figures among them.
 * A section that renders its figure conditionally (the filmstrip, on a package
 * with no keyframes) leaves a gap in the sequence rather than renumbering the
 * ones after it — a stable number is worth more than a gapless one.
 */
export const FIGURE = {
  heroOverlay: 1,
  scenarioEvent: 2,
  filmstrip: 3,
  selectedFrame: 4,
  strategyChart: 5,
  compareWipe: 6,
} as const;

export interface FigureProps {
  /** Its number in the document's sequence — pass a `FIGURE` member. */
  number: number;
  /** The bundle's own words for what this figure shows. */
  caption: ReactNode;
  /** Break out of the text measure to the column's full width. */
  wide?: boolean;
  /** A layout class for the figure's own contents (e.g. the overlay stack). */
  className?: string | undefined;
  children: ReactNode;
}

export function Figure({
  number,
  caption,
  wide = false,
  className,
  children,
}: FigureProps): JSX.Element {
  const classes = ["figure"];
  if (wide) classes.push("figure-wide");
  if (className !== undefined) classes.push(className);

  return (
    <figure className={classes.join(" ")}>
      {children}
      <figcaption className="figure-caption">
        <span className="figure-number">Figure {number}</span>
        <span aria-hidden="true"> — </span>
        {caption}
      </figcaption>
    </figure>
  );
}
