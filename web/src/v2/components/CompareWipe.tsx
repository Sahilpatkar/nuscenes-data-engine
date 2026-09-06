import { useState, type CSSProperties } from "react";

import { assetUrl } from "../assetUrl";
import type { ArmLabel, ImageRef } from "../../data/types";

/**
 * One frame, drawn twice — the before/after wipe of story contract §1.4's first
 * evidence tier.
 *
 * FORKED FROM THE ROOT EDITION FOR ONE REASON: its `<img>` srcs are
 * document-relative and 404 from `/v2/`, so all four are resolved through
 * `assetUrl`. Two smaller changes serve the report frame: the root element is a
 * `<div>` rather than a `<figure>` (the numbered `Figure` around it is the
 * figure, and figures do not nest usefully), and the caption is that Figure's
 * rather than this component's. Every mechanic below is the root edition's:
 *
 * The control is a real `<input type="range">`, not a drag handler: it is
 * keyboard-operable for free (arrows move one step, Home/End jump to the ends),
 * it announces itself as a slider, and it works before any JavaScript of ours
 * runs. The visible handle is the line across the image, drawn at the same
 * percentage the clip uses, so the seam and the control are never out of step.
 *
 * The wipe itself is one CSS custom property: the after image sits on top of the
 * before image and is clipped to the left `--wipe` of the frame. There is NO
 * transition on the clip — the seam follows the pointer or the arrow key, which
 * is user-driven motion, not animation, and a reader who asked for reduced
 * motion still gets the full control.
 *
 * Under ~480 px the wipe is replaced by the static side-by-side pair (the live
 * tour's own layout for this step). The swap is a media query over two blocks of
 * markup, not a resize listener: at any width exactly one of them is rendered,
 * and the hidden one is out of the accessibility tree with it.
 */

/** Arrow-key granularity: 20 stops across the frame, which a keyboard can walk. */
const STEP = 5;
const START = 50;

export interface CompareWipeProps {
  before: ImageRef;
  after: ImageRef;
  beforeLabel: ArmLabel;
  afterLabel: ArmLabel;
}

export function CompareWipe({
  before,
  after,
  beforeLabel,
  afterLabel,
}: CompareWipeProps): JSX.Element {
  const [wipe, setWipe] = useState(START);

  const frameStyle = { "--wipe": `${wipe}%` } as CSSProperties;

  return (
    <div className="compare">
      <div className="compare-frame" style={frameStyle}>
        <img
          className="compare-img"
          src={assetUrl(before.src)}
          width={before.width}
          height={before.height}
          alt={before.alt}
          decoding="async"
        />
        <img
          className="compare-img compare-after"
          src={assetUrl(after.src)}
          width={after.width}
          height={after.height}
          alt={after.alt}
          decoding="async"
        />

        <span className="compare-handle" aria-hidden="true" />

        {/* Each end label sits over the image it names — story label above, the raw
            id under it (the spec's honesty rule). */}
        <span className="compare-end compare-end-after">
          <span className="compare-end-label">{afterLabel.label}</span>
          <span className="compare-end-id mono">{afterLabel.id}</span>
        </span>
        <span className="compare-end compare-end-before">
          <span className="compare-end-label">{beforeLabel.label}</span>
          <span className="compare-end-id mono">{beforeLabel.id}</span>
        </span>

        <input
          className="compare-range"
          type="range"
          min={0}
          max={100}
          step={STEP}
          value={wipe}
          onChange={(event) => setWipe(Number(event.target.value))}
          aria-label={`Wipe between ${beforeLabel.label} and ${afterLabel.label}`}
          aria-valuetext={`${wipe}% ${afterLabel.label}`}
        />
      </div>

      {/* The same pair, side by side, for a frame too narrow to wipe inside. */}
      <div className="compare-static">
        {[
          { image: before, model: beforeLabel },
          { image: after, model: afterLabel },
        ].map(({ image, model }) => (
          <figure className="compare-static-item" key={model.id}>
            <img
              src={assetUrl(image.src)}
              width={image.width}
              height={image.height}
              alt={image.alt}
              decoding="async"
            />
            <figcaption>
              <span className="compare-end-label">{model.label}</span>{" "}
              <span className="compare-end-id mono">{model.id}</span>
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}
