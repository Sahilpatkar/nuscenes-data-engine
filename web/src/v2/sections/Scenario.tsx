import { useState } from "react";

import { assetUrl } from "../../assetUrl";
import { CanCurves } from "../components/charts/CanCurves";
import { FIGURE, Figure } from "../components/Figure";

import { Filmstrip } from "../../components/Filmstrip";
import { Learned } from "../../components/Learned";
import type { Scenario as ScenarioData } from "../../data/types";

/**
 * Step 3 — one failure, or a recurring driving scenario?
 *
 * Figure 2 answers "what is this": the top-ranked event of the flagship scenario
 * query as a situation first (its title, one plain sentence, what is near the
 * vehicle, its conditions), with the row it came from — scene, severity figure,
 * preset — as small technical metadata under it. All of it is a ruled fact
 * block underneath the figure rather than a card beside it: the report reads
 * down one column.
 *
 * Figure 3 answers "is it a moment or a manoeuvre": a sentence saying what to
 * notice, then five keyframes against the CAN speed and acceleration behind
 * them, then the sentence that reads them. The pair shares one active step —
 * point at a thumb and both curves mark it; point at a curve and the thumb
 * lights. The sync
 * is the original edition's mechanic exactly: one `active` index owned here and
 * handed to both halves, starting on the event's own frame (`is_current`), which
 * is also the one the strip rings permanently. Nothing reverts on mouse-out: a
 * reader who scrubbed to t+2 meant to be at t+2.
 */
export function Scenario({ data }: { data: ScenarioData }): JSX.Element {
  const { steps } = data.filmstrip;
  const current = steps.findIndex((step) => step.is_current);
  const [active, setActive] = useState(current < 0 ? 0 : current);

  return (
    <>
      <Figure number={FIGURE.scenarioEvent} wide caption={data.image.alt}>
        <img
          className="event-image"
          src={assetUrl(data.image.src)}
          width={data.image.width}
          height={data.image.height}
          // The figure caption above renders this image's bundle alt text
          // verbatim, so a non-empty alt here would be announced twice inside
          // one <figure> (final-review M2). The figcaption is the description.
          alt=""
          decoding="async"
        />
      </Figure>

      <div className="event-facts">
        <h3 className="event-head">{data.event.title}</h3>
        <p>{data.event.summary}</p>
        <p>{data.event.facts}</p>
        <ul className="chip-row">
          {data.event.chips.map((chip) => (
            <li className="chip" key={chip}>
              {chip}
            </li>
          ))}
        </ul>
        <p className="mono event-preset">
          <span className="muted">scene</span> {data.event.scene_name}
          <span aria-hidden="true"> · </span>
          {data.event.severity_caption}
          <span aria-hidden="true"> · </span>
          <span className="muted">preset</span> {data.preset}
        </p>
      </div>

      {steps.length > 0 ? (
        <>
          <p className="measure">{data.timeline_lede}</p>
          <Figure number={FIGURE.filmstrip} wide caption={data.filmstrip.caption}>
            <div className="sync">
              <Filmstrip steps={steps} active={active} onActivate={setActive} />
              <CanCurves
                steps={steps}
                speedTitle={data.filmstrip.speed_title}
                accelTitle={data.filmstrip.accel_title}
                active={active}
                onActivate={setActive}
              />
            </div>
          </Figure>
          {data.timeline_sentence ? <p className="measure">{data.timeline_sentence}</p> : null}
        </>
      ) : null}

      <p className="key-line">
        {data.found_sentence}
        <br />
        {data.night_sentence}
      </p>
      <p className="mono muted">{data.parity_caption}</p>
      <p className="measure">{data.mechanism_sentence}</p>

      <Learned text={data.takeaway} />
    </>
  );
}
