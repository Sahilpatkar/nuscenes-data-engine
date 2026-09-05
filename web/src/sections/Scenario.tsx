import { useState } from "react";

import { CanCurves } from "../components/charts/CanCurves";
import { Filmstrip } from "../components/Filmstrip";
import { Learned } from "../components/Learned";
import type { Scenario as ScenarioData } from "../data/types";

/**
 * Step 3 — one failure, or a recurring driving scenario?
 *
 * The event card answers "what is this": the top-ranked event of the flagship
 * scenario query, its severity, what is near the ego, and its conditions. The
 * synced pair under it answers "is it a moment or a manoeuvre": five keyframes
 * against the CAN speed and acceleration behind them, sharing one active step —
 * point at a thumb and both curves mark it; point at a curve and the thumb lights.
 *
 * The active step starts on the event's own frame (`is_current`), which is also
 * the one the strip rings permanently. Nothing here reverts on mouse-out: a
 * reader who scrubbed to t+2 meant to be at t+2.
 */
export function Scenario({ data }: { data: ScenarioData }): JSX.Element {
  const { steps } = data.filmstrip;
  const current = steps.findIndex((step) => step.is_current);
  const [active, setActive] = useState(current < 0 ? 0 : current);

  return (
    <>
      <div className="event-card">
        <img
          className="event-image"
          src={data.image.src}
          width={data.image.width}
          height={data.image.height}
          alt={data.image.alt}
          decoding="async"
        />
        <div className="event-facts">
          <h3 className="event-head">
            {data.event.scene_name} · {data.event.severity_caption}
          </h3>
          <p>{data.event.facts}</p>
          <ul className="chip-row">
            {data.event.chips.map((chip) => (
              <li className="chip" key={chip}>
                {chip}
              </li>
            ))}
          </ul>
          <p className="mono event-preset">
            <span className="muted">preset</span> {data.preset}
          </p>
        </div>
      </div>

      {steps.length > 0 ? (
        <div className="sync">
          <Filmstrip steps={steps} active={active} onActivate={setActive} />
          <CanCurves
            steps={steps}
            speedTitle={data.filmstrip.speed_title}
            accelTitle={data.filmstrip.accel_title}
            active={active}
            onActivate={setActive}
          />
          <p className="mono muted sync-caption">{data.filmstrip.caption}</p>
        </div>
      ) : null}

      <p className="mono muted">{data.parity_caption}</p>
      <p className="key-line">{data.night_sentence}</p>
      <p className="measure">{data.mechanism_sentence}</p>

      <Learned text={data.takeaway} />
    </>
  );
}
