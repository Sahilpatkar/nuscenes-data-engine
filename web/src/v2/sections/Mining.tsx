import { useState } from "react";

import { assetUrl } from "../../assetUrl";
import { CanCurves } from "../components/charts/CanCurves";
import { FactList } from "../components/FactList";
import { FIGURE, Figure } from "../components/Figure";

import { Filmstrip } from "../../components/Filmstrip";
import type { Report, Scenario } from "../../data/types";

/**
 * 03. Context-aware failure mining: what the search matches on, the top-ranked
 * event, the query / result / validation facts, and the evidence: five keyframes
 * against the CAN speed and acceleration behind them, sharing one active step
 * (point at a thumb and both curves mark it; the strip starts on the event's own
 * frame). The figures carry the argument; there is no sentence narrating them.
 */
export function Mining({ data, report }: { data: Scenario; report: Report }): JSX.Element {
  const { steps } = data.filmstrip;
  const current = steps.findIndex((step) => step.is_current);
  const [active, setActive] = useState(current < 0 ? 0 : current);

  return (
    <>
      <p className="lede measure section-lede">{report.mining.lede}</p>

      <Figure number={FIGURE.scenarioEvent} wide caption={report.mining.event_caption}>
        <img
          className="event-image"
          src={assetUrl(data.image.src)}
          width={data.image.width}
          height={data.image.height}
          alt={data.image.alt}
          decoding="async"
        />
      </Figure>

      <FactList facts={report.mining.facts} />

      {steps.length > 0 ? (
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
      ) : null}
    </>
  );
}
