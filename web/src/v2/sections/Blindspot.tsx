import { StatTile } from "../components/StatTile";

import { Learned } from "../../components/Learned";
import { Ticks } from "../../components/Ticks";
import type { Blindspot as BlindspotData } from "../../data/types";

/**
 * Step 1 — the opening beat, story contract §1.1, in its three fixed beats:
 *
 *   (a) the problem in plain language, before a single number — set as the
 *       report's serif italic lede, the way a paper states its finding before
 *       it tabulates it;
 *   (b) the numbers — the three baseline metrics as a ruled row of tiles, the
 *       counted miss line, and the sentence that ties them to the arm they were
 *       recorded for;
 *   (c) the purpose: what the system exists to do about it. It closes the
 *       section, set between two rules as the one line of the page that is about
 *       the system rather than the measurement, because by the end of step 1 the
 *       viewer must know what this thing *is*.
 *
 * The accented tile is the last card. The exporter writes them in a fixed order
 * (overall → night → night pedestrian, `build_blindspot`), so the last one is the
 * night-pedestrian metric: the case the rest of the story chases. On a package
 * whose eval wrote no per-class night metric the exporter drops that card rather
 * than shipping "nan", and the accent falls on the night metric instead — still
 * the worst reading on the row, which is the point the accent is making.
 */
export function Blindspot({ data }: { data: BlindspotData }): JSX.Element {
  const lastCard = data.cards.length - 1;

  return (
    <>
      <p className="lede measure section-lede">{data.problem_sentence}</p>

      <div className="stat-row">
        {data.cards.map((card, index) => (
          <StatTile
            key={card.label}
            label={card.label}
            value={card.value}
            detail={card.delta}
            accent={index === lastCard}
          />
        ))}
      </div>

      <p className="key-line">{data.miss_sentence}</p>
      <p className="measure">
        <Ticks text={data.derived_sentence} />
      </p>

      <Learned text={data.takeaway} />

      <div className="pull-block">
        <hr className="rule" />
        <p className="pull-line">{data.purpose_sentence}</p>
        <hr className="rule" />
      </div>
    </>
  );
}
