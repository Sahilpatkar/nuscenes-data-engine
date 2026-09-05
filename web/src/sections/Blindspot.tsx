import { Learned } from "../components/Learned";
import { StatTile } from "../components/StatTile";
import { Ticks } from "../components/Ticks";
import type { Blindspot as BlindspotData } from "../data/types";

/**
 * Step 1 — the opening beat, story contract §1.1, in its three fixed beats:
 *
 *   (a) the problem in plain language, before a single number;
 *   (b) the numbers — the three baseline metrics, the counted miss line, and the
 *       sentence that ties them to the arm they were recorded for;
 *   (c) the purpose: what the system exists to do about it. It closes the section
 *       because by the end of step 1 the viewer must know what this thing *is*.
 *
 * The accented tile is the last card. The exporter writes the three in a fixed
 * order (overall → night → night pedestrian, `build_blindspot`), so the last one
 * is the night-pedestrian metric: the case the rest of the story chases.
 */
export function Blindspot({ data }: { data: BlindspotData }): JSX.Element {
  const lastCard = data.cards.length - 1;

  return (
    <>
      <p className="lede section-lede">{data.problem_sentence}</p>

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

      <p className="pull-line">{data.purpose_sentence}</p>
    </>
  );
}
