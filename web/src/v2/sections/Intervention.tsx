import { StrategyBar } from "../components/charts/StrategyBar";
import { FIGURE, Figure } from "../components/Figure";
import { StatTile } from "../components/StatTile";

import { Ticks } from "../../components/Ticks";
import type { Intervention as InterventionData } from "../../data/types";

/**
 * Step 5 — the intervention (story contract §1.3): what the mining changed about
 * the training set, and how that choice compares with the other ways of spending
 * the same budget.
 *
 * The contract's demand is structural, not decorative: the fairness statement is
 * what makes the chart a comparison rather than a leaderboard, so it is not prose
 * that happens to sit nearby — it is the box the chart is drawn inside. In this
 * edition that box is a hairline rectangle (no fill, no shadow): the lead, the
 * clauses as mono chips, and directly under them Figure 5 and its caption, all in
 * ONE container. The control clause — the arm every other bar is measured
 * against — is the one chip in cobalt. The chart cannot be read without reading
 * what was held constant.
 *
 * Beats: the four cards → the night-share line that says what changed → the fair
 * comparison box (chips, chart, the charted arms' raw ids) → the winner, which is
 * the sentence the section lands on.
 *
 * No tile is accented here. The section's one point of emphasis is the
 * highlighted bar and the winner sentence that names it; a fifth accent in the
 * card row would only compete with them.
 */
export function Intervention({ data }: { data: InterventionData }): JSX.Element {
  /* Which clause is the control clause: the exporter appends it last and joins it
     with an em dash (`_fairness_parts` / `_fairness_sentence`), and writes it at
     all only when the control arm is one of the charted ones. Reading it back off
     the sentence — rather than assuming "the last chip" — means a package without
     a charted control emphasises nothing instead of promoting a held-constant
     clause into the role. */
  const last = data.fairness_parts[data.fairness_parts.length - 1];
  const control =
    last !== undefined && data.fairness_sentence.endsWith(`— ${last}.`) ? last : null;

  return (
    <>
      <div className="stat-row is-quad">
        {data.cards.map((card) => (
          <StatTile key={card.label} label={card.label} value={card.value} detail={card.delta} />
        ))}
      </div>

      {data.night_share_line ? (
        <p className="night-shares">
          <NightShares line={data.night_share_line} />
        </p>
      ) : null}

      <div className="fairness figure-wide">
        <p className="eyebrow">Fair comparison</p>

        <ul className="chip-row fairness-parts">
          {data.fairness_parts.map((part) => (
            <li className={part === control ? "chip is-control" : "chip"} key={part}>
              {part}
            </li>
          ))}
        </ul>

        <Figure
          number={FIGURE.strategyChart}
          caption={<Ticks text={data.charted_ids_caption} />}
        >
          <div className="overflow-x strategy-scroll">
            <StrategyBar chart={data.chart} />
          </div>
        </Figure>
      </div>

      {/* The landing line — and a package where no arm has a night delta to rank
          gets no line at all rather than an empty accent bar. */}
      {data.winner_sentence ? (
        <div className="pull-block">
          <hr className="rule" />
          <p className="pull-line">
            <Ticks text={data.winner_sentence} />
          </p>
          <hr className="rule" />
        </div>
      ) : null}

      {data.similarity_caption ? (
        <p className="muted measure">{data.similarity_caption}</p>
      ) : null}
      {data.spread_sentence ? <p className="measure">{data.spread_sentence}</p> : null}
    </>
  );
}

/**
 * The night-share comparison, with each arm's share picked out of its clause.
 *
 * The tour writes this line as `Label: **31% night**` per clause and Streamlit
 * bolds it; the exporter strips the markers ("Streamlit markdown is not copy"),
 * so the report restores the same emphasis with its own type. Nothing is reworded
 * or reordered: each clause is split at its first ": ", the halves are rendered
 * back to back, and a clause with no separator is rendered whole.
 */
function NightShares({ line }: { line: string }): JSX.Element {
  const separator = " · ";
  return (
    <>
      {line.split(separator).map((clause, index) => {
        const at = clause.indexOf(": ");
        return (
          <span key={clause}>
            {index === 0 ? null : separator}
            {at < 0 ? (
              clause
            ) : (
              <>
                {clause.slice(0, at + 2)}
                <strong>{clause.slice(at + 2)}</strong>
              </>
            )}
          </span>
        );
      })}
    </>
  );
}
