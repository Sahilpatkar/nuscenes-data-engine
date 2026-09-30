import { StatTile } from "../components/StatTile";

import type { Report } from "../../data/types";

/** Chrome: the heading over the study-design comparison. No fact in it. */
const DESIGN_HEADING = "Study design";

/**
 * 01. Baseline failure analysis: the finding in one sentence, what the metric
 * means, the three slice values, and the exact gaps between them. Then the study
 * design, baseline against the engine, and the question the experiment answers,
 * so a reader knows what was built before the first figure.
 */
export function Baseline({ data }: { data: Report }): JSX.Element {
  const { baseline, design } = data;
  const last = baseline.cards.length - 1;

  return (
    <>
      <p className="lede measure section-lede">{baseline.lede}</p>
      <p className="muted measure small">{baseline.metric_definition}</p>

      <div className="stat-row">
        {baseline.cards.map((card, index) => (
          <StatTile key={card.label} label={card.label} value={card.value} accent={index === last} />
        ))}
      </div>

      <p className="key-line">{baseline.comparison}</p>

      <section className="design" aria-labelledby="design-heading">
        <h3 id="design-heading" className="eyebrow">
          {DESIGN_HEADING}
        </h3>
        <div className="overflow-x">
          <table className="design-table">
            <thead>
              <tr>
                <td />
                {design.columns.map((column) => (
                  <th scope="col" key={column}>
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {design.rows.map((row) => (
                <tr key={row.label}>
                  <th scope="row">{row.label}</th>
                  <td>{row.baseline}</td>
                  <td>{row.engine}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="measure goal">{design.goal}</p>
      </section>
    </>
  );
}
