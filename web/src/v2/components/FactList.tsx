import type { LabelledValue } from "../../data/types";

/**
 * Labelled facts under a figure: "Baseline: pedestrian missed", one per row.
 * A description list, because each row IS a term and its value; the labels and
 * values are the bundle's (`report.json`), never typed here.
 */
export function FactList({ facts }: { facts: readonly LabelledValue[] }): JSX.Element {
  return (
    <dl className="fact-list">
      {facts.map((fact) => (
        <div className="fact-row" key={fact.label}>
          <dt>{fact.label}</dt>
          <dd>{fact.value}</dd>
        </div>
      ))}
    </dl>
  );
}
