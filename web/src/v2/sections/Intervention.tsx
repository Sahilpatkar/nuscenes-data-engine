import type { Report } from "../../data/types";

/**
 * 05. Training intervention: what was added, how its night coverage compares with
 * the two reference strategies, and what every arm held constant. The targeted
 * arm's share is the one set in the accent.
 */
export function Intervention({ report }: { report: Report }): JSX.Element {
  const { intervention } = report;

  return (
    <>
      <p className="key-line">{intervention.headline}</p>

      <dl className="fact-list shares">
        {intervention.night_shares.map((share) => (
          <div className={share.highlight ? "fact-row is-accent" : "fact-row"} key={share.label}>
            <dt>{share.label}</dt>
            <dd className="tnum">{share.value}</dd>
          </div>
        ))}
      </dl>

      <p className="measure">{intervention.fairness}</p>
    </>
  );
}
