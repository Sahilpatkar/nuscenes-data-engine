/**
 * A left-to-right sequence of stages (the selection path, the pipeline, the
 * result summary): an ordered list, so a screen reader hears the order, with the
 * arrows drawn by CSS rather than typed as text. The last stage is the outcome
 * and the only one in the accent. Stage names come from the bundle.
 */
export function Chain({
  steps,
  numbered = true,
  label,
}: {
  steps: readonly string[];
  numbered?: boolean;
  label: string;
}): JSX.Element {
  const last = steps.length - 1;
  return (
    <div className="chain-band">
      <div className="chain-scroll overflow-x">
        <ol className="chain" aria-label={label}>
          {steps.map((step, index) => (
            <li className={index === last ? "chain-step is-outcome" : "chain-step"} key={step}>
              {numbered ? (
                <span className="chain-num tnum">{String(index + 1).padStart(2, "0")}</span>
              ) : null}
              <span className="chain-text">{step}</span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
