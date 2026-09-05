import type { WhyFrame as WhyFrameData } from "../data/types";

/**
 * Step 4 — the centrepiece (story contract §1.2): what makes this a data engine
 * rather than "the model failed, retrain it".
 *
 * Four beats, loudest first:
 *   (a) the lede — the system SEARCHES the training pool for the diagnosed
 *       failure instead of adding more images at random;
 *   (b) the plain-language chain as the section's visual device: four stages,
 *       arrows between them, the outcome accented. Horizontal on desktop (it
 *       scrolls inside itself rather than widening the page), stacked on mobile
 *       where the arrows turn to point down;
 *   (c) the concrete frame it picked, with the recorded reasons as chips;
 *   (d) a native `<details>` fold with the implementation chain, the seven
 *       recorded factors and the sentence that says no single one of them chose
 *       the frame. Native, so it opens with no JavaScript and prints open.
 *
 * The section wears the page's panel band (`tone="panel"` on Section) — the one
 * step given a surface of its own, because it is the step the story turns on.
 */
export function WhyFrame({ data }: { data: WhyFrameData }): JSX.Element {
  const lastStage = data.selection_chain.length - 1;

  return (
    <>
      <p className="lede section-lede">{data.lede_sentence}</p>

      <div className="chain-scroll overflow-x">
        <ol className="chain">
          {data.selection_chain.map((stage, index) => (
            <li
              className={index === lastStage ? "chain-step is-outcome" : "chain-step"}
              key={stage}
            >
              <span className="chain-num mono">{String(index + 1).padStart(2, "0")}</span>
              <span className="chain-text">{stage}</span>
            </li>
          ))}
        </ol>
      </div>

      <figure className="frame-figure">
        <img
          src={data.image.src}
          width={data.image.width}
          height={data.image.height}
          alt={data.image.alt}
          decoding="async"
        />
        {data.train_pool_note ? (
          <figcaption className="mono muted">{data.train_pool_note}</figcaption>
        ) : null}
      </figure>

      <ul className="chip-row chip-row-accent">
        {data.chips.map((chip) => (
          <li className="chip" key={chip}>
            {chip}
          </li>
        ))}
      </ul>

      <details className="fold">
        <summary>How selection works</summary>
        <div className="fold-body">
          <p className="mono fold-path">{data.selection_path}</p>

          <dl className="factors">
            {data.factors.map((factor) => (
              <div className="factor" key={factor.label}>
                <dt>{factor.label}</dt>
                <dd>
                  {factor.value}
                  {factor.flag === null ? null : (
                    <span className={factor.flag ? "flag is-yes" : "flag is-no"} aria-hidden="true">
                      {factor.flag ? " ✓" : " ✗"}
                    </span>
                  )}
                </dd>
              </div>
            ))}
          </dl>

          <p className="measure">{data.mechanism_sentence}</p>
        </div>
      </details>

      {data.flagship_rank_sentence ? (
        <p className="key-line">{data.flagship_rank_sentence}</p>
      ) : null}
      {data.weak_rejected_sentence ? (
        <p className="measure">{data.weak_rejected_sentence}</p>
      ) : null}

      <p className="mono muted architecture-strip">{data.architecture_strip}</p>
    </>
  );
}
