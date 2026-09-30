import { assetUrl } from "../../assetUrl";
import { Chain } from "../components/Chain";

import type { Report, WhyFrame } from "../../data/types";

/** Chrome: the chain's accessible name. No fact in it. */
const CHAIN_LABEL = "Selection path";

/**
 * 04. Targeted training-data selection: exactly what the acquisition function
 * does, the four-stage path, and, folded shut, one selected frame with the
 * per-frame facts behind its pick (community, failure mass and rank, quota,
 * which pass took it, similarity-degree rank). The fold keeps those numbers
 * one click away instead of on the page by default.
 */
export function Selection({ data, report }: { data: WhyFrame; report: Report }): JSX.Element {
  const caption = data.train_pool_note ?? data.image.alt;

  return (
    <>
      <p className="lede measure section-lede">{report.selection.lede}</p>

      <Chain steps={report.selection.chain} label={CHAIN_LABEL} />

      <details className="fold">
        <summary>Acquisition details</summary>
        <div className="fold-body">
          <figure className="fold-figure">
            <img
              src={assetUrl(data.image.src)}
              width={data.image.width}
              height={data.image.height}
              // The figcaption below describes the frame; when it falls back to
              // this image's own alt, a non-empty alt would be read twice.
              alt={data.train_pool_note != null ? data.image.alt : ""}
              decoding="async"
            />
            <figcaption className="figure-caption">{caption}</figcaption>
          </figure>

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

          <p className="measure">{data.selection_path}</p>
          <p className="measure">{data.mechanism_sentence}</p>
          {data.flagship_rank_sentence ? (
            <p className="measure">{data.flagship_rank_sentence}</p>
          ) : null}
        </div>
      </details>
    </>
  );
}
