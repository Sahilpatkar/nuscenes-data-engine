import { Chain } from "../components/Chain";
import { StrategyBar } from "../components/charts/StrategyBar";
import { CompareWipe } from "../components/CompareWipe";
import { FIGURE, Figure } from "../components/Figure";
import { StatTile } from "../components/StatTile";

import { DeepLink } from "../../components/DeepLink";
import { OverlayLegend } from "../../components/OverlayLegend";
import { Ticks } from "../../components/Ticks";
import metaJson from "../../data/meta.json";
import type { DeepLink as DeepLinkRef, Intervention, Meta, Report, Verdict } from "../../data/types";

const meta = metaJson as Meta;

/* Chrome: headings and accessible names. No fact in any of them. */
const LIMITATIONS_HEADING = "Limitations";
const TOOLING_HEADING = "Interactive tooling";
const PIPELINE_LABEL = "Pipeline";
const SUMMARY_LABEL = "Result summary";

/** The live app's deep pages, in the order a reader would go deeper. */
const TOOLING_PATHS = [
  "failures",
  "scenarios",
  "active_learning",
  "weak_supervision",
  "chat_replay",
  "tour",
] as const;

function tooling(): DeepLinkRef[] {
  return TOOLING_PATHS.map((path) => meta.deep_links.find((link) => link.url_path === path)).filter(
    (link): link is DeepLinkRef => link !== undefined,
  );
}

/**
 * 06. Results: the quantitative result first (the diagnosed slice, then night
 * overall, then the overall trade-off), the acquisition-strategy comparison, one
 * qualitative before/after, the pipeline and its result in one line each, the
 * limitations, and the doors into the live app.
 */
export function Results({
  report,
  intervention,
  verdict,
}: {
  report: Report;
  intervention: Intervention;
  verdict: Verdict;
}): JSX.Element {
  const { results } = report;
  const links = tooling();

  return (
    <>
      <div className="stat-row">
        {results.cards.map((card, index) => (
          <StatTile
            key={card.label}
            label={card.label}
            value={card.value}
            detail={card.delta}
            accent={index === 0}
          />
        ))}
      </div>
      <p className="muted measure small">
        <Ticks text={results.overall_note} />
      </p>

      <Figure number={FIGURE.strategyChart} caption={<Ticks text={intervention.charted_ids_caption} />}>
        <div className="overflow-x strategy-scroll">
          <StrategyBar chart={intervention.chart} />
        </div>
      </Figure>

      <Figure
        number={FIGURE.compareWipe}
        wide
        caption={
          <>
            {results.example_caveat}
            {verdict.held_out_caption ? ` ${verdict.held_out_caption}.` : null}
          </>
        }
      >
        <CompareWipe
          before={verdict.images.baseline}
          after={verdict.images.arm}
          beforeLabel={verdict.models.baseline}
          afterLabel={verdict.models.arm}
        />
        <OverlayLegend items={verdict.legend} />
      </Figure>

      <Chain steps={results.pipeline} numbered={false} label={PIPELINE_LABEL} />
      <Chain steps={results.summary} numbered={false} label={SUMMARY_LABEL} />

      <section className="limitations" aria-labelledby="limitations-heading">
        <h3 id="limitations-heading" className="eyebrow">
          {LIMITATIONS_HEADING}
        </h3>
        <ul className="limitation-list">
          {results.limitations.map((line) => (
            <li className="measure" key={line}>
              {line}
            </li>
          ))}
        </ul>
      </section>

      {links.length > 0 ? (
        <nav className="crossref-block" aria-labelledby="tooling-heading">
          <h3 id="tooling-heading" className="eyebrow">
            {TOOLING_HEADING}
          </h3>
          <ul className="crossref-links">
            {links.map((link) => (
              <li key={link.url}>
                <DeepLink href={link.url}>{link.label}</DeepLink>
              </li>
            ))}
          </ul>
        </nav>
      ) : null}
    </>
  );
}
