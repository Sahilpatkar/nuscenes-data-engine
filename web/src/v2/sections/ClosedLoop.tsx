import { StatTile } from "../components/StatTile";

import { DeepLink } from "../../components/DeepLink";
import { Ticks } from "../../components/Ticks";
import metaJson from "../../data/meta.json";
import type { ClosedLoop as ClosedLoopData, DeepLink as DeepLinkRef, Meta } from "../../data/types";

const meta = metaJson as Meta;

/**
 * The report's cross-references: every deep dive in the live instrument, in the
 * order the story walked them (the tour first, then the five pages behind steps
 * 1–6). The LABELS are the exporter's own (`meta.deep_links`), so the report
 * never invents a description of a page it does not own, and a path this package
 * did not export is skipped rather than guessed at.
 *
 * They live in this section because a report gathers its references at the end.
 */
const CROSSREF_PATHS = [
  "tour",
  "failures",
  "scenarios",
  "active_learning",
  "weak_supervision",
  "chat_replay",
] as const;

/** SITE-EDITORIAL COPY — the heading over the references. No fact in it. */
const CROSSREF_HEADING = "Go deeper in the live app";

function crossrefs(): DeepLinkRef[] {
  return CROSSREF_PATHS.map((path) =>
    meta.deep_links.find((link) => link.url_path === path),
  ).filter((link): link is DeepLinkRef => link !== undefined);
}

/**
 * Step 7 — the loop, closed (story contract §1.5).
 *
 * The headline is the exporter's guarded one ("… → measurable improvement" only
 * where the night delta really is a gain), and it opens the section as its serif
 * line. Where the guard did not hold the exporter writes null, and the section
 * simply starts on the numbers: the frame's own numbered title already says what
 * step this is, so nothing has to be invented to fill the gap.
 *
 * Then the three numbers the whole story earned; the four questions it can now
 * answer — including the one about what failed, which is written as plainly as
 * the ones about what worked — set as a ruled 2×2 grid, hairline-separated the
 * way a printed table of results is; and the thesis those four answers are the
 * evidence for. The thesis is the report's last sentence: the page's largest
 * pull line, with nothing after it but the cross-references.
 *
 * The accented tile is the last card: the exporter writes the relative-gain card
 * last (`_result_hero_cards`), and the relative gain is the number the story has
 * been chasing since step 1's third tile — the two accents are the same claim,
 * six steps apart.
 */
export function ClosedLoop({ data }: { data: ClosedLoopData }): JSX.Element {
  const lastCard = data.cards.length - 1;
  const links = crossrefs();

  return (
    <>
      {data.headline ? <p className="loop-headline">{data.headline}</p> : null}

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

      <div className="answers">
        {data.answers.map((answer) => (
          <section className="answer" key={answer.question}>
            <h3 className="answer-question">{answer.question}</h3>
            {answer.sentences.map((sentence) => (
              <p className="answer-line" key={sentence}>
                <Ticks text={sentence} />
              </p>
            ))}
          </section>
        ))}
      </div>

      <div className="pull-block">
        {/* The one accent rule the report spends, over its last claim. */}
        <hr className="rule rule-accent" />
        <p className="thesis">{data.closing_thesis}</p>
        <hr className="rule" />
      </div>

      {links.length > 0 ? (
        <nav className="crossref-block" aria-labelledby="crossref-heading">
          <h3 id="crossref-heading" className="eyebrow">
            {CROSSREF_HEADING}
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
