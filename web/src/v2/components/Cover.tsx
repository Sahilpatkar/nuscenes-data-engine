import { reportIndex } from "./SectionFrame";

import metaJson from "../../data/meta.json";
import type { Meta } from "../../data/types";

const meta = metaJson as Meta;

/**
 * The report's masthead: the project's name, the title, one sentence on what the
 * project does, the package metadata, and the contents. It states the project
 * once; the name, title and description each say something the others do not.
 *
 * SITE-EDITORIAL COPY: the three constants below are the site's own fixed words
 * and the only prose on this page the bundle does not supply. DEK is, verbatim,
 * the `<meta name="description">` of `web/index.html` (a test pins the pair).
 * None of them contains a number, so nothing here can drift from the package;
 * the version, build date and commit are read from `meta.json`.
 */
const SERIES = "Perception Data Engine";

const COVER_TITLE = "Improving perception models through targeted data selection";

const DEK =
  "A data engine that identifies failure modes in an object detector, mines relevant " +
  "training samples from nuScenes, and measures the effect of targeted retraining.";

/** Abbreviated-commit length — the dateline is the one place the commit appears. */
const SHA_CHARS = 7;

/* The cover owns both anchors and exports them, because the running head links
   to them: one definition, so an anchor cannot go stale in one file only. */
/** The contents block — where the running head points while you are up here. */
export const CONTENTS_ID = "contents";
/** The top of the report, and the wordmark's destination. */
export const COVER_ID = "top";

export interface CoverEntry {
  id: string;
  step: number;
  title: string;
}

export function Cover({ entries }: { entries: readonly CoverEntry[] }): JSX.Element {
  const { built_at: builtAt, git_sha: gitSha, version } = meta.package;
  /* The stamp's date half. `split` can hand back an empty array to the type
     checker, so the whole stamp is the fallback — a longer line, never a wrong
     one. */
  const builtOn = builtAt.split("T")[0] ?? builtAt;

  return (
    <section className="cover" id={COVER_ID} aria-labelledby="cover-title">
      <div className="page">
        <hr className="rule" />

        <p className="eyebrow cover-series">{SERIES}</p>

        <h1 id="cover-title" className="cover-title">
          {COVER_TITLE}
        </h1>

        <p className="lede measure cover-dek">{DEK}</p>

        <p className="tnum cover-dateline">
          package v{version}
          <span aria-hidden="true"> · </span>
          built {builtOn}
          <span aria-hidden="true"> · </span>
          git {gitSha.slice(0, SHA_CHARS)}
        </p>

        <nav className="cover-contents" id={CONTENTS_ID} aria-labelledby="contents-heading">
          <h2 id="contents-heading" className="eyebrow contents-heading">
            Contents
          </h2>
          <ol className="contents-list">
            {entries.map((entry) => (
              <li key={entry.id}>
                <a className="contents-link" href={`#${entry.id}`}>
                  <span className="contents-index mono" aria-hidden="true">
                    {reportIndex(entry.step)}
                  </span>
                  <span className="contents-dash" aria-hidden="true">
                    ·
                  </span>
                  <span className="contents-title">{entry.title}</span>
                </a>
              </li>
            ))}
          </ol>
        </nav>


        <hr className="rule" />
      </div>
    </section>
  );
}
