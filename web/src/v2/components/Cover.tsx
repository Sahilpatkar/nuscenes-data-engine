import { reportIndex } from "./SectionFrame";

import metaJson from "../../data/meta.json";
import type { Meta } from "../../data/types";

const meta = metaJson as Meta;

/**
 * The report's masthead: rule, series line, title, dek, dateline, contents, the
 * two cross-references, rule.
 *
 * SITE-EDITORIAL COPY — the three constants below are the site's OWN fixed
 * words, not the package's, and they are the only prose on this page that the
 * bundle does not supply:
 *
 *   · SERIES is the running series line a report carries above its title;
 *   · COVER_TITLE is the promise the root edition already makes in its own
 *     landing headline — the two editions are one story and title it once;
 *   · DEK is, verbatim, the `<meta name="description">` of `web/v2/index.html`,
 *     so what a search result promises and what the cover says are the same
 *     sentence.
 *
 * None of the three contains a number or a result, so nothing here can drift
 * away from the package. Everything that IS a fact — the version, the build date,
 * the commit, the live app's address — is read from `meta.json`.
 */
const SERIES = "nuScenes Data Engine — Technical report";

const COVER_TITLE = "From model failure to better training data";

const DEK =
  "The nuScenes data-engine story as a research report: where the detector fails, " +
  "which data fixes it, and whether the intervention worked — every number derived " +
  "from the committed experiment package.";

/** Abbreviated-commit length — the dateline's, not the colophon's, which is full. */
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
  /* Vite's base: "/" in dev and preview, "/nuscenes-data-engine/" on Pages —
     always with a trailing slash, and always the ROOT edition's document. */
  const rootEdition = import.meta.env.BASE_URL;

  return (
    <section className="cover" id={COVER_ID} aria-labelledby="cover-title">
      <div className="page">
        <hr className="rule" />

        <p className="eyebrow cover-series">{SERIES}</p>

        <h1 id="cover-title" className="cover-title">
          {COVER_TITLE}
        </h1>

        <p className="lede measure cover-dek">{DEK}</p>

        <p className="mono cover-dateline">
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
                    —
                  </span>
                  <span className="contents-title">{entry.title}</span>
                </a>
              </li>
            ))}
          </ol>
        </nav>

        <ul className="crossrefs">
          <li>
            <span className="eyebrow crossref-label">Live instrument</span>
            <a href={meta.streamlit_base} target="_blank" rel="noopener noreferrer">
              the Streamlit app
              <span aria-hidden="true"> ↗</span>
            </a>
          </li>
          <li>
            <span className="eyebrow crossref-label">Original edition</span>
            <a href={rootEdition}>the story site at the site root</a>
          </li>
        </ul>

        <hr className="rule" />
      </div>
    </section>
  );
}
