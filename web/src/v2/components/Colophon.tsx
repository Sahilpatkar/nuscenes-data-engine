import metaJson from "../../data/meta.json";
import type { Meta } from "../../data/types";

const meta = metaJson as Meta;

const REPO_URL = "https://github.com/Sahilpatkar/nuscenes-data-engine";
const AUTHOR = "Sahil Patkar";

/**
 * The end-of-report colophon: how the document was made, what it is made of, and
 * where its two siblings live.
 *
 * The prose is the site's own — the exporter's NAME, the package stamp, the
 * dataset, the licence and the live app's address are all read from `meta.json`,
 * so a re-export moves them without anyone editing this file. The stamp is the
 * FULL commit here; the cover's dateline abbreviates it, which is the printed
 * convention (a short date up front, the whole record at the back).
 */
export function Colophon(): JSX.Element {
  const { attribution, exporter, package: pkg, streamlit_base: streamlitBase } = meta;
  const rootEdition = import.meta.env.BASE_URL;

  return (
    <footer className="colophon">
      <div className="page">
        <hr className="rule" />

        <h2 className="eyebrow colophon-heading">Colophon</h2>

        <p className="measure">
          Every number in this report is derived from the committed package at export time by{" "}
          <span className="mono">{exporter}</span>; nothing on the page is typed by hand. A test
          re-runs the exporter and fails the build if a single byte drifts.
        </p>

        <p className="mono colophon-stamp">
          package v{pkg.version}
          <span aria-hidden="true"> · </span>
          git {pkg.git_sha}
          <span aria-hidden="true"> · </span>
          built {pkg.built_at}
        </p>

        <p className="measure">
          This is the report edition; the original edition lives{" "}
          <a href={rootEdition}>at the site root</a>; the full instrument is{" "}
          <a href={streamlitBase} target="_blank" rel="noopener noreferrer">
            the live Streamlit app
          </a>
          . All three read the same package, so they cannot disagree on a number.
        </p>

        <p className="measure colophon-attribution">
          Data: {attribution.dataset} — {attribution.citation}. Licensed {attribution.license};
          this project is non-commercial and shares alike.{" "}
          <a href={attribution.url} target="_blank" rel="noopener noreferrer">
            {attribution.url}
          </a>
        </p>

        <p className="mono colophon-author">
          {AUTHOR}
          <span aria-hidden="true"> · </span>
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer">
            github.com/Sahilpatkar/nuscenes-data-engine
          </a>
        </p>
      </div>
    </footer>
  );
}
