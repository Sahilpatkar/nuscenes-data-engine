import metaJson from "../../data/meta.json";
import type { Meta } from "../../data/types";

const meta = metaJson as Meta;

const REPO_URL = "https://github.com/Sahilpatkar/nuscenes-data-engine";
const AUTHOR = "Sahil Patkar";

/**
 * The report's end matter: the dataset attribution the nuScenes licence asks
 * for, and the author line. Nothing else — the report sources nothing on the
 * page (no provenance sentence, no package stamp, no edition note; the cover's
 * dateline is the one place the package version appears, and the cover's
 * cross-references are the one door to the original edition).
 *
 * The attribution is read from `meta.json`, so a re-export moves it without
 * anyone editing this file.
 */
export function Colophon(): JSX.Element {
  const { attribution } = meta;

  return (
    <footer className="colophon">
      <div className="page">
        <hr className="rule" />

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
