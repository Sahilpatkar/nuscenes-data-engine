import { DeepLink } from "./DeepLink";
import metaJson from "../data/meta.json";
import type { Meta } from "../data/types";

const meta = metaJson as Meta;

/** The tour's deep link, if the exporter wrote one — the other front-end's door. */
const tourLink = meta.deep_links.find((link) => link.url_path === "tour") ?? null;

const REPO_URL = "https://github.com/Sahilpatkar/nuscenes-data-engine";

/**
 * Provenance, attribution, and the honest note that this is one of two readings
 * of the same package. Every fact here is read from meta.json — the version, the
 * sha and the build stamp are the package's own, not the site's.
 */
export function Footer(): JSX.Element {
  const { attribution, exporter, package: pkg, streamlit_base: streamlitBase } = meta;

  return (
    <footer className="site-footer">
      <div className="page">
        <p className="eyebrow">Provenance</p>
        <p className="measure">
          Every number on this page is derived from the committed package at export time by{" "}
          <span className="mono">{exporter}</span>; nothing on the page is typed by hand. A test
          re-runs the exporter and fails the build if a single byte drifts.
        </p>

        <ul className="footer-facts mono">
          <li>
            <span className="muted">package</span> v{pkg.version}
          </li>
          <li>
            <span className="muted">git</span> {pkg.git_sha}
          </li>
          <li>
            <span className="muted">built</span> {pkg.built_at}
          </li>
        </ul>

        <p className="measure">
          Two front-ends, one package: this site is the story, the live Streamlit app is the full
          instrument — the same tables, the same numbers, read two ways.
        </p>
        {tourLink ? <DeepLink href={tourLink.url}>{tourLink.label}</DeepLink> : null}

        <p className="measure footer-attribution">
          Data: {attribution.dataset} — {attribution.citation}. Licensed {attribution.license};
          this project is non-commercial and shares alike.{" "}
          <a href={attribution.url} target="_blank" rel="noopener noreferrer">
            {attribution.url}
          </a>
        </p>

        <p className="footer-author mono">
          Sahil Patkar ·{" "}
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer">
            github.com/Sahilpatkar/nuscenes-data-engine
          </a>{" "}
          ·{" "}
          <a href={streamlitBase} target="_blank" rel="noopener noreferrer">
            live app
          </a>
        </p>
      </div>
    </footer>
  );
}
