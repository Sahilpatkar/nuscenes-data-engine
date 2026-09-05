import { ThemeToggle } from "./ThemeToggle";
import heroFrameJson from "../data/hero_frame.json";
import metaJson from "../data/meta.json";
import type { HeroFrame, Meta } from "../data/types";

const hero = heroFrameJson as HeroFrame;
const meta = metaJson as Meta;

/**
 * The deck's title slide, translated to the web: eyebrow, the one promise, the
 * one-sentence description, the two doors out (live app, source), and the frame
 * the whole story is about.
 *
 * SITE_LEDE is the site's own fixed description — the same sentence as the
 * `<meta name="description">` in index.html, and the only copy on this page that
 * is not read from the bundle. It contains no number and no result, so nothing
 * here can drift away from the package.
 */
const SITE_LEDE =
  "A closed-loop AV perception data engine: it finds where a detector fails, " +
  "mines targeted training data, retrains, and measures the result.";

const REPO_URL = "https://github.com/Sahilpatkar/nuscenes-data-engine";

/** The story's first section — where the scroll cue and the rail both point. */
const FIRST_SECTION = "blindspot";

export function Landing(): JSX.Element {
  const image = hero.images.arm;

  return (
    <header className="landing">
      <div className="page landing-inner">
        <div className="landing-top">
          <p className="eyebrow">nuScenes Data Engine</p>
          <ThemeToggle />
        </div>

        <div className="landing-main">
          <div className="landing-copy">
            <h1 className="landing-title">
              From model failure to <span className="accent">better training data</span>
            </h1>
            <p className="lede landing-lede">{SITE_LEDE}</p>
            <div className="landing-actions">
              <a
                className="cta cta-primary"
                href={meta.streamlit_base}
                target="_blank"
                rel="noopener noreferrer"
              >
                Open the live app
                <span aria-hidden="true">↗</span>
              </a>
              <a className="cta" href={REPO_URL} target="_blank" rel="noopener noreferrer">
                Source on GitHub
                <span aria-hidden="true">↗</span>
              </a>
            </div>
          </div>

          <figure className="landing-figure">
            <img
              src={image.src}
              width={image.width}
              height={image.height}
              alt={image.alt}
              decoding="async"
            />
            {hero.held_out_caption ? (
              <figcaption className="mono muted">{hero.held_out_caption}</figcaption>
            ) : null}
          </figure>
        </div>

        <a className="scroll-cue mono" href={`#${FIRST_SECTION}`}>
          <span>Start the story</span>
          {/* Decoration only: the arrow (and its drift) is hidden under
              prefers-reduced-motion, the link itself always works. */}
          <span className="scroll-cue-arrow" aria-hidden="true">
            ↓
          </span>
        </a>
      </div>
    </header>
  );
}
