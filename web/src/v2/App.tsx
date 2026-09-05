import { assetUrl } from "./assetUrl";
import { ThemeToggle } from "./components/ThemeToggle";

import heroFrameJson from "../data/hero_frame.json";
import interventionJson from "../data/intervention.json";
import type { HeroFrame, Intervention, StrategyBar } from "../data/types";

/* The bundle, typed once — the same eight files the root edition reads, so the
   two editions cannot disagree on a number. The JSON's inferred types are widened
   (a `kind` field is `string`, not the union), so each import is asserted into its
   exported interface here and never re-typed downstream. */
const heroFrame = heroFrameJson as HeroFrame;
const intervention = interventionJson as Intervention;

/* SPECIMEN — replaced in Tasks 2-3 ------------------------------------------
   A proof sheet for the identity, not a page of the report: the type ramp, the
   token palette, the hairline rules, and one rough bar sample drawn purely by the
   chart tokens over the real bundle. Every figure and label below comes from the
   bundle; the only numbers written here are SVG geometry. The seven sections, the
   report chrome and the real charts arrive in Tasks 2 and 3, when this file and
   the SPECIMEN block at the foot of base.css are deleted.
   -------------------------------------------------------------------------- */

const SPECIMEN_INDEX = "01";
const SPECIMEN_SECTION = "Specimen";

/** The surface + mark tokens, in the order the proof sheet shows them. */
const SWATCHES: readonly string[] = [
  "--ground",
  "--panel",
  "--panel-2",
  "--line",
  "--ink",
  "--muted",
  "--accent",
  "--chart-accent",
  "--chart-muted",
  "--chart-pos",
  "--chart-neg",
  "--chart-zero",
];

const VIEW = { width: 640, height: 250, left: 16, right: 16, top: 26, bottom: 54 } as const;
const PLOT_W = VIEW.width - VIEW.left - VIEW.right;
const PLOT_H = VIEW.height - VIEW.top - VIEW.bottom;
const MAX_BAR_WIDTH = 24;
const VALUE_LABEL_ROOM = 14;

type PlottedBar = StrategyBar & { delta_night: number };

/** A bar the package never recorded is not drawn: zero would be a claim. */
const plotted: PlottedBar[] = intervention.chart.bars.filter(
  (bar): bar is PlottedBar => bar.delta_night !== null,
);

/**
 * The decimals the PACKAGE recorded, so labels neither invent precision nor round
 * a recorded difference away (v1's rule, kept).
 */
function decimals(values: readonly number[]): number {
  return values.reduce((most, value) => {
    const text = String(value);
    const dot = text.indexOf(".");
    return Math.max(most, dot < 0 ? 0 : text.length - dot - 1);
  }, 0);
}

/** A signed reading: "+" on a gain, the number's own "-" on a loss, none on 0. */
function signed(value: number, places: number): string {
  const fixed = value.toFixed(places);
  if (Number(fixed) === 0) return fixed.replace("-", "");
  return value > 0 ? `+${fixed}` : fixed;
}

export default function App(): JSX.Element {
  const places = decimals(plotted.map((bar) => bar.delta_night));
  const extent = plotted.reduce((most, bar) => Math.max(most, Math.abs(bar.delta_night)), 0);
  const scale = extent > 0 ? PLOT_H / 2 / extent : 0;
  const zeroY = VIEW.top + PLOT_H / 2;
  const slot = plotted.length > 0 ? PLOT_W / plotted.length : PLOT_W;
  const barWidth = Math.min(MAX_BAR_WIDTH, slot * 0.5);

  return (
    <main className="page specimen">
      <p className="eyebrow">nuScenes Data Engine — report edition</p>

      <h1 className="cover-title">Paper, ink, and one cobalt accent</h1>
      <p className="lede measure">
        A type and colour specimen for the report edition — the same committed bundle the
        root story site reads, set as a research report rather than a deck.
      </p>
      <div className="specimen-toolbar">
        <ThemeToggle />
        <span className="mono">Source Serif 4 · Source Sans 3 · Source Code Pro</span>
      </div>

      <section className="specimen-block" aria-labelledby="specimen-type">
        <hr className="rule" />
        <div className="specimen-head">
          <h2 id="specimen-type" className="section-title">
            <span className="section-index">{SPECIMEN_INDEX} / </span>
            {SPECIMEN_SECTION}
          </h2>
          <p className="eyebrow">Type ramp · the measure</p>
        </div>
        <p className="measure">
          Body text sets at seventeen pixels over a sixty-five character measure, in Source
          Sans 3. Long-form reading is the whole point of this edition: the column stays
          narrow, the rules do the work the panels used to do, and figures break out wider
          than the text they belong to. Numbers set with{" "}
          <span className="tnum">tabular figures</span> wherever a column has to line up.
        </p>
        <p className="measure">
          Raw identifiers, axis ticks and provenance detail set in the mono voice —{" "}
          <span className="mono">{plotted.map((bar) => bar.arm).join(" · ")}</span> — so a
          reader can always tell the story label from the thing the package actually stored.
        </p>
        <div className="footnotes">
          <p className="footnote">
            <span className="footnote-mark">1</span>
            Footnotes carry provenance in this edition: a short rule, small type, the muted
            ink — and the bundle's own wording, never a paraphrase.
          </p>
          {heroFrame.provenance.slice(0, 1).map((entry, index) => (
            <p className="footnote" key={entry.sentence}>
              <span className="footnote-mark">{index + 2}</span>
              <span className="mono">{entry.kind}</span> — {entry.sentence}: {entry.detail}
            </p>
          ))}
        </div>
      </section>

      <section className="specimen-block" aria-labelledby="specimen-palette">
        <hr className="rule" />
        <div className="specimen-head">
          <h2 id="specimen-palette" className="section-title">
            <span className="section-index">02 / </span>Palette
          </h2>
          <p className="eyebrow">Validated per surface</p>
        </div>
        <ul className="specimen-swatches">
          {SWATCHES.map((token) => (
            <li key={token} className="specimen-swatch">
              <div className="specimen-chip" style={{ background: `var(${token})` }} />
              <span className="mono">{token}</span>
            </li>
          ))}
        </ul>
        <div className="specimen-head">
          <p className="eyebrow">Rules</p>
        </div>
        <hr className="rule" />
        <hr className="rule-short" style={{ marginTop: "0.6rem" }} />
        <hr className="rule rule-accent" style={{ marginTop: "0.6rem" }} />
      </section>

      <section className="specimen-block" aria-labelledby="specimen-figure">
        <hr className="rule" />
        <div className="specimen-head">
          <h2 id="specimen-figure" className="section-title">
            <span className="section-index">03 / </span>Figure
          </h2>
          <p className="eyebrow">Chart tokens only</p>
        </div>
        <figure className="specimen-figure figure-wide">
          <svg
            className="specimen-svg"
            viewBox={`0 0 ${VIEW.width} ${VIEW.height}`}
            role="img"
            aria-label={intervention.chart.title}
          >
            <line
              className="specimen-zero"
              x1={VIEW.left}
              x2={VIEW.width - VIEW.right}
              y1={zeroY}
              y2={zeroY}
            />
            {plotted.map((bar, index) => {
              const centre = VIEW.left + slot * index + slot / 2;
              const height = Math.abs(bar.delta_night) * scale;
              const up = bar.delta_night >= 0;
              const label = signed(bar.delta_night, places);
              return (
                <g key={bar.arm}>
                  <rect
                    className={`specimen-bar${bar.highlight ? " is-highlight" : ""}`}
                    x={centre - barWidth / 2}
                    y={up ? zeroY - height : zeroY}
                    width={barWidth}
                    height={height}
                  />
                  <text
                    className="specimen-value"
                    x={centre}
                    y={up ? zeroY - height - VALUE_LABEL_ROOM / 2 : zeroY + height + VALUE_LABEL_ROOM}
                    textAnchor="middle"
                  >
                    {label}
                  </text>
                  <text
                    className="specimen-tick"
                    x={centre}
                    y={VIEW.height - VIEW.bottom / 2}
                    textAnchor="middle"
                  >
                    {bar.arm}
                  </text>
                </g>
              );
            })}
          </svg>
          <figcaption className="specimen-caption measure">
            <span className="mono">Figure 1</span> — {intervention.chart.title} (
            {intervention.chart.y_title}). Emphasis is carried by cobalt against bronze;
            sign is carried by the zero rule and the signed labels, never by hue.
          </figcaption>
        </figure>
        <figure className="specimen-figure">
          <img
            src={assetUrl(heroFrame.images.arm.src)}
            width={heroFrame.images.arm.width}
            height={heroFrame.images.arm.height}
            alt={heroFrame.images.arm.alt}
            className="figure-wide"
          />
          <figcaption className="specimen-caption measure">
            <span className="mono">Figure 2</span> — the same asset the root edition serves,
            addressed through <span className="mono">assetUrl</span> so it resolves from{" "}
            <span className="mono">/v2/</span>.
          </figcaption>
        </figure>
      </section>
    </main>
  );
}
