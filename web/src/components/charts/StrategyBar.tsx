import { useId, useState } from "react";

import type { StrategyBar as StrategyBarDatum, StrategyChart } from "../../data/types";

/**
 * The five acquisition strategies against the baseline, on the diagnosed night
 * metric — the tour's own five-bar chart (`strategy_coverage`), not the deep
 * page's thirteen.
 *
 * SIGN IS NEVER CARRIED BY HUE. The two chart hues here say *emphasis* (which arm
 * the story follows), not direction: every bar is `--chart-muted` except the
 * highlighted one, which is `--chart-accent`. Amber and red are neighbours under
 * deuteranopia (tokens.css, the standing obligation), so direction is carried
 * twice and geometrically instead — a negative bar hangs BELOW the zero rule, and
 * every bar wears a signed value label. Read in greyscale, the chart still says
 * which arm lost night mAP.
 *
 * Marks follow the dataviz method: ≤ 24 px bars with ≥ 8 px gaps, the zero rule
 * in `--chart-zero`, at most two gridlines (the domain's own ends — the direct
 * labels are the reading, the axis is only a reference), story labels wrapped to
 * two lines rather than rotated, and no legend for what the labels already name.
 *
 * Interaction: pointing at OR tabbing to a bar's band raises a tooltip with the
 * story label, the raw arm id and the value. The bands are focusable and carry
 * that same reading as their accessible name; the screen-reader-only table under
 * the chart is the twin for reading the whole series at once.
 *
 * A bar whose delta the package never recorded is not drawn — plotting a missing
 * measurement at zero would claim the arm made no difference. It keeps its row in
 * the table twin, marked as not recorded.
 */

const VIEW = { width: 640, height: 336, left: 54, right: 18, top: 34, bottom: 78 } as const;
const PLOT_W = VIEW.width - VIEW.left - VIEW.right;
const PLOT_H = VIEW.height - VIEW.top - VIEW.bottom;

/** The method's bar cap, and the gap that keeps the bars reading as separate. */
const MAX_BAR_WIDTH = 24;
const MIN_BAR_GAP = 8;
/** Three intervals: a domain that ends on round numbers without wasting height. */
const TICK_INTERVALS = 3;
/** IBM Plex Mono at the 10 px `.chart-step` size advances ≈ 6 px per character. */
const LABEL_ADVANCE = 6;
/** The band a direct value label occupies above its bar, in viewBox units. */
const VALUE_LABEL_ROOM = 20;

function round(value: number): number {
  return Math.round(value * 1e6) / 1e6;
}

/** 1, 2, 5 × 10ⁿ — the step sizes that read as round numbers on an axis. */
function niceStep(rough: number): number {
  if (!(rough > 0)) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const normalized = rough / magnitude;
  const snapped = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return snapped * magnitude;
}

/**
 * The decimals the PACKAGE recorded, so the labels neither invent precision nor
 * round a recorded difference away. Every bar is then written to the same width,
 * which is what makes a column of deltas comparable at a glance.
 */
function decimals(values: readonly number[]): number {
  return values.reduce((most, value) => {
    const text = String(value);
    const dot = text.indexOf(".");
    return Math.max(most, dot < 0 ? 0 : text.length - dot - 1);
  }, 0);
}

/** A signed reading: "+" on a gain, the number's own "-" on a loss, no sign on 0. */
function signed(value: number, places: number): string {
  const fixed = value.toFixed(places);
  if (Number(fixed) === 0) return fixed.replace("-", "");
  return value > 0 ? `+${fixed}` : fixed;
}

/**
 * A story label over two lines, never rotated and never shortened: the words that
 * fit go on the first line, everything left goes on the second. A label too long
 * for two lines overruns its band rather than being truncated — the tour's rule is
 * that a chart never drops or clips the name of an arm.
 */
function wrapLabel(label: string, maxChars: number): [string, string] {
  const words = label.split(" ");
  let first = "";
  let index = 0;
  while (index < words.length) {
    const word = words[index] ?? "";
    const candidate = first === "" ? word : `${first} ${word}`;
    if (first !== "" && candidate.length > maxChars) break;
    first = candidate;
    index += 1;
  }
  return [first, words.slice(index).join(" ")];
}

interface PlottedBar {
  bar: StrategyBarDatum;
  value: number;
}

export function StrategyBar({ chart }: { chart: StrategyChart }): JSX.Element | null {
  const titleId = useId();
  const [active, setActive] = useState<number | null>(null);

  const plotted: PlottedBar[] = chart.bars.flatMap((bar) =>
    bar.delta_night === null ? [] : [{ bar, value: bar.delta_night }],
  );
  const places = decimals(plotted.map((entry) => entry.value));

  // `.sr-only` goes on a block WRAPPER, never on the <table> itself: a table box
  // cannot shrink below its min-content width (~438px here), so an absolutely
  // positioned one kept stretching the document's scrollWidth past a 360px
  // viewport — a horizontal scrollbar on every phone. A <div> honours the 1px
  // clip; the table inside keeps every bit of its semantics.
  const table = (
    <div className="sr-only">
      <table>
        <caption>{chart.title}</caption>
        <thead>
          <tr>
            <th scope="col">Strategy</th>
            <th scope="col">Arm</th>
            <th scope="col">{chart.y_title}</th>
          </tr>
        </thead>
        <tbody>
          {chart.bars.map((bar) => (
            <tr key={bar.arm}>
              <th scope="row">{bar.strategy}</th>
              <td>{bar.arm}</td>
              <td>{bar.delta_night === null ? "not recorded" : signed(bar.delta_night, places)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  // Nothing plottable: the caption and the sentences around the chart still say
  // what this package holds, and an empty frame would only imply zeros.
  if (plotted.length === 0) return null;

  const values = plotted.map((entry) => entry.value);
  const highest = Math.max(0, ...values);
  const lowest = Math.min(0, ...values);
  const step = niceStep((highest - lowest) / TICK_INTERVALS);
  let lo = round(Math.floor(lowest / step) * step);
  let hi = round(Math.ceil(highest / step) * step);
  // A bar that reaches the frame has nowhere to put its own label: where the
  // extreme value lands exactly on the domain's end, the domain gains a step.
  if (highest !== 0 && hi === highest) hi = round(hi + step);
  if (lowest !== 0 && lo === lowest) lo = round(lo - step);
  const span = hi - lo || 1;

  const count = plotted.length;
  const band = PLOT_W / count;
  const barWidth = Math.min(MAX_BAR_WIDTH, Math.max(band - MIN_BAR_GAP, 1));
  const maxChars = Math.max(6, Math.floor((band - MIN_BAR_GAP) / LABEL_ADVANCE));

  const centre = (index: number): number => VIEW.left + band * (index + 0.5);
  const y = (value: number): number => VIEW.top + (PLOT_H * (hi - value)) / span;
  const zeroY = y(0);
  /** The domain's own ends — two hairlines at most, and never one at zero (the
      zero rule is a mark of its own, not a gridline). */
  const gridlines = [lo, hi].filter((tick) => tick !== 0);

  const tip = active === null ? undefined : plotted[active];
  // Centred over its bar, except at the ends, where a centred tooltip would leave
  // the plot: there it hangs from the near edge instead of being clipped away.
  const tipShift = active === 0 ? "-8px" : active === count - 1 ? "calc(-100% + 8px)" : "-50%";

  return (
    <figure className="chart strategy-chart">
      <figcaption className="chart-title" id={titleId}>
        {chart.title}
      </figcaption>

      <div className="strategy-plot">
        {/* No `role="img"` on the svg: its bands are focusable, and a focusable
            element inside an image role is a thing an assistive technology is told
            to ignore and a keyboard still lands on. The figcaption names it. */}
        <svg
          className="chart-svg"
          viewBox={`0 0 ${VIEW.width} ${VIEW.height}`}
          aria-labelledby={titleId}
        >
          {/* The unit, written horizontally above the scale it belongs to: a
              rotated axis title is one more thing to tilt your head at. */}
          <text className="chart-tick" x={0} y={14}>
            {chart.y_title}
          </text>

          {gridlines.map((tick) => (
            <g key={tick}>
              <line
                className="chart-grid"
                x1={VIEW.left}
                x2={VIEW.left + PLOT_W}
                y1={y(tick)}
                y2={y(tick)}
              />
              <text className="chart-tick" x={VIEW.left - 8} y={y(tick) + 3.5} textAnchor="end">
                {signed(tick, places)}
              </text>
            </g>
          ))}

          <line
            className="chart-zero"
            x1={VIEW.left}
            x2={VIEW.left + PLOT_W}
            y1={zeroY}
            y2={zeroY}
          />
          <text className="chart-tick" x={VIEW.left - 8} y={zeroY + 3.5} textAnchor="end">
            {signed(0, places)}
          </text>

          {plotted.map(({ bar, value }, index) => (
            <rect
              key={`bar-${bar.arm}`}
              className={bar.highlight ? "bar is-highlight" : "bar"}
              x={centre(index) - barWidth / 2}
              y={Math.min(y(value), zeroY)}
              width={barWidth}
              height={Math.abs(y(value) - zeroY)}
            />
          ))}

          {/* Direct labels: above a gain, below a loss — the second reading of the
              sign, and the reason the chart needs no value axis to be read. */}
          {plotted.map(({ bar, value }, index) => (
            <text
              key={`value-${bar.arm}`}
              className="chart-value"
              x={centre(index)}
              y={value < 0 ? y(value) + 14 : y(value) - 8}
              textAnchor="middle"
            >
              {signed(value, places)}
            </text>
          ))}

          {plotted.map(({ bar }, index) => {
            const [first, second] = wrapLabel(bar.strategy, maxChars);
            return (
              <g
                key={`label-${bar.arm}`}
                className={index === active ? "chart-step is-active" : "chart-step"}
              >
                <text x={centre(index)} y={VIEW.top + PLOT_H + 26} textAnchor="middle">
                  {first}
                </text>
                {second === "" ? null : (
                  <text x={centre(index)} y={VIEW.top + PLOT_H + 38} textAnchor="middle">
                    {second}
                  </text>
                )}
              </g>
            );
          })}

          {/* A full-height band per bar, far larger than the bar it activates —
              and focusable, so the tooltip is reachable without a pointer. */}
          {plotted.map(({ bar, value }, index) => (
            <rect
              key={`hit-${bar.arm}`}
              className="chart-hit"
              x={VIEW.left + band * index}
              y={VIEW.top}
              width={band}
              height={PLOT_H}
              tabIndex={0}
              // `graphics-symbol`, not `img`: this band IS focusable, and a
              // focusable element inside an `img` role is exactly the thing an
              // assistive technology is told to treat as opaque. The graphics
              // role keeps the band a labelled unit of the chart.
              role="graphics-symbol"
              aria-label={`${bar.strategy} (${bar.arm}): ${signed(value, places)}`}
              onPointerEnter={() => setActive(index)}
              onPointerLeave={() => setActive(null)}
              onFocus={() => setActive(index)}
              onBlur={() => setActive(null)}
            />
          ))}
        </svg>

        {tip === undefined || active === null ? null : (
          <div
            className="bar-tip"
            aria-hidden="true"
            style={{
              left: `${(centre(active) / VIEW.width) * 100}%`,
              // Above the bar AND above its direct label — the tooltip repeats
              // the value, so covering the label with it would be the one place
              // the reading disappeared.
              top: `${(Math.max(Math.min(y(tip.value), zeroY) - VALUE_LABEL_ROOM, 0) / VIEW.height) * 100}%`,
              transform: `translate(${tipShift}, -100%)`,
            }}
          >
            <span className="bar-tip-label">{tip.bar.strategy}</span>
            <span className="bar-tip-id mono muted">{tip.bar.arm}</span>
            <span className="bar-tip-value mono">{signed(tip.value, places)}</span>
          </div>
        )}
      </div>

      {table}
    </figure>
  );
}
