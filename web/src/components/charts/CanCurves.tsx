import { useId } from "react";

import type { FilmstripStep } from "../../data/types";

/**
 * The event's CAN speed and longitudinal acceleration across the filmstrip's five
 * keyframes — the right half of the synced pair, marking the same `active` step
 * the thumbs do.
 *
 * Two charts, never one: speed (km/h) and acceleration (m/s²) are different
 * scales, and a dual axis would invent a relationship between them. Both read
 * their titles from the bundle, which is also where the CAN-vs-ego-pose wording
 * is decided (`speed_is_can`) — this component never names the signal itself.
 *
 * Marks follow the dataviz method: a 2 px line in `--chart-muted`, 8 px markers
 * with a 2 px surface ring, hairline solid gridlines in `--chart-grid`, the y = 0
 * rule on the acceleration chart (the series runs negative — without it a hard
 * deceleration reads as merely "lower"), and exactly ONE direct value label: the
 * active step's. No label on every point, no tick spam, no legend for a single
 * series (the title names it).
 *
 * Interaction: pointing at a chart activates the nearest step (a full-height hit
 * band per step, far larger than the marker), which moves the thumbs too. The
 * keyboard path is the filmstrip's own arrow keys, and every value is also in the
 * table twin below — the marker label is never the only way to read a number.
 */

const VIEW = { width: 360, height: 196, left: 42, right: 16, top: 26, bottom: 30 } as const;
const PLOT_W = VIEW.width - VIEW.left - VIEW.right;
const PLOT_H = VIEW.height - VIEW.top - VIEW.bottom;
/** Two intervals: three or four gridlines, which is a scale — not tick spam. */
const TICK_INTERVALS = 2;

function round(value: number): number {
  return Math.round(value * 1e6) / 1e6;
}

/** A pre-rounded reading, with the trailing zeros a fixed 2 dp leaves behind. */
function format(value: number): string {
  const fixed = (Object.is(value, -0) ? 0 : value).toFixed(2);
  return fixed.includes(".") ? fixed.replace(/\.?0+$/, "") : fixed;
}

/** 1, 2, 5 × 10ⁿ — the step sizes that read as round numbers on an axis. */
function niceStep(rough: number): number {
  if (!(rough > 0)) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const normalized = rough / magnitude;
  const snapped = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return snapped * magnitude;
}

interface Domain {
  lo: number;
  hi: number;
  ticks: number[];
}

function niceDomain(values: readonly number[], includeZero: boolean): Domain {
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (includeZero) {
    lo = Math.min(lo, 0);
    hi = Math.max(hi, 0);
  }
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  const step = niceStep((hi - lo) / TICK_INTERVALS);
  lo = Math.floor(lo / step) * step;
  hi = Math.ceil(hi / step) * step;
  const ticks: number[] = [];
  for (let value = lo; value <= hi + step / 2; value += step) ticks.push(round(value));
  return { lo, hi, ticks };
}

interface CurveProps {
  title: string;
  labels: readonly string[];
  values: readonly (number | null)[];
  /** True where 0 is a meaningful boundary (acceleration), not just a number. */
  includeZero: boolean;
  active: number;
  onActivate: (index: number) => void;
}

function Curve({
  title,
  labels,
  values,
  includeZero,
  active,
  onActivate,
}: CurveProps): JSX.Element | null {
  const titleId = useId();
  const readings = values.filter((value): value is number => value !== null);
  // A signal this package never recorded gets no chart at all, rather than an
  // empty frame implying the reading was zero.
  if (readings.length === 0) return null;

  const { lo, hi, ticks } = niceDomain(readings, includeZero);
  const count = values.length;
  const x = (index: number): number =>
    count < 2 ? VIEW.left + PLOT_W / 2 : VIEW.left + (PLOT_W * index) / (count - 1);
  const y = (value: number): number => VIEW.top + (PLOT_H * (hi - value)) / (hi - lo);

  // Split at the gaps: a step with no reading breaks the line rather than being
  // bridged across, which would draw a measurement that does not exist.
  const runs: { index: number; value: number }[][] = [];
  let run: { index: number; value: number }[] = [];
  values.forEach((value, index) => {
    if (value === null) {
      if (run.length > 0) runs.push(run);
      run = [];
      return;
    }
    run.push({ index, value });
  });
  if (run.length > 0) runs.push(run);

  const activeValue = values[active] ?? null;
  const band = count < 2 ? PLOT_W : PLOT_W / (count - 1);
  const zeroShown = includeZero && lo <= 0 && hi >= 0;

  return (
    <figure className="chart">
      <figcaption className="chart-title" id={titleId}>
        {title}
      </figcaption>
      <svg
        className="chart-svg"
        viewBox={`0 0 ${VIEW.width} ${VIEW.height}`}
        role="img"
        aria-labelledby={titleId}
      >
        {ticks.map((tick) => (
          <g key={tick}>
            <line
              className="chart-grid"
              x1={VIEW.left}
              x2={VIEW.left + PLOT_W}
              y1={y(tick)}
              y2={y(tick)}
            />
            <text className="chart-tick" x={VIEW.left - 8} y={y(tick) + 3.5} textAnchor="end">
              {format(tick)}
            </text>
          </g>
        ))}

        {zeroShown ? (
          <line
            className="chart-zero"
            x1={VIEW.left}
            x2={VIEW.left + PLOT_W}
            y1={y(0)}
            y2={y(0)}
          />
        ) : null}

        {activeValue === null ? null : (
          <line
            className="chart-rule"
            x1={x(active)}
            x2={x(active)}
            y1={VIEW.top}
            y2={VIEW.top + PLOT_H}
          />
        )}

        {runs.map((points) =>
          points.length < 2 ? null : (
            <polyline
              key={`run-${points[0]?.index ?? 0}`}
              className="chart-line"
              points={points.map((point) => `${x(point.index)},${y(point.value)}`).join(" ")}
            />
          ),
        )}

        {values.map((value, index) =>
          value === null ? null : (
            <circle
              key={labels[index] ?? index}
              className={index === active ? "chart-dot is-active" : "chart-dot"}
              cx={x(index)}
              cy={y(value)}
              r={index === active ? 5.5 : 4}
            />
          ),
        )}

        {activeValue === null ? null : (
          <text
            className="chart-value"
            x={x(active)}
            y={Math.max(y(activeValue) - 11, VIEW.top - 8)}
            textAnchor={active === 0 ? "start" : active === count - 1 ? "end" : "middle"}
          >
            {format(activeValue)}
          </text>
        )}

        {labels.map((label, index) => (
          <text
            key={label}
            className={index === active ? "chart-step is-active" : "chart-step"}
            x={x(index)}
            y={VIEW.top + PLOT_H + 17}
            textAnchor={index === 0 ? "start" : index === count - 1 ? "end" : "middle"}
          >
            {label}
          </text>
        ))}

        {labels.map((label, index) => (
          <rect
            key={`hit-${label}`}
            className="chart-hit"
            x={Math.max(x(index) - band / 2, VIEW.left)}
            y={VIEW.top}
            width={Math.min(band, PLOT_W)}
            height={PLOT_H}
            onPointerEnter={() => onActivate(index)}
            onClick={() => onActivate(index)}
          />
        ))}
      </svg>
    </figure>
  );
}

export interface CanCurvesProps {
  steps: readonly FilmstripStep[];
  speedTitle: string;
  accelTitle: string;
  active: number;
  onActivate: (index: number) => void;
}

export function CanCurves({
  steps,
  speedTitle,
  accelTitle,
  active,
  onActivate,
}: CanCurvesProps): JSX.Element {
  const labels = steps.map((step) => step.label);

  return (
    <div className="curves">
      <Curve
        title={speedTitle}
        labels={labels}
        values={steps.map((step) => step.can_speed_kmh)}
        includeZero={false}
        active={active}
        onActivate={onActivate}
      />
      <Curve
        title={accelTitle}
        labels={labels}
        values={steps.map((step) => step.accel_mps2)}
        includeZero
        active={active}
        onActivate={onActivate}
      />

      {/* The table twin: every plotted value, reachable without the chart.
          `.sr-only` sits on the wrapper, not on the <table>: a table box cannot
          shrink below its min-content width, so an absolutely positioned one
          stretched the document's scrollWidth past a 360px viewport. A block
          wrapper honours the 1px clip and the table keeps its semantics. */}
      <div className="sr-only">
        <table>
          <caption>Filmstrip step readings</caption>
          <thead>
            <tr>
              <th scope="col">Step</th>
              <th scope="col">{speedTitle}</th>
              <th scope="col">{accelTitle}</th>
            </tr>
          </thead>
          <tbody>
            {steps.map((step) => (
              <tr key={step.token}>
                <th scope="row">{step.label}</th>
                <td>{step.can_speed_kmh === null ? "not recorded" : format(step.can_speed_kmh)}</td>
                <td>{step.accel_mps2 === null ? "not recorded" : format(step.accel_mps2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
