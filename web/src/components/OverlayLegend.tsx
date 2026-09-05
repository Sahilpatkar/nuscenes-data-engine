import type { LegendItem } from "../data/types";

/**
 * What the colours on an overlay image mean — the app's own `LEGEND_ITEMS`
 * wording, each chip carrying a swatch in the exact RGB the exporter burned into
 * the JPEG.
 *
 * The swatch colour is data, not theme: it comes from the bundle, and the chip it
 * sits on is deliberately theme-invariant (`--image-chip`, the night frame's own
 * tone in both modes). White and yellow marks would vanish on warm paper, so the
 * legend keeps the image's ground under them in light mode too.
 */
export function OverlayLegend({ items }: { items: readonly LegendItem[] }): JSX.Element | null {
  if (items.length === 0) return null;

  return (
    <ul className="legend">
      {items.map((item) => {
        const [red, green, blue] = item.rgb;
        return (
          <li className="legend-chip" key={item.wording}>
            <span
              className="legend-swatch"
              aria-hidden="true"
              style={{ backgroundColor: `rgb(${red ?? 0}, ${green ?? 0}, ${blue ?? 0})` }}
            />
            <span>{item.wording}</span>
          </li>
        );
      })}
    </ul>
  );
}
