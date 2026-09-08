/**
 * One number, with the label that says what it measures — the report's tile.
 *
 * Same contract as the root edition's (the story's stats arrive pre-formatted
 * from the bundle; the tile formats nothing and computes nothing), a different
 * voice: no card, no fill — a hairline over a small-caps label and the value set
 * in the serif, the way a table of results reads on paper.
 *
 * `accent` marks the one tile a section is ABOUT — it tints the rule and the
 * value in cobalt. At most one per row: two accents and the emphasis is gone.
 * The tint is never the only thing that distinguishes it; the label always says
 * what the number is.
 */
export interface StatTileProps {
  label: string;
  value: string;
  /** A delta or a caveat under the value; omitted where the bundle has none. */
  detail?: string | undefined;
  accent?: boolean;
}

export function StatTile({ label, value, detail, accent = false }: StatTileProps): JSX.Element {
  return (
    <div className={accent ? "stat-tile is-accent" : "stat-tile"}>
      <p className="stat-label">{label}</p>
      <p className="stat-value">{value}</p>
      {detail ? <p className="stat-note mono">{detail}</p> : null}
    </div>
  );
}
