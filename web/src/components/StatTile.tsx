/**
 * One number, with the label that says what it measures.
 *
 * The generic tile the whole story uses: the blind spot's three baseline metrics,
 * and (from step 5 on) the intervention and result cards, which carry a delta
 * line as well. Every string arrives pre-formatted from the bundle — the tile
 * formats nothing and computes nothing.
 *
 * `accent` marks the one tile the section is *about* (the night-pedestrian metric
 * the whole story chases). At most one per row: two accents and the emphasis is
 * gone.
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
    <div className={accent ? "stat stat-tile is-accent" : "stat stat-tile"}>
      <p className="stat-label">{label}</p>
      <p className="stat-value">{value}</p>
      {detail ? <p className="stat-note">{detail}</p> : null}
    </div>
  );
}
