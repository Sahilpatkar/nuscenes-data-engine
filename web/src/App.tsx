import { ThemeToggle } from "./components/ThemeToggle";

/* SPECIMEN — replaced in Task 3 by the scrollytelling shell.
   Its only job is to put every token and primitive on one screen so the two
   themes can be checked side by side before any story content exists.
   Numbers below are deliberate placeholders, never results. */

const SURFACES = ["--ground", "--panel", "--panel-2", "--line"];
const INK = ["--ink", "--muted", "--accent", "--accent-deep", "--accent-strong", "--cyan", "--pass", "--fail"];
const CHART = ["--chart-accent", "--chart-muted", "--chart-pos", "--chart-neg", "--chart-zero", "--chart-grid"];

function Swatches({ title, tokens }: { title: string; tokens: readonly string[] }): JSX.Element {
  return (
    <div>
      <p className="eyebrow" style={{ marginBottom: 10 }}>
        {title}
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 10 }}>
        {tokens.map((token) => (
          <div key={token} style={{ border: "1px solid var(--line)", background: "var(--panel)" }}>
            <div style={{ height: 52, background: `var(${token})` }} />
            <div className="mono" style={{ fontSize: "0.72em", padding: "8px 10px", color: "var(--muted)" }}>
              {token}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App(): JSX.Element {
  return (
    <main
      style={{
        maxWidth: 1080,
        margin: "0 auto",
        padding: "clamp(24px, 4vw, 56px)",
        display: "grid",
        gap: "clamp(28px, 4vw, 48px)",
      }}
    >
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 24 }}>
        <p className="eyebrow">Specimen · design system</p>
        <ThemeToggle />
      </header>

      <section style={{ display: "grid", gap: 16 }}>
        <p className="eyebrow">Step 00 — eyebrow</p>
        <h1>
          From model failure to <span style={{ color: "var(--accent-strong)" }}>better training data</span>
        </h1>
        <p className="lede">
          Display type is Syne; this lede is IBM Plex Sans one step up from body size, held to a short measure so
          the opening beat reads as a single thought.
        </p>
        <p className="measure">
          Body copy sits at the reading measure. The system finds where a detector fails, mines the training pool
          for examples related to that failure, retrains, and measures whether the intervention actually worked —
          which is the sentence the type scale has to carry without shouting.
        </p>
        <h2>Section heading, balanced across lines</h2>
        <h3>Subheading in Syne 700</h3>
        <p className="mono muted" style={{ fontSize: "0.82em" }}>
          Mono is for identifiers, captions and provenance: scene-1084 · graph_rate_night
        </p>
      </section>

      <section style={{ display: "grid", gap: 18 }}>
        <Swatches title="Surfaces" tokens={SURFACES} />
        <Swatches title="Ink, brand, status" tokens={INK} />
        <Swatches title="Chart slots (validated)" tokens={CHART} />
      </section>

      <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 14 }}>
        <div className="stat">
          <span className="stat-label">Placeholder metric</span>
          <span className="stat-value num" style={{ color: "var(--accent-strong)" }}>
            0.1234
          </span>
          <span className="stat-note">Specimen value — not a result.</span>
        </div>
        <div className="stat">
          <span className="stat-label">Placeholder gain</span>
          <span className="stat-value num" style={{ color: "var(--chart-pos)" }}>
            +0.0123
          </span>
          <span className="stat-note">Positive delta slot.</span>
        </div>
        <div className="stat">
          <span className="stat-label">Placeholder loss</span>
          <span className="stat-value num" style={{ color: "var(--chart-neg)" }}>
            −0.0123
          </span>
          <span className="stat-note">Negative delta slot.</span>
        </div>
      </section>

      <section className="panel" style={{ display: "grid", gap: 12 }}>
        <p className="eyebrow">Bordered panel</p>
        <p className="measure">
          Panels carry one border, one background, and no shadow — the same flat vocabulary as the deck, so the
          two front-ends look like one project.
        </p>
        <div className="chip-row">
          <span className="chip">night</span>
          <span className="chip">pedestrian</span>
          <span className="chip">low-confidence</span>
          <span className="chip" style={{ borderColor: "var(--accent-strong)" }}>
            highlighted chip
          </span>
        </div>
        <p className="mono" style={{ fontSize: "0.8em" }}>
          <a href="https://nuscenes-data-engine-sahil.streamlit.app">Link colour on a panel</a>
        </p>
      </section>
    </main>
  );
}
