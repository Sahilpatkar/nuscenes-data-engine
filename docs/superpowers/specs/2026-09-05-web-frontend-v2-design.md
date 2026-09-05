# Story site v2 — the research-report edition (`/v2/`)

**Status:** approved 2026-09-05 (plan-mode session; user decisions: one more version,
designed and implemented entirely in code with professional fonts, not reusing the
deck-derived identity; ships at `/v2/` under the existing GitHub Pages site while
the v1 story site stays the root page, source byte-for-byte untouched; direction =
research-report editorial, light-first with a first-class tuned dark).

## Why

v1 wears the presentation deck's night-navy + amber identity. v2 is a second,
fully code-designed reading of the same story — a research report: paper and ink,
a fine serif, hairline rules, numbered sections and figures, one restrained accent.
It renders the **same committed bundle** (`web/src/data/*.json` +
`web/public/story/`), so the two editions cannot disagree on a number.

## Binding rules

- **All five story contracts of the v1 spec** (`2026-09-04-web-frontend-v1-design.md`
  §1) hold structurally in v2: the three opening beats; the centerpiece lede +
  plain-language selection chain with the mechanism folded; the fairness strip
  structurally bound to the strategy chart; the "One example" / "The aggregate
  result" evidence tiers; the closing thesis.
- **Zero-touch v1**: no file under `web/src/` outside `web/src/v2/` changes, and no
  v1 section/component/style file changes at all. The ONLY shared file edited is
  `web/vite.config.ts` (the MPA input map). v1's dist chunk hashes may shift from
  shared-chunk splitting; its source may not.
- Honesty rules carry over: v2 TSX contains no numeric result literals; every
  displayed figure/sentence comes from the bundle fields; the dataviz method binds
  the restyled charts (sign never by hue alone, direct labels, sr-only table twins,
  validated palette per surface); reduced motion never hides content.

## 1. MPA wiring

- `web/vite.config.ts`: `build.rollupOptions.input = { main: <index.html>, v2:
  <v2/index.html> }` with ESM-safe `fileURLToPath` entries. `web/v2/index.html` →
  `dist/v2/index.html` → served at `…/nuscenes-data-engine/v2/`. Shared chunks emit
  base-prefixed root-absolute URLs — correct by construction. One npm project, one
  build; ci.yml and pages.yml need zero changes.
- **Asset paths:** the bundle's `story/…` srcs are document-relative and would 404
  from `/v2/`. `web/src/v2/assetUrl.ts` — `import.meta.env.BASE_URL + src` — is
  used by every v2 `<img>` (and the colophon back-link uses `BASE_URL` itself). No
  asset duplication.
- New code lives in `web/v2/index.html` + `web/src/v2/**` (main.tsx, vite-env.d.ts,
  assetUrl.ts, App.tsx, styles/{tokens,base}.css, components/**, sections/**,
  hooks/useActiveSection.ts).

## 2. The identity

- **Tokens** (`src/v2/styles/tokens.css`, plain `:root` — loaded only by the v2
  document): v1's audited three-layer pattern, **light-first**: bare `:root` =
  paper/ink + `color-scheme: light`; twin DARK SET blocks (textually identical
  between markers) in `@media (prefers-color-scheme: dark) { :root:not([data-theme]) }`
  and `:root[data-theme="dark"]`; `:root[data-theme="light"]` closes precedence.
  Shares `src/theme.ts` and the same `"theme"` localStorage key; same pre-paint
  script in the v2 head.
- **Palette** (starting values; every chart slot passes the dataviz validator
  against each mode's worst surface — light `#EFECE4`, dark `#22262E` — with the
  run recorded inline): light — ground `#F7F5F0`, panel `#FFFFFF`, panel-2
  `#EFECE4`, line `#D8D3C8`, ink `#1A1C21`, muted `#5B5E66`, accent **deep cobalt
  `#2447B2`**; dark — `#14161B` / `#1B1E24` / `#22262E` / `#333844` / `#E9E7E1` /
  `#9A9DA6` / `#93ACEE`. Chart slots keep v1's names: `--chart-accent` cobalt,
  `--chart-muted` **bronze** `#8C6A1D` / `#B08A3E`, `--chart-pos/neg` green/red,
  `--chart-zero/grid`. Rationale: an oxide-red accent would sit hue-adjacent to
  `--chart-neg` and spread a CVD hazard everywhere; cobalt-vs-bronze is a
  blue/yellow-axis emphasis pair — the safest available — and cobalt is the
  fountain-pen mark of printed research. v1's theme-invariant `--image-chip` trio is
  kept verbatim (legend swatch RGBs are burned into the JPEGs).
- **Type — the Source superfamily** (one foundry, cohesive, professional): Source
  Serif 4 (display 600/700, italic 400 for ledes and figure captions; true optical
  size axis), Source Sans 3 (text 400/500/600), Source Code Pro (data mono
  400/500). Google Fonts URL:
  `https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&family=Source+Sans+3:wght@400;500;600&family=Source+Code+Pro:wght@400;500&display=swap`
  Scale: body 17px/1.6, 65ch measure; cover title `clamp(2.1rem, 4.5vw, 3rem)`
  serif 700; section heads `clamp(1.45rem, 2.6vw, 1.95rem)` serif 600; eyebrows
  Source Sans 600 · 0.78rem · 0.08em uppercase; stat values Source Serif 600; arm
  ids/axes/provenance Source Code Pro; `tabular-nums` on every numeric context.
- **Texture & layout:** hairline rules where v1 had panels; each section opens with
  a full-measure rule, a mono two-digit index + the exact tour title ("01 / We
  found a blind spot" — the numbering is frame; the title string stays the pinned
  one), stage + act small-caps right-aligned; figures carry numbered captions
  ("Figure 3 — {bundle caption}"); provenance renders as a footnote block under a
  short rule. One centered column (~76ch container, 65ch text measure); figures and
  charts break out to ~60rem. **No rail** — a slim sticky report header: small-caps
  wordmark left; "02 / 07 — {current title}" scroll-spy + ThemeToggle right; a 1px
  `--accent` scroll-progress underline. The centerpiece's emphasis is a full-bleed
  `--panel-2` band between rules; the fairness strip is mono chips in a hairline
  box directly above the chart. Motion: opacity-only fade (shared `useReveal`),
  clean under reduced motion.

## 3. Share vs fork (zero-touch-v1 beats DRY)

Share as-is: `src/data/*` + `types.ts`, `src/theme.ts`, `src/hooks/useReveal.ts`,
`components/{Ticks,Learned,OverlayLegend,DeepLink}.tsx` (pure bundle-copy
renderers; v2's base.css restyles their class names within the v2 document).
Fork into `src/v2/`: ThemeToggle, Section (numbered header + footnote provenance),
the contents header (own `useActiveSection`), StatTile (serif values), Landing →
report cover, Footer → colophon (back-link to the root edition + the live app),
Filmstrip + CompareWipe (mechanics verbatim; forked only for `assetUrl`), both
charts (restyle only; all dataviz obligations survive), all seven sections, and
`App.tsx` (repeats the SECTIONS literal block so the titles/stages stay greppable
by the copy-contract tests).

## 4. Tests, docs, out of scope

- `tests/test_web_v2.py` (source-level, importing `test_web_export.py`'s helpers):
  (1) the seven step titles/stages in `v2/App.tsx` equal `tour.py::_STEPS` (step 7
  = "Closed the loop" / whole-loop stage); (2) the five contract sentences appear
  in no v2 TSX (they come from bundle fields, whose names must appear); (3) no
  numeric result literals in v2 TSX; (4) the "How selection works" fold label and
  "Fair comparison" lead are the tour's own; (5) every `src={` in v2 TSX routes
  through `assetUrl(`.
- Docs: README + DEMO.md gain a Report-edition mention placed to avoid the lines
  owned by the open `story-site-live` PR (its docs merge first when possible).
- Out of scope: any v1 visual change, any data/exporter change, deep-dive surfaces,
  a link from the v1 page to /v2/ (discoverability is docs + the v2 colophon).

## 5. Gates

Consolidated review (dataviz audit of the new palette per surface; the DARK-SET
twin-block diff audit; a11y; a scoped `git diff` proving zero v1-source touches) +
fix round; browser walk at `/v2/` in both themes + reduced motion + 360 px, plus a
root-page glance proving v1 renders unchanged; final whole-branch review; push; PR.
After merge, Pages redeploys automatically; verify the live `/v2/` and record its
URL in the docs if a placeholder was used.
