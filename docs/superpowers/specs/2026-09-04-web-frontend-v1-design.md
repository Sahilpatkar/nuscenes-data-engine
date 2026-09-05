# Web front-end v1 — the story site (`web/`)

**Status:** approved 2026-09-04 (plan-mode session; user decisions: custom front-end in
addition to the deployed Streamlit app, story-first v1, light + dark both first-class,
GitHub Pages deploy; five story contracts added during plan review — opening beat,
centerpiece, fair comparison, evidence hierarchy, closing thesis). The Streamlit demo
(phases 1–10, live at https://nuscenes-data-engine-sahil.streamlit.app/) stays deployed
and untouched except the small tour framing mirror in §6.

## Why

The Streamlit app is the full instrument, but its chrome is Streamlit's. This phase
builds a **designed** telling of the same story: a single scrollytelling page with a
real visual identity (the presentation deck's night-navy + amber + Syne/IBM Plex),
first-class light and dark themes, and hand-rolled charts — deployed on GitHub Pages,
deep-linking into the live app for every deep dive. Two front-ends, one committed
package, one set of derived numbers.

## Honesty rules (binding, inherited from the demo phases)

- **Every number is derived at export time** from `demo_data/` (v0.8) by
  `web/build_data.py`; React components contain no numeric result literals. The
  committed JSON bundle is guarded by a pytest that re-runs the exporter and asserts
  byte-equality — drift against the package goes red.
- **Superlatives and headlines computed, never asserted** (winner sentence, closed-loop
  headline guard, similarity caption condition) — re-derived with the same rules as
  `views/tour.py` and pinned by tests to the shipped tour's recorded output.
- Absolute + relative gains always together (`filters.gain_text`); the weak-supervision
  negative result is a required field; the 0.135 low-conf disclaimer chain survives;
  provenance (package `git_sha`, `built_at`, exporter name) is visible in the footer;
  nuScenes CC BY-NC-SA 4.0 attribution as in the README.
- Story labels front arm ids with the raw id beside them in small text, exactly as the
  tour does.

## 1. Story contracts (user review, binding)

1. **Opening beat (blind spot):** three beats in order — (a) plain-language problem:
   "The detector looked reasonable overall — but performance dropped sharply at night,
   especially for pedestrians."; (b) the numbers (three stat tiles 0.2477 / 0.1667 /
   0.0826 + the 60-of-62 line); (c) the causal objective: "The goal of the system:
   automatically find failures like this and turn them into better training data." The
   viewer must know what the system *is* by the end of the section.
2. **Centerpiece (why-this-frame):** the differentiator from "model failed → retrain".
   Visible story: lede "Instead of randomly adding more images, the system searches the
   training pool for examples related to the diagnosed failure." → the plain-language
   chain as the section's visual device: **Failed validation frame → relevant driving
   scenario → candidate training frames → targeted retraining set** → the concrete
   frame with its reason chips. The implementation-oriented chain and the mechanism
   details (similarity communities, graph context, CAN signals, quotas, failure-mass
   scoring, night floor) fold underneath. Largest visual weight after the landing.
3. **Fair comparison (intervention):** the fairness statement ("Same detector · same
   8,535-frame budget · same training configuration · scored on the same held-out
   split — Random sample is the control.") is a design element visually bound to the
   strategy chart — a bordered strip whose clauses render as mono chips, the control
   clause emphasized — not secondary prose. Budget clause keeps its derived condition.
4. **Evidence hierarchy (verdict):** two labeled tiers — **"One example"** (the
   before/after compare + low-conf 0.30 → 0.47 callout, caveat "one hand-approved
   frame — illustrative, not the metric") then **"The aggregate result"** (the
   `gain_text` sentence, 0.0826 → 0.1171, +0.0345 absolute / +41.8 % relative, with
   stat-tile weight). The structure itself answers "did you just pick a flattering
   image?".
5. **Closing thesis:** "Instead of blindly retraining the model, the system diagnoses
   where it fails, finds the data that can address the weakness, and measures whether
   the intervention actually works." — everywhere the thesis appears (site and tour).

## 2. Architecture

- **Stack:** Vite + React + TypeScript in `web/`; runtime deps `react`/`react-dom`
  only; hand-rolled token CSS (no framework); hand-rolled SVG charts per the dataviz
  method (thin marks, ≤ 24 px bars, 2 px lines, direct labels, one amber accent, zero
  rule, hover tooltips; palette validated against both surfaces with the dataviz
  validator, never eyeballed).
- **Exporter** `web/build_data.py` (strict mypy): sys.path-imports `app/demo/filters.py`
  (pandas-only) and `app/demo/render.py` (`draw_overlay`, `STYLE_*`, `LEGEND_ITEMS`;
  imports streamlit at module top but makes no calls at import). Emits one typed JSON
  per section into `web/src/data/` (`meta`, `blindspot`, `hero_frame`, `scenario`,
  `why_frame`, `intervention`, `verdict`, `closed_loop`) and ~11 pre-rendered images
  (~0.7 MB, scale-0.6 JPEGs: hero baseline+arm overlays, exemplar baseline+arm,
  flagship GT overlay, five filmstrip thumbs, selected-frame overlay) into
  `web/public/story/`. Canonical JSON: `sort_keys=True`, floats rounded at source,
  `\n` endings. Every image ref is `{src, width, height, alt}`.
- **Committed bundle + guard:** the generated JSON/images are committed;
  `tests/test_web_export.py` re-runs the exporter (JSON byte-equality; image set +
  dimensions) and pins the tour-twin sentences to the shipped tour's recorded output.
- **Page:** single route, hash anchors, fixed 7-dot step rail. Landing → the seven
  sections (per §1 contracts; scenario section has the filmstrip↔CAN-curves scrub
  sync; verdict uses an accessible range-input CompareWipe with a side-by-side
  fallback under ~480 px) → footer (provenance, attribution, both-front-ends note).
  Scroll reveals via one IntersectionObserver hook — pure enhancement, disabled under
  `prefers-reduced-motion`, content always in the DOM.
- **Tokens** (`web/src/styles/tokens.css`): bare `:root` = the deck's dark set
  (design-primary, `color-scheme: dark`); `@media (prefers-color-scheme: light)
  { :root:not([data-theme]) }` and `:root[data-theme="light"]` = the warm-paper light
  set (`#FAF7F1` ground, navy-as-ink `#1C2233`, accent promoted to `#C77A14` for text;
  bright amber `#F2A33A` survives as large marks); `:root[data-theme="dark"]` explicit
  override. Pre-paint inline head script applies `localStorage.theme`; the toggle
  cycles light/dark, unset = follow system. Legend swatches use the exact `STYLE_*`
  RGBs on an image-toned chip (the colours are burned into the JPEGs — theme-invariant).
- **Deep links:** the pinned Streamlit `url_path`s against
  `https://nuscenes-data-engine-sahil.streamlit.app`, emitted by the exporter, styled
  as story sentences (the tour's "PageName — clause" shape, no digits in labels).
- **CI/deploy:** `ci.yml` gains a `web` job (node 22, `npm ci`, `npm run typecheck`,
  `npm run build` with `BASE_PATH=/nuscenes-data-engine/`); new
  `.github/workflows/pages.yml` builds `web/dist` and deploys via
  `actions/upload-pages-artifact` + `actions/deploy-pages` on pushes to main touching
  `web/**` (+ manual dispatch). Hermetic: no Python in the deploy job — the committed
  bundle is the contract. **Manual user step:** repo Settings → Pages → Source =
  "GitHub Actions". `pyproject.toml` mypy `files` gains `web` (mypy_path for the
  filters import; targeted override if resolution collides).

## 3. Tour framing mirror (`app/demo/views/tour.py`, the one app change)

So both front-ends tell one story: step 0 gains the three-beat opening (problem
sentence subheader; purpose line after the derived sentence); step 3 gains the lede +
plain-language chain with `_SELECTION_PATH` demoted into the fold; step 4's fairness
sentence moves into a bordered chip strip directly above the chart; step 5 gains the
two evidence-tier labels; step 6's `_CLOSING_THESIS` becomes the §1.5 sentence. All
pinned walk tests update in the same commit; every other honesty line unchanged.

## 4. Not in v1 (binding)

Deep-dive pages (Failure Explorer, 13-arm chart/table, quotas), the interactive graph
panel, chat replay, semantic search, parquet-in-browser, client-side recomputation,
Playwright e2e, analytics. Each is a labeled deep link into the live Streamlit app.

## 5. Docs and gates

README + docs/DEMO.md present the two front-ends honestly (one package, two readings);
the Pages URL is recorded after the first deploy (like the Streamlit handshake). Gates:
consolidated review after the code tasks (exporter honesty vs the shipped tour; the
dataviz anti-patterns catalog; a both-themes token audit — no colour defined only
inside a media/data-theme block) + fix round; a browser walk in light + dark +
reduced-motion + 360 px; final whole-branch review; push; PR.
