# Story site v2 Implementation Plan (the research-report edition)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Reviews are BATCHED: one consolidated review after Task 3, the final whole-branch review before push.**

**Goal:** A second, fully code-designed edition of the story site at `/v2/` — research-report editorial (paper/ink, Source superfamily, hairline rules, cobalt accent), light-first with first-class dark — rendering the SAME committed bundle. Spec: `docs/superpowers/specs/2026-09-05-web-frontend-v2-design.md` (its zero-touch-v1 rule, identity, and contract obligations are binding).

**Architecture:** Vite MPA (`vite.config.ts` input map — the only shared file edited); all new code under `web/v2/index.html` + `web/src/v2/**`; shares the bundle, `theme.ts`, `useReveal`, and the four pure renderers; forks every visual component. `assetUrl.ts` (`import.meta.env.BASE_URL + src`) fixes the document-relative image srcs from `/v2/`.

**Tech Stack:** the existing web project (Vite 6, React 18, TS strict; no new npm deps). Fonts: Source Serif 4 + Source Sans 3 + Source Code Pro (URL in the spec).

**Working branch:** `web-frontend-v2` (cut from main). One commit per task. Python baseline: 836 passed / 3 deselected (2 Neo4j-gated skips when the local stack is down). The open `story-site-live` PR owns README's Story-site line (~:22) and DEMO.md's Story-site-URL (~:49) + walk-stamp cells — doc mentions go elsewhere (README's two-front-ends paragraph ~:38-46; a `### Report edition (/v2/)` subsection AFTER DEMO.md's Pages runbook) unless it merges first.

**Verified facts (do not re-derive):**
- `web/package.json` is `"type": "module"` → vite.config entries via `fileURLToPath(new URL(p, import.meta.url))`, not `__dirname`. Vite MPA preserves input paths → `web/v2/index.html` emits `dist/v2/index.html`; shared chunks land in `dist/assets/` referenced base-prefixed root-absolute. Dev URL: `http://localhost:5173/v2/`.
- Bundle image refs are document-relative (`"src": "story/hero-arm.jpg"`, e.g. `web/src/data/hero_frame.json:27`) — correct at root, 404 from `/v2/` without `assetUrl`.
- v1 conventions to mirror (read these files first): `web/src/styles/tokens.css` (three-layer pattern, twin-block markers, validator-run comment, `--image-chip*` trio), `web/index.html` (pre-paint theme script), `web/src/App.tsx` (SECTIONS literal block shape the copy tests grep), `tests/test_web_export.py` (helpers `_tour_steps`, `_only`, `_source`; the contract-sentence constants; the v1 copy-binding tests to mirror for v2).
- Shared-safe components (import from `../..`): `data/*.json` + `data/types.ts`, `theme.ts`, `hooks/useReveal.ts`, `components/Ticks.tsx`, `components/Learned.tsx`, `components/OverlayLegend.tsx`, `components/DeepLink.tsx`. Everything else visual is forked into `src/v2/`.
- Dataviz validator: `/private/tmp/claude-501/bundled-skills/2.1.240/932f0323748d00a11c6a1109ed701162/dataviz/scripts/validate_palette.js` (node; `--mode light|dark`); if absent, compute WCAG ratios by hand and say so. Emphasis pair = chart-muted bronze vs chart-accent cobalt; polarity pair = pos/neg; validate per mode against light `#EFECE4` / dark `#22262E` (worst surfaces).
- v1's charts: `web/src/components/charts/{StrategyBar,CanCurves}.tsx` — geometry, signed direct labels, sr-only div-wrapped table twins, focusable `graphics-symbol` bands, `LABEL_ADVANCE` char-width constant (retune for Source Code Pro if label collision appears). CompareWipe: range-input clip-path + <480px static fallback. Filmstrip: roving tabindex + shared activeStep.
- The five contract sentences ship as bundle fields (`blindspot.problem_sentence`/`purpose_sentence`, `why_frame.lede_sentence`/`selection_chain`, `intervention.fairness_parts`/`fairness_sentence`, `verdict.example_label`/`example_caveat`/`aggregate_label`, `closed_loop.closing_thesis`).

---

### Task 1: MPA scaffold + tokens + type specimen (identity first)

**Files:** modify `web/vite.config.ts`; create `web/v2/index.html`, `web/src/v2/main.tsx`, `web/src/v2/vite-env.d.ts`, `web/src/v2/assetUrl.ts`, `web/src/v2/styles/tokens.css`, `web/src/v2/styles/base.css`, `web/src/v2/App.tsx` (SPECIMEN), `web/src/v2/components/ThemeToggle.tsx`.

- [ ] vite.config input map (ESM-safe). `v2/index.html`: `<html lang="en">`, the Source fonts URL + preconnects, the same pre-paint theme script as v1's index.html, `<title>nuScenes Data Engine — report edition</title>`, meta/OG, `<div id="root">`, module script `/src/v2/main.tsx`.
- [ ] `tokens.css` per spec §2 (light-first three layers; twin DARK SET markers; chart slots; validator runs recorded inline with pass ratios; `--image-chip*` verbatim from v1). `base.css`: reset scoped to the document, type scale, hairline-rule utilities, `.page`/measure/figure-breakout classes, footnote styles, focus-visible in accent, reduced-motion guards.
- [ ] Specimen App: type ramp (cover/section/eyebrow/body/mono), palette + rule samples, one restyled StrategyBar over the real `intervention.json` via the shared chart? NO — charts are forked in Task 3; specimen may inline a rough SVG bar sample styled by tokens instead (no result literals — use the bundle data via import). Mark `{/* SPECIMEN — replaced in Task 2/3 */}`.
- [ ] Gate: `cd web && npm run typecheck && npm run build && BASE_PATH=/nuscenes-data-engine/ npm run build` — assert `dist/v2/index.html` exists, its asset URLs carry the base prefix, and `dist/index.html` (v1) still emits. Headless eyeball of both themes (playwright install→use→uninstall pattern). `git diff --stat` shows NO v1 source file except vite.config.ts.
- [ ] Commit `web: v2 scaffold + research-report token system (Source superfamily, cobalt on paper)`.

### Task 2: report chrome

**Files:** create `web/src/v2/components/{Header,SectionFrame,Cover,Colophon,StatTile}.tsx`, `web/src/v2/hooks/useActiveSection.ts`; rewrite `web/src/v2/App.tsx` (SECTIONS literal block with the EXACT tour titles/stages, cover → seven numbered section frames with stubbed bodies + per-section provenance footnotes → colophon).

- [ ] Sticky header (wordmark · "0N / 07 — {title}" scroll-spy · ThemeToggle · 1px accent progress underline), hash anchors, keyboard accessible. Cover = report masthead (title, dek from the meta description text, date/package line from `meta.json`, links to the live app + root edition via `assetUrl`/`BASE_URL`). SectionFrame: full-measure rule, mono index + pinned title, stage/act small-caps, children, footnote-style provenance from the bundle. Colophon: provenance, attribution, back-links, author.
- [ ] Gate: typecheck + both builds; headless shot both themes. Commit `web: v2 report chrome (contents header, cover, numbered section frames, colophon)`.

### Task 3: the seven sections + figures + charts

**Files:** create `web/src/v2/sections/{Blindspot,MissedPedestrian,Scenario,WhyFrame,Intervention,Verdict,ClosedLoop}.tsx`, `web/src/v2/components/{Filmstrip,CompareWipe}.tsx`, `web/src/v2/components/charts/{CanCurves,StrategyBar}.tsx`; wire into App (specimen deleted).

- [ ] Fork from the v1 sections keeping ALL mechanics/contracts; restyle to the report idiom (figures numbered sequentially across the document; the centerpiece band; the fairness hairline box above the chart; the two evidence tiers as report sub-heads; footnote provenance handled by SectionFrame). Every `<img>` via `assetUrl`. Charts restyled with the v2 slots + Source Code Pro labels (retune LABEL_ADVANCE if needed); sr-only twins, signed labels, graphics-symbol bands preserved.
- [ ] Gate: typecheck + both builds; headless walk of /v2/ both themes + 360px scrollWidth check; `uv run --no-sync pytest -q tests/test_web_export.py` untouched-green. Commit `web: v2 story sections (report figures, restyled charts, evidence tiers)`.

**>>> CONSOLIDATED REVIEW GATE (Tasks 1–3): the five contracts structurally present at /v2/; dataviz audit of the new palette (validator numbers reproduced); DARK-SET twin-block diff audit; no result literals in v2 TSX; every img through assetUrl; a11y (focus, roving tabindex, range input, reduced motion, heading order); zero-touch-v1 (`git diff main...HEAD -- web/src ':(exclude)web/src/v2'` empty except nothing; vite.config.ts the only shared edit). Fix round. <<<**

### Task 4: tests + docs

**Files:** create `tests/test_web_v2.py`; modify `README.md`, `docs/DEMO.md` (placement per the sequencing note in the header).

- [ ] The five source-level tests per spec §4, importing `test_web_export.py`'s helpers; run RED against a doctored mutation (e.g. temporarily…— no: verify at least one test fails if a title is renamed, by inspection or a scratch mutation run). README two-front-ends paragraph gains the report-edition sentence + URL (placeholder "…/v2/ — live after the next Pages deploy" if merged before deploy); DEMO.md `### Report edition (/v2/)` subsection (what it is, same bundle, zero drift, where the code lives).
- [ ] Full gates: `uv run ruff check .`, `uv run mypy`, `uv run pytest -q` (836 + new), npm typecheck + both builds. Commit `web: v2 copy-contract tests + report-edition docs`.

### Task 5: operational gates

- [ ] Browser walk at /v2/ (both themes, reduced motion, 360px, wipe/scrub/fold, console clean) + root-page glance (v1 unchanged); screenshots for the record; final whole-branch review; push `web-frontend-v2`; PR link (body: v1 source untouched, vite.config.ts the only shared edit, same bundle, zero data drift). After merge: Pages auto-redeploys; curl-verify `…/v2/`; record the URL if a placeholder was used.
