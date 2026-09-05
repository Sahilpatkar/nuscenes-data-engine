# Web front-end v1 Implementation Plan (the story site)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Reviews are BATCHED: one consolidated review after Task 7, the final whole-branch review before push.**

**Goal:** A designed scrollytelling site (`web/`, Vite + React + TS, GitHub Pages) telling the demo's 7-step story with first-class light/dark themes, every number exported from `demo_data/` v0.8 and guarded by CI, deep links into the live Streamlit app — per spec `docs/superpowers/specs/2026-09-04-web-frontend-v1-design.md` (its five story contracts are binding).

**Architecture:** `web/build_data.py` (strict-typed Python) derives one JSON per section + ~11 pre-rendered overlay/thumb images from the package by reusing `app/demo/filters.py` + `app/demo/render.py`; the committed bundle is guarded by `tests/test_web_export.py`. The site is a single page of seven sections + landing + footer with a token design system (dark design-primary from the deck, warm-paper light), hand-rolled SVG charts, and an IntersectionObserver reveal hook. One small app change mirrors the story framing into the Streamlit tour.

**Tech Stack:** Vite 6 + React 18 + TypeScript strict; runtime deps react/react-dom only; Google Fonts (Syne, IBM Plex Sans/Mono); GitHub Actions Pages deploy. Python side: pandas + PIL via the existing venv.

**Working branch:** `web-frontend-v1` (cut from main at df3e91d). One commit per task. Baseline: 805 passed / 3 deselected (2 Neo4j-gated skips when the local stack is down). NOTE: the `demo-live-url` PR (README:20 live line, DEMO.md live stamps) is still open — do not touch those exact lines; the user merges that PR independently.

**Verified facts (do not re-derive):**
- Import pattern for the exporter: `tests/test_demo_filters.py:11-12` (`sys.path.insert(0, str(APP_DEMO))` then `from filters import …`). `app/demo/filters.py` imports only pandas. `app/demo/render.py` imports streamlit/altair at module top but executes no calls at import; `draw_overlay(image, gt_boxes, predictions, *, mode, scale, pseudo_boxes=None)` is pure PIL; `STYLE_*` RGBs: GT `(46,204,64)` solid w2, FN `(255,133,27)` dash w3, TP `(255,255,255)` w1, low-conf `(255,220,0)` dash w2, FP `(255,65,54)` w2; `LEGEND_ITEMS` wording in render.py.
- Derivation helpers (all in `app/demo/filters.py`): `failure_flags`, `visible_gt`, `visible_gt_boxes`, `gt_for_render`, `rank_events`, `severity_caption`, `filmstrip_steps` (FilmstripCurve.speed_is_can), `parity_short`, `tour_frame_candidates`, `reason_chips`, `selection_factors`, `frame_quota_before`, `tour_strategies`, `strategy_coverage`, `fixed_boxes`, `upgrade_callout`, `gain_text`, `relative_gain`, `arm_story_label`.
- Tour copy source of truth: `app/demo/views/tour.py` — constants `_HELD_OUT_CAPTION`, `_BRIDGE_TO_MINING`, `_MINING_MECHANISM`, `_SELECTION_PATH`, `_ARCHITECTURE_STRIP`, `_CLOSING_THESIS`, `_HERO_HONESTY_LINE`, headline subheaders per step, `_night_share_line`, `_best_night_arm_sentence(arms, *, baseline)` (excludes baseline; no-win branch), `_similarity_sentence`, `_result_headline` guard, `_night_ped_sentence`, `_result_hero_cards`, the four `_result_*` answers, `_STEPS` link labels. The exporter re-derives the private-sentence twins with the same rules and tests pin them to these recorded outputs.
- Package anchors (v0.8): hero `5994f34b836043b3b5be191bceba2e3e` (crop 81 KB); exemplar `00ec313e6a6b45cf8d8b5022a0dbe4a6`? — NO: read `demo_data/al_exemplars.json` for the exact first token (`00ec313e…`); flagship event `5767aa11…` scene-1084 with neighbour tokens in `scenario_events.parquet`; step-4 selected token computed by `tour_frame_candidates` (first with a crop: `00740c25…` on v0.8); arm table numbers 0.2477/0.1667/0.0826, 60-of-62, 1,500/368/31 % vs 13 %/0 %, winner `graph_rate_night` +0.0101, `rate_strat` +0.0069 best score-based, night-ped 0.0826 → 0.1171 (+41.8 %), callout `('pedestrian','low-conf 0.30','0.47')`, weak retention 18.2 %.
- Streamlit page URLs: `https://nuscenes-data-engine-sahil.streamlit.app` + `/tour`, `/failures`, `/scenarios`, `/active_learning`, `/weak_supervision`, `/chat_replay` (pinned `url_path`s in `app/demo/main.py`).
- Deck tokens (`docs/presentation/nuscenes-data-engine-deck.html` head): ground `#0B1020`, panel `#141A2E`, panel-2 `#1B2238`, line `#2A3350`, ink `#EDE9E1`, muted `#8A93A6`, amber `#F2A33A`, amber-deep `#C77A14`, cyan `#5BC0EB`, pass `#3DDC97`, fail `#FF5A5F`; fonts link `https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap`.
- CI: `.github/workflows/ci.yml` quality job uses `uv sync --extra dev --extra data --extra engine --extra serve` then ruff/mypy/pytest — the exporter guard runs there for free. mypy config in `pyproject.toml` (`files = ["src/nuscenes_data_engine", "app/demo"]`).
- Dataviz validator: `/private/tmp/claude-501/bundled-skills/2.1.240/932f0323748d00a11c6a1109ed701162/dataviz/scripts/validate_palette.js` (run with node; `--mode light|dark`); its skill's mark specs are the chart contract. (Path is session-local — if absent, re-derive contrast with any WCAG checker and say so.)
- GitHub Pages: project site base path `/nuscenes-data-engine/`; workflows `actions/configure-pages@v5`? (use the current upload-pages-artifact@v3 + deploy-pages@v4 pattern); Source = GitHub Actions is a manual repo setting.

---

### Task 1: scaffold + tokens + typography (design first)

**Files:** create `web/package.json`, `web/package-lock.json` (via `npm install`), `web/vite.config.ts`, `web/tsconfig.json`, `web/index.html`, `web/.gitignore` (`node_modules/`, `dist/`), `web/src/main.tsx`, `web/src/App.tsx` (specimen), `web/src/styles/tokens.css`, `web/src/styles/base.css`, `web/src/theme.ts`, `web/src/components/ThemeToggle.tsx`.

- [ ] Scaffold Vite React-TS manually (no `npm create` interactivity): package.json with react/react-dom + dev typescript/vite/@vitejs/plugin-react; `vite.config.ts` reads `process.env.BASE_PATH ?? "/"`; strict tsconfig with `resolveJsonModule`.
- [ ] `index.html`: the deck's Google Fonts link + preconnect; the pre-paint theme script (reads `localStorage.theme`, sets `data-theme`); `<title>nuScenes Data Engine — from model failure to better training data</title>`; meta description + OG tags.
- [ ] `tokens.css` per spec §2 (three-layer, both palettes complete, `color-scheme` per theme); `base.css` (reset, type scale: Syne display / Plex body / Plex mono data, `text-wrap: balance` on headings, 65ch measure, tabular-nums).
- [ ] Chart slot tokens validated: run the dataviz validator for the categorical pair (muted bar vs amber accent) and status colors against BOTH surfaces; record the passing hexes as CSS tokens with a comment citing the run.
- [ ] Specimen `App.tsx`: type scale, tokens, both themes via the toggle — a temporary page proving the identity before content.
- [ ] `npm run build` + `npm run typecheck` pass locally (add a `typecheck` script). Commit `web: scaffold + token design system (dark-first deck identity, warm-paper light)`.

### Task 2: exporter + committed bundle + guard

**Files:** create `web/build_data.py`, `web/src/data/types.ts`, generated `web/src/data/*.json`, generated `web/public/story/*.jpg`; modify `pyproject.toml` (mypy files/mypy_path); tests `tests/test_web_export.py`.

- [ ] Failing tests first (`tests/test_web_export.py`): exporter produces the eight JSON files with required keys and no NaN/None in display strings; tour-twin pins (winner sentence == the tour's on v0.8 incl. `(\`graph_rate_night\`, +0.0101 night)`; fairness parts with the 8,535 budget clause; closed-loop headline "measurable improvement"; night-ped `gain_text` string; callout tuple; chips list; the five §1-contract fixed sentences present in their fields); deep links start with the Streamlit base; every image ref's file exists with the stated dimensions; re-running the exporter is byte-stable (run twice, compare).
- [ ] Implement `web/build_data.py` per the spec schema (canonical JSON writer; images via `draw_overlay` at scale 0.6, JPEG q85; only the ~11 assets). Run it; commit the generated bundle. Add the byte-equality guard test (committed vs regenerated into tmp).
- [ ] `uv run ruff check .`, `uv run mypy` (with the pyproject addition), `uv run pytest -q` green. Commit `web: data exporter + committed story bundle + CI guard`.

### Task 3: scrollytelling shell

**Files:** create `web/src/components/Section.tsx`, `useReveal.ts`, `StepRail.tsx`, `Landing.tsx`, `Footer.tsx`, `DeepLink.tsx`; rewrite `App.tsx` (specimen → real shell with stubbed sections rendering each bundle's headline + provenance).

- [ ] Landing (title, thesis, hero image, scroll affordance, links to the live app + GitHub), 7 anchored sections with eyebrow/step labels, footer (meta.json provenance + CC BY-NC-SA attribution + both-front-ends note), step rail with active tracking, reveals disabled under `prefers-reduced-motion`. Typecheck + build green. Commit `web: scrollytelling shell (sections, step rail, landing, footer)`.

### Task 4: sections 1–4

**Files:** create `web/src/components/StatTile.tsx`, `sections/Blindspot.tsx`, `sections/MissedPedestrian.tsx`, `components/OverlayLegend.tsx`, `sections/Scenario.tsx`, `components/Filmstrip.tsx`, `components/charts/CanCurves.tsx`, `sections/WhyFrame.tsx`.

- [ ] Blindspot per contract §1.1 (three beats). MissedPedestrian: overlay image (arm by default or a baseline/arm tab-toggle — keep simple: show the ARM overlay with the claims list; both images are in the bundle), claims, legend chips, held-out + bridge sentences. Scenario: event card + filmstrip↔curves scrub sync (hover/focus/tap; amber marker; direct labels; reduced-motion safe). WhyFrame per contract §1.2 (lede → plain chain visual → frame + chips → `<details>` fold with factors + implementation chain + mechanism note). Typecheck + build. Commit `web: story sections 1-4 (blind spot, missed pedestrian, scenario, centerpiece)`.

### Task 5: sections 5–7

**Files:** create `web/src/components/charts/StrategyBar.tsx`, `sections/Intervention.tsx`, `components/CompareWipe.tsx`, `sections/Verdict.tsx`, `sections/ClosedLoop.tsx`.

- [ ] StrategyBar per dataviz specs (≤24 px bars, zero rule, direct value+label per bar, amber highlight on the tour arm, hover tooltip); fairness strip per contract §1.3 bound to the chart; winner sentence below. Verdict per contract §1.4 (labeled tiers; CompareWipe range input + ≤480 px side-by-side fallback; legend; honesty line). ClosedLoop: three hero StatTiles, four answers, thesis (§1.5), deep links. Typecheck + build. Commit `web: story sections 5-7 (intervention, verdict, closed loop)`.

### Task 6: tour framing mirror (the one app change)

**Files:** modify `app/demo/views/tour.py`; tests `tests/test_demo_app.py`.

- [ ] Per spec §3: step 0 three-beat opening (subheader → problem sentence; purpose line after the derived sentence); step 3 lede + plain-language chain visible, `_SELECTION_PATH` into the fold; step 4 fairness chip strip above the chart (bordered container; "Random sample is the control" emphasized; derived budget condition unchanged); step 5 evidence-tier labels ("One example — one hand-approved frame, illustrative, not the metric" / bold "The aggregate result" lead-in); step 6 `_CLOSING_THESIS` = the §1.5 sentence. Update every affected pin (walk tests, result test) in the same commit; run the full suite. Commit `demo: tour mirrors the story-site framing (opening beats, centerpiece chain, fairness strip, evidence tiers, thesis)`.
- [ ] Then regenerate the web bundle if any exporter-pinned tour sentence changed (`uv run python web/build_data.py`; the guard test tells you) and commit the delta with it.

### Task 7: CI + deploy + docs

**Files:** modify `.github/workflows/ci.yml` (web job); create `.github/workflows/pages.yml`; modify `README.md`, `docs/DEMO.md`.

- [ ] ci.yml `web` job: setup-node 22 + npm cache (web/package-lock.json), `npm ci`, `npm run typecheck`, `npm run build` (BASE_PATH set), working-directory web. pages.yml per spec §2 (paths filter `web/**`, workflow_dispatch, upload-pages-artifact v3 + deploy-pages v4, permissions pages/id-token).
- [ ] README: a "Story site" line beside the live-demo line (placeholder URL "pending first deploy"; do NOT touch README:20's exact live-demo line — the open `demo-live-url` PR owns it); docs/DEMO.md: a short "Two front-ends" section (one package, two readings) + the Pages runbook (enable Pages once: Settings → Pages → Source = GitHub Actions) + phase row. Commit `web: CI + GitHub Pages deploy + two-front-ends docs`.

**>>> CONSOLIDATED REVIEW GATE (Tasks 1–7): exporter honesty vs the shipped tour (pinned twins), the five story contracts implemented as specified, dataviz anti-patterns catalog over the three charts, both-themes token audit (no colour defined only inside a media/data-theme block; body paints its ground), a11y (focus states, alt text, reduced motion, the range input), bundle size (< 2 MB total). Fix round. <<<**

### Task 8: operational gates

- [ ] Browser walk (`npm run dev` + Playwright temporarily in `.venv` or plain manual via screenshots): both themes, reduced motion, 360 px width, the wipe, the scrub sync, anchor nav; screenshots for the record.
- [ ] `uv run ruff check . && uv run mypy && uv run pytest -q` + `npm run typecheck && npm run build`; final whole-branch review → push `web-frontend-v1` → PR link. After merge + the user enabling Pages: watch the pages.yml run via the public Actions API, verify https://sahilpatkar.github.io/nuscenes-data-engine/ in both themes, then a small follow-up records the URL in README/DEMO.md.
