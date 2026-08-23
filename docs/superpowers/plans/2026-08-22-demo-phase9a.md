# Demo Phase 9a Implementation Plan (guided tour, outcome-first Overview, trust chrome)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Reviews are BATCHED: one consolidated review after Task 7, the final whole-branch review before push.**

**Goal:** A 2–3 minute "Start here" guided path through the model-improvement loop, an outcome-first Overview, and the loop breadcrumb / provenance / "what we learned" chrome on every page — spec `docs/superpowers/specs/2026-08-22-demo-phase9a-design.md`. App-only; package stays v0.7.

**Architecture:** `views/tour.py` (seven steps, `tour_step` state, Back/Next mutate state before rendering) reuses the existing page helpers (`draw_overlay`, `gt_for_render`, `visible_gt`, `rank_events`, `severity_caption`, `selection_factors`, `fixed_boxes`, `bar_chart`, `metric_cards`) over the same loaders; `nav.py` holds the page registry so views can deep-link without importing `main`. New `render.py` primitives: `loop_breadcrumb`, `provenance`, `chip_row`, `learned`; new `filters.py` helpers: `parity_short`, `tour_frame_candidates`.

**Tech Stack:** streamlit 1.59 (`st.navigation` sections, `st.page_link`, `st.switch_page`, `:color-badge[…]` markdown, `st.container(border=True)`), pandas, altair via `render.bar_chart`, Streamlit AppTest. Strict mypy + ruff. No new dependency.

**Working branch:** `demo-phase-9a` (stacked on `demo-phase-8` until #8 merges; rebase then). One commit per task. Baseline: 675 passed / 3 deselected.

**Verified facts (do not re-derive):**
- AppTest 1.59.2: in-script `st.switch_page` is not sticky across `at.run()` → tests open the tour with `at.switch_page("views/tour.py")`; `st.page_link` → `at.get("page_link")`; `st.badge`/`:orange-badge[x]` render as markdown; `at.sidebar.expander` works; `StreamlitPage.url_path` is public.
- Pinned widget keys (must survive): `failure_model`, `failure_type_select`, `scenario_preset_*`, `scenario_select_*`, `scenario_filmstrip`, `al_exemplar`, `al_model`, `al_frame_select_*`, `al_gallery_show_all`, `ws_*`, `chat_replay_graded`. Pinned strings: Overview footer sentences, `HERO_CAPTION` text, Scenario `parity_caption` wording, `_GRAPH_LEGEND`, the "val frames match the current filters" caption, the Failure Explorer fix-pair + FN captions. `test_demo_no_longer_promises_phase_8` forbids the literal "Phase 8" in `app/demo/*.py`.
- Page seams: `overview.py` (151 lines, static; `_HERO_CAPTION` :25, `_hero_overlay` :32, `render` :56); `failures.py` sidebar filters :90-182; `scenarios.py` `render` :557 (`_render_parity_line` :158, `_render_viewer` :388, `_render_filmstrip` :490, `_render_graph_panel` :248); `active_learning.py` `render` :532 (`_render_story` :124 beats, `_render_selected_frame` :292 factor panel, `_render_before_after` :452); `weak_supervision.py` `render` :482 (`_render_cards`, `_render_decomposition`); `chat_replay.py` `_render_header`. `data.py` loaders: `load_overview`, `load_al_results`, `load_frame_manifest`, `load_gt_boxes`, `load_predictions`, `events_available/load_events`, `load_subgraphs(preset)`, `load_al_exemplars`, `al_explain_available/load_al_explain/load_al_explain_validation`, `load_al_communities`, `load_weak_loss`, `load_weaksup`, `crop_path/thumb_path/frame_image_path`, `STALE_PACKAGE_NOTE`.
- Real-data anchors: hero `5994f34b…` (night + rain; pedestrian at 11.13 m missed by baseline, `graph_rate_night` low_conf 0.135); flagship rank-1 event `5767aa11…` (scene-1084, −4.46 m/s², 12 pedestrians within 10 m); AL-selected `0f70138f…` (flagship rank 12, community #10823 all-night, `pick_pass="night"`, quota 23 → 87, `weak_verdict="rejected"`); exemplar `00ec313e…` (pedestrian low-conf 0.30 → 0.47); arm table: baseline 0.2477/0.1667/night-ped 0.0826, `graph_rate_night` Δnight +0.0101 Δoverall +0.0254 `n_scenes` 368 `night_share` 0.309, `random` 501/0.127/−0.0048, `mined` 219/0.000/+0.0072, `graph` best overall +0.0344, `weak_graph_rate_night` Δnight −0.0262; weak headline 0.1824 (0.497/0.321), crowding 3.87 vs 7.61.
- Fixture (`tests/test_demo_app.py::built_demo_data`): hero `v0`, events `s1`/`v1`, AL-selected `wA`/`g28`/`g29` (explain staged; `wA` picked in the night pass with floor 1), exemplar `v0`, weak pair `random`, models `baseline` + `graph_rate_night`; variants `built_demo_data_without_explain`, `_without_chat_replay`. `_reset_demo_app_modules` + `_<page>_apptest` helper pattern.

---

### Task 1: `render.py` + `filters.py` helpers

**Files:** modify `app/demo/render.py`, `app/demo/filters.py`; tests `tests/test_demo_render.py`, `tests/test_demo_filters.py`.

- [ ] Failing tests: `loop_breadcrumb_text(["Mine"])` lights only Mine (`:orange-badge[Mine]`, others `:gray-badge`), `None` lights none, unknown stage raises; `provenance_text("recorded", "x")` contains the icon, the fixed sentence and `x`; unknown kind raises; `chip_row([])` emits nothing; `parity_short(30, 30, True)` == `"30 events found · SQL 30 / Graph 30 ✓"`, `(30, 29, False)` ends `"✗ mismatch recorded"`, `(4, None, None)` → `"Graph n/a (GT-only preset)"`, `(1, 1, True, noun="event")` singular; `tour_frame_candidates` orders night first, then pedestrian GT count desc, then token, and drops frames without an explain row.
- [ ] RED → implement per spec §3 (`LOOP_STAGES`, `loop_breadcrumb_text/loop_breadcrumb`, `PROVENANCE`, `provenance_text/provenance`, `chip_row`, `learned`; `parity_short`, `tour_frame_candidates`) → GREEN; ruff; mypy.
- [ ] Commit `demo: loop breadcrumb, provenance, chip/learned primitives, parity_short, tour candidates`.

### Task 2: `nav.py`, sectioned navigation, tour skeleton (steps 0–1)

**Files:** create `app/demo/nav.py`, `app/demo/views/tour.py`; modify `app/demo/main.py`, `app/demo/views/overview.py` (`_HERO_CAPTION` → `HERO_CAPTION`); tests `tests/test_demo_app.py`.

- [ ] Failing tests: `at.switch_page("views/tour.py")` renders title "Guided tour", caption "Step 1 of 7", `tour_back` disabled, `tour_next` enabled; clicking `tour_next` → "Step 2 of 7" with the hero image, `tour_hero_model` radio, and `HERO_CAPTION`'s pinned substring; every existing page test still passes (sectioned navigation keeps `url_path`s).
- [ ] RED → implement spec §1 machinery + steps 0 and 1 → GREEN; ruff; mypy.
- [ ] Commit `demo: guided tour skeleton (nav registry, sectioned navigation, steps 0-1)`.

### Task 3: tour steps 2–5

**Files:** modify `app/demo/views/tour.py`; tests `tests/test_demo_app.py`.

- [ ] Failing tests: extend the walk — step 3 shows the fixture's flagship event (`s1`) with `parity_short` text and "at night"; step 4 shows the AL-selected frame's `selection_factors` lines (`"night pass (night floor 1)"`), provenance "reproduced"; step 5 shows the Δnight chart (`at.get("vega_lite_chart")`) and the mined-frames cards; step 6 shows `tour_exemplar_model` radio and the upgraded-boxes table; `test_tour_degrades_without_optional_groups`: without explain → absent note at step 4; events parquet removed → step 3 shows the stale-package note, no exception.
- [ ] RED → implement spec §1 steps 2–5 → GREEN; ruff; mypy.
- [ ] Commit `demo: guided tour steps 2-5 (scenario event, why selected, retrain, before/after)`.

### Task 4: result screen (step 6)

**Files:** modify `app/demo/views/tour.py`; tests.

- [ ] Failing tests: step 7 shows the four answer headings, interpolated numbers from the fixture (`+0.0300`-style delta, the headline retention), ≥ 6 `page_link`s, `tour_next` disabled, `tour_restart` returns to step 1.
- [ ] RED → implement spec §2 → GREEN.
- [ ] Commit `demo: guided tour result screen`.

### Task 5: Overview outcome-first

**Files:** modify `app/demo/views/overview.py`; tests.

- [ ] Failing tests: `overview_start_tour` button exists and clicking it renders the tour title; loop badges in `at.markdown`; exactly the four flagship metric labels; "Camera keyframes" still reachable inside the "Dataset scale" expander; "Architecture (for technical reviewers)" expander present and containing "TRINITY"; pinned footer sentences and the flagship/retention captions still present.
- [ ] RED → implement spec §4 → GREEN.
- [ ] Commit `demo: outcome-first Overview (CTA, loop strip, four flagship metrics, scale + architecture drawers)`.

### Task 6: chrome on every page

**Files:** modify `views/failures.py`, `views/scenarios.py`, `views/active_learning.py`, `views/weak_supervision.py`, `views/chat_replay.py`; tests.

- [ ] Failing tests: each page's markdown carries the breadcrumb with the right lit stages; provenance captions present where spec §3 places them; Scenario viewer header shows `parity_short` for a dynamics preset (and not for a model preset); AL and Weak pages show a "What we learned" container with the interpolated numbers; all existing assertions unchanged.
- [ ] RED → implement → GREEN; ruff; mypy; full suite.
- [ ] Commit `demo: loop breadcrumb, provenance labels, what-we-learned callouts on every page`.

### Task 7: docs, screenshots script, walkthrough

**Files:** `docs/DEMO.md`, `README.md`, `scripts/demo_screenshots.py`.

- [ ] DEMO.md "Guided tour (Phase 9)" section + success-walk remap + phase row 9a; README wording + gallery; screenshots script `PAGES`/`NO_CLICK`.
- [ ] Commit `demo: Phase 9a docs (guided tour, success walk remap, screenshots)`.

**>>> CONSOLIDATED REVIEW GATE (Tasks 1–7): honesty of every tour sentence against the package, derived numbers, degradation paths, pinned keys/strings, AppTest coverage, docs. Fix round. <<<**

### Task 8: operational

- [ ] Browser walkthrough of the tour on the committed package (timed ≤ 3 min reading pace); screenshots re-captured incl. `demo-tour.png`; commit; `ruff check . && mypy && pytest -q`.
- [ ] Final whole-branch review → push `demo-phase-9a` → PR link (stacked on #8 until it merges).
