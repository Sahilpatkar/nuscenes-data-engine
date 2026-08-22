# Demo Phase 9a — guided tour, outcome-first Overview, trust chrome

**Status:** approved 2026-08-22 (plan-mode session; the user's 15-item UX brief,
split into 9a/9b with the user). Phases 1–8 shipped (package v0.7). Goal: make the
demo read as **one guided perception-data workflow** rather than a dashboard of
features — a 2–3 minute default path, an outcome-first landing page, and the trust
chrome (loop breadcrumb, provenance labels, plain-English lessons) on every page.
App-only: no exporter or package change.

## Why

A recruiter lands on a six-page app and has to discover its structure. The brief asks
for: a prominent "Explore a model failure →" path (night weakness → missed pedestrian
→ scenario mining → selected frames → retraining → improvement) ending in a result
screen; an Overview that shows the loop first and only 3–4 flagship metrics; a
persistent "Diagnose → Mine → Train → Evaluate" breadcrumb; provenance labels
(recomputed / recorded experiment / reproduced selection); a "What did we learn?"
sentence per experiment; the SQL ↔ Graph parity as a small trust indicator; and one
collapsible architecture drawer instead of technology in the navigation. Phase 9b
takes the deep-page visual hooks (Failure Explorer, one-screen Scenario Search with
the CAN curve and progressive graph path, acquisition-strategy chart + reason chips,
Weak Supervision three views + crowding trend, held-out weak-arm predictions).

## Honesty rules (verified against v0.7)

- The hero `5994f34b836043b3b5be191bceba2e3e` is the only night val frame where a
  pedestrian goes miss → hit, and the hit is a **low-confidence claim (conf 0.135)**
  that the matching rule counts as a hit — the tour says so and never calls it an
  exemplar (`fixed_boxes` returns no row for it).
- No per-frame "failure rate" exists for pool frames (1,249 of 1,500 selected frames
  received no routed failure mass); the tour's selection facts are `selection_factors`
  only (night, community size/night members/mass rank/quota, pick pass, degree rank).
- Every number is read from the package; "what we learned" sentences are prose that
  interpolates table values and is skipped when a named arm is absent.

## 1. Navigation and the tour page

- New `app/demo/nav.py` (streamlit-only): `register(pages)`, `page(url_path)` →
  `StreamlitPage` (keyed by `url_path`, `""` → `overview`; `ValueError` on unknown).
- `main.py`: sectioned `st.navigation({"Start here": [overview, tour], "Explore the
  loop": [failures, scenarios, active_learning, weak_supervision], "Ask":
  [chat_replay]})`; every existing `url_path` unchanged; the tour is
  `views/tour.py` with `url_path="tour"` (AppTest opens it via
  `switch_page("views/tour.py")` — an in-script `st.switch_page` is not sticky across
  `at.run()`).
- `views/tour.py`: `_STEPS: tuple[_Step, ...]` (`key, title, stage, render`), a
  `_TourData` dataclass loaded once per run, `st.session_state["tour_step"]` clamped
  to range, top nav row `st.button("← Back", key="tour_back")` /
  `st.button("Next →", key="tour_next", type="primary")` that mutate `tour_step`
  **before** the step renders (no reliance on `st.rerun()`), `st.caption(f"Step {i+1}
  of 7 · {title}")`, `loop_breadcrumb(step.stage)`, the step inside
  `st.container(border=True)` (one visual + ≤ 3 sentences + one provenance line), a
  "Go deeper" `st.page_link` row, `st.button("Restart", key="tour_restart")` on the
  last step.

Steps (stage → content; every value from the package):

| # | Title | Content |
|---|---|---|
| 0 | The weakness: night (Diagnose) | baseline overall / night / night-pedestrian mAP50-95 from `active_learning_results`; "N of M night val frames carry a baseline miss" via `failure_flags` |
| 1 | One missed pedestrian (Diagnose) | hero crop, `st.radio("Model", [baseline, arm], key="tour_hero_model")`, `draw_overlay` with `gt_for_render`; `HERO_CAPTION`; fact line "baseline: no claim · graph_rate_night: 0.135 (below the 0.40 hit floor — a low-confidence claim)" |
| 2 | Find more like it (Mine) | flagship preset top event (`rank_events`), GT overlay (train-pool — said), `severity_caption`, `n_peds_within_10m` / `min_dist_pedestrian_m`, filmstrip thumbs, `parity_short` from the preset's subgraph payload, "k of n at night" |
| 3 | Why this frame was picked (Mine) | first `tour_frame_candidates` token with a crop (night first, then pedestrian GT count), GT overlay, `selection_factors` lines, conditional sentences only when true (it is itself a flagship event; the weak verifier later rejected it), `provenance("reproduced")`; explain group absent → the absent note |
| 4 | Retrain on what was found (Train) | cards: 1,500 frames (`n_train_images` delta), `n_scenes`, `night_share`, 7,035 → 8,535; compact `bar_chart` of `delta_night` for all arms, arm highlighted |
| 5 | Same kind of frame, after (Evaluate) | first exemplar token, two-state `st.radio(key="tour_exemplar_model")`, overlay, `fixed_boxes` table, night mAP sentence, the hero honesty line |
| 6 | What we found, added, gained — and what failed (Evaluate, all lit) | four bordered answers (§2), deep links to all six pages, Restart |

## 2. Result screen (step 6) — derived sources

1. **What weakness did we find?** baseline night 0.1667 vs overall 0.2477, night
   pedestrian 0.0826; "N of M night val frames carry a baseline miss" (`failure_flags`).
2. **What data did we add?** 1,500 frames, 368 scenes, 30.9 % night (`n_train_images`,
   `n_scenes`, `night_share` of the arm) vs random 501 / 12.7 % and similarity 219 / 0 %.
3. **Did the model improve?** night +0.0101 (0.1667 → 0.1768), night pedestrian 0.0826 →
   0.1171, overall +0.0254; "best overall arm is `graph` (+0.0344) — the night arm
   trades overall for night" (`delta_overall.idxmax()`).
4. **What failed along the way?** weak labels kept 18.2 % of the GT gain
   (`weak_loss_decomposition` headline row; 49.7 % dropped frames / 32.1 % label
   noise), the verifier kept sparse frames (3.87 vs 7.61 GT boxes/frame),
   `weak_graph_rate_night` is the worst night arm (−0.0262), the hero recovery is a
   conf-0.135 claim.

## 3. Trust chrome (`render.py`, `filters.py`)

- `LOOP_STAGES = ("Diagnose", "Mine", "Train", "Evaluate")`; pure
  `loop_breadcrumb_text(active: Sequence[str] | None) -> str`
  (`:orange-badge[Diagnose] → :gray-badge[Mine] → …`; `None` = nothing lit; unknown
  stage → `ValueError`) + `loop_breadcrumb(active, *, caption=None)`.
- `PROVENANCE = {"recomputed": (icon, "recomputed in this app from the package
  tables"), "recorded": (icon, "recorded experiment output — shipped as the offline
  pipeline produced it"), "reproduced": (icon, "reproduced selection — re-derived by
  demo al-explain and validated equal to the run")}`; pure `provenance_text(kind,
  detail="")` + `provenance(kind, detail="")` (`st.caption`).
- `chip_row(chips, *, color="blue")` (one markdown line of `:{color}-badge[…]`;
  no-op on empty) and `learned(text)` (bordered container, ":material/school: **What
  we learned** — …"; deliberately not `st.info`).
- `filters.parity_short(sql_count, cypher_count, parity, *, noun="events")` →
  "30 events found · SQL 30 / Graph 30 ✓" / "… ✗ mismatch recorded" / "Graph n/a
  (GT-only preset)"; singular noun at 1.
- `filters.tour_frame_candidates(manifest, explain, gt, *, arm) -> list[str]`.

Page → stages: Overview `None` (whole loop, caption "Detect weakness → Find useful
data → Retrain → Measure impact"); tour = the step's stage; Failure Explorer
`Diagnose`; Scenario Search `Mine`; Active Learning `Mine, Train, Evaluate`; Weak
Supervision `Train, Evaluate`; Ask the Dataset `Diagnose`.

Provenance placement: Overview cards recorded, hero recomputed; Failure detail
recomputed; Scenario parity recorded, overlay recomputed (semantic gallery keeps its
`recorded_banner`); AL chart/story recorded, why-selected reproduced, before/after
recomputed; Weak cards/loss/crowding recorded, frame panels recomputed; Ask header
recorded.

"What we learned" callouts in 9a: Active Learning — the weak-arm night inversion
sentence (the similarity-mining sentence ships with 9b's strategy chart); Weak
Supervision — the loss-split sentence. Scenario viewer header gains `parity_short`
(the preset header's `parity_caption` stays — test-pinned).

## 4. Overview, outcome-first

Order: title + mission → CTA `st.button("Explore a model failure →",
key="overview_start_tour", type="primary")` (sets `tour_step = 0`,
`st.switch_page(nav.page("tour"))`) + one-line promise of the 2–3 minute path →
loop strip (`loop_breadcrumb(None, caption=…)` + four one-number beats: night 0.1667
vs overall 0.2477 · 1,500 frames / 368 scenes · 7,035 → 8,535 images · night
+0.0101) → four flagship metrics (Best night gain; Night pedestrian mAP 0.083 →
0.117; Weak-sup share of GT gain 18 %; Graph = SQL flagship 30 = 30) → "The model at
work" hero unchanged (`HERO_CAPTION` public, text unchanged) → `st.expander("Dataset
scale")` with the four scale cards → `st.expander("Architecture (for technical
reviewers)")` restating `docs/PROJECT.md` §2 (two machines, component map, CI —
no new claims) and carrying the existing flagship/retention captions verbatim →
footer captions unchanged (test-pinned).

## 5. Docs, screenshots, tests

- `docs/DEMO.md`: "Guided tour (Phase 9)" section, success-criteria walk remapped
  (Q1 → loop strip + step 0; Q2 → step 1 + Failure detail; Q3 → step 2; Q6 → step 3;
  Q7 → steps 4–5; Q10 → step 6 + the breadcrumb), phase row 9a; `README.md` ("a
  guided tour plus six pages", gallery gains `docs/img/demo-tour.png`);
  `scripts/demo_screenshots.py` `PAGES` += `("tour", "demo-tour")`, `NO_CLICK` += `"tour"`.
- Tests: pure helpers (`loop_breadcrumb_text`, `provenance_text`, `chip_row`,
  `parity_short`, `tour_frame_candidates`); AppTests — `test_tour_walks_every_step`
  (seven steps, Back/Next disabled at the ends, one pinned substring per step, ≥ 6
  page links on the result screen), `test_tour_degrades_without_optional_groups`,
  Overview outcome-first + CTA → tour, breadcrumb badges on every page. All existing
  widget keys survive; new keys are additive. `test_demo_no_longer_promises_phase_8`
  forbids the literal "Phase 8" in `app/demo/*.py` — comments say "Phase 9".
- Batched gates: one consolidated review after Task 7 (code + docs), a fix round, one
  final whole-branch review; a timed browser walkthrough of the tour.

## 6. Out of scope (Phase 9b)

Failure Explorer restructure (advanced-filters fold, model toggle beside the image,
image-first layout), Scenario one-screen layout + CAN curve + progressive graph path,
acquisition-strategy chart + reason chips, Weak Supervision three views + crowding
trend (`vlm_count_buckets` export) + held-out weak-arm predictions, package 0.8.
