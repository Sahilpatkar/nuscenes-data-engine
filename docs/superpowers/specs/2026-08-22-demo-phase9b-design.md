# Demo Phase 9b — deep-page hooks, held-out weak-arm predictions, package 0.8

**Status:** approved 2026-08-22 (the second half of the Phase 9 plan the user approved in
plan mode on 2026-08-21; decisions taken there: two phases, and **variant (i)** — run the
two weak-arm checkpoints over the held-out val frames rather than stating the weak result
only as a number). Phase 9a (guided tour, outcome-first Overview, trust chrome) merged
2026-08-22 as PR #22. This phase changes the package (v0.8) and four deep pages.

## Why

The 9a tour gives the demo one guided path. The deep pages it links to still read as
feature dashboards: the Failure Explorer buries the model choice in a sidebar and shows
the detail below a grid; Scenario Search stacks six panels full-width with no motion
cue; the graph draws all 27 nodes at once; Active Learning shows every arm but never
*why* graph-aware mining beat similarity mining, and "why this frame" is a list of
labels; Weak Supervision shows retention numbers but never the three pictures that
explain them (what the VLM pseudo-labelled, what the verifier decided, what the
weak-trained detector then did on a frame it never saw). The user's brief asks for each
of these as a visual hook. Everything stays derived from the package.

## Honesty rules (verified against v0.7 and the 2026-08-22 research brief)

- **"CAN curve" must be CAN.** `scenario_events.parquet`'s per-step `speed_t_*` columns
  come from `ego_pose.speed_mps`, not the CAN bus (`demo/events.py:160-164`); only
  `accel_t_*` is CAN. v0.8 carries `can_speed_kmh` for the event frame and
  `can_speed_t_minus2 … can_speed_t_plus2` for the neighbours (same join and shift loop
  as accel; `canbus.parquet` has `can_speed_kmh` for all 34,149 keyframes). The curve
  plots CAN speed (km/h) and CAN longitudinal acceleration (m/s²) over the five
  keyframe steps; the axis is the nominal step label (keyframes are ~0.5 s apart — the
  caption says so). `speed_mps`/`speed_t_*` stay in the table for the existing readouts.
- **Bucket counts are frame × class pairs.** `eval_count_buckets` pools all ten
  `autolabel.schema.COUNT_FIELDS` (cars, trucks, buses, trailers, construction
  vehicles, motorcycles, bicycles, pedestrians, traffic cones, barriers), so `n`
  (38,042 / 8,868 / 2,494 / 456 = 49,860 = 4,986 × 10) counts pairs, not frames — the
  chart caption says "frame–class pairs" (spec amendment 2026-08-23, review: this note
  and the chart caption both said "the five classes", which is the count-vote table's
  set of *detector* classes, not the VLM's ten count fields; §5's own invariant
  `n.sum() == n_ok × len(COUNT_FIELDS)` was right all along). The MAEs are
  **recomputed at build** from `data/autolabel/labels.parquet` (`parse_status ==
  "ok"`, 4,986 of 5,000 rows) and `annotations.parquet`; the documented figures (0.08 / 0.87 / 2.81 / 6.67 in
  `docs/AUTOLABEL_EVAL.md`) are a cross-check, never a source — no 4-dp constant is
  written into code.
- **Reason chips state facts that exist.** The only per-frame selection facts are the
  `al_selection_explain` columns (night, community id/size/night members/mass rank/
  quota, pick pass, degree rank, routed failures) and the frame's visible GT. There is no
  per-frame failure rate (1,249 of 1,500 selected frames routed no failure mass); a test
  asserts no chip contains the word "rate". Chips exist only for the 78 curated frames
  with `al_selected_by == "graph_rate_night"`; the other 172 keep the absent note.
- **Weak-arm predictions are offline inference, labelled as such.** `demo infer` runs
  the training runs' own checkpoints (`mlruns/artifacts/8d953609…`,
  `…/c10a4fed…`, yolov8n @640, both on this Mac) over the 125 held-out val frames on
  CPU; the package carries them as two more models (`weak_graph_rate_night`,
  `weak_graph_rate_night_gt`). Provenance: `recorded` with the detail "demo infer, CPU,
  checkpoint of the training run". The `_gt` twin is the control: same frames, GT
  labels instead of pseudo labels. Captions compare the two detectors' boxes on the
  frame shown; the *effect* sentences still come from `active_learning_results` /
  `weak_loss_decomposition`, never from one frame.
- **Five models change derived sentences.** Re-running `demo infer` rewrites the three
  staged tables every number downstream reads; the hero story ("the shadowed car
  defeats all three models", `docs/DEMO.md:186`, `overview.HERO_CAPTION`) and the
  exemplar validation are **re-verified from the new tables**, and any sentence that
  counts models is rewritten from data or made count-free.
- **Computed, never asserted, superlatives** (9a rule): the strategy-comparison lesson
  is emitted only when the graph arm really has more scenes and a higher night share
  than the similarity arm; otherwise a neutral comparison sentence.

## 1. Failure Explorer as the hook (`views/failures.py`)

- Sidebar keeps `st.sidebar.header("Filters")`, then **primary** filters in this order:
  Lighting, Rain, Category, Size, Distance to ego (m), Failure type
  (`failure_type_select`). The **model radio leaves the sidebar**: it is instantiated in
  the detail header (`st.radio("Model", models, horizontal=True, key="failure_model")`);
  `render()` reads `st.session_state.get("failure_model", default)` before filtering so
  the grid and detail follow the selection on the rerun. `st.sidebar.expander("Advanced
  filters")` holds Curation bucket (`st.multiselect`) and Sort by (`st.selectbox`,
  moved out of the main column).
- Layout: title → breadcrumb → caption → **detail first**, then the grid. Auto-select:
  if `st.session_state["failure_token"]` is unset or not in the filtered set, the
  detail shows the first frame of the sorted filtered set (the caption says
  "auto-selected — click View on another frame"); an explicit View click wins.
- Detail: `st.columns([3, 2])` — left the overlay image with the view-mode radio
  (`st.radio("View", …, horizontal=True, key="failure_view_mode")`) directly under it;
  right the model radio, the four metric cards, per-model `n_preds`, the exemplar badge,
  the fix-pair caption. The per-box `st.dataframe` moves into `st.expander("Per-box
  detail")` below the columns.
- Pinned strings and keys survive verbatim: the `"{n} / {total} val frames match the
  current filters"` caption, `"No frames match these filters"`, the fix-pair caption,
  the exemplar badge, `provenance("recomputed", …)`, the Diagnose breadcrumb,
  `failure_model`, `failure_type_select`, `failure_select_{token}`, `failure_token`.
- With five models the radio shows `filters.model_label(...)` text per option
  (`format_func`); the raw model name stays the widget value.

## 2. Scenario Search one-screen viewer + CAN curve (`views/scenarios.py`, `tour.py`)

- `filters.FILMSTRIP_STEPS = (("t_minus2","t-2"), ("t_minus1","t-1"), (None,"current"),
  ("t_plus1","t+1"), ("t_plus2","t+2"))` becomes the single definition;
  `scenarios._BEFORE_STEPS/_CURRENT_STEP/_AFTER_STEPS` and `tour._FILMSTRIP_STEPS` are
  derived from it (the copy-contract test compares all three to `filters`).
- `filters.filmstrip_steps(row) -> list[FilmstripStep]`, `FilmstripStep(label, token,
  can_speed_kmh, accel_mps2, is_current)` — one entry per step whose token is present
  (NA neighbours at scene edges dropped; the current step always present), in time
  order. Falls back to `speed_mps` × 3.6 **only** when the package predates v0.8
  (`can_speed_*` columns absent) and then the chart title says "ego speed" — so an old
  package never mislabels the series.
- `render.line_chart(frame, *, x, y, order, y_title, selected=None, zero_line=False,
  height=160) -> alt.LayerChart`: `mark_line(point=True)` over a nominal x with explicit
  `sort=order`, a vertical `mark_rule` at `selected` (the filmstrip slider's step), an
  optional zero rule, `labelOverlap=False`. Unknown `selected` → `ValueError`.
- Viewer layout (one screen): header line (scene · `severity_caption` · `parity_short`
  · provenance) → `st.columns([3, 2])`: the frame | the two curves (CAN speed, CAN
  accel; rule at the slider step) → **one metric row** (CAN speed, min longitudinal
  accel, min pedestrian distance, pedestrians within 10 m, lighting/rain, model
  result or the not-curated note; amendment 2026-08-23, browser walk: the six
  cards render three to a row — a sixth-width st.metric clipped the speed/accel/
  model values — and the not-curated sentence stays the viewer's single caption
  beside the frame) → filmstrip + `scenario_filmstrip` slider (drives
  the curve rule too) → graph panel (§3). The existing pinned notes
  (`_GRAPH_LEGEND`, `_GRAPH_ABSENT_NOTE`, `_GRAPH_EVENT_ABSENT_NOTE`,
  `_MODEL_GT_ONLY_NOTE`), the compact parity line and `recorded experiment output`
  caption keep their text.
- Tour step 2 draws the same two curves directly under its filmstrip (no slider; rule
  at the current step).

## 3. Progressive graph path (`filters.path_steps`, `views/scenarios.py`)

- The subgraph JSON already carries `path`: a list of node-id lists — segment 0
  `[scene, sample, egopose]`, then one `[sample, object, category]` per matching
  object. `filters.path_steps(event) -> list[PathStep]`, `PathStep(label, node_ids,
  facts)`, ordered: **Scene** (facts: name, location, night/rain) → **Keyframe**
  (timestamp, night/rain) → one step per matching object **nearest first** by
  `meta.distance_to_ego_m` (label "Pedestrian · 2.1 m", facts: category, distance,
  visibility) → **Ego pose** (facts: CAN speed km/h, min longitudinal accel, hard
  braking yes/no). Labels are prefixed with their 1-based index so `select_slider`
  options are unique. The caption under the panel says the CAN reading lives on the
  EgoPose node (the export has no separate CAN node).
- Widgets: `st.select_slider("Reveal the matched path", options=labels,
  value=labels[-1], key="scenario_path_step")` and `st.toggle("Show context nodes",
  value=False, key="scenario_graph_context")`.
- Rendering: on-path nodes are always in the node set (layout stays stable across
  steps); nodes in steps ≤ the selected step are amber, later on-path nodes are hollow
  (white fill, grey border, grey label); context nodes (`on_path == False`) and their
  edges are omitted unless the toggle is on. The detail column shows the selected
  step's facts as a two-column table; clicking a node still shows its `meta` table
  (existing behaviour, takes precedence while a node is selected).
- `subgraph_narrative` and the click → meta behaviour are unchanged.

## 4. Active Learning — strategy comparison + reason chips (`views/active_learning.py`, `tour.py`)

- `filters.strategy_coverage(arms, *, strategies: Mapping[str, str]) -> pd.DataFrame`
  (`arm, strategy, n_scenes, night_share, delta_night, n_train_images`, one row per
  present arm in the mapping's order; absent arms skipped). Page constant
  `_STRATEGIES = {"random": "Random sample", "mined": "Similarity mining",
  "graph_rate_night": "Graph-aware mining"}`; design notes per strategy are a fixed
  page constant (how the strategy *works*), the numbers are the results.
- Section "Three ways to pick 1,500 frames" above "Every arm, one chart": two
  `bar_chart`s side by side (scenes covered; night share, `highlight="graph_rate_night"`)
  + a `learned(...)` sentence computed from the table: emitted as "Similarity mining
  found near-duplicates: {mined.n_scenes} scenes, {mined.night_share:.0%} night,
  {mined.delta_night:+.4f}; graph-aware mining spread the same budget over
  {graph.n_scenes} scenes at {graph.night_share:.0%} night and took the best night gain
  ({graph.delta_night:+.4f})" only when graph > mined on both scenes and night share
  **and** graph has the max `delta_night`; otherwise a neutral sentence that states the
  three (scenes, night share, Δnight) triples.
- `filters.reason_chips(row, gt_rows, *, n_communities, night_floor=None,
  quota_before=None) -> list[str]`: `night` · `{k} pedestrian GT box(es)` (visible GT,
  omitted at 0) · `community #{c} · {size} frames` · `mass rank {r} of
  {n_communities}` (omitted when the rank is NA) · `quota {before} → {after}` or
  `quota {after}` · `night-pass pick (floor {n})` / `main-pass pick` / `seeded
  backfill` · `degree rank {r} of {size}` (omitted when NA) · `{n} routed failures`
  only when `n_failures_routed > 0`. Never the word "rate".
- `chip_row(reason_chips(...))` renders above the factor lines on the AL per-frame
  panel and on tour step 3; `quota_before` comes from `community_jump` when the
  communities table has both quota columns.

## 5. Count-bucket export + package 0.8 (`demo/exporters.py`, `build.py`, `data.py`)

- `exporters.export_vlm_count_buckets(*, labels_path, processed_dir, out_dir) ->
  pd.DataFrame` reuses `data_engine.autolabel.evaluate.gt_counts` and
  `eval_count_buckets` (no re-implementation): rows with `parse_status == "ok"`, one
  group per `model`, output columns `model, bucket, n, mae` with `bucket` ordered
  `0, 1-3, 4-9, 10+`; writes `vlm_count_buckets.parquet`. Invariant: per model,
  `n.sum() == n_ok × len(COUNT_FIELDS)`.
- `run_build` wiring right after `export_weak_verifier_by_class`, from the **Phase-6b
  table only** (`data/autolabel/labels.parquet` — the first of the VLM label tables the
  build already hashes; the weak run's table is not a count-accuracy eval). Absent →
  `validation["vlm_count_buckets"] = "absent"` with a warning; present →
  `"included"` and the file hashed as an input. `_PACKAGE_VERSION = "0.8"`;
  `tests/test_demo_export.py:1417` pins `"0.8"`.
- `data.load_vlm_count_buckets()` follows the graceful-empty shape
  (`_EMPTY_VLM_COUNT_BUCKETS_COLUMNS = ["model","bucket","n","mae"]`).
  `STALE_PACKAGE_NOTE` is unchanged (it describes the Phase-7 tables, still true); the
  Weak page shows its own `_BUCKETS_ABSENT_NOTE` ("count-bucket chart needs demo_data
  >= 0.8 — rerun demo build") when the table is empty.
- Events schema (§2): `demo/events.py` reads `can_speed_kmh` with the canbus columns
  and shifts it per neighbour; `build.py` keeps `_FILMSTRIP_NEIGHBOR_COLUMNS` as is.

## 6. Weak Supervision — one frame, three views + held-out predictions (`views/weak_supervision.py`)

- **Variant (i)** `configs/demo.yaml models:` gains `weak_graph_rate_night: {run:
  8d953609142d44abb443f882e6f00735, imgsz: 640}` and `weak_graph_rate_night_gt: {run:
  c10a4fedc8c04e43a3d47fbb0bdd623c, imgsz: 640}` (both yolov8n, trained on TRINITY,
  checkpoints synced to `mlruns/`). `demo infer` re-runs over the curated set (five
  checkpoints; ~5 min each on CPU) and rewrites the staged `predictions.parquet`,
  `gt_boxes.parquet` (`matched_<model>` × 5), `frame_manifest.parquet`
  (`n_preds_<model>` × 5, `fixes_fn_vs_<a>_<b>` × 20). `filters.model_label` gains the
  two entries ("weak_graph_rate_night (pseudo labels, yolov8n)" / "…_gt (GT-labelled
  twin, yolov8n)"), and the stale "torch is not in the local venv" docstrings in
  `demo/infer.py` are corrected.
- `filters.weak_showcase_token(manifest, weak_labels, gt) -> str | None`: an
  `accepted` train-pool frame with ≥ 1 pseudo box and ≥ 1 visible pedestrian GT box;
  most pseudo boxes first, then token. `filters.weak_result_token(manifest, gt, preds,
  *, weak_arm, gt_arm) -> str | None`: a val frame, night first, where
  `fixed_boxes(gt, preds, baseline=weak_arm, arm=gt_arm)` is non-empty (the GT twin
  catches a box the weak arm misses); then most such boxes, then token. Both return
  `None` on an older package (no `matched_weak_*` columns) and the page shows a note.
- Page order: cards (unchanged) → **"One frame, three views"**: row 1 (train pool,
  `weak_showcase_token`) GT boxes | pseudo labels (`draw_overlay(pseudo_boxes=…)`) |
  verdict + count vote (`weak_frame_summary` table); row 2 (held-out val,
  `weak_result_token`) GT | weak-trained detector | GT-trained twin with `st.radio(…,
  key="ws_result_model")` over the pair and the `fixed_boxes` per-box table; provenance
  lines (recomputed for overlays; recorded for the inference) → loss decomposition
  (unchanged) → **count-bucket chart** (`bar_chart` of `mae` by `bucket`, caption with
  `n` as frame–class pairs) + `learned("Why crowded frames defeat the VLM — …")`
  interpolating the four MAEs → crowding (unchanged) → galleries (the existing model
  panel folds the weak pair behind `st.expander("Weak-arm detections on this frame")`)
  → story. Existing keys (`ws_*_show_all`, `ws_*_select_*`) and pinned notes survive.

## 7. Docs, screenshots, tests

- `docs/DEMO.md`: package layout rows (`vlm_count_buckets.parquet`, the `can_speed_*`
  columns, five-model columns), builder bullets (`export_vlm_count_buckets`, the two
  weak models in `demo infer`), page sections for the four pages ("(Phase 3, 9b)" etc.),
  success-criteria walk rows that now land on the new hooks, phase row 9b `shipped`;
  `README.md` phase line "1–9b" and gallery captions; all seven screenshots re-shot.
- Tests: pure helpers (`filmstrip_steps`, `path_steps`, `strategy_coverage`,
  `reason_chips`, `weak_showcase_token`, `weak_result_token`, `line_chart`,
  `export_vlm_count_buckets`, events `can_speed_*`); AppTests per page (Failure
  Explorer: model radio outside the sidebar, `Advanced filters` expander, auto-select,
  `failure_view_mode`; Scenario: ≥ 2 charts in the viewer, `scenario_path_step`
  options start with "1 · Scene", context toggle changes the node count; AL: third and
  fourth charts, chips never contain "rate"; Weak: three-views rows, `ws_result_model`,
  bucket chart or the absent note); copy contract extended to `FILMSTRIP_STEPS`; the
  fixture gains the two weak models' predictions/matches and a tiny
  `autolabel/labels.parquet` with `category_name`/`visibility_token` annotations.
- Batched gates: one consolidated review after Task 8, a fix round, one final
  whole-branch review; a timed browser walkthrough of the four pages and the tour.

## 8. Out of scope

Live inference, any LLM call, new presets, a CAN node in the graph export, the
Overview/tour structure (9a), chat replay changes.
