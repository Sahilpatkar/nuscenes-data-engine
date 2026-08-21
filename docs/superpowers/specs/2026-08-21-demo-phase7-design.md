# Demo Phase 7 — Active Learning + Weak Supervision pages

**Status:** approved 2026-08-21 (Phase 7 of the approved 8-phase demo plan; master
spec `docs/DEMO_PLAN.md` §8–§10; Phases 1–6 merged as PRs #14–#19, package v0.5).
Goal: replace the two remaining story stubs with real pages that answer success
criteria 6–10 of `docs/DEMO_PLAN.md` — *how active learning chooses frames, whether
retraining helped, how well VLM supervision worked and why it lost to GT, and how the
loop closes* — every number derived at export time, never hand-typed.

## Why

The package already carries every arm's mAP, both weak-retention pairs, 30 curated
`al_selected` and 40 `weak_accepted`/`weak_rejected` train-pool frames with crops and
GT, and 53 val frames where `graph_rate_night` catches a box `baseline` misses. What it
lacks is (a) the per-frame selection facts the AL run never persisted (community,
routed failure mass, pick pass), (b) the rejected-side crowding number (documented
7.61 GT boxes/frame) and the loss decomposition (documented ~50% / ~32% / 18%), and
(c) the VLM pseudo boxes and count votes for the curated weak frames. Decision
(user, 2026-08-21): re-derive (a) at export against the local Neo4j/LanceDB and ship
it only if it reproduces the experiment exactly; compute (b) and (c) with pure
exporters over local artifacts.

## 1. `demo al-explain` — re-derive the `graph_rate_night` selection, validated

New `demo` sub-command (`src/nuscenes_data_engine/demo/al_explain.py` + `cli.py`),
optional like `demo subgraphs`: needs the Neo4j container (GDS 2.13.11 — the build
the AL code was written against) and the local LanceDB. It stages
`data/demo_curation/al_explain/` and is consumed by `demo build` as a declared-
optional input group (absent / partial-error / included, exactly the
`_include_subgraphs` pattern).

Inputs and parameters come from the artifacts, never re-typed:
`configs/active_learning.yaml` (`n_mine`, `route_k`, `sweep.top_k_failures`, the
arm's `quotas.is_night` night floor, `seed`, `split.channel`, engine config for the
LanceDB path), `data/active_learning/{split,failures}.parquet`,
`data/processed/samples.parquet`.

Procedure (mirrors `active_learning/graph_mining.py::run_graph_mining` for the
`graph_rate_night` arm, reusing its functions — no AL code is modified):

1. `_louvain_communities(driver, sorted(pool_frames), database)` — GDS Louvain with
   `concurrency: 1` over the `SIMILAR_TO` projection, with the same transient
   `_al_pool` marking and cleanup the run used (no persistent graph change).
2. Failure-mass routing through a demo-side `route_failure_mass_per_frame(...)` that
   performs the same neighbour queries as `route_failure_mass` but also keeps
   per-frame accumulators (`n_failures_routed`, `mass_routed` per pool frame). The
   export asserts its per-community totals equal `route_failure_mass(...)`'s output
   (|Δ| < 1e-9) — the original function stays the oracle.
3. `select_by_mass(...)` with the arm's night floor; pick-pass attribution by
   re-applying the same two passes with the real `allocate_by_mass`: a selected
   token is `night` if it is among the top-`night_quota[c]` night members of its
   community by degree, else `main` if among the main-pass take, else `backfill`;
   the union must equal the selected set.

**Validation — the export fails loudly (nothing staged) unless:** the re-derived
selected set equals `data/active_learning/graph_rate_night.parquet` (all 1500
tokens, exact set equality) AND the re-derived diagnostics equal
`communities_graph_rate_night.json` row-for-row (`community`, `size`,
`night_members`, `quota` exactly; `mass` to 4 dp). Reproduction is the only evidence
that the per-frame facts are the experiment's, so there is no "approximately
matches" mode.

Staged outputs:

- `al_selection_explain.parquet` — one row per selected frame (1500):
  `sample_data_token, arm, is_night, scene_name, community, community_size,
  community_night_members, community_mass, community_mass_rank` (1 = heaviest),
  `community_quota` (this arm), `degree, degree_rank_in_community` (1 = highest),
  `pick_pass ∈ {night, main, backfill}, n_failures_routed, mass_routed`.
- `al_explain_validation.json` — `{selected_match, n_selected, communities_match,
  n_communities, mass_total, gds_version, config: {n_mine, route_k, top_k,
  night_floor, seed, channel}}`.

`demo build` copies both into `demo_data/`, hashes them as inputs and outputs, and
records `validation.al_explain ∈ {"absent", "included"}`. Absent is the ordinary
state on a fresh clone; the page degrades honestly (§3d).

## 2. Pure exporters — run inside `demo build`, local parquets only

All extend `src/nuscenes_data_engine/demo/exporters.py` (and `build.py` wiring);
inputs are hashed in the manifest; sorted, deterministic writes. `_PACKAGE_VERSION
= "0.6"`.

**`active_learning_results.parquet`** (13 rows) gains: `round_order` (insertion
index in `results.json` — the export currently loses it), `family` (`baseline`,
`random`, `mined`, `rate`, `strat`, `rate_strat`, `graph`, `graph_rate`,
`graph_rate_night`, `weak`), `overall_map50`, `night_map50`, `night_precision`,
`night_recall`, `night_ped_map5095` (night `per_class.pedestrian` — the mechanism:
baseline 0.0826 → `graph_rate_night` 0.1171, `weak_graph_rate_night` 0.0203),
`day_map5095`, `rain_map5095`, `clear_map5095` (from `slices`), `n_boxes`
(nullable; weak arms only), and mined-set composition `n_scenes`, `night_share`,
`rain_share` via `active_learning/report.py::arm_composition` (nullable where an
arm has no token list). Existing columns and their values are unchanged.

**`weak_supervision_results.parquet`** (2 rows) gains: `n_rejected`, `tolerance`,
`conf`, `n_unparsed`, `gt_boxes_per_candidate_frame`, and
**`gt_boxes_per_rejected_frame`** computed with the verified recipe: read
`data/processed/annotations.parquet` `[sample_data_token, category_group]`, keep
`category_group.notna()` (the five detector classes), count per token, reindex over
the arm's candidate tokens (`<arm>.parquet`, missing → 0), split by membership in
`<arm>_accepted.parquet`. This reproduces the documented 3.87 / 7.61 (`random`) and
3.41 / 6.68 (`graph_rate_night`); the stale docstring recipe (`samples.n_boxes`,
which gives 9.94) is corrected in place. A test pins the accepted-side mean against
the summary's `mean_gt_boxes_per_accepted_frame` so the two recipes cannot drift.

**`weak_loss_decomposition.parquet`** — one row per base arm that has both
`weak_<base>` and `weak_<base>_gt` in `results.json` (today: `random`,
`graph_rate_night`), from `overall.mAP50-95`: `gt_gain` (base − baseline),
`weak_gt_gain`, `weak_gain`, `retention = weak_gain / gt_gain`,
`dropped_frame_cost = gt_gain − weak_gt_gain`, `dropped_frame_share`,
`label_cost = weak_gt_gain − weak_gain`, `label_share`, plus `headline` (True for
the documented `random` pair only — the attribution rule from `export_overview`).
Reproduces 0.0169 (49.7%) / 0.0109 (32.1%) / 18.2% and 0.0107 (42.1%) / 0.0047
(18.5%) / 39.4%; `retention` must equal `overview_metrics.json`'s `by_base_arm` to
4 dp (asserted at build).

**`weak_verifier_by_class.parquet`** — long table from the two
`*_pseudo_summary.json`: `arm, category_group, n_rejected_disagreements,
n_accepted_mutual_zero, mutual_zero_share` (= n / `n_accepted`; pedestrian 870/1101
= 79.0% for `graph_rate_night`).

**`weak_labels.parquet`** — the weak arm's (`curation.weak_arm`) verified pseudo
boxes for curated tokens only: `sample_data_token, category_group, x_min, y_min,
x_max, y_max, score`; native 1600×900, `draw_overlay`-ready. Accepted frames with
no rows are the "mutual zero" case and are shown as such.

**`vlm_counts.parquet`** — one row per curated `weak_accepted`/`weak_rejected`
token from `data/active_learning/autolabel_weak/labels.parquet`: `sample_data_token,
parse_status, label_confidence, vlm_time_of_day, vlm_weather, vlm_car, vlm_truck,
vlm_bus, vlm_pedestrian, vlm_bicycle` (from `cars/trucks/buses/pedestrians/
bicycles`) and `gt_car … gt_bicycle` (from `annotations.parquet` `category_group`).
If `labels.parquet` carries more than one row for a token, keep the first
`parse_status == "ok"` row in file order and assert uniqueness afterwards.

**`frame_manifest.parquet`** gains `weak_verdict` (`accepted` / `rejected` for
tokens in the weak arm's candidate pool, else NA) and `al_selected_by` (the AL arm
name for tokens in `curation.al_arm`'s selected parquet, else NA) — derived from the
local arm parquets at build, not hand-tagged.

**`al_communities.parquet`** — 97 rows from the two persisted diagnostics files
(`communities_graph_rate.json`, `communities_graph_rate_night.json`): `community,
size, night_members, mass, quota_graph_rate, quota_graph_rate_night`. Build asserts
the two files agree on `size`/`night_members`/`mass` and that each quota column sums
to `n_mine`. No Neo4j needed — the community chart works even when `al-explain` is
absent.

**`al_exemplars.json`** — `{"arm": "graph_rate_night", "baseline": "baseline",
"tokens": [...]}` from `configs/demo.yaml` `al.exemplar_tokens` (4–6 hand-approved
tokens). Build fails unless every token is a `val` row of `frame_manifest.parquet`
with `fixes_fn_vs_<baseline>_<arm> == True` and has predictions for all three models;
a null/empty list is allowed only when curation is absent (same rule as
`hero.token`). The tokens are hand-picked; everything said about them on the page is
generated from `gt_boxes` + `predictions` (no hand-written captions — the Phase-3
hero-caption lesson).

## 3. Active Learning page — `app/demo/views/active_learning.py`

Registered in `main.py` with `url_path="active_learning"`; `stubs.py` keeps only
`chat_replay`. Sections, top to bottom:

**(a) Story view.** `render.story_arrows` (written in Phase 1, unused until now) with
the doc's six beats — Problem (baseline night mAP50-95 vs overall, from the table) →
Hypothesis → Acquisition (`graph_rate_night`: mass-weighted communities + night
floor of 375, from `al_explain_validation.json`/config when present, else the
documented mechanism in words) → Training (n_train_images 7,035 → 8,535) →
Evaluation (same 6,019-image val split) → Result (`+0.0101` night, derived).

**(b) Arm chart.** Altair bars of `delta_night` and `delta_overall` for all 13 arms
in `round_order`, `graph_rate_night` in the accent colour, weak arms greyed, zero
line drawn; an expander table adds `night_ped_map5095`, `night_share`, `n_scenes`.
Honest naming: the table's best *overall* arm is `graph` (+0.0344); the yolov8m
checkpoint the demo calls "champion" is a different, model-size champion and is
labelled "champion (yolov8m @960)" wherever it appears.

**(c) How `graph_rate_night` chooses.** From `al_communities.parquet`: a chart of
community mass vs quota (points coloured by night share), and the quota under
`graph_rate` vs `graph_rate_night` per community; the all-night community's jump
(83 → 323, ≈3.9×) is computed from the table and stated in a caption.

**(d) Why was this frame selected?** Gallery (thumbs) of the curated `al_selected`
frames → click → crop with GT (`draw_overlay`, mode `gt`; train-pool frames carry no
predictions and the caption says so) and a factor panel from
`al_selection_explain.parquet`: night ✓/✗; community (id, size, night members, mass
rank of 97, quota); picked in the night pass or the main pass; degree rank ("3rd of
916 by similarity degree"); failures routed to this frame (count, mass). When the
explain group is absent: the same gallery with night/scene/GT-count facts and an
explicit note "per-frame community and routed mass are not included in this package
(`demo al-explain`)" — never a generic "high failure rate ✓".

**(e) Before / after.** Selectbox over `al_exemplars.json` tokens (thumb + scene +
night), radio `baseline` / `graph_rate_night` / `champion (yolov8m @960)` over the
Phase-3 overlay (`draw_overlay`, mode `overlay`, 0.6 crop), and a per-box table
built by a pure `filters.fixed_boxes(gt, preds, *, baseline, arm)` helper: every
visible GT box where the arm's claim beats the baseline's — baseline `none` (no
prediction ≥ floor) or `low-conf 0.22` → arm `0.61 (tp)` / `low-conf 0.41` — with
class and distance. Below it the arm-level line: night mAP50-95 0.1667 → 0.1768 on
the held-out val split. The doc's `Failure → Selected → Added → Retrained →
Improved` arrows close the section.

## 4. Weak Supervision page — `app/demo/views/weak_supervision.py`

Registered with `url_path="weak_supervision"`. Sections:

**(a) Headline cards.** Share of GT gain retained: **18%** (`random` pair, the
documented headline, `headline == True`) and **39.4%** (`graph_rate_night` pair —
fulfilling Overview's live promise, whose sentence becomes present tense); verifier
retention 64% / 73%; night pedestrian mAP50-95 `graph_rate_night` 0.117 →
`weak_graph_rate_night` 0.020 with "worst night result of the 13 arms" derived
from the table (min `delta_night`).

**(b) Loss decomposition.** Stacked altair bars per base arm: retained /
dropped-frame cost / label cost, labelled with shares (≈50% / ≈32% / 18% for the
headline pair; 42% / 19% / 39% for the other).

**(c) Crowded-frame bias.** Accepted vs rejected GT boxes/frame per arm (3.87 vs
7.61; 3.41 vs 6.68) as paired bars, plus by-class rejected disagreements and
mutual-zero shares from `weak_verifier_by_class.parquet` (pedestrian 79%).

**(d) Visual comparison.** Tabs *accepted* / *rejected* over the 40 curated weak
frames (`weak_verdict`): crop with GT (green) and VLM pseudo boxes in a new style
(`STYLE_VLM`: blue solid, label `VLM 0.73`) via a `draw_overlay` `vlm_boxes=`
extension that keeps the existing signature backward-compatible; verdict badge
(accepted-with-zero-boxes flagged "accepted — 0 pseudo boxes (mutual zero)";
rejected frames have no boxes by construction and the caption says the verifier
compared the VLM's counts to the *detector's*, which are not in the package); a VLM
count vs GT count table per class from `vlm_counts.parquet`; and the downstream
result stated at arm level (`weak_graph_rate_night` Δnight −0.0262) — train-pool
frames carry no predictions and the page says so.

**(e) Story arrows** for the weak-sup experiment: Hypothesis → Labelling (Qwen2.5-VL
counts, conf ≥ 0.5, tolerance 1 — from the summaries) → Verification (retention) →
Training → Result (18% of the GT gain; loss split).

## 5. Charts, app contract, docs

Charts use **altair** through `st.altair_chart` — already a hard dependency of
Streamlit (no new wheel), now declared in `app/demo/requirements.txt` and allowed
(and asserted present) by `test_demo_requirements_stay_minimal`; the AST
no-backend-import guard is unchanged. A small `render.bar_chart(df, *, x, y,
highlight=...)`-style helper keeps the two pages' charts consistent. `data.py` gains
loaders for the new tables with the graceful-absence pattern (`load_semsearch`).

`docs/DEMO.md`: Phase-7 section (runbook `docker compose up -d neo4j` → `demo
al-explain` → `demo build` v0.6; what reproduction means; the corrected rejected-side
recipe), package-layout rows, validation bullets (exemplar tokens, al-explain
reproduction, decomposition-vs-overview agreement), phase table (7 shipped).
`docs/ACTIVE_LEARNING.md`: one line noting the demo's reproduction check of the
`graph_rate_night` selection. `exporters.py`'s stale Phase-7 TODO is removed.

## 6. Testing

- Exporters on tmp fixtures: AL table columns + `round_order`/`family`; loss
  decomposition reproduces the documented shares from a miniature `results.json`;
  rejected-side recipe (detector classes only — a fixture with non-detector
  annotations proves `samples.n_boxes` would be wrong); by-class table; `weak_labels`
  / `vlm_counts` filtering + uniqueness; `al_communities` agreement + quota-sum
  assertions; exemplar validation failures (non-val, flag False, missing
  predictions); manifest columns; determinism with all new groups staged; size
  budget.
- `al_explain` pure parts: per-frame routing totals == `route_failure_mass` on a
  synthetic table; pick-pass attribution covers the selected set exactly; the
  validation comparator rejects a one-token difference. One live
  `@pytest.mark.integration` test runs the full reproduction against the local
  Neo4j/LanceDB and self-skips without them.
- AppTest: both pages render with the explain group present and absent; exemplar
  selectbox + radio; per-box table rows; weak tabs; the 39.4% card; no
  `at.exception`. CI stays network/torch-free.
- Batched review gates as in Phase 6 (one consolidated review after the code tasks,
  one final whole-branch review), then a real-browser check of both pages.

## 7. Out of scope

Chat replay gallery, deployment, README link (Phase 8); per-frame explanations for
the non-graph arms (their runs persisted only `_distance`/`cluster`); re-running VLM
labelling or the detector over the pool; timeline scrubbing; any modification of
`src/nuscenes_data_engine/active_learning/` beyond imports; Cytoscape/animation.
