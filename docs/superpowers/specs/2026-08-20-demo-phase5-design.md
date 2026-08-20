# Demo Phase 5 — scenario search + synchronized event viewer + semantic gallery

**Status:** approved 2026-08-20 (Phase 5 of the approved demo plan; Phases 1-4 merged
as PRs #14-#17). The demo doc's §§4-6 minus the graph visualization (Phase 6): preset
driving-scenario queries return ranked events; clicking one opens the synchronized
event viewer (camera + ego dynamics + scene context + model context) with a
t−2…t+2 filmstrip.

## Why

The graph/CAN story ("hard braking near pedestrians → 30 events, identical in SQL
and Cypher") is the project's flagship, but the demo currently shows it only as an
Overview number. Phase 5 makes it explorable: pick a scenario, see the events, scrub
the moments around one.

## 1. The events artifact — `demo export events` → `scenario_events.parquet`

Computed at build time from `canbus` ⋈ `ego_pose` ⋈ `annotations_3d` ⋈ `samples`
(CAM_FRONT keyframes are 1:1 with canbus rows — verified 34,149 = 34,149):

- per-event columns: `sample_data_token, sample_token, scene_name, timestamp,
  speed_mps, accel_long_min_mps2, is_hard_braking, min_dist_pedestrian_m,
  min_dist_vehicle_m, min_dist_cyclist_m, n_peds_within_10m, is_night, is_rain,
  preset_tags (list), in_curated_set (bool)`, plus filmstrip context:
  `t_minus2..t_plus2` neighbor `sample_data_token`s (same scene, timestamp-ordered
  CAM_FRONT keyframes; NA at scene edges) and per-neighbor `speed_mps`/`accel`.
- **Two preset families, honestly separated:**
  - *dynamics presets* (dataset-wide, GT-only): hard braking near pedestrians
    (the flagship — count asserted == 30 at export); night scenes with nearby
    pedestrians; nearby cyclists during high-speed driving; rain scenes with
    nearby vulnerable road users.
  - *model-result presets* (curated-val-only — predictions exist only for the 125
    curated frames; the UI labels this scope): false-negative pedestrians at
    night; low-confidence detections during braking.
- Only events matching ≥1 preset are exported, ranked within each preset by its
  own severity key (e.g. flagship: strongest braking first), capped at 30 events
  per preset (a `preset_rank_<name>` column carries the order; the cap is logged).
- Thumbnails: the build exports LanceDB thumbs for every exported event token AND
  its filmstrip neighbors (~≤900 additional thumbs ≈ 7 MB — within budget).

## 2. The page — `app/demo/views/scenarios.py` (replaces the stub)

- Six preset buttons with one-line descriptions; selecting one shows ranked event
  cards (thumb + braking g, min pedestrian distance, speed, lighting, and — for
  model-result presets — the model verdict), with the family's scope stated
  ("all 34,149 keyframes" vs "125 curated val frames").
- The flagship preset shows the parity badge: "30 events — identical count in
  DuckDB SQL and Neo4j Cypher" (SQL computed at export and asserted; Cypher
  sourced from GRAPH.md until Phase 6, labelled as in Phase 1).
- **Event viewer** (detail state, same session-state idiom as the Failure
  Explorer): the event frame (crop through `draw_overlay` when the token is
  curated-val — GT + selected-model predictions; thumb otherwise, GT boxes only
  scaled 0.16), an ego-dynamics panel (speed, peak decel, hard-braking flag), a
  scene-context panel (min distances per class, peds within 10 m, lighting/rain),
  a model-context panel (per-model n_preds + verdicts when curated, else an
  honest "not in the curated prediction set"), and the **filmstrip**: t−2…t+2
  thumbnails via `st.select_slider`, the selected step showing its own
  speed/accel readout under the strip.
- The semantic gallery section: `demo export semsearch` runs the canned SigLIP
  queries offline (torch locally) → `semantic_search_results.parquet` (query,
  rank, token, score) + thumbs; rendered as a "Recorded semantic search" section
  on this page with the recorded-not-live banner (Phase-1 `recorded_banner`).
- Overview gains nothing; the stub table in main.py just points at the real page.

## 3. Testing (batched per the leaner Phase-5 process)

- Exporter pure functions: preset predicates each unit-tested on synthetic
  fixtures (incl. the flagship == 30 assertion path and the per-preset cap);
  neighbor resolution (scene edges → NA; ordering by timestamp); in_curated_set
  flag; determinism (two builds byte-identical).
- Page: filter/selection logic kept in pure helpers where practical; one AppTest
  smoke per state (preset list → cards render; event viewer renders for a curated
  and a non-curated event; filmstrip slider changes the shown readout) against a
  tiny fixture package.
- Build integration: events + semsearch artifacts in the manifest with
  validations (flagship count, thumbs coverage for all filmstrip tokens).
- CI stays torch-free (semsearch export is operational, like curation).
- **Process note: reviews are batched** — one consolidated spec+quality review
  after the code tasks, plus the final whole-branch review. Two review gates
  total, not two per task.

## 4. Out of scope

Interactive graph rendering (Phase 6 — the page leaves a labelled slot for it);
scenario BUILDER with arbitrary filters (deferred per the master plan; presets
only); live semantic search in the public app; any new inference or rsync
(events use existing GT/predictions; semsearch uses the local store).
