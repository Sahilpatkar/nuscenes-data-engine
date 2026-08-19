# Demo Phase 3 — Failure Explorer + overlay renderer + real hero

**Status:** approved 2026-08-18 (Phase 3 of the approved demo plan; Phases 1-2 merged
as PRs #14/#15). Goal: the demo's first "wow" page — see exactly where and why the
detector fails, with GT/prediction overlays whose visual language every later page
reuses.

## Why

The package now carries everything the page needs: 250 curated frames with crops,
1,595 GT boxes with per-model matched flags, and 3,758 predictions speaking
`failures.parquet`'s exact TP/FP/FN language. Phase 3 renders it.

## 1. Package enrichment (small builder change, rebuild v0.2 → v0.3)

- `gt_boxes.parquet` gains `distance_to_ego_m` (joined from
  `annotations_3d.parquet` on `annotation_token` — probed 100% joinable) and
  `size_bucket` (small/medium/large from bbox area at documented thresholds:
  <32² px small, <96² px medium, else large — the COCO convention).
- The hero becomes a real overlay subject: `configs/demo.yaml` `hero:` changes to
  `{token: <hand-picked exemplar sample_data_token>}` — chosen from the 53
  `fixes_fn_vs_baseline_graph_rate_night` frames; `demo build` copies that frame's
  crop to `sample_frames/hero.jpg` and records `hero_token` in
  `overview_metrics.json` so the app can render its boxes live. The mosaic path
  and `models.<hero.run>` indirection are removed (dated amendment supersedes the
  Phase-1 spec's mosaic hero).
- Everything else in the builder is untouched; manifest/validation as before.

## 2. The overlay renderer — `app/demo/render.py`

`draw_overlay(image, gt_boxes, predictions, *, mode, scale) -> PIL.Image` — pure
PIL, no matplotlib. Fixed visual language (module constants, reused by Phases 5-8):

| element | style |
|---|---|
| GT (matched) | green solid, 2px |
| GT unmatched by selected model (= FN) | orange dashed, 3px |
| prediction tp | white solid thin + conf label |
| prediction fp | red solid, 2px + conf label |
| prediction low_conf | yellow dotted, 2px + conf label |

`mode ∈ {"gt", "pred", "overlay"}`: gt-only, predictions-only, or both. `scale`
maps 1600×900 coords onto the given image (0.6 for crops, 0.16 for thumbs).
Labels: category + conf at 2dp, drawn with PIL's default font (no font files).
Dashed/dotted lines implemented via segment drawing (PIL has no native dash) — a
small pure helper, unit-tested by pixel-sampling a rendered synthetic image.

## 3. The page — `app/demo/views/failures.py`

- **Val frames only** (125): train_pool frames carry no predictions by design; a
  caption states this and points at the later pages that use them.
- Filters (sidebar, all pandas predicates over the three-table join):
  lighting (day/night), rain, model (baseline default / graph_rate_night /
  champion), object class, size bucket, distance-to-ego range slider, failure
  type (has FN / has FP / has low-conf / clean for the selected model), curation
  bucket.
- Results: sortable frame grid (thumbs + failure-count captions) → click →
  detail view: the crop rendered through `draw_overlay` with a GT/Pred/Overlay
  `st.radio` toggle, plus a metadata panel (scene, lighting, rain, bucket labels,
  per-model n_preds, per-box table with category/conf/status/distance) and an
  exemplar badge when `fixes_fn_vs_<selected>_<other>` is true.
- Selection state via `st.session_state` (the Phase-1 idiom); loaders in
  `data.py` gain `load_frame_manifest()`, `load_gt_boxes()`, `load_predictions()`
  (st.cache_data, same pattern as existing loaders).
- FN definition shown in the UI matches the pipeline: GT unmatched by the
  selected model (low-conf claims count as matches — a caption says so).

## 4. Overview hero swap — `app/demo/views/overview.py`

The hero image renders live through `draw_overlay` (overlay mode, baseline model)
using `hero_token`'s boxes, captioned "baseline yolov8n: the orange dashed box is
a miss the night-targeted retrain catches — explore more in the Failure Explorer".
Falls back to the plain crop if the token's boxes are absent.

## 5. Testing

- Renderer: pure-function tests on synthetic images — box pixels present at
  expected scaled coords for each style/mode, dashed helper produces gaps,
  no-boxes case returns the image unchanged (pixel-compare).
- Page logic: extract the filter function as pure
  (`filter_frames(manifest, gt, preds, **criteria) -> DataFrame`) and unit-test
  every criterion incl. the per-model failure-type predicates; AppTest smoke:
  page renders against a tiny package (DEMO_DATA_DIR override, Phase-1 pattern)
  without exception, and the detail view renders one overlay.
- Builder: distance/size enrichment tested on fixtures (join correctness, bucket
  boundaries at exactly 32²/96²); hero-token flow (config → crop copy →
  overview_metrics.hero_token) tested; v0.3 rebuild against real data must keep
  all Phase-1/2 outputs byte-identical EXCEPT gt_boxes (+2 columns),
  overview_metrics (+hero_token), hero.jpg (now a crop) — verified in the
  operational task.
- AST/no-backend-import test already covers new view files automatically; mypy
  covers app/demo.

## 6. Out of scope

Scenario search/event viewer (5); graph viz (6); AL/weak-sup pages (7); chat
(4/8); any new curation or inference; publishing (the licensing notice shipped in
Phase 2).
