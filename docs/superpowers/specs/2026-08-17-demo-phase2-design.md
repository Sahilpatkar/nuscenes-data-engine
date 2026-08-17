# Demo Phase 2 — curated frames, TRINITY rsync, local inference, predictions package

**Status:** approved 2026-08-17 (Phase 2 of the approved demo plan; master spec
`docs/DEMO_PLAN.md`; Phase-1 foundation merged as PR #14). Goal: close the one real
artifact gap — per-frame model predictions — so Phase 3's Failure Explorer and
Phase 7's before/after comparison have data.

## Why

No per-frame prediction dump exists anywhere in the repo (the AL sweep persisted
only per-frame *counts*). Producing one needs full-res images (TRINITY-only) plus
the checkpoints (already local in `mlruns/`). Decision locked in the master plan:
rsync ~250 curated frames down once, infer locally on CPU, write parquet + crops.

## 1. Curation — `demo curate`

`src/nuscenes_data_engine/demo/curate.py`, seeded and deterministic (same config →
same token list, asserted by test). Buckets with quotas in `configs/demo.yaml`
(defaults; sum ≈ 250):

| bucket | source | quota |
|---|---|---|
| `night_failure` | `failures.parquet` top `failure_score`, `is_night` | 60 |
| `day_failure` | `failures.parquet` top `failure_score`, not night | 40 |
| `hard_braking` | `canbus.is_hard_braking` ⋈ samples (keyframes with CAM_FRONT) | 40 |
| `al_selected` | `graph_rate_night` arm's mined tokens ∩ frames with images | 30 |
| `weak_accepted` / `weak_rejected` | night-champion accepted parquet / candidates − accepted | 20 + 20 |
| `clean_success` | `failures.parquet` lowest `failure_score`, `n_gt ≥ 3` | 20 |
| `semantic` | canned SigLIP queries (fog/glare, crowded night crossing) via local LanceDB | 20 |

A token may satisfy several buckets; `frame_manifest.parquet` carries
`curation_buckets` as a list and tokens are deduplicated. **Two populations are
kept distinct and labelled** (`split` column): `val` frames (from `failures.parquet`
— models never trained on them; the only legitimate basis for before/after claims)
and `train_pool` frames (AL/weak-sup buckets — used for selection-explanation and
weak-label visuals, never for accuracy comparison). Output:
`data/demo_curation/frame_manifest.parquet` + `rsync_filelist.txt` (image paths
relative to the nuScenes root, from `samples.parquet.filename`).

## 2. Transfer — documented rsync (one-time per curation change)

```bash
rsync -av --files-from=data/demo_curation/rsync_filelist.txt \
  trinity:/data/ggare/datasets/nuscenes/ data/raw/demo_frames/
```

`data/raw/` is gitignored staging; nuScenes data on TRINITY is read-only (CLAUDE.md)
— rsync only reads it. ~250 CAM_FRONT JPEGs ≈ 40 MB. Fallback if TRINITY is
unreachable: Phases 3+ degrade to GT-only overlays on LanceDB thumbnails; the spec's
outputs below are then absent and `demo build` reports which optional inputs are
missing rather than failing (curation artifacts are a *declared-optional* input
group until present, then hashed like everything else).

## 3. Inference + matching — `demo infer`

`src/nuscenes_data_engine/demo/infer.py`; ultralytics imported lazily (CI stays
torch-free; local run needs the `train` extra installed — documented in DEMO.md).
For each model in `configs/demo.yaml` `models:` (baseline, `graph_rate_night`,
champion `yolov8m`) run over the staged `val`-split frames:

- predictions at native resolution, kept above a floor conf (0.05);
- IoU-matching against `annotations.parquet` GT (same 1600×900 space) using the
  SAME parameters as the AL sweep (`active_learning/sweep.py` — the implementer
  reads and reuses its IoU threshold and low-conf cutoff so the demo's TP/FP/FN
  language means the same thing as `failures.parquet`'s);
- outputs: `predictions.parquet` (`sample_token, model, category, x1..y2, conf,
  status ∈ {tp, fp, low_conf}, matched_ann_id`), `gt_boxes.parquet` (GT rows for
  curated tokens + `matched_<model>` flags; an FN for model m = row with
  `matched_m = false`), and 960×540 JPEG crops for every curated frame (val and
  train_pool) into `data/demo_curation/crops/`.
- **exemplar auto-identification** for Phase 7: tokens where baseline has an FN
  that `graph_rate_night` matches (and vice versa), written as boolean columns
  (`fixes_fn_vs_baseline`, ...) on `frame_manifest` — hand-approval of which to
  feature stays in `configs/demo.yaml` (Phase 7).

`train_pool` frames get crops only (no inference claims). Weak-label boxes for
Phase 7 come from the existing `graph_rate_night_pseudo_labels.parquet` — nothing
new to compute.

## 4. Package integration — `demo build`

Build gains a `curation` step: copies `frame_manifest`/`gt_boxes`/`predictions`
parquets and crops from `data/demo_curation/` into `demo_data/`, exports LanceDB
thumbnails for every curated token (Phase-1 exporter, bulk run), and hashes all of
it as inputs/outputs in the manifest. Validation additions: every `val` manifest
token must appear in `predictions.parquet` for every configured model; every
curated token must have either a crop or a thumbnail; size budget unchanged
(expected ≈ 60 MB total — crops ~35 MB + thumbs ~20 MB). When
`data/demo_curation/` is absent entirely, build skips the group with a prominent
log line and the manifest records `"curation": "absent"` — Phase-1 outputs still
build (the TRINITY-fallback path).

Overview's hero stays the mosaic; the real-overlay hero swap is Phase 3 (needs the
overlay renderer). No `app/demo` changes in this phase.

## 5. Testing

- Curation: deterministic selection (same seed+config → identical token list),
  bucket quotas respected on a synthetic fixture, dedup with multi-bucket labels,
  val/train_pool split correctness, filelist paths relative and unique.
- Matching: IoU matcher unit-tested on synthetic boxes (exact hit, near-miss at
  the threshold boundary, double-match prevention — one GT consumes one pred),
  status assignment incl. the low-conf cutoff, FN flags derived per model.
- Infer: end-to-end behind `pytest.importorskip("ultralytics")` on 1-2 staged
  fixture images IF cheap; otherwise the matcher is the tested unit and the model
  call is a thin adapter (state which in the plan).
- Build integration: absent-curation fallback (skip + manifest note), validation
  failures (val token missing predictions), size budget with crops.
- CI stays torch-free; all new tests green in the torch-free env.

## 6. Out of scope

Failure Explorer UI (Phase 3); any app/demo change; hero swap; chat work (Phase 4);
committing staged images or anything under `data/raw/`; publishing imagery
(licensing gate remains Phase 8).
