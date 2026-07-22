# Monitoring (Phase 5) — drift detection over serving inputs

The serving API's inputs are compared against a training-data reference on four
per-image features; [Evidently](https://www.evidentlyai.com/) decides, per feature and
overall, whether the distributions have drifted.

## Features

| Feature | Source | Why |
|---|---|---|
| `brightness` | mean of the Rec.601 gray conversion (`cv2.COLOR_BGR2GRAY`) | catches day→night / exposure shifts; gray-mean approximates perceived brightness better than HSV V (max-channel), which overweights saturated pixels |
| `width`, `height` | image shape | catches resolution / crop changes (nuScenes is constant 1600×900 — any change is drift) |
| `n_boxes` | GT boxes (reference) / predicted detections (serving) | catches scene-content shifts (night frames average far fewer visible objects) |

One function ([`image_brightness`](../src/nuscenes_data_engine/monitoring/features.py))
is used both offline (reference build) and online (per-request capture), so there is no
train/serve skew in the feature definition.

## Where the two sides come from

- **Reference** — built where the images live (the GPU server):
  `monitor build-reference --condition day` samples N images (default 2000) from
  `samples.parquet`, reads them from `NUSCENES_DATAROOT`, and writes
  `data/processed/monitoring_reference.parquet`, which is rsynced to the infra machine
  with the rest of `data/processed/`. Without dataset access (`--no-images`) brightness
  is NaN and the report degrades to metadata-only.
- **Current** — the serving API appends one JSONL row per request
  (`{ts, image_width, image_height, n_detections, brightness, latency_ms, model_version}`)
  to `SERVING_CAPTURE_PATH` (default `data/monitoring/requests.jsonl`, gitignored,
  volume-mounted in Docker; blank disables). `monitor report --current` also accepts any
  feature parquet — that's how the night simulation below works.

## Running it

```bash
# report vs the live serving capture:
make monitor

# explicit inputs + output dir:
uv run nuscenes-data-engine monitor report \
    --reference data/processed/monitoring_reference.parquet \
    --current   data/processed/monitoring_night_current.parquet \
    --out-dir   runs/monitoring/night-drift
```

Outputs: `drift_report.html` (full Evidently report) and `drift_summary.json`
(`{columns: {feature: {drift_detected, score}}, n_drifted, share_drifted, dataset_drift}`)
— the JSON is the machine-readable hook for automation. The report is an observation,
not a gate: the CLI always exits 0.

## Demo: simulated night drift

Reference = 2000 day images (brightness mean 106.8); current = 2000 night images
(mean 29.0), built with the same pipeline via `--condition night`. Result
(`drift_summary.json`):

```json
{
  "columns": {
    "brightness": { "score": 13.1, "drift_detected": true },
    "width":      { "score": 0.0,  "drift_detected": false },
    "height":     { "score": 0.0,  "drift_detected": false },
    "n_boxes":    { "score": 0.55, "drift_detected": true }
  },
  "n_drifted": 2,
  "share_drifted": 0.5,
  "dataset_drift": true
}
```

Brightness collapses (day ≫ night) and the per-image detection count drops, so both
columns fail Evidently's per-column drift tests and the dataset-level flag trips
(`drift_share: 0.25` — 1 of 4 columns suffices). `width`/`height` stay clean, as they
should. Scores are Evidently's auto-selected drift statistic (K-S p-value for small
samples, normalized Wasserstein distance at this sample size — higher = more drift).

## Notes / limitations

- **Evidently 0.7.x API** (locked at 0.7.21): `Report`/`Dataset`/`DataDefinition` +
  `presets.DataDriftPreset`. Most online examples show the 0.4 or 0.8 APIs. Gotcha:
  `snapshot.save_html()` silently writes nothing when handed a `pathlib.Path` — pass
  `str`.
- `n_boxes` compares ground-truth box counts (reference) with predicted detection
  counts (serving) — a deliberate proxy. The night simulation compares GT to GT, so it
  is apples-to-apples there.
- The JSONL capture is a plain synchronous append (uvicorn's default single worker);
  fine at demo scale, a queue/rotation would come before real traffic.
- Future work: schedule `monitor report` (cron/Dagster), alert on
  `dataset_drift: true`, and a Terraform-provisioned cloud deployment.
