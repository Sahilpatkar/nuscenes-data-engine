# DATA (Phase 1 deliverable — placeholder)

> Populated during Phase 1. Documents the processed dataset schema and validation rules.

## Source

- Dataset: [nuScenes](https://www.nuscenes.org/) `v1.0-trainval`
- Location (read-only, not vendored): `/data/ggare/datasets/nuscenes/`
- 850 scenes · 34,149 keyframe samples · 6 cameras + LiDAR · 23 categories

## Relational schema (nuScenes)

`scene → sample → sample_data → sample_annotation`, joined with `calibrated_sensor`,
`ego_pose`, and `category`.

## Processed tables (to document)

- `annotations.parquet` — one row per (camera keyframe, projected 2D box).
- `samples.parquet` — per-keyframe scene/weather/sensor context.

_TODO(Phase 1): column-level schema, dtypes, and the 2D-projection procedure._

## Validation rules (Great Expectations)

_TODO(Phase 1): schema conformance, null checks, boxes within image bounds, valid
category labels, per-scene sample counts._

## Reproduce

```bash
make ingest    # nuScenes -> Parquet + 2D projections
make validate  # Great Expectations suites
```
