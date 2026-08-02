# Phase B — Geo-spatial graph (ego-pose + 3D geometry)

**Status:** approved 2026-07-24. Extends Phase 6e (context graph). Turns the relational
context graph into a true geo-spatial map by ingesting the ego-pose + 3D annotation
geometry the pipeline currently discards.

## Goal & scope (approved)

All four capabilities, **BEV** distance, **end-to-end** (ingest → SQL → graph):
distance-to-ego, ego trajectory & speed, object 3D geometry, cross-frame tracking.

Refinements the user pinned:
- Separate **`EgoPose`**, **`ObjectObservation`**, **`ObjectInstance`** nodes.
- **One `ObjectObservation` per unique 3D `sample_annotation`** (per instance, per keyframe).
- **All raw 3D observations**, not only camera-visible projections.
- **BEV distance + ego-relative coords as properties** (no materialized `NEAR_EGO` edge).
- **Constraints created before the full load.**
- **`--limit-scenes` + resumable/idempotent batching.**

Geometry uses GT metadata only (ego-pose/annotation records, no LiDAR blobs), so it runs
on TRINITY where the devkit + dataset live; parquet syncs down; the graph builds here.

## New parquet tables (`ingestion/geometry.py` → `ingest-geometry` CLI, runs on TRINITY)

Reference ego pose per keyframe = the `LIDAR_TOP` sample_data's `ego_pose` (the canonical
keyframe pose; 3D annotations are in that global frame).

- **`ego_pose.parquet`** (~34,149): `ego_pose_token, sample_token, scene_token, timestamp,
  x, y, z, heading` (yaw from quaternion), `speed_mps` (‖Δ(x,y)‖ / Δt between consecutive
  keyframes in the scene; null for the first), `location, is_night, is_rain`.
- **`annotations_3d.parquet`** (~1.2M): `annotation_token, sample_token, scene_token,
  instance_token, category_name, category_group, x, y, z, width, length, height, yaw,
  vx, vy, speed_mps` (`nusc.box_velocity`), `num_lidar_pts, num_radar_pts,
  visibility_token, distance_to_ego_m` (BEV), `ego_rel_x` (forward), `ego_rel_y` (left),
  `location, is_night, is_rain`. Distinct grain from the existing per-camera 2D
  `annotations.parquet`.
- **`instances.parquet`** (~200k): `instance_token, category_name, category_group,
  scene_token, n_annotations`.

Pure geometry helpers (TDD'd): quaternion→yaw; BEV `distance = √(dx²+dy²)`; ego-relative
`ego_rel_x = dx·cosθ + dy·sinθ`, `ego_rel_y = −dx·sinθ + dy·cosθ` (θ = ego heading);
per-scene ego speed from pose deltas.

## Graph model (`graph/model.py` projections + `graph/builder.py` passes)

Nodes (unique key in parens), all MERGE-idempotent:
- **`EgoPose(token=ego_pose_token)`** `{x, y, z, point:point({x,y}), heading, speed_mps, timestamp}`
- **`ObjectObservation(token=annotation_token)`** `{category, group, x, y, z, point,
  width, length, height, yaw, vx, vy, speed_mps, num_lidar_pts, num_radar_pts,
  visibility, distance_to_ego_m, ego_rel_x, ego_rel_y}`
- **`ObjectInstance(token=instance_token)`** `{category, group, n_annotations}`

Edges:
- `(Sample)-[:AT_POSE]->(EgoPose)`
- `(Sample)-[:HAS_OBJECT]->(ObjectObservation)`
- `(ObjectObservation)-[:OF_CATEGORY]->(Category)` (reuses 6e Category nodes)
- `(ObjectInstance)-[:OBSERVED_AS]->(ObjectObservation)`
- `(ObjectInstance)-[:IN_SCENE]->(Scene)` · `(ObjectInstance)-[:OF_CATEGORY]->(Category)`

No `NEAR_EGO` edge — `distance_to_ego_m` / `ego_rel_x/y` are properties (filterable,
indexed). Trajectory = the existing `NEXT` Sample chain via `AT_POSE` ego points.

Constraints/indexes (created **before** load): uniqueness on `EgoPose.token`,
`ObjectObservation.token`, `ObjectInstance.token`; secondary on
`ObjectObservation.distance_to_ego_m`, `ObjectObservation.category`; Neo4j **point
index** on `EgoPose.point` + `ObjectObservation.point`.

Resumability: the ~1.2M `ObjectObservation` load skips samples already carrying
`HAS_OBJECT` (analogous to the SIMILAR_TO resume). `graph build --limit-scenes` for dev.

## SQL (`chat/catalog.py`)

Register `ego_poses`, `annotations_3d`, `instances` DuckDB views + schema-prompt entries.

## Flagship query — both surfaces

*"pedestrians within 5 m of ego at night"*
- SQL: `SELECT count(*) FROM annotations_3d WHERE category_group='pedestrian' AND distance_to_ego_m < 5 AND is_night`
- Cypher: `MATCH (s:Sample)-[:HAS_OBJECT]->(a:ObjectObservation)-[:OF_CATEGORY]->(:Category{group:'pedestrian'}) WHERE s.is_night AND a.distance_to_ego_m < 5 RETURN count(a)`

## Payoff: retire the "no ego-pose/geometry" caveats

`catalog.py`, `graph/guard.py`, `DATASET_CHAT.md`, `PROJECT.md` future-scope #2 all
currently say distance questions can't be answered — Phase B makes that obsolete.

## Build order

1. `ingestion/geometry.py` pure helpers (yaw/distance/ego-rel/speed) — TDD.
2. `flatten_geometry` + `run_geometry_ingestion` + `ingest-geometry` CLI + `data.yaml`.
3. Ingest on TRINITY (`--limit-scenes` smoke first) → sync-down parquet.
4. `catalog.py` views + schema prompt.
5. `graph/model.py` projections (ego/observation/instance) — TDD.
6. `graph/schema.py` constraints/point-indexes; `graph/builder.py` passes (constraints
   first, resumable observation load, `--limit-scenes`).
7. `graph/guard.py` schema prompt + distance example; retire "no ego-pose" caveats.
8. Tests (pure projections + geometry math + smoke marker), docs (GRAPH.md/DATA.md/PROJECT.md).
9. Live verify the flagship query end-to-end in SQL + Cypher.
