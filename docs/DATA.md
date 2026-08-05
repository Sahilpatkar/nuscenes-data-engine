# DATA — processed nuScenes metadata (Phase 1)

How raw nuScenes is turned into a validated, versioned, analytics-ready dataset, and
how to reproduce it.

## Source

- Dataset: [nuScenes](https://www.nuscenes.org/) `v1.0-trainval`
- Location (read-only, **not** vendored): `/data/ggare/datasets/nuscenes/`
  (configurable via `NUSCENES_DATAROOT` / `configs/data.yaml`)
- 850 scenes · 34,149 keyframe samples · 6 cameras + LiDAR · 23 categories

## Pipeline

```
raw nuScenes JSON  ──►  flatten + 3D→2D projection  ──►  Parquet  ──►  Great Expectations  ──►  DVC
(devkit tables)        (src/…/ingestion)                (data/processed)   (src/…/validation)     (MinIO)
```

1. **Flatten** (`ingestion/parse.py`) — walk `scene → sample → sample_data` for each of
   the 6 cameras, joining scene/log/weather context. Uses the devkit's
   `NuScenes.get_sample_data`, which transforms each 3D box into the camera frame.
2. **Project** (`ingestion/projection.py`) — project each 3D box's corners to pixels with
   `view_points`, take the axis-aligned extent of the positive-depth corners, and clip to
   the image. Boxes below `visibility_min` or `min_box_area_px` are dropped.
3. **Write Parquet** (`ingestion/parquet.py`) — two columnar tables (below).
4. **Validate** (`validation/expectations.py`) — Great Expectations suites.
5. **Version** — DVC tracks the Parquet with a MinIO (S3) remote.

## Output schema

### `data/processed/samples.parquet` — one row per (keyframe, camera) image

| column | type | notes |
|---|---|---|
| `sample_data_token` | str | unique per image (PK) |
| `sample_token` | str | keyframe the image belongs to |
| `channel` | str | camera, e.g. `CAM_FRONT` |
| `filename` | str | image path relative to the dataroot |
| `width`, `height` | int | 1600 × 900 |
| `timestamp` | int | microseconds |
| `n_boxes` | int | projected 2D boxes in this image |
| `scene_token`, `scene_name`, `scene_description` | str | scene context |
| `log_token`, `location` | str | e.g. `singapore-onenorth` |
| `is_night`, `is_rain` | bool | parsed from `scene_description` (for Phase 3 slices) |

### `data/processed/annotations.parquet` — one row per projected 2D box

| column | type | notes |
|---|---|---|
| `annotation_token` | str | nuScenes `sample_annotation` token |
| `sample_data_token` | str | FK → samples (the image) |
| `sample_token`, `channel` | str | |
| `category_name` | str | fine-grained nuScenes category (23 values) |
| `category_group` | str \| null | coarse detector class (car/truck/bus/pedestrian/bicycle), null if not a target |
| `visibility_token` | str | `1`–`4` (v0-40 … v80-100) |
| `num_lidar_pts`, `num_radar_pts` | int | points in the 3D box |
| `x_min, y_min, x_max, y_max` | float | 2D box in pixels, clipped to the image |
| `bbox_area` | float | pixels² |
| scene/weather context | | same columns as samples |

## Validation rules (Great Expectations)

Run with `make validate` (or `nuscenes-data-engine validate`):

- **Schema** — required columns exist in both tables.
- **Nulls** — tokens, category, and box coordinates are non-null.
- **Box bounds** — `0 ≤ x ≤ 1600`, `0 ≤ y ≤ 900`, `x_max > x_min`, `y_max > y_min`, `bbox_area > 0`.
- **Labels** — `category_name` ∈ the 23 nuScenes categories; `category_group` ∈ the 5
  detector classes (nulls allowed); `visibility_token` ∈ {1,2,3,4}.
- **Images** — width/height are exactly 1600×900; `channel` is one of the 6 cameras;
  `sample_data_token` is unique.
- **Per-scene counts** — each scene has a plausible number of camera images (15–60 keyframes × 6).

## Reproduce

Compute runs on the GPU server (infra-free); versioning happens on the infra machine.
See the README "Two-machine topology" section.

**On the GPU server** — produce + validate the Parquet (no MinIO/MLflow/Docker needed):

```bash
uv sync --extra data --extra dev
make ingest      # or: nuscenes-data-engine ingest [--limit-scenes N]
make validate    # Great Expectations suites (exit non-zero on failure)
# outputs: data/processed/{samples,annotations}.parquet
```

**On the local infra machine** — sync the Parquet off the server, then version it in MinIO:

```bash
rsync -a user@gpu-server:/home/mgaur/sahil/nuscenes_project/data/processed/ ./data/processed/
make infra-up                                  # MinIO + MLflow via docker compose
dvc add data/processed/samples.parquet data/processed/annotations.parquet
dvc push                                       # upload to the MinIO remote
git add data/processed/*.dvc && git commit -m "data: version processed dataset"
```

Data version = the hashes in the committed `*.dvc` files; `dvc pull` restores the exact
Parquet from MinIO for any commit. The GPU server never runs `dvc push`.

## Data-availability manifest (Phase 6)

The metadata references every sensor blob, but this server's copy is partial. The
audited inventory:

| Present | Missing (referenced by metadata, absent on disk) |
|---|---|
| All metadata JSON tables | All 5 RADAR sensors (keyframes + sweeps) |
| All 6 cameras — keyframes **and** sweeps | LIDAR_TOP sweeps |
| LIDAR_TOP keyframes | |

`uv run nuscenes-data-engine manifest` cross-checks every `sample_data` record against
the filesystem (one directory listing per referenced directory — minutes, not hours on
NFS) and writes `data/processed/availability.parquet`: one row per record with
`channel`, `modality`, `is_key_frame`, `scene_name`, and `present`. Downstream stages
(the embedding job, future auto-labeling) filter on it instead of trusting metadata
paths; the CLI exits non-zero if any **camera keyframe** — the working set — is
missing.

## Geometry tables (Phase B)

`uv run nuscenes-data-engine ingest-geometry` (runs where the devkit + dataset live;
GT metadata only, no LiDAR blobs) writes three more Parquet tables — the ego pose and
3D world geometry the Phase 1 pipeline projects to 2D and discards:

- **`ego_pose.parquet`** — one row per keyframe (34,149): `ego_pose_token, sample_token,
  scene_token, timestamp, x, y, z` (map metres), `heading` (rad), `speed_mps` (from
  consecutive-keyframe pose deltas), `location, is_night, is_rain`. Reference pose per
  keyframe = the `LIDAR_TOP` sample_data's `ego_pose`.
- **`annotations_3d.parquet`** — one row per 3D GT box per keyframe (1,166,187):
  `annotation_token, sample_token, instance_token, category_name, category_group, x, y, z,
  width, length, height, yaw, vx, vy, speed_mps` (`nusc.box_velocity`), `num_lidar_pts,
  num_radar_pts, visibility_token, distance_to_ego_m` (BEV), `ego_rel_x` (forward),
  `ego_rel_y` (left), + scene/weather context. Distinct grain from the per-camera 2D
  `annotations.parquet`: this is one row per physical object per keyframe.
- **`instances.parquet`** — one row per tracked object (64,386): `instance_token,
  category_name, category_group, scene_token, n_annotations`.

These back the DuckDB `ego_pose`/`annotations_3d`/`instances` chat views and the Neo4j
`EgoPose`/`ObjectObservation`/`ObjectInstance` nodes (see [GRAPH.md](GRAPH.md)), unlocking
distance-to-ego, trajectory/speed, 3D-size, and cross-frame-tracking queries.

## CAN-bus table (Phase B)

`uv run nuscenes-data-engine ingest-canbus` (also devkit-only, runs where the dataset
lives) parses the separate [nuScenes can_bus expansion](https://www.nuscenes.org/nuscenes)
into a keyframe-aligned dynamics table — the vehicle's own speed/steering/brake signals,
independent of GT-derived pose:

- **`canbus.parquet`** — one row per keyframe (34,149, unique `sample_token`):
  - `sample_token`, `timestamp` — join key (µs).
  - `has_canbus` (bool) — `False` for keyframes with no CAN data (see below); every other
    CAN column is null on those rows.
  - `can_speed_kmh` — nearest `vehicle_monitor.vehicle_speed` (km/h).
  - `steering_deg`, `steering_speed` — nearest `vehicle_monitor.steering` / `steering_speed`.
  - `brake_pedal` — nearest `vehicle_monitor.brake`, **raw 0–126 platform scale** (not a
    physical unit).
  - `brake_switch`, `throttle`, `yaw_rate`, `left_signal`, `right_signal` — remaining
    nearest `vehicle_monitor` fields, as-is from the devkit.
  - `accel_long_min_mps2`, `accel_long_max_mps2` — min/max longitudinal acceleration
    (m/s², vehicle-frame x-forward) over a **±0.5 s window** of the 50 Hz `pose` CAN
    stream centered on the keyframe timestamp; null if the window contains no messages.
  - `can_vel_mps` — speed (m/s) from the `pose` message nearest the keyframe (independent
    of `can_speed_kmh`'s `vehicle_monitor` source).
  - `is_hard_braking` (bool | null) — `accel_long_min_mps2 <= -3.0 m/s²`; null (not
    `False`) when the window is empty, so it composes correctly with SQL `WHERE
    is_hard_braking`.
  - `scene_token`, `scene_name`, `location`, `is_night`, `is_rain` — scene/weather context,
    same convention as the other Phase B tables.

**Alignment.** `vehicle_monitor` is a ~2 Hz stream; each keyframe takes the nearest
message within a **600 ms** tolerance (`nearest_message`), null beyond that. The
acceleration/velocity fields instead use every `pose` message inside a **±0.5 s** window
around the keyframe, since a single nearest-sample would miss the peak deceleration of a
braking event.

**Missing CAN data.** On the full `v1.0-trainval` run, all 15 no-CAN scenes (597
keyframes; scenes 161-168 and 170-176) are in the devkit's `can_bus.can_blacklist`.
Scenes whose CAN files are merely missing on disk are handled identically (`has_canbus =
False`, all CAN columns null) — this is a defensive path, not one exercised by this run,
since every excluded scene here is blacklisted rather than file-missing. Either way,
**no keyframe is ever dropped**.

**Config** (`configs/data.yaml`):

```yaml
canbus:                      # CAN-bus keyframe alignment
  window_s: 0.5               # pose-accel window around each keyframe
  hard_braking_mps2: -3.0     # is_hard_braking threshold on accel_long_min
  monitor_tolerance_ms: 600   # max age of the nearest vehicle_monitor message
```

**CLI:**

```bash
uv run nuscenes-data-engine ingest-canbus [--limit-scenes N]
# outputs: data/processed/canbus.parquet
```

**Validation note — independent speed cross-check.** `run_canbus_ingestion` correlates
`can_speed_kmh / 3.6` (vehicle CAN) against `ego_pose.speed_mps` (GT-pose deltas) — two
independent sources that should agree if keyframe alignment is correct. Measured on the
full run: **Pearson r = 0.999** (same 0.999 on the 5-scene smoke run), logged as
`speed_correlation_vs_gt` in the ingestion summary.
