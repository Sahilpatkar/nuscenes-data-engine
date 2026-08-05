# CAN-bus ingestion — ego dynamics (steering / braking / speed)

**Status:** approved 2026-08-05. Closes the piece Phase B left out of scope: the nuScenes
CAN-bus expansion (confirmed present on TRINITY at `/data/ggare/datasets/nuscenes/can_bus/`,
7,832 files) becomes keyframe-aligned ego-dynamics data — Parquet → DuckDB → Neo4j
`EgoPose` properties → chat queries like the project plan's *"find scenes with hard
braking near pedestrians"*.

## Goal & scope (approved)

Decisions the user pinned:
- **Keyframe-aligned summary only** — one row per keyframe (34,149), no raw 50–100 Hz
  stream storage (~35M rows, YAGNI). Every roadmap query works off the summary.
- **CAN signals live on the existing `EgoPose` nodes** — new properties, no new node
  type; "hard braking near pedestrians" stays a two-hop MATCH.
- **`is_hard_braking` is deceleration-based**: min longitudinal (vehicle-frame
  x-forward) acceleration within ±0.5 s of the keyframe ≤ −3.0 m/s², both knobs
  configurable. The brake-pedal position is stored as a raw column but does not
  define the flag.

## Ingestion (`ingestion/canbus.py` → `ingest-canbus` CLI, runs on TRINITY)

Mirrors the Phase B `geometry.py` pattern: pure alignment helpers (TDD'd, torch-free
CI) + `flatten_canbus` walking keyframes with the devkit's `NuScenesCanBus`.

Per keyframe at timestamp *t* (µs):
- **Nearest `vehicle_monitor` message** (~2 Hz stream) within `monitor_tolerance_ms`
  (default 600): `can_speed_kmh, steering_deg, steering_speed, brake_pedal` (raw
  0–126 platform scale), `brake_switch, throttle, yaw_rate, left_signal,
  right_signal`. No message within tolerance → these columns null.
- **`pose` window** (50 Hz stream) [*t* − `window_s`, *t* + `window_s`] (default
  0.5 s): `accel_long_min_mps2`, `accel_long_max_mps2` (accel[0], vehicle-frame
  x-forward), and `can_vel_mps` (‖vel‖ of the nearest pose message). Empty window →
  null.
- **`is_hard_braking`** = `accel_long_min_mps2 <= hard_braking_mps2` (default −3.0);
  null when the window is empty.
- **Missing scenes**: the devkit `can_blacklist` (~10 scenes) and any scene whose
  files are absent get `has_canbus = False` with all CAN columns null — no rows
  dropped, 34,149 rows always.

Output: `data/processed/canbus.parquet`, keyed `sample_token` (+ `scene_name`,
`timestamp`, `location`, `is_night`, `is_rain` denormalized like the other Phase B
tables). Config block in `configs/data.yaml`:

```yaml
canbus:
  window_s: 0.5              # pose-accel window around each keyframe
  hard_braking_mps2: -3.0    # is_hard_braking threshold on accel_long_min
  monitor_tolerance_ms: 600  # max age of the nearest vehicle_monitor message
```

Ingestion logs (not gates) a sanity cross-check: Pearson correlation of
`can_speed_kmh / 3.6` against the Phase B GT-derived `ego_pose.speed_mps` over
matched keyframes — a wiring check for free, since the two speeds come from
independent sources.

## SQL + graph + chat

- **DuckDB (`chat/catalog.py`)**: register a `canbus` view over the parquet;
  conditional caveat (like the geometry tables): when the parquet is absent the
  schema prompt says so, when present the columns are listed and any "no dynamics
  data" caveat is retired.
- **Neo4j (`graph/builder.py`)**: a new idempotent pass —
  `(Sample {token})-[:AT_POSE]->(e:EgoPose)` `SET e.can_speed_kmh, e.steering_deg,
  e.brake_pedal, e.throttle, e.yaw_rate, e.accel_long_min_mps2,
  e.accel_long_max_mps2, e.is_hard_braking, e.has_canbus` — batched with the
  existing `run_write_batches`, naturally resumable (SET is idempotent),
  `--limit-scenes` respected. `graph/schema.py`: one range index on
  `EgoPose.accel_long_min_mps2`. No new constraints or node types.
- **Chat guard (`graph/guard.py` + SQL schema prompt)**: CAN properties documented;
  flagship example added in both languages —
  Cypher: `MATCH (e:EgoPose {is_hard_braking: true})<-[:AT_POSE]-(s:Sample)
  -[:HAS_OBJECT]->(o:ObjectObservation)-[:OF_CATEGORY]->(:Category {group:
  'pedestrian'}) WHERE o.distance_to_ego_m < 10 RETURN count(DISTINCT s)`;
  SQL: the same count via `canbus JOIN annotations_3d USING (sample_token)`.

## Acceptance (the Phase B pattern)

1. Pure helpers green in CI (nearest-message selection incl. tolerance edge,
   window stats, flag thresholding, blacklist nulls).
2. TRINITY: `ingest-canbus --limit-scenes 5` smoke, then the full run (JSON-only,
   minutes); rsync `canbus.parquet` down.
3. Infra Mac: DuckDB view queryable; graph pass over the full 34,149 `EgoPose`
   nodes; the flagship "hard braking near pedestrians (< 10 m)" query returns
   **identical counts in SQL and Cypher**.
4. Speed cross-check correlation reported in the run log and quoted in docs.

## Docs

`DATA.md` (canbus table schema + config), `GRAPH.md` (EgoPose CAN properties +
flagship Cypher), `ANALYTICS.md` (SQL example), `DATASET_CHAT.md` (retired caveat +
example question), `PROJECT.md` (§9 roadmap item, §2 component map mention).

## Explicitly out of scope

- No raw-stream parquet (pose/steering at 50–100 Hz) — revisit only when a consumer
  exists.
- No new event flags beyond `is_hard_braking` (no is_turning/is_accelerating).
- No changes to the 2D ingestion, training, or AL pipelines.
- `zoe_veh_info` / `zoesensors` / `ms_imu` / `route` streams unused.
