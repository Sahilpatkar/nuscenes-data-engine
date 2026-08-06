# Phase 6e — Knowledge Graph

A **Neo4j context graph** built entirely from data already on disk — the processed
Parquet tables, the Phase 6b VLM labels, and the Phase 6a SigLIP/LanceDB vectors. No
nuScenes devkit re-ingestion. It turns the already-denormalized token columns back into an
explicit property graph so the chat agent can answer **multi-relationship** questions,
the dataset can be **browsed visually** in Neo4j Browser, and active learning can sample
for **graph diversity**.

Why a graph on top of DuckDB + LanceDB: SQL is great for counts and aggregates, vectors
for "looks-like" retrieval, but questions that chain several relationships
("what else rides along with bicycles at night in Singapore?", "which frames are visually
nearest to the ones that contain a construction vehicle?", temporal "what came next?")
are awkward as SQL joins. Those are the graph's home turf.

## Graph model

```
              (Location)
                  ^ IN_LOCATION
   NEXT           |
 (Sample)──▶(Sample)──IN_SCENE──▶(Scene)◀──IN_SCENE──(Frame)
    ▲                                                   │  ├─HAS_HAZARD──▶(Hazard)
    └────────────IN_SAMPLE──────────────────────────────┘  ├─HAS_CONDITION▶(NotableCondition)
                                                            ├─SIMILAR_TO {score,rank}▶(Frame)
                                                            └─CONTAINS {count,avg_visibility,…}▶(Category)
                                                                       (Category)─CO_OCCURS_WITH {n_frames}─▶(Category)
```

**Nodes** (unique key in parens): `Scene(token)` · `Sample(token)` (keyframe joining the 6
cameras) · `Frame(token = sample_data_token)` · `Location(name)` · `Category(name)` ·
`Hazard(text)` · `NotableCondition(text)`. Condition flags (`is_night`/`is_rain`) and the
VLM fields (`vlm_time_of_day`, `vlm_weather`, `vlm_label_confidence`, the 10 object counts)
are **properties** — on `Scene` and `Frame` respectively. The open-vocabulary VLM
`hazards`/`notable_conditions` become nodes (many-to-many is the graph's strength).

**Relationships:** `IN_SCENE`, `IN_SAMPLE`, `IN_LOCATION`, `NEXT {dt_us}`,
`CONTAINS {count, avg_visibility, min_visibility, total_bbox_area, max_bbox_area,
num_lidar_pts, num_radar_pts}`, `HAS_HAZARD`, `HAS_CONDITION`,
`CO_OCCURS_WITH {n_frames}` (per canonical category-name order), `SIMILAR_TO {score, rank}`
(SigLIP kNN within one camera channel).

Annotations are **not** nodes: the ~1.1M boxes collapse to one `CONTAINS` edge per (frame,
category) with the box stats as edge aggregates — the graph stays about relationships, not
individual boxes.

### Scale (full v1.0-trainval build)

| Nodes | | Relationships | |
|---|---|---|---|
| Frame | 204,894 | CONTAINS | 382,086 |
| Sample | 34,149 | IN_SCENE | 239,043 |
| Scene | 850 | IN_SAMPLE | 204,894 |
| NotableCondition | 2,473 | NEXT | 33,299 |
| Category | 23 | HAS_CONDITION | 12,306 |
| Hazard | 111 | HAS_HAZARD | 1,347 |
| Location | 4 | CO_OCCURS_WITH | 220 |
| | | SIMILAR_TO | 341,490 (CAM_FRONT, k=10) |
| EgoPose (6B) | 34,149 | AT_POSE | 34,149 |
| ObjectObservation (6B) | 1,166,187 | HAS_OBJECT | 1,166,187 |
| ObjectInstance (6B) | 64,386 | OBSERVED_AS | 1,166,187 |

## Code

- `src/nuscenes_data_engine/data_engine/graph/model.py` — pure projections (Parquet
  DataFrame → node/edge dicts); the graph's correctness is unit-tested here, no DB needed.
- `graph/builder.py` — dependency-ordered, batched `UNWIND`-MERGE passes (idempotent).
- `graph/knn.py` — `SIMILAR_TO` edges via a vectorized in-memory kNN over the LanceDB
  vectors (cosine = dot on the L2-normalized SigLIP vectors; ~25 s for all 34,149
  CAM_FRONT frames), written incrementally so the pass is resumable.
- `graph/schema.py` — constraints/indexes. `graph/connection.py` — driver + batch helpers.
- `graph/guard.py` — the read-only Cypher guard + the agent's graph-schema prompt.
- `graph/queries.py` — the canned query library (`graph query --canned <name>`).

## How to run

The graph lives in the compose stack (infra machine). Build it after `make sync-down`
brings `data/processed` + `data/lancedb` local.

```bash
docker compose up -d neo4j                 # Browser at http://localhost:7474 (Bolt :7687)
make graph-build                           # or: uv run nuscenes-data-engine graph build
#   --skip-knn         skip the SIMILAR_TO pass
#   --edges similar    rebuild only one derived pass (repeatable)
#   --rebuild          delete all nodes/rels first
#   --knn-k 10 --channel CAM_FRONT

uv run nuscenes-data-engine graph stats
uv run nuscenes-data-engine graph query --canned top_co_occurrence
uv run nuscenes-data-engine graph query "MATCH (s:Scene) WHERE s.is_night RETURN count(*)"
```

The build is idempotent/resumable — every write is a `MERGE` on a key, and the long
`SIMILAR_TO` pass skips frames already linked, so re-running only fills gaps.

Default connection: `bolt://localhost:7687`, user `neo4j` (`NEO4J_URI`/`NEO4J_USER`/
`NEO4J_PASSWORD` in `.env`; keep `NEO4J_PASSWORD` in sync with `NEO4J_AUTH`). Inside compose
the api container reaches the graph at `bolt://neo4j:7687`.

## Example: relationship questions SQL finds awkward

`graph query --canned top_co_occurrence` (categories that share the most frames):

```
a                       | b            | frames
human.pedestrian.adult  | vehicle.car  | 47326
vehicle.car             | vehicle.truck| 43627
movable_object.trafficcone | vehicle.car | 20929
```

`--canned night_categories` (most-boxed categories in night frames), `--canned
top_hazards` (VLM hazards), `--canned similar_bicycles` (SigLIP neighbours of
bicycle-containing frames), `--canned location_conditions`. All live in `graph/queries.py`.

## Chat integration (`run_cypher`)

When the graph is reachable, the Phase 6c chat agent gains a fourth tool, `run_cypher`,
alongside `run_sql`/`search_frames`/`show_frames`. The agent's system prompt describes the
graph schema and routes relationship/path/co-occurrence/similarity/temporal questions to
Cypher, counts/aggregates to SQL. If Neo4j is down, the tool is simply not offered and chat
runs SQL + vector only — same graceful degradation as a missing LanceDB store.

```bash
uv run nuscenes-data-engine chat \
  "Which categories most often co-occur with bicycles in night scenes in Singapore?"
```

## Cypher safety

`run_cypher` mirrors the DuckDB SQL guard's philosophy:

1. **Hard guard** — the query runs in a Neo4j **read-mode transaction**; any write clause
   is rejected by the server with `Neo.ClientError.Statement.AccessMode`, even if the
   denylist below were bypassed.
2. **Denylist** (defense-in-depth, a fast explanatory error before the DB): write keywords
   (`CREATE/MERGE/DELETE/DETACH/SET/REMOVE/DROP/FOREACH`, `LOAD CSV`, `PERIODIC COMMIT`) and
   any procedure `CALL` not on a read-only allowlist (`gds.*.stream`, `db.labels`,
   `db.schema.*`) — so `apoc.*` and `gds.*.write/.mutate/.drop` are refused.
3. A **row cap** (100) with a truncation flag, and a **query timeout**.

Errors are returned to the model as data, so it repairs its own Cypher.

## Active learning (graph diversity)

A graph-native acquisition arm for Phase 6d: GDS **Louvain** community detection over the
pool's `SIMILAR_TO` subgraph yields appearance communities without KMeans; the labeling
budget is allocated across communities — size-proportional via the shared
`autolabel.sampling.allocate` helper for the round-1 arm, or ∝ smoothed-rate failure mass
via `allocate_by_mass` for the round-3 arms — and each community contributes its
most-connected (representative) frames.

```bash
docker compose up -d neo4j && make graph-build            # needs the SIMILAR_TO edges
uv run nuscenes-data-engine al graph-mine                 # -> data/active_learning/graph.parquet
GPU_DEVICES=0 scripts/gpu-run.sh --bg al run --arm graph  # train + evaluate on the GPU server
uv run nuscenes-data-engine al report                     # 4-arm comparison
```

It plugs in as a new `arm="graph"` against the existing mined/random control (same leakage
guards, same shared val split). Graph-mining runs on the infra machine (Neo4j lives there);
`graph.parquet` then ships to the GPU server for training, exactly like `mined.parquet`.

**Why it helps:** the 6d KMeans mined arm collapsed to near-duplicates around failure
centroids — 219 distinct scenes, **0% night** — which is why it lost to the equal-budget
random control. Graph-diversity selection spreads across **473 scenes (12.1% night)**,
matching random's diversity (501 scenes, 12.7% night) while staying structured and
interpretable. See `docs/ACTIVE_LEARNING.md` for the trained-arm mAP comparison.

## Geo-spatial layer (Phase B)

The graph is now geo-spatial as well as relational: `ingest-geometry` persists the GT
**ego pose + all 1.17M 3D boxes** (world position, size, heading, velocity, BEV
distance-to-ego, ego-relative coords, instance tracking), and `graph build` adds three
node types keyed off the keyframe `Sample`:

- **`EgoPose`** (34,149) `{x, y, z, point, heading, speed_mps, location, is_night, is_rain,
  has_canbus, can_speed_kmh, steering_deg, brake_pedal, throttle, yaw_rate,
  accel_long_min_mps2, accel_long_max_mps2, is_hard_braking}` — `(Sample)-[:AT_POSE]->(EgoPose)`;
  trajectory = the `NEXT` chain of ego points. The CAN properties come from a second,
  idempotent `SET` pass keyed off `(Sample)-[:AT_POSE]->(EgoPose)` (`canbus.parquet` →
  `graph/builder.py`'s `_CANBUS` pass), run after the base `EgoPose` load; a range index on
  `accel_long_min_mps2` backs the hard-braking filter. The applied count is read back from
  the graph after the write (`MATCH (e:EgoPose) WHERE e.has_canbus IS NOT NULL RETURN
  count(e)`) rather than trusted from the batch size, so a silent no-op (e.g. a stale graph
  missing the matching `EgoPose`) is detectable — full run: `canbus: 34149,
  canbus_applied: 34149`. Because Neo4j `SET`ting a property to `null` **removes** it,
  `{is_hard_braking: true}` naturally excludes both no-CAN keyframes and keyframes whose
  accel window was empty (no ambiguous `false`).
- **`ObjectObservation`** (1,166,187) `{category, x, y, z, point, width/length/height, yaw,
  speed_mps, distance_to_ego_m, ego_rel_x (forward), ego_rel_y (left), num_lidar_pts,
  visibility, location, is_night, is_rain}` — `(Sample)-[:HAS_OBJECT]->(ObjectObservation)-[:OF_CATEGORY]->(Category)`.
- **`ObjectInstance`** (64,386) `{category, n_annotations}` — one physical object tracked
  across keyframes: `(ObjectInstance)-[:OBSERVED_AS]->(ObjectObservation)`, `-[:IN_SCENE]->(Scene)`.

Point indexes on `EgoPose.point` + `ObjectObservation.point` back spatial ops; a
`distance_to_ego_m` index backs the common proximity filter (no materialized `NEAR_EGO`
edge). Distances are **bird's-eye** (ground-plane) metres; `point.distance()` reproduces
the stored value.

The project plan's flagship — *"pedestrians within 5 m of ego at night"* — answers
identically (183) in SQL and Cypher:

```cypher
MATCH (o:ObjectObservation)-[:OF_CATEGORY]->(:Category {group:'pedestrian'})
WHERE o.is_night AND o.distance_to_ego_m < 5 RETURN count(o)
```
```bash
docker compose up -d neo4j
make sync-down                                  # or: rsync the geometry parquet down
uv run nuscenes-data-engine ingest-geometry     # on TRINITY (devkit); --limit-scenes for dev
uv run nuscenes-data-engine graph build          # adds the geo passes; --limit-scenes / --skip-geometry
```

The observation load is idempotent and resumable (a keyframe already carrying `HAS_OBJECT`
is skipped).

CAN-bus dynamics are now ingested too (see the `EgoPose` CAN properties above). A second
flagship — *"hard braking with a pedestrian within 10 m"* — answers identically (**30**)
in SQL (`canbus` `JOIN` `annotations_3d`) and Cypher:

```cypher
MATCH (e:EgoPose {is_hard_braking: true})<-[:AT_POSE]-(s:Sample)
      -[:HAS_OBJECT]->(o:ObjectObservation)-[:OF_CATEGORY]->(c:Category)
WHERE c.group = 'pedestrian' AND o.distance_to_ego_m < 10
RETURN count(DISTINCT s)   // 30 — identical to the SQL form
```
```sql
SELECT count(DISTINCT c.sample_token)
FROM canbus c JOIN annotations_3d a USING (sample_token)
WHERE c.is_hard_braking AND a.category_group = 'pedestrian'
  AND a.distance_to_ego_m < 10;   -- 30, identical to the Cypher above
```

> The `graph_smoke` test builds a tiny graph end-to-end against a live Neo4j and **deletes
> all nodes** on cleanup — run it against a throwaway/dev instance, not a graph you want to
> keep.
