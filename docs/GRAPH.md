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
| | | SIMILAR_TO | CAM_FRONT × k |

## Code

- `src/nuscenes_data_engine/data_engine/graph/model.py` — pure projections (Parquet
  DataFrame → node/edge dicts); the graph's correctness is unit-tested here, no DB needed.
- `graph/builder.py` — dependency-ordered, batched `UNWIND`-MERGE passes (idempotent).
- `graph/knn.py` — `SIMILAR_TO` edges from the LanceDB vectors (reuses `data_engine/store.py`).
- `graph/schema.py` — constraints/indexes. `graph/connection.py` — driver + batch helpers.
- `graph/guard.py` — the read-only Cypher guard + the agent's graph-schema prompt.
- `graph/queries.py` — the canned query library (`graph query --canned <name>`).

## How to run

The graph lives in the compose stack (infra machine). Build it after `make sync-down`
brings `data/processed` + `data/lancedb` local.

```bash
docker compose up -d neo4j                 # Browser at http://localhost:7474 (Bolt :7687)
make graph-build                           # or: uv run nuscenes-data-engine graph build
#   --skip-knn         skip the (slower) SIMILAR_TO pass
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

The `SIMILAR_TO` + `CO_OCCURS_WITH` structure enables a graph-native acquisition strategy
for Phase 6d: GDS community detection (Louvain) over the similarity subgraph yields
appearance/failure clusters without KMeans, budget is allocated across communities with the
existing `autolabel.sampling.allocate` helper, and each community contributes a
representative frame. It plugs in as a new `arm="graph"` against the existing mined/random
control (see `docs/ACTIVE_LEARNING.md`). *(Scaffolded as a follow-on; the chat + exploration
paths above are the shipped core.)*

## Known limitation — Phase B (no geometry yet)

The graph is **relational/semantic**, not geo-spatial: ingestion still discards ego-pose and
3D box geometry (see `docs/DATASET_CHAT.md`, `chat/catalog.py`), so distance-to-ego and
trajectory questions remain out of scope. The seam is `ingestion/parse.py` — it already
holds the loaded devkit handle, so a parallel `flatten_geometry` emitting `ego_pose.parquet`
+ `annotations_3d.parquet` would let the same batched-MERGE machinery add `EgoPose` nodes
(Neo4j `point`), per-box `Annotation` nodes with `distance_m`, and
`(Sample)-[:AT_POSE]->(EgoPose)-[:OBSERVES {distance_m}]->(Annotation)` — unlocking the
project plan's "pedestrians within 5 m of ego at night".

> The `graph_smoke` test builds a tiny graph end-to-end against a live Neo4j and **deletes
> all nodes** on cleanup — run it against a throwaway/dev instance, not a graph you want to
> keep.
