# Demo Phase 6 — interactive graph: subgraph export + agraph panel

**Status:** approved 2026-08-20 (Phase 6 of the approved demo plan; Phases 1-5 merged
as PRs #14-#18). The demo doc's #1 visual priority: "why did this query return this
driving event?" as a query-focused subgraph with the matched path highlighted and
click-to-inspect node metadata — and the flagship Cypher count computed against the
live graph instead of sourced from docs.

## Why (and what was verified first)

Neo4j is populated locally (volume intact; `graph stats` healthy) and GRAPH.md's
verbatim flagship Cypher returns **30 live** — identical to the SQL count the events
exporter asserts. The real schema (from `graph/guard.py::graph_schema_prompt`):
`(Sample)-[:AT_POSE]->(EgoPose {CAN props incl. is_hard_braking, accel_*,
speed_mps})`, `(Sample)-[:HAS_OBJECT]->(ObjectObservation {category,
distance_to_ego_m, ego_rel_x/y, visibility, ...})-[:OF_CATEGORY]->(Category {name,
group})`, `(Sample)-[:IN_SCENE]->(Scene)-[:IN_LOCATION]->(Location)`,
`(Sample)-[:NEXT {dt_us}]->(Sample)`. Category matching is on `group` (the demo doc's
conceptual diagram used different names — the spec follows the real schema).
`streamlit-agraph` 0.0.45 installs cleanly.

## 1. Subgraph export — `demo/subgraph_export.py` + `demo subgraphs` CLI

Neo4j is an **operational** dependency, handled like curation/semsearch: a CLI
command stages the artifact under `data/demo_curation/graph_subgraphs/`, and
`demo build` copies + validates it when present or records `"subgraphs": "absent"`
when not. `demo build` never requires Neo4j.

For each of the six presets (events from the staged `scenario_events.parquet`):
- **Count query** (dynamics presets only): a parameterized Cypher equivalent of the
  preset predicate (flagship = GRAPH.md's verbatim query; the other three built from
  the same threshold config). Export records `sql_count` (from the events exporter's
  pre-cap count, re-derived the same way) and `cypher_count`. The flagship's parity
  is ASSERTED (export fails on mismatch); the other three are recorded with a
  `parity: true|false` flag and a warning on mismatch — a schema-coverage gap there
  is a finding to report, not something to hide or to block on. Model presets carry
  `cypher_count: null` and a note: the model verdict is not in the graph.
- **Per-event subgraph** for each exported event (≤30 per preset): the Sample, its
  Scene and Location, its EgoPose (with CAN properties), its ObjectObservations
  (the predicate-matching ones marked `on_path`; non-matching ones capped at the 12
  nearest by distance so a crowded frame stays legible) with their Categories, and
  the NEXT-prev/next Samples (temporal context, 1 hop). Nodes carry
  `{id, label, group, on_path, meta{...}}` (meta = the properties the demo doc lists
  per node type: Sample timestamp/scene/lighting/weather; EgoPose speed/accel/
  braking; ObjectObservation class/distance/ego_rel/visibility), edges
  `{source, target, label}`, and `path` = the ordered node ids of the matched
  Scene→Sample→EgoPose and Sample→ObjectObservation→Category chains.
- Output: `graph_subgraphs/<preset>.json` = `{preset, count_cypher, sql_count,
  cypher_count, parity, events: {sample_data_token: {nodes, edges, path}}}`. Sizes
  are small (≈15-40 nodes per event).
- The flagship's computed `cypher_count` also flows into `overview_metrics.json`:
  when the staged subgraphs are present, `flagship.cypher` = the computed value and
  `cypher_source = "computed (neo4j, demo subgraphs)"`; otherwise the existing
  sourced value + label remain (the Phase-1 contract, now with a real upgrade path).
  Build validation asserts sql == cypher for the flagship whenever computed.

## 2. The panel — in `views/scenarios.py`'s Phase-6 slot

- `streamlit-agraph` renders the selected event's subgraph: `on_path` nodes in the
  accent color and larger, off-path nodes faded; edge labels = relationship types;
  a fixed legend (Scene / Sample / EgoPose / ObjectObservation / Category /
  Location). No animation (static highlight — per the master plan's YAGNI call).
- Clicking a node returns its id → a metadata panel (`st.json`-style key/value
  list from `meta`) beside the graph. When no node is selected, the panel shows the
  path narrative: "Scene → Sample → EgoPose (hard braking −4.5 m/s²) → pedestrian at
  3.8 m".
- The preset header gains the parity line: "Cypher: 30 · SQL: 30 ✓" (or the
  recorded mismatch, plainly) for dynamics presets; "graph holds GT only — model
  verdict comes from the prediction set" for model presets.
- When the package has no subgraphs (`"absent"`), the slot shows an honest note
  ("graph export not included in this package") instead of disappearing.
- Dependencies: `streamlit-agraph` added to `app/demo/requirements.txt` AND to the
  repo's `serve` extra (so AppTest/CI can import it); the requirements-minimal test
  gains the new allowed name; the AST no-backend-import guard is unaffected
  (`streamlit_agraph` is not a backend module).

**Amendment (2026-08-20, visual check).** Rendered in a real browser, the panel as
specified was unreadable, so three details differ from the bullets above.
`physics=True`, not `False`: vis.js is handed no seeded coordinates, so with physics
off it scattered ~30 nodes into a random pile — the "static highlight (no
animation)" call above is about the path COLORING, which is unchanged, not about the
layout solver. Node labels are semantic rather than type-plus-token-prefix
(`adult 9.9 m`, `ego -4.5 m/s²`, `scene-1084`, `human.pedestrian.adult` in full) —
head-truncated ids made every `human.pedestrian.*` category render identically, and
a hex token told the viewer nothing; the graph type and the raw token moved to the
node's hover title. And edge labels are drawn only for the structural relationships
(`AT_POSE`, `IN_SCENE`, `IN_LOCATION`, `NEXT`) — the up-to-14 `HAS_OBJECT`/
`OF_CATEGORY` labels around the Sample hub were pure clutter, since their endpoints
already name themselves.

## 3. Testing (batched reviews, as in Phase 5)

- Exporter: Cypher text builders (threshold injection; the flagship text equals
  GRAPH.md's verbatim query modulo whitespace — asserted against the doc file so
  they cannot drift), JSON assembly from fake driver records (injected `run_query`
  callable returning dicts), path extraction, node/edge dedup, the 12-nearest cap,
  `on_path` marking, parity flag + flagship assertion path, model-preset nulls.
- A live-Neo4j integration test, skipped unless the bolt URI is reachable (the
  repo's `graph_smoke` precedent) — runs the flagship count and one event subgraph.
- Build: absent/present handling; overview `cypher_source` switch; flagship parity
  assertion when computed.
- Page: AppTest smoke with a tiny subgraph JSON — no exception, parity line and
  narrative panel present. `streamlit-agraph` is a custom component that AppTest
  does not render; the smoke asserts the surrounding Streamlit elements and the
  live `streamlit run` check in the operational task covers the component itself
  (documented limitation).
- Two review gates: one consolidated review after the code tasks, the final
  whole-branch review after the operational task.

## 4. Out of scope

Cytoscape.js / custom JS components; animated traversal; a scenario BUILDER over
the graph; GDS/community views (the AL pages in Phase 7 use the precomputed
community JSONs, not live GDS); any graph rebuild (the volume is current — if
`demo subgraphs` ever finds the flagship ≠ 30, the fix is to rebuild via `graph
build`, reported as an operational step, not code in this phase).
