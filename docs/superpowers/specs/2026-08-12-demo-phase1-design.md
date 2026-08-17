# Demo Phase 1 — shell, Overview page, artifact-builder core

**Status:** approved 2026-08-12 (Phase 1 of the approved 8-phase demo plan; master spec:
`docs/DEMO_PLAN.md`, architecture decisions locked in the plan-mode session of
2026-08-11). Goal: `streamlit run app/demo/main.py` shows a real Overview page fed
entirely from a committed `demo_data/` v0.1, with the other pages registered as
labelled stubs so the five-page navigation ships now.

## Why

The engineering is done; the presentation layer is not. The public demo must run on
Streamlit Community Cloud with zero backend, so it reads only a committed artifact
package built offline by this repo's pipeline. Phase 1 establishes the three load-
bearing structures every later phase extends: the `demo` CLI builder, the
`demo_data/` package with its validated manifest, and the `app/demo/` app shell.

## 1. The builder — `demo` typer sub-app + `src/nuscenes_data_engine/demo/`

New package `src/nuscenes_data_engine/demo/exporters.py` (pure functions, pytest,
no torch; lancedb imported lazily) and a `demo` sub-app in `cli.py`:

- `demo export overview` → `demo_data/overview_metrics.json`. Every number is
  **derived from local artifacts at export time, never hardcoded**:
  - dataset scale: row counts from `data/processed/{samples,annotations,annotations_3d,canbus}.parquet`;
  - headline results: best night delta (+0.0101 arm `graph_rate_night`) and weak-sup
    retention (18%) read from `data/active_learning/results.json`;
  - CAN validation `r`: computed as the Pearson correlation of `canbus.speed_mps`
    against ego-pose-derived speed on the keyframe join (the documented 0.999);
  - flagship consistency: the hard-braking-near-pedestrian count computed via the
    documented SQL over local parquet (the documented 30). The Cypher twin is NOT
    computed in Phase 1 (Neo4j is a Phase 6 dependency); the JSON stores
    `{"sql": <computed>, "cypher": 30, "cypher_source": "docs/GRAPH.md"}` and
    Phase 6 replaces the sourced value with a computed one.
- `demo export al` → `active_learning_results.parquet` (arm, round, overall/night
  mAP50-95, delta vs baseline, n_selected) reshaped from `results.json`
  (reuse patterns from `active_learning/report.py`).
- `demo export weaksup` → `weak_supervision_results.parquet` (per-arm retention,
  loss decomposition, boxes-per-frame accepted/rejected) from the weak-sup
  summaries + `results.json`.
- `demo export hero` → copies one curated `val_batch*_pred.jpg` mosaic from
  `mlruns/artifacts/<baseline-run>/artifacts/ultralytics_run/` into
  `demo_data/sample_frames/hero.jpg` (interim hero; replaced by a real overlay in
  Phase 3). Run id comes from `configs/demo.yaml`, not hardcoded.
- `demo export thumbs --tokens <file>` → exports LanceDB thumbnails for a token
  list into `demo_data/sample_frames/thumbs/<token>.jpg`. Phase 1 ships the
  exporter + tests; the bulk run happens in Phase 2 when the curated manifest
  exists. (lancedb lazily imported; test with a tmp LanceDB table.)
- `demo build` → runs all registered exporters in order, then writes
  `demo_data/manifest.json`: built_at, git sha, per-input SHA256, per-output row
  counts, and validation. **Validation fails loudly** if: a declared input is
  missing, the flagship SQL count ≠ 30, or `demo_data/` exceeds the size budget
  (`configs/demo.yaml`, default 100 MB). Exporters sort before writing and use
  fixed parquet settings so `demo build` is deterministic given identical inputs.

`configs/demo.yaml` (Phase 1 keys): model-run registry (baseline/graph_rate_night/
champion run ids), hero source run, size budget, paths. Later phases add their
sections; the config is the single place demo curation is parameterized.

**Amendment (2026-08-14, at plan-writing):** Phase 1 exposes only `demo build` on
the CLI; the per-artifact `demo export <name>` subcommands listed above stay
internal functions until a later phase actually needs a partial rebuild (YAGNI —
`demo build` takes seconds at Phase-1 scope). The exporter functions and their
tests are unchanged.

## 2. The package — `demo_data/`

Committed to the repo root (gitignore exception). Phase 1 contents:
`manifest.json`, `overview_metrics.json`, `active_learning_results.parquet`,
`weak_supervision_results.parquet`, `sample_frames/hero.jpg`. Size after Phase 1:
well under 1 MB + the hero JPEG (~500 KB). The <100 MB budget is enforced from day
one so image-heavy later phases cannot creep past it silently.

## 3. The app shell — `app/demo/`

- `main.py`: `st.navigation`/`st.Page` explicit registry — Overview (real),
  Failure Explorer, Scenario Search, Active Learning, Weak Supervision, Ask the
  Dataset (recorded) — the last five as stubs rendering their one-line purpose and
  "ships in Phase N". Page functions are plain importable functions.
- `data.py`: `st.cache_data` loaders (`load_manifest()`, `load_overview()`,
  `load_al_results()`, `load_weaksup()`); a missing `demo_data/` renders a clear
  "run `nuscenes-data-engine demo build`" message rather than a traceback.
- `render.py`: `metric_cards(...)` (st.columns + st.metric), `story_arrows([...])`
  (the doc §9 vertical narrative primitive), `results_table(df)`. The overlay
  renderer and chart helper land in Phases 3-4.
- `views/overview.py`: per `docs/DEMO_PLAN.md` §1 — project description blockquote,
  four scale metrics, four headline results, hero image with caption, and a short
  "how to read this demo" footer naming the recorded-vs-live distinction.
- `app/demo/requirements.txt`: `streamlit`, `pandas`, `pyarrow`, `pillow` only
  (agraph added in Phase 6). **The demo app never imports `src/` or `requests`** —
  enforced by a repo test that imports every `app/demo` module and asserts
  `nuscenes_data_engine` and `requests` are absent from `sys.modules`, plus a
  documented bare-venv smoke command in `docs/DEMO.md`.

**Amendment (2026-08-14):** `results_table` was not built (Overview needed no
table — its four headline results render as metric cards, and the two per-arm
comparison tables are deferred to the pages that actually need them, Phases 3/7);
`recorded_banner` was added instead (`render.py`), unused by Overview but there for
Phase-8 (chat replay gallery) reuse.

## 4. Testing

- Exporter unit tests on tmp fixtures: overview numbers derived (not literal),
  determinism (two builds byte-identical given same inputs), manifest validation
  failure modes (missing input; wrong flagship count; size over budget), thumbs
  exporter against a tmp LanceDB table (importorskip lancedb), al/weaksup reshape
  correctness against a miniature results.json.
- The no-`src/`-import test above.
- CI stays torch-free and network-free. `uv run pytest -q`, `ruff check .`, bare
  `mypy` all green.
- Manual: `uv run nuscenes-data-engine demo build` against the real artifacts, then
  `streamlit run app/demo/main.py` — Overview renders every number from
  `demo_data/`, stubs navigate.

## 5. Explicitly out of scope (later phases)

Curated frames/predictions/crops (2); Failure Explorer + overlay renderer (3);
chat streaming/charts/probe fix (4); scenario events + event viewer + semantic
gallery (5); subgraph export + agraph (6); AL/weak-sup pages (7); chat replays,
licensing gate, deployment (8). No docker-compose changes; no changes to the
existing `app/streamlit_app.py`.

## 6. New docs

`docs/DEMO.md` started in this phase: the builder runbook (`demo build`), the
package layout, the app-run command, and the bare-venv smoke check. Extended each
phase.
