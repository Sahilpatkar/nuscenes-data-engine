# Public demo

A self-contained Streamlit app ([app/demo/](../app/demo/)) that presents the project's
real results from a committed artifact package — no backend, no GPU, no databases.
The full plan is [DEMO_PLAN.md](DEMO_PLAN.md); the build lands in phases, each its own
spec/plan/branch cycle.

## Run it

```bash
uv run streamlit run app/demo/main.py
```

The app reads only `demo_data/`. If the package is missing, the app says so and
points at the builder.

## The artifact builder

```bash
uv run nuscenes-data-engine demo build          # parameters: configs/demo.yaml
```

`demo build` **wipes and fully regenerates** `demo_data/` from local artifacts, then
writes `manifest.json` (git sha anchored to this repo, SHA256 of every input and
output, row counts). It **fails loudly** — no manifest is written — if:

- a declared input is missing,
- the flagship hard-braking-near-pedestrians SQL count ≠ 30 (the documented
  SQL/Cypher parity number), or
- the package exceeds the size budget (100 MB; currently 0.49 MB).

Every number in the package is **derived at export time**, never hardcoded: dataset
scale = live parquet row counts; CAN-speed validation r = live correlation of the
wheel-speed signal (`can_speed_kmh/3.6`) against GT ego-pose speed; weak-supervision
retention is computed per weak/GT arm pair and published with explicit arm
attribution (headline = the documented 18% `random` pair; the 39.4%
`graph_rate_night` pair is carried separately). The flagship *Cypher* twin is
sourced from [GRAPH.md](GRAPH.md) until Phase 6 computes it against a live graph —
`overview_metrics.json` labels it as sourced.

Rebuilds are deterministic: identical inputs produce byte-identical outputs (only
`manifest.json`'s `built_at` differs — expect that one-line diff on any rebuild).

## Package layout (Phase 1)

| file | contents |
|---|---|
| `manifest.json` | provenance: git sha, input/output SHA256s, counts, validation |
| `overview_metrics.json` | scale, headline results, flagship parity, CAN r |
| `active_learning_results.parquet` | all 13 arms: overall/night mAP50-95 + deltas |
| `weak_supervision_results.parquet` | per-arm verifier retention + box stats |
| `sample_frames/hero.jpg` | interim hero (baseline val-batch mosaic; real overlay in Phase 3) |

## Streamlit-Cloud contract

The demo app never imports `src/nuscenes_data_engine`, `requests`, `torch`,
`lancedb`, `neo4j`, or `duckdb` — enforced by an AST test
(`tests/test_demo_app.py`) and by strict mypy (the app is inside the repo's mypy
scope). Its full dependency set is [app/demo/requirements.txt](../app/demo/requirements.txt)
(streamlit, pandas, pyarrow, pillow). Bare-venv smoke check:

```bash
uv venv "$TMPDIR/demo-venv" --python 3.11
VIRTUAL_ENV="$TMPDIR/demo-venv" uv pip install -r app/demo/requirements.txt
cd app/demo && "$TMPDIR/demo-venv/bin/python" -c "import data, render; print('clean')" && cd ../..
rm -rf "$TMPDIR/demo-venv"
```

## Phase status

| phase | scope | status |
|---|---|---|
| 1 | builder core, `demo_data/` v0.1, app shell + Overview | **shipped** |
| 2 | curated frames, TRINITY rsync, local inference, predictions | pending |
| 3 | Failure Explorer + GT/pred overlays | pending |
| 4 | chat upgrades (local stack): probe fix, streaming, charts | pending |
| 5 | scenario search + synchronized event viewer | pending |
| 6 | interactive graph (subgraph export + agraph) | pending |
| 7 | active-learning + weak-supervision pages | pending |
| 8 | chat replay gallery, licensing gate, deployment | pending |
