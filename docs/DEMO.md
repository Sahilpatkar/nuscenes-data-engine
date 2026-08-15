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
  SQL/Cypher parity number),
- the documented weak-retention headline (the `random` pair) is missing from
  `results.json`, or
- the package exceeds the size budget (100 MB; currently 0.49 MB).

`demo build` only runs where the local pipeline artifacts already exist —
`data/processed/`, `data/active_learning/`, and `mlruns/` are all gitignored, so a
fresh clone cannot rebuild the package (those come from running the ingestion and
active-learning phases locally, not from git). What a fresh clone *can* always do is
verify the committed `demo_data/` against `manifest.json`'s SHA256s without rebuilding
anything.

Every number in the package is **derived at export time**, never hardcoded: dataset
scale = live parquet row counts; CAN-speed validation r = live correlation of the
wheel-speed signal (`can_speed_kmh/3.6`) against GT ego-pose speed; weak-supervision
retention is computed per weak/GT arm pair and published with explicit arm
attribution (headline = the documented 18% `random` pair; the 39.4%
`graph_rate_night` pair is carried separately). The flagship *Cypher* twin is
sourced from [GRAPH.md](GRAPH.md) until Phase 6 computes it against a live graph —
`overview_metrics.json` labels it as sourced.

Rebuilds are deterministic: identical inputs produce byte-identical outputs —
`manifest.json`'s `built_at` and `git_sha` are the only fields that vary run-to-run
(`git_sha` whenever HEAD moves between builds). `git_sha` is necessarily the *parent*
commit: a package can't contain the sha of the commit that adds it, so the committed
manifest always names the commit it was built from, not the one that carries it.

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
`lancedb`, `neo4j`, or `duckdb` — enforced by an AST test (`tests/test_demo_app.py`)
that parses every `app/demo/*.py` file's imports and rejects those names; it's a
contributor guardrail, not a sandbox (a dynamic `importlib` call would slip past it).
Strict mypy also covers `app/demo` (see `[tool.mypy] files` in `pyproject.toml`) —
that's real type-checking of the app code, not an import-policing mechanism. Its full
dependency set is [app/demo/requirements.txt](../app/demo/requirements.txt)
(streamlit, pandas, pyarrow, pillow). Bare-venv smoke check:

```bash
uv venv "$TMPDIR/demo-venv" --python 3.11
VIRTUAL_ENV="$TMPDIR/demo-venv" uv pip install -r app/demo/requirements.txt
cd app/demo
"$TMPDIR/demo-venv/bin/python" -c "import data, render; print('data/render import clean')"
# views/ need a live Streamlit runtime to import (st.cache_data etc.), so the honest
# static check there is that every file at least parses.
"$TMPDIR/demo-venv/bin/python" -c "import ast, pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('.').rglob('*.py')]; print('all files parse clean')"
cd ../..
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
