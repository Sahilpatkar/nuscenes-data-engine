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
- the package exceeds the size budget (100 MB; currently 20.06 MB).

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

## Package layout (Phases 1-2)

| file | contents |
|---|---|
| `manifest.json` | provenance: git sha, input/output SHA256s, counts, validation |
| `overview_metrics.json` | scale, headline results, flagship parity, CAN r |
| `active_learning_results.parquet` | all 13 arms: overall/night mAP50-95 + deltas |
| `weak_supervision_results.parquet` | per-arm verifier retention + box stats |
| `sample_frames/hero.jpg` | hand-picked night exemplar crop (`hero_token` in overview_metrics; baseline misses a shadowed car + pedestrian that graph_rate_night catches) — rendered live with overlays on Overview |
| `frame_manifest.parquet` | 250 curated frames: buckets, val/train_pool split, failure stats, per-model prediction counts (`n_preds_<model>`, 0 = ran-and-found-nothing), exemplar flags (`fixes_fn_vs_<a>_<b>` — b fixes a's misses: True where model a has an FN that model b matched) |
| `gt_boxes.parquet` | GT boxes (1600×900 coords) + per-model `matched_<model>` flags (NA = not evaluated) + `distance_to_ego_m` + `size_bucket` (COCO 32²/96²) + `below_visibility_min` (all-False today; parity-defensive) |
| `predictions.parquet` | 3,758 predictions × 3 models (baseline/graph_rate_night @640, champion @960 — per-row `imgsz`), status ∈ tp/fp/low_conf matched with the AL sweep's exact semantics |
| `sample_frames/crops/` | 250 × 960×540 crops (0.6 scale of native) |
| `sample_frames/thumbs/` | 250 × 256×144 LanceDB thumbnails |

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

## Curated frames + predictions (Phase 2)

One-time per curation change (parameters: `configs/demo.yaml` `curation:`):

```bash
uv sync --extra train --extra engine     # torch/ultralytics for inference + SigLIP for the semantic bucket
uv run nuscenes-data-engine demo curate  # 250 tokens -> data/demo_curation/ (+ rsync_filelist.txt)
rsync -a --files-from=data/demo_curation/rsync_filelist.txt \
  TRINITY:/data/ggare/datasets/nuscenes/ data/raw/demo_frames/   # ~44 MB, read-only source
uv run nuscenes-data-engine demo infer   # 3 checkpoints over the val frames (CPU, ~15 min)
uv run nuscenes-data-engine demo build   # package v0.2 with the curation group
```

Facts about the shipped run: 250 frames (125 val / 125 train_pool), all eight buckets
at quota; matching reuses the AL sweep's exact parameters (IoU 0.5, conf_hit 0.4,
conf floor 0.05, visibility ≥ 2) via the property-tested box-level matcher, so
tp/fp/FN here mean what `failures.parquet` means. A val frame with `n_preds_<model>=0`
means the model ran and detected nothing (2 (token, model) pairs on one frame — a
night frame both yolov8n models totally missed). Without TRINITY access, `demo build`
skips the curation group and still produces the Phase-1 package
(`validation.curation: "absent"`).

## Failure Explorer (Phase 3)

`app/demo/views/failures.py` — filters (lighting, rain, model, class, size bucket,
distance-to-ego, failure type, curation bucket) over the 125 **val** frames only
(train_pool frames carry no predictions by design; later pages use them). The
GT/Predictions/Overlay toggle renders through the shared `draw_overlay` visual
language: GT green, misses orange-dashed, FPs red, low-confidence yellow-dotted.
FN here means what `failures.parquet` means: GT unmatched by the selected model,
with low-confidence claims counting as matches. Leaving the distance slider at
full extent applies no distance filter (narrowing it excludes zero-GT frames —
stated in the widget's help). Exemplar badges credit the model that *catches* a
box another model missed.

## Dataset attribution & license

The demo package (`demo_data/sample_frames/`) contains imagery **derived from the
nuScenes dataset** (© nuTonomy / Motional): 960×540 crops, 256×144 thumbnails, and a
validation mosaic. nuScenes is licensed **CC BY-NC-SA 4.0** (non-commercial,
attribution, share-alike); the derived visuals here are shared for non-commercial
research/portfolio demonstration under those same terms, and the underlying dataset
is **not** redistributed by this repository.

> Caesar et al., *"nuScenes: A multimodal dataset for autonomous driving"*,
> CVPR 2020 — https://www.nuscenes.org

## Phase status

| phase | scope | status |
|---|---|---|
| 1 | builder core, `demo_data/` v0.1, app shell + Overview | **shipped** |
| 2 | curated frames, TRINITY rsync, local inference, predictions | **shipped** |
| 3 | Failure Explorer + GT/pred overlays | **shipped** |
| 4 | chat upgrades (local stack): probe fix, streaming, charts | pending |
| 5 | scenario search + synchronized event viewer | pending |
| 6 | interactive graph (subgraph export + agraph) | pending |
| 7 | active-learning + weak-supervision pages | pending |
| 8 | chat replay gallery, licensing gate, deployment | pending |
