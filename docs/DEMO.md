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
- the package exceeds the size budget (100 MB; currently 22.98 MB).

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

## Package layout (Phases 1-5)

| file | contents |
|---|---|
| `manifest.json` | provenance: git sha, `package_version`, input/output SHA256s, counts, validation |
| `overview_metrics.json` | scale, headline results, flagship parity, CAN r |
| `active_learning_results.parquet` | all 13 arms: overall/night mAP50-95 + deltas |
| `weak_supervision_results.parquet` | per-arm verifier retention + box stats |
| `sample_frames/hero.jpg` | hand-picked night exemplar crop (`hero_token` in overview_metrics; baseline misses a shadowed car AND a pedestrian — graph_rate_night recovers only the pedestrian, a low-confidence hit; the car defeats all three models) — rendered live with overlays on Overview |
| `frame_manifest.parquet` | 250 curated frames: buckets, val/train_pool split, failure stats, per-model prediction counts (`n_preds_<model>`, 0 = ran-and-found-nothing), exemplar flags (`fixes_fn_vs_<a>_<b>` — b fixes a's misses: True where model a has an FN that model b matched) |
| `gt_boxes.parquet` | GT boxes (1600×900 coords) + per-model `matched_<model>` flags (NA = not evaluated) + `distance_to_ego_m` + `size_bucket` (COCO 32²/96²) + `below_visibility_min` (all-False today; parity-defensive) |
| `predictions.parquet` | 3,758 predictions × 3 models (baseline/graph_rate_night @640, champion @960 — per-row `imgsz`), status ∈ tp/fp/low_conf matched with the AL sweep's exact semantics |
| `sample_frames/crops/` | 250 × 960×540 crops (0.6 scale of native) |
| `sample_frames/thumbs/` | 617 × 256×144 LanceDB thumbnails (250 curated + 367 for events, filmstrip neighbors, semsearch) |
| `scenario_events.parquet` | 126 preset-tagged keyframes (6 presets, capped 30 each): ego dynamics, per-class min distances, `preset_tags`, `preset_rank_<name>`, t−2…t+2 filmstrip neighbor tokens + readouts, `in_curated_set` |
| `semantic_search_results.parquet` | recorded SigLIP results: 4 canned queries × up to 8 front-camera hits (query, rank, sample_data_token, score, k — k drives the gallery's "n of k" caption) |

## Picking the hero token

The Overview page's hero image is a hand-picked exemplar crop, not a mosaic or an
algorithmically-chosen frame. To change it: pick a `sample_data_token` from
`frame_manifest.parquet`'s val rows with `fixes_fn_vs_baseline_graph_rate_night ==
True` (a frame where `graph_rate_night` catches a false negative `baseline`
misses), ideally with 2-6 GT boxes for legibility, set `configs/demo.yaml`
`hero.token` to it, and rebuild. `demo build` fails loudly on a null token, but
only once curation is included — a null token is the ordinary, expected state on
a fresh clone (`data/demo_curation/` is gitignored) or before `demo curate` +
`demo infer` have run; a *non-null* token with curation absent is still an error
(the config references curated data that isn't there).

The hero's caption (`app/demo/views/overview.py`'s `_HERO_CAPTION`) is a fixed
string describing what THIS SPECIFIC token's boxes actually show — it is not
generated from the data. Picking a different token means rewriting that string to
match its ground truth, or the caption will describe a frame the viewer isn't
looking at (this is exactly the bug the final Phase 3 review round caught and
fixed: the caption claimed a single miss the retrain catches, when the real frame
has two misses and the retrain only catches one of them).

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
distance-to-ego, failure type, curation bucket — multi-select, any-overlap: a
frame matches if any of its own buckets is selected) over the 125 **val** frames
only (train_pool frames carry no predictions by design; later pages use them).
The filtered grid can be sorted by failure count (n_fn+n_fp+n_low_conf for the
selected model, worst first), distance to the nearest GT box (nearest first, a
zero-GT frame sorts last), or scene name. The GT/Predictions/Overlay toggle
renders through the shared `draw_overlay` visual language: GT green, misses
orange-dashed, FPs red, low-confidence yellow-dotted. FN here means what
`failures.parquet` means: GT unmatched by the selected model, with low-confidence
claims counting as matches. Leaving the distance slider at full extent applies no
distance filter (narrowing it excludes zero-GT frames — stated in the widget's
help). Exemplar badges credit the model that *catches* a box another model
missed.

## Scenario Search (Phase 5)

`app/demo/views/scenarios.py` — six preset scenario queries in two honestly-separated
families: **dynamics presets** over all 34,149 keyframes from GT alone (hard braking
near pedestrians — the flagship, asserted at exactly 30 during `demo build` and
shown with the SQL/Cypher parity badge read from `overview_metrics.json`; night
pedestrians; fast cyclists; rain VRUs) and **model-result presets** over the 125
curated val frames only, because that is where predictions exist (false-negative
pedestrians at night: 4; low-confidence detections during braking: 2 — the
scope is stated on the page; the N is the card count). Each preset caps at 30 ranked by its own severity (cards show
that quantity; braking is only called braking when the acceleration is negative).
The **event viewer** renders curated events through the Phase-3 overlay renderer and
non-curated ones as GT-only thumbnails with an explicit "not in the curated
prediction set" caption; ego/context/model panels; and a t−2…t+2 filmstrip (strip +
slider + per-step speed/accel readout). The **semantic gallery** is recorded
(`demo semsearch`, SigLIP offline, `semsearch.oversample: 16` to survive the 5/6
non-front-camera store) and labelled as such; per-query hit counts are shown.
The interactive graph traversal slot on this page lands in Phase 6.

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
| 5 | scenario search + synchronized event viewer | **shipped** |
| 6 | interactive graph (subgraph export + agraph) | pending |
| 7 | active-learning + weak-supervision pages | pending |
| 8 | chat replay gallery, licensing gate, deployment | pending |
