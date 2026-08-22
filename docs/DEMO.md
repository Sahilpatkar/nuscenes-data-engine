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
  `results.json`,
- a staged `graph_subgraphs/` group is partial (some of the six presets missing),
  stale (a preset's event keys differ from the events this build ranks — re-run
  `demo subgraphs`), or its flagship Cypher count ≠ the flagship SQL count,
- an `al.exemplar_tokens` entry is not a val frame with a confident `graph_rate_night`
  detection that `baseline` missed or only low-conf claimed (or the list is empty while
  curation is present),
- `weak_loss_decomposition.retention` disagrees with `overview_metrics.json`'s per-pair
  weak retention, the two community diagnostics files disagree, or a quota column does
  not sum to `n_mine`,
- a staged `al_explain/` group is partial, belongs to another arm, or did not record
  `selected_match` and `communities_match` as true, or
- the package exceeds the size budget (100 MB; currently 25.43 MB).

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
computed against the live Neo4j graph by `demo subgraphs` (the query text is
asserted verbatim against [GRAPH.md](GRAPH.md)); `overview_metrics.json` records
`cypher_source: "computed (neo4j, demo subgraphs)"` when the subgraph group is
included, and falls back to the GRAPH.md value — labelled as sourced — when it is
absent.

Rebuilds are deterministic: identical inputs produce byte-identical outputs —
`manifest.json`'s `built_at` and `git_sha` are the only fields that vary run-to-run
(`git_sha` whenever HEAD moves between builds). `git_sha` is necessarily the *parent*
commit: a package can't contain the sha of the commit that adds it, so the committed
manifest always names the commit it was built from, not the one that carries it.

## Package layout (Phases 1-8)

| file | contents |
|---|---|
| `manifest.json` | provenance: git sha, `package_version`, input/output SHA256s, counts, validation |
| `overview_metrics.json` | scale, headline results, flagship parity, CAN r |
| `active_learning_results.parquet` | all 13 arms: overall/night mAP50-95 + deltas, plus (Phase 7) round order, family, mAP50, night precision/recall, night pedestrian mAP, day/rain/clear slices, `val_images`, mined-set composition (`n_scenes`, `night_share`, `rain_share`) |
| `weak_supervision_results.parquet` | per-arm verifier retention + box stats, plus (Phase 7) rejected-side and candidate-pool GT boxes/frame, tolerance, conf, n_unparsed |
| `al_communities.parquet` | 97 Louvain communities: size, night members, routed failure mass, quota under `graph_rate` and `graph_rate_night` (`is_backfill` sentinel flag) |
| `al_selection_explain.parquet` + `al_explain_validation.json` | optional (`demo al-explain`): per selected frame — community, its size / night members / mass rank / quota, degree rank, night-pass vs main-pass pick, routed failures; the validation record proving the run reproduced |
| `al_exemplars.json` | hand-approved before/after tokens (validated at build) |
| `weak_loss_decomposition.parquet` | per base arm: GT gain, dropped-frame cost/share, label cost/share, retention (headline flag) |
| `weak_verifier_by_class.parquet` | per arm × class: rejected disagreements, accepted mutual-zero count/share |
| `weak_labels.parquet` | verified pseudo boxes (baseline-detector proposals) for curated tokens, native 1600×900 |
| `vlm_counts.parquet` | per curated frame with a weak verdict: the VLM's per-class count vote, parse status, label confidence, scene call; GT counts alongside |
| `sample_frames/hero.jpg` | hand-picked night exemplar crop (`hero_token` in overview_metrics; baseline misses a shadowed car AND a pedestrian — graph_rate_night recovers only the pedestrian, a low-confidence hit; the car defeats all three models) — rendered live with overlays on Overview |
| `frame_manifest.parquet` | 250 curated frames: buckets, val/train_pool split, failure stats, per-model prediction counts (`n_preds_<model>`, 0 = ran-and-found-nothing), exemplar flags (`fixes_fn_vs_<a>_<b>` — b fixes a's misses: True where model a has an FN that model b matched), `weak_verdict` (accepted / rejected for frames in the weak arm's candidate pool, else NA) and `al_selected_by` (the AL arm that selected the frame, else NA) |
| `gt_boxes.parquet` | GT boxes (1600×900 coords) + per-model `matched_<model>` flags (NA = not evaluated) + `distance_to_ego_m` + `size_bucket` (COCO 32²/96²) + `below_visibility_min` (all-False today; parity-defensive) |
| `predictions.parquet` | 3,758 predictions × 3 models (baseline/graph_rate_night @640, champion @960 — per-row `imgsz`), status ∈ tp/fp/low_conf matched with the AL sweep's exact semantics |
| `sample_frames/crops/` | 250 × 960×540 crops (0.6 scale of native) |
| `sample_frames/thumbs/` | 659 × 256×144 LanceDB thumbnails (250 curated + 367 for events, filmstrip neighbors, semsearch + 42 frames retrieved in the recorded chat session) |
| `scenario_events.parquet` | 126 preset-tagged keyframes (6 presets, capped 30 each): ego dynamics, per-class min distances, `preset_tags`, `preset_rank_<name>`, t−2…t+2 filmstrip neighbor tokens + readouts, `in_curated_set` |
| `semantic_search_results.parquet` | recorded SigLIP results: 4 canned queries × up to 8 front-camera hits (query, rank, sample_data_token, score, k — k drives the gallery's "n of k" caption) |
| `graph_subgraphs/<preset>.json` | 6 presets × per-event subgraphs from the live graph: nodes (`on_path` flag + properties), edges, the matched path (nearest matching object first), off-path observations capped at the nearest 12; plus the preset's count Cypher and `sql_count` / `cypher_count` / `parity` on the full keyframe population (model presets: `cypher_count` null — their verdict comes from predictions, not the graph) |
| `chat_replays.json` | recorded chat session (`demo chat-record`): per replay `{id, kind (eval/showcase), question, answer, model, provider, steps [{tool, input, output}], frames (7 projected columns, no thumbnail bytes), charts, checks (graded cases only), latency_s, error}` |
| `chat_replay_summary.json` | that recording's provenance: model, provider, `recorded_at`, `git_sha`, `n_eval`, `n_passed`, `n_showcase`, `search_available`, `graph_available`, `max_turns` |

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
(streamlit ≥ 1.59 — `st.altair_chart(width="stretch")` — pandas, pyarrow, pillow,
altair (already a Streamlit dependency, declared explicitly because the Phase-7 pages
import it), streamlit-agraph — the latter imported lazily inside the graph panel, so
its absence degrades to a warning, not a crash). That file is also the one
Community Cloud installs: Cloud searches the **entrypoint's directory first** and only
then the repository root, taking the first of `uv.lock`, `Pipfile`, `environment.yml`,
`requirements.txt`, `pyproject.toml` it finds — and `app/demo/requirements.txt` sits
next to `app/demo/main.py`. The repo's root `uv.lock` (the full project, torch
included) is therefore never consulted, and there is deliberately **no root
`requirements.txt`**. Both halves of that are pinned by
`test_streamlit_config_and_requirements_for_cloud` in `tests/test_demo_app.py`.
Bare-venv smoke check:

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
near pedestrians — the flagship, asserted at exactly 30 during `demo build`; night
pedestrians; fast cyclists; rain VRUs — each header carries its own SQL/Cypher
parity line read from `graph_subgraphs/<preset>.json`, stating the full-population
count and, only when the 30-card cap actually cuts it down, how many of the 30 are
shown — the flagship's own 30-of-30 has nothing to cut) and **model-result
presets** over the 125 curated val frames only, because that is where predictions
exist (false-negative pedestrians at night: 4; low-confidence detections during
braking: 2 — the scope is stated on the page; the N is the card count). Each
preset caps at 30 ranked by its own severity (cards show that quantity; braking
is only called braking when the acceleration is negative).
The **event viewer** renders curated events through the Phase-3 overlay renderer and
non-curated ones as GT-only thumbnails with an explicit "not in the curated
prediction set" caption; ego/context/model panels; and a t−2…t+2 filmstrip (strip +
slider + per-step speed/accel readout). The **semantic gallery** is recorded
(`demo semsearch`, SigLIP offline, `semsearch.oversample: 16` to survive the 5/6
non-front-camera store) and labelled as such; per-query hit counts are shown.

## Interactive graph (Phase 6)

One-time per events change, with the Neo4j graph built per [GRAPH.md](GRAPH.md):

```bash
docker compose up -d neo4j
uv run nuscenes-data-engine demo subgraphs   # -> data/demo_curation/graph_subgraphs/<preset>.json
uv run nuscenes-data-engine demo build       # package v0.5 with the graph group
```

`demo subgraphs` recomputes the six presets' events, runs each preset's **count
Cypher** against the live graph (the flagship's is the GRAPH.md query verbatim,
asserted by a test; a flagship mismatch fails the export), records
`sql_count`/`cypher_count`/`parity` per dynamics preset (shipped run: 30/30,
786/786, 92/92, 766/766; a mismatch is recorded, not hidden, and the page renders
it as a warning), and exports one subgraph per ranked event: the matched path
(Scene → Sample → EgoPose → matching objects → Category), the t±1 `NEXT`
neighbours, the Location, and the nearest 12 other observations as context. Model-
result presets get GT-only subgraphs with `cypher_count: null` — their verdict is
not graph-derivable and the page says so.

The **panel** (event viewer, `app/demo/views/scenarios.py`) renders that subgraph
with `streamlit-agraph`: orange = the matched path, grey = context, semantic
labels (`adult 6.2 m`, `ego -4.5 m/s²`, `scene-1084`), force-directed layout;
clicking a node shows its properties, otherwise a one-line narrative of the path
(nearest matching object first, so it agrees with the card's caption; a
model-result preset's path stops at the EgoPose, since its verdict isn't in the
graph). Without a staged `graph_subgraphs/`, `demo build` records
`validation.subgraphs: "absent"`, the page shows an honest "not included in this
package" note, and Overview keeps the sourced Cypher value.

## Active Learning + Weak Supervision (Phase 7)

Both pages read only the package. One optional step re-derives the per-frame
selection facts the AL run never persisted, with the Neo4j graph built per
[GRAPH.md](GRAPH.md):

```bash
docker compose up -d neo4j
uv run nuscenes-data-engine demo al-explain   # -> data/demo_curation/al_explain/ (~90 s)
uv run nuscenes-data-engine demo build        # package v0.6
```

`demo al-explain` re-runs the `graph_rate_night` selection with the experiment's own
functions and config (`configs/active_learning.yaml`: Louvain over the `SIMILAR_TO`
projection with GDS `concurrency: 1`, failure-mass routing through LanceDB, then
`select_by_mass` with the 375-frame night floor) and **stages nothing unless it
reproduces the run exactly** — the 1,500 selected tokens must equal
`data/active_learning/graph_rate_night.parquet` and the 97-community table must equal
`communities_graph_rate_night.json` row for row. Reproduction is the only evidence that
the per-frame facts are the experiment's, so there is no approximate mode. The shipped
run reproduced (`al_explain_validation.json`: `selected_match`, `communities_match`,
GDS 2.13.11, passes `{night: 375, main: 1125}`).

**Active Learning page** (`app/demo/views/active_learning.py`): the experiment story
(Problem → Hypothesis → Acquisition → Training → Evaluation → Result, every number from
the table); Δnight / Δoverall for all 13 arms in round order (`graph_rate_night`
highlighted, weak arms greyed; the table's best *overall* arm is `graph`, and the
yolov8m checkpoint the demo calls "champion (yolov8m @960)" is a different, model-size
champion); how `graph_rate_night` chooses (community failure mass → quota; the all-night
community's quota 83 → 323 is computed from `al_communities.parquet`); a "why was this
frame selected?" gallery over the arm's curated frames (night first, paged) whose
factor panel states the real mechanism — community mass → quota, then
similarity-degree rank within the community; routed mass per frame is context only,
because most selected frames received none; and a before/after section over
hand-approved exemplars (`configs/demo.yaml` `al.exemplar_tokens`, validated at build:
each must have a visible GT box where `graph_rate_night` reaches a confident detection
`baseline` missed or only claimed at low confidence) with a per-box table generated
from `predictions.parquet`. Without the `al_explain` group the gallery still renders
and says per-frame community and routed mass are not in the package.

**Weak Supervision page** (`app/demo/views/weak_supervision.py`): the documented 18%
headline (`random` pair) next to the 39.4% `graph_rate_night` pair, verifier retention,
the night-pedestrian collapse (0.117 → 0.020; "worst night result of the 13 arms" is
computed, not quoted), the loss split (≈50% dropped frames / ≈32% label noise / 18%
retained, recomputed from `active_learning_results.parquet`), the crowded-frame bias
(accepted 3.87 vs rejected **7.61** GT boxes/frame; 3.41 vs 6.68 for the night arm —
the rejected side is a new export computed over the five detector classes in
`annotations.parquet`; `samples.n_boxes` counts every class and gives 9.94), the
verifier's by-class rejections and mutual-zero rates, and accepted / rejected galleries.
The blue boxes there are **pseudo labels: baseline-detector proposals at conf ≥ 0.5,
kept because the VLM's per-class counts agreed within ±1** — the VLM emits counts,
never boxes, so the number on a box is the detector's confidence
([DEMO_PLAN.md](DEMO_PLAN.md)'s "VLM-generated labels" is shorthand the page
deliberately corrects). Each frame shows the VLM's count vote against GT; the verifier
itself compared those counts to the detector's, which the package does not carry.
`vlm_counts.parquet` merges both label tables the run used
(`data/autolabel/labels.parquet` + `data/active_learning/autolabel_weak/labels.parquet`)
and covers every curated frame with a verdict.

## Ask the Dataset (recorded) (Phase 8)

The public app never calls an LLM. `app/demo/views/chat_replay.py` replays **one
recorded session** with the real tool-calling chat agent
([DATASET_CHAT.md](DATASET_CHAT.md)), exported into the package. Recording is a
paid, one-off run against a live provider:

```bash
docker compose up -d neo4j                    # so the agent is offered run_cypher
# dry run first — --limit caps the TOTAL questions, showcase first, so 5 = exactly
# the five showcase questions (the ones whose `expect` is asserted):
uv run nuscenes-data-engine demo chat-record --provider anthropic --limit 5
uv run nuscenes-data-engine demo chat-record --provider anthropic   # full run
uv run nuscenes-data-engine demo build                              # package v0.7
```

The full run is ≈25 Claude answers — the graded eval cases plus the five showcase
questions — at **≈ $4 at Opus list price**. It stages
`data/demo_curation/chat_replays/`; `demo build` validates and copies both files into
`demo_data/` and exports a thumbnail for every retrieved frame.

**The shipped recording** (`chat_replay_summary.json`, 2026-08-21, `claude-opus-4-8`):
25 replays — 5 showcase (chart, Cypher, semantic search, frames, slices; every
expectation met) and 20 graded, **18/20 passed**; the two misses
(`labels_parse_ok_count`, `max_instance_keyframes`) failed the `grounded` check, the
same failure mode the earlier documented run had, and are shown on the page as such.
42 distinct frames were retrieved, 210 s of agent time, no errors. It was recorded
twice: the first run (19/20) exposed a Phase-4 bug in the agent's step summary (every
successful `make_chart` step read "repeat (skipped)"), fixed in this phase, and the
run was repeated so the shipped steps read "charted: …" as they should — roughly
$8 of Claude API in total. `docs/DATASET_CHAT.md`'s 17/20 is the earlier
pre-registered run of the same 20 cases (re-graded under grounding v2); this is a
fresh session, so the two figures are different runs, not a contradiction.

**What gets recorded.** Two question sets, answered in one session:

- **Graded** — the cases of `configs/chat_eval.yaml`, run and graded by the chat-eval
  harness's *own* `grade_case`, so the `checks` dict on the page (`english`,
  `tool_use`, `grounded`, `numeric`/`expected`, `frames`) is the harness's verdict,
  not the demo's. The **stored question is the one the model was actually sent** — a
  config question reworded after a recording would otherwise show a viewer a question
  the model never saw.
- **Showcase** — `configs/demo.yaml` `chat_replay.showcase`: five hand-written
  questions, each declaring an `expect` (`chart` / `cypher` / `frames` / `search` /
  `none`) that the recorder **asserts** against the result. A missed expectation
  fails the whole run: a dud showcase is re-worded and re-recorded, never shipped.

One replay is `{id, kind, question, answer, model, provider, steps, frames, charts,
checks, latency_s, error}` — the frames keep seven projected columns and drop the
thumbnail bytes (the package ships the JPEGs as ordinary 256×144 thumbs instead). A
question the provider errors on is recorded with its `error` and the run continues.

**What the page shows, and its honesty rules.** Header cards (model, questions
recorded, graded pass rate, tools exercised), then every showcase replay in full —
answer, charts, retrieved-frame gallery, and an "Agent steps" expander with the SQL
and Cypher it ran — then a selectbox over the graded cases labelled ✓ / ✗. The rules
the page is built around:

- Nothing is generated at view time. The word-by-word reveal of an answer is
  **cosmetic**; the recording stores finished text, not a token stream. The first
  showcase answer types out once per browser session; every other answer renders
  statically.
- **Raw SQL result rows are not stored.** A step shows the query and its row count —
  exactly what the live chat UI shows — and the page never reconstructs a table the
  package does not carry.
- **Failed cases are shown, not hidden**: a ✗ names the check it failed and the
  reference value it was graded against ("reference 66").
- The summary's `search_available` / `graph_available` flags are stated on the page
  when false, so a recording made without the search engine or the graph says so
  rather than looking like an agent that chose not to use them.
- No recording in the package (a fresh clone, or any pre-0.7 package) is an ordinary
  state, not an error: the page says "no recorded sessions in this package".

## Deploy (Streamlit Community Cloud)

The demo is a static-artifact app: no secrets, no backend, no GPU. Deploying it is a
form, filled in once.

1. Sign in with GitHub at [share.streamlit.io](https://share.streamlit.io).
2. **New app** → **Deploy a public app from GitHub**.
3. Repository `Sahilpatkar/nuscenes-data-engine`, branch `main`, **Main file path**
   `app/demo/main.py`.
4. **Advanced settings** → **Python version 3.11** (matches `.python-version`; Cloud
   defaults to 3.12). Leave **Secrets** empty — the app reads nothing from
   `st.secrets`.
5. Deploy, then paste the resulting URL into the README's **Live demo** line and into
   the "Live URL" note at the end of this section.

**Which dependency file Cloud installs.** Community Cloud searches the **entrypoint's
directory first** and only then the repository root, taking the first of `uv.lock`,
`Pipfile`, `environment.yml`, `requirements.txt`, `pyproject.toml` it finds — and it
uses **one** dependency file, not a merge of several. `app/demo/requirements.txt` sits
next to `app/demo/main.py`, so that is the file Cloud installs (six light wheels; see
"Streamlit-Cloud contract" above). The repo's root `uv.lock` — the full project,
torch included — is never consulted while that file exists, which is exactly why the
test suite asserts its presence *and* asserts that no root `requirements.txt` was
added (a second dependency file Cloud would never read).

**Footprint.** Cloud clones the whole repository (≈55 MB, of which `demo_data/` is
25 MB) and installs the six wheels above; nothing else is downloaded at runtime. The
loaders in `app/demo/data.py` are `st.cache_data`-wrapped, and the biggest table in
memory is a frame parquet of a few MB — comfortably inside the free tier's 1 GB.
Expect a cold start of roughly 30 s (clone + pip install + first script run), then
sub-second page switches.

**Redeploying.** Cloud watches the deployed branch: pushing to `main` redeploys.
After a package rebuild, remember that `manifest.json`'s `git_sha` names the *parent*
commit by design (a package cannot contain the sha of the commit that adds it) — that
is not a stale deploy.

**What the deployment claims.** Overview's footer renders
[DEMO_PLAN.md](DEMO_PLAN.md)'s credibility statement verbatim — *"Results shown here
were generated by the full offline pipeline. The public application serves curated
experiment outputs for reproducibility and demonstration."* — which is the honest
description of this deployment: it serves artifacts, it does not run the pipeline.

**Live URL:** _pending deploy_ (record it here and in README.md's "Live demo" line).

### Screenshots

The README gallery's six PNGs are produced by `scripts/demo_screenshots.py`, a manual
Playwright tool that is deliberately **not** a project dependency:

```bash
uv pip install playwright                     # temporary, into .venv
uv run streamlit run app/demo/main.py --server.headless true --server.port 8599  # another shell
.venv/bin/python scripts/demo_screenshots.py  # -> docs/img/demo-*.png
uv pip uninstall playwright
```

It drives the machine's installed Google Chrome (`channel="chrome"`, so no browser
download), shoots a 1200×900 viewport per page, opens the first frame-detail panel
where a page has one — the Active learning and Weak supervision pages have none, so
those two are captured at the top of the page (scroll position 0, title visible)
instead — and **exits non-zero if any page rendered a Streamlit exception** — a
broken page cannot quietly become a README screenshot. Re-run it whenever a page
changes visibly; any capture over 300 KB is quantized to a 256-colour palette PNG by
the script itself, no manual compression step needed.

These PNGs contain nuScenes-derived imagery and are covered by the attribution
section below, exactly as the packaged frames are.

## Success-criteria walk

[DEMO_PLAN.md](DEMO_PLAN.md)'s ten questions, and the page and on-screen section that
answers each. Status is filled in when the walk is actually performed: the local
column after a browser pass over the committed package, the live column against the
deployed URL.

| # | Question (DEMO_PLAN.md) | Page | Section / element that answers it | Status |
|---|---|---|---|---|
| 1 | What problem does the project solve? | Overview | the mission blockquote under the title, then **Scale** and **Headline results** cards | local: 2026-08-21 · live: pending |
| 2 | Where does the baseline perception model fail? | Failure Explorer (+ Overview) | the sidebar filters (lighting, rain, model, class, size bucket, distance, failure type, curation bucket) over the 125 val frames, and **Detail**'s GT / Predictions / Overlay toggle; Overview's **The model at work** hero shows one such miss | local: 2026-08-21 · live: pending |
| 3 | How does the system find difficult data? | Scenario Search | the six **preset** buttons + ranked card grid, and **Recorded semantic search** | local: 2026-08-21 · live: pending |
| 4 | Why is the graph useful? | Scenario Search | each preset header's SQL/Cypher **parity line** (flagship 30 = 30), and the event viewer's **Interactive graph** panel with the matched path | local: 2026-08-21 · live: pending |
| 5 | What does CAN-bus data add? | Scenario Search (+ Overview) | the event viewer's ego panel and t−2…t+2 **filmstrip** with per-step speed/accel readouts; Overview's **CAN speed vs ego-motion** card (r) | local: 2026-08-21 · live: pending |
| 6 | How does active learning choose frames? | Active Learning | **How `graph_rate_night` chooses: community mass → quota**, then **Why was this frame selected?** per-frame factor panel | local: 2026-08-21 · live: pending |
| 7 | Did targeted retraining improve performance? | Active Learning | the story's **Result** beat, **Every arm, one chart** (13 arms in round order), and **Before / after** exemplars | local: 2026-08-21 · live: pending |
| 8 | How well did VLM-generated supervision work? | Weak Supervision | the retention cards (GT gain retained + verifier retention) and **What the VLM saw** accepted/rejected galleries | local: 2026-08-21 · live: pending |
| 9 | Why did weak supervision underperform GT? | Weak Supervision | **Where the rest of the gain went** (dropped-frame cost vs label cost) and **What the verifier's rule selects for** (the crowding bias) | local: 2026-08-21 · live: pending |
| 10 | How does the project form a closed model-improvement loop? | Active Learning + Weak Supervision (+ Overview) | the two story-arrow narratives — Problem → Hypothesis → Acquisition → Training → Evaluation → Result, and Hypothesis → Labelling → Verification → Training → Result — closed by Overview's footer credibility statement | local: 2026-08-21 · live: pending |

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
| 4 | chat upgrades (local stack): probe fix, streaming, charts | **shipped** |
| 5 | scenario search + synchronized event viewer | **shipped** |
| 6 | interactive graph (subgraph export + agraph) | **shipped** |
| 7 | active-learning + weak-supervision pages | **shipped** |
| 8 | recorded chat replay, deployment scaffolding, README + screenshots | **shipped** |
