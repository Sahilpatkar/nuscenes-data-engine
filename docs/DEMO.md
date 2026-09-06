# Public demo

A self-contained Streamlit app ([app/demo/](../app/demo/)) that presents the project's
real results from a committed artifact package — no backend, no GPU, no databases.
The full plan is [DEMO_PLAN.md](DEMO_PLAN.md); the build lands in phases, each its own
spec/plan/branch cycle.

## Two front-ends, one package

The committed package is read two ways, and neither reading carries a number of its
own:

- **The Streamlit app** (`app/demo/`, the rest of this document) is the **full
  instrument**: every frame, every arm, every scenario event, the interactive graph
  panel and the recorded chat session, with the guided tour as its default path.
- **The story site** (`web/`, Vite + React + TypeScript) is a **designed
  scrollytelling reading** of the same seven-step story — one page, seven sections,
  hand-rolled SVG charts, first-class light and dark themes. It holds no deep-dive
  pages of its own: every deep dive is a labelled deep link back into the live
  Streamlit app.

Every figure the site renders is exported at build time by `web/build_data.py`, which
imports the app's own `filters.py` / `render.py` helpers and writes eight JSON files
(`web/src/data/`) plus eleven pre-rendered overlays and thumbs (`web/public/story/`).
That bundle is committed, and `tests/test_web_export.py` re-runs the exporter and
asserts it comes back **byte-identical** — JSON byte-for-byte, images by name and
dimensions — so a package or helper change that moves a number turns the suite red
instead of drifting the site away from the app. The same tests pin the sentences the
exporter re-derives (the computed winner, the fairness budget clause, the closed-loop
headline guard, the `gain_text` sentence) to the shipped tour's own recorded output.

```bash
uv run python web/build_data.py   # regenerate the bundle from demo_data/ (commit the delta)
cd web && npm install && npm run dev
```

### GitHub Pages runbook

One-time, by hand: repo **Settings → Pages → Source = "GitHub Actions"**.

From then on `.github/workflows/pages.yml` builds `web/dist` (with
`BASE_PATH=/nuscenes-data-engine/`, the project-site base) and deploys it on every push
to `main` touching `web/**` — plus manual `workflow_dispatch`. That workflow is
hermetic: no Python, no `demo_data/`, no exporter run, because the committed bundle is
the contract and CI's `quality` job is what guards it. `ci.yml` also gained a `web` job
(node 22 → `npm ci`, `npm run typecheck`, `npm run build` with the same base path), so
a change that breaks the site fails the PR rather than the deploy.

**Story site URL:** _pending first deploy_

### Report edition (`/v2/`)

A second presentation of the same story, served one directory down at
`…/nuscenes-data-engine/v2/`: the same seven steps in a research-report idiom — paper
and ink with a cobalt accent, the Source superfamily (Serif 4 / Sans 3 / Code Pro),
hairline rules, numbered sections and figures, footnoted provenance, light-first with
a first-class dark — designed entirely in code rather than from the deck's identity.
Its code is `web/src/v2/` behind the MPA entry `web/v2/index.html`; one Vite build
emits both editions (`dist/index.html` and `dist/v2/index.html`) from the one npm
project, so the Pages workflow above needs no change.

Zero data drift is structural rather than promised: the report edition imports the
**same** `web/src/data/*.json` and `web/public/story/` bundle the root edition does —
no second export, no second set of numbers — and `tests/test_web_v2.py` pins its copy
the way `test_web_export.py` pins the root edition's: the seven step titles and stages
against `views/tour.py`, the five story-contract sentences read from bundle fields
instead of typed out, no recorded figure as a literal anywhere in the sources, the
selection fold and fairness lead against their tour literals, and every image routed
through `assetUrl` (the base-prefix helper that keeps the bundle's document-relative
srcs resolving from `/v2/`).

**Report edition URL:** _live after the next Pages deploy_ —
…/nuscenes-data-engine/v2/

## Run it

```bash
uv run streamlit run app/demo/main.py
```

The app reads only `demo_data/`. If the package is missing, the app says so and
points at the builder.

## Guided tour (Phase 9a, restructured in Phase 10, framing mirrored in Phase 11)

The demo's default path. `views/tour.py` (`url_path="tour"`) is a seven-screen walk
through the whole model-improvement loop, told as *problem → intervention → impact*:
the page itself renders as **"From model failure to better training data"** (with the
subtitle caption "See how the data engine finds a perception weakness, mines targeted
AV scenarios, and measures whether retraining fixes it."), while the sidebar entry
`main.py` registers stays **"Guided tour"**. It is reachable from the Overview page's
`st.button("Explore a model failure →", key="overview_start_tour")` CTA, or directly
from the "Start here" nav section. It is built for a 2-3 minute pace: each step leads
with a headline, shows one visual, states its numbers in a few sentences and names
where they came from (`render.provenance`), inside a bordered container; steps 1 and 3
carry a "what we learned" takeaway (`render.learned`), and mechanism detail folds
into expanders instead of filling the screen. `st.session_state["tour_step"]` is the
page's only state, moved by the `tour_back` / `tour_next` buttons (mutated *before*
the step renders, never via `st.rerun()`), with `tour_restart` returning to step 1
from the result screen.

Phase 11 mirrors the story site's framing into five of the steps (the story
contracts of `docs/superpowers/specs/2026-09-04-web-frontend-v1-design.md` §1, applied to
the tour in its §3): step 1's three-beat opening, step 4's lede and plain-language
chain, step 5's **Fair comparison** strip, step 6's two evidence tiers, and step
7's closing thesis. Those fixed sentences are word-for-word the same on both
front-ends — `web/build_data.py` ships them as bundle fields and
`tests/test_web_export.py` pins them.

| # | Step | Stage | What it shows | Deep-links to |
|---|---|---|---|---|
| 1 | We found a blind spot | Diagnose | the three-beat opening: the plain-language problem as the headline ("The detector looked reasonable overall — but performance dropped sharply at night, especially for pedestrians."), then the numbers that establish it — the baseline's overall / night / night-pedestrian mAP50-95 cards, the bold count of night val frames carrying a miss, one derived sentence — then the purpose line "The goal of the system: automatically find failures like this and turn them into better training data.", and the takeaway "Aggregate metrics hide important failure slices." | Failure Explorer |
| 2 | What the failure looks like | Diagnose | headline "Here the baseline misses a pedestrian at night.", the hero crop with the badge legend under it, the model radio (`tour_hero_model`, story labels), what each model claimed about the missed pedestrian, and the bridge into mining ("Finding one failure is easy — the hard part is finding the rest of the dataset where the same thing happens.") | Failure Explorer |
| 3 | Where else does this happen? | Mine | headline "Is this one bad photo, or a recurring driving scenario?", the flagship scenario preset's top-ranked event (severity facts, chips, filmstrip, CAN curves), how many matching events are at night, one sentence on what the search matches on — driving context, not similar-looking images — and the takeaway "Perception failures must be analyzed in driving context." | Scenario Search |
| 4 | What data should we add? | Mine | headline "Thousands of candidate training frames — which are worth labelling?" over the lede "Instead of randomly adding more images, the system searches the training pool for examples related to the diagnosed failure.", one AL-selected frame, its `reason_chips` row and the plain-language selection chain (**Failed validation frame → relevant driving scenario → candidate training frames → targeted retraining set**), with the same chain in the package's own vocabulary (failed val frame → similarity community → representative train-pool frames → selected for retraining), the seven `selection_factors` and the causal closing sentence folded into **How selection works**; the flagship-rank / weak-rejected sentences, the `demo al-explain` provenance and the system-path caption stay outside the fold | Active Learning |
| 5 | We changed the training data | Train | headline "We did not just add data — we changed what the model trains on.", the mined-set cards (frames mined, scenes covered, night share, training images), the night-share comparison against the comparator arms this package ran, the bordered **Fair comparison** strip directly above the chart (the held-constant clauses as chips — "Same detector", the derived budget clause, "same training configuration", "scored on the same held-out split" — plus the accent chip "Random sample is the control", written only when the control arm is one of the charted ones), and the tour's five-strategy **Night mAP50-95 vs baseline, by acquisition strategy** chart (baseline, the random control, similarity mining, the best score-based arm and the arm the tour follows — an id this package has no row for is dropped) with story labels on the axis, the charted arms' raw ids captioned under it, the computed best-intervention sentence and the scene-spread sentence | Active Learning |
| 6 | Did it fix the failure? | Evaluate | headline "Did the targeted retraining fix the kind of failure we started with?" over two labelled evidence tiers — **One example** (the caption "One example — one hand-approved frame, illustrative, not the metric", the derived before/after callout cards and the hand-approved exemplar drawn twice side by side, baseline vs arm, each captioned with its story label *and* its raw id, with one badge legend under the pair) then **The aggregate result:** (that lead-in, written only where this package has such a number, over the `gain_text` night-pedestrian sentence and the night mAP result on the held-out split) — and the upgraded-boxes table folded into **Technical details — per-box claims** | Active Learning |
| 7 | Closed the loop | all four stages lit | the guarded headline ("Closed the loop: weakness → targeted data → measurable improvement" only when the arm's recorded night Δ is a gain, else "… measured result"), three hero cards (targeted frames added / night share of the mined data vs the random control / night-pedestrian mAP50-95 gain), the four bordered answers restating steps 1-6's numbers, the closing thesis ("Instead of blindly retraining the model, the system diagnoses where it fails, finds the data that can address the weakness, and measures whether the intervention actually works."), and six "go deeper" links across every page | Overview, Failure Explorer, Scenario Search, Active Learning, Weak Supervision, Ask the Dataset |

Phase 10's shared vocabulary lives in `filters.py` / `render.py`, so no step writes a
name or a number by hand:

- **Story labels** (`filters.arm_story_label`) give the tour readable experiment names
  — `graph_rate_night` → "Graph + night targeting", `random` → "Random sample —
  control", `mined` → "Similarity mining" — falling back to `model_label` for anything
  unmapped (the weak checkpoints keep their technical labels). The raw arm id stays
  visible wherever a story label fronts one: the strategy chart's id caption, step 6's
  two overlay captions, and the backticked ids inside the precise claims themselves.
  Readable names are tour-scope only — the deep pages keep `model_label`, raw ids, the
  full 13-arm chart and the per-arm table.
- **`render.legend()`** is the app's one overlay legend: five badge chips ("green —
  ground truth", "orange dashed — GT box the model missed", "white — true positive",
  "yellow dotted — low-confidence claim (below the hit floor)", "red — false
  positive"), rendered wherever a full prediction overlay is drawn — tour steps 2 and
  6, the Failure Explorer detail, the Overview hero overlay, the Active Learning
  before/after panel, and Weak Supervision's held-out row. The three prose legend
  captions those pages used to carry are gone, and the copy-contract test now checks
  that none of them comes back (Weak Supervision keeps its separate sentence saying
  what a *pseudo* box is, which is a different claim). GT-only frames — tour steps 3
  and 4 — get no legend, since it would name prediction colours those images never
  draw.
- **`filters.gain_text`** states a before/after with its absolute and relative gain
  together — "Night pedestrian mAP50-95 0.0826 → 0.1171 (+0.0345 absolute, +41.8%
  relative)" on package v0.8 — and drops the relative clause when there is no honest
  base to divide by (`relative_gain` returns `None` at `before <= 0`).
- **"Go deeper" links** read as story sentences rather than page names (e.g. "Failure
  Explorer — see every night frame the baseline missed"), and none of them names a
  count: a static label is the one string on the screen nothing recomputes.

Honesty notes (verified against package v0.8, restated from the design spec):

- The hero frame (`5994f34b836043b3b5be191bceba2e3e`) is the only night val frame
  where a pedestrian goes miss → hit, and the hit is a **low-confidence claim (conf
  0.135)** that the matching rule counts as a hit — the tour says so at steps 2, 6 and
  7 and never calls it an exemplar (`fixed_boxes` returns no row for it; the hand-
  approved exemplars at step 6 are the confident recoveries).
- No per-frame "failure rate" exists for pool frames (1,249 of the arm's 1,500
  selected frames received no routed failure mass) — step 4's selection facts, inside
  the **How selection works** fold, are `selection_factors` only: night, community
  size / night members / mass rank / quota, pick pass, and degree rank within the
  community.
- Every number on every step is read from the package, the same tables the deep
  pages read; nothing is computed only for the tour.
- The tour degrades honestly when an optional group is missing: without
  `scenario_events.parquet` step 3 shows "needs demo_data >= 0.4 — rerun demo
  build" instead of an event; without a staged `al_explain/` group step 4 shows
  "per-frame community and routed mass are not included in this package (demo
  al-explain)" instead of a selection reason. Neither case raises.

Widget/state keys, for contributors extending the tour: `tour_step` (session state,
the current step index) and buttons `tour_back` / `tour_next` / `tour_restart` /
`tour_hero_model` (step 2's model radio) / `tour_open_event` / `tour_open_al_frame` /
`tour_open_exemplar`. Phase 10 retired `tour_exemplar_model`: step 6 draws the
baseline and the arm side by side instead of toggling between them, and
`tour_open_exemplar` is how a viewer gets to the Active Learning page's own radio for
model-by-model inspection.

Three buttons pre-select an item on the page they open, rather than just linking to
it (unlike the "Go deeper" `st.page_link` row, which carries no state):
`tour_open_event` (step 3) sets `scenario_preset` + `scenario_token` and opens
Scenario Search on that event; `tour_open_al_frame` (step 4) sets `al_frame_token`
and opens Active Learning on that frame; `tour_open_exemplar` (step 6) sets
`al_exemplar` and opens Active Learning on that exemplar.

### Trust chrome

Loop breadcrumb (`render.loop_breadcrumb`, `LOOP_STAGES = ("Diagnose", "Mine",
"Train", "Evaluate")`): every page carries it, lighting the stage(s) it belongs to.

| Page | Stage(s) lit |
|---|---|
| Overview | none (whole loop; caption "Detect weakness → Find useful data → Retrain → Measure impact") |
| Guided tour | the current step's stage — all four together on the result screen |
| Failure Explorer | Diagnose |
| Scenario Search | Mine |
| Active Learning | Mine, Train, Evaluate |
| Weak Supervision | Train, Evaluate |
| Ask the Dataset | Diagnose |

Provenance captions (`render.provenance`) name where a number on screen came from,
one of three kinds:

- **recomputed** — derived live in this app from the package's own tables (e.g. an
  overlay drawn from `gt_boxes.parquet` + `predictions.parquet`).
- **recorded** — the offline pipeline's own output, shipped and shown as-is (e.g.
  `active_learning_results.parquet` rows).
- **reproduced** — re-derived by `demo al-explain` and validated equal to the run
  that actually selected the frames (the "why selected" panels).

The Scenario event viewer's header now also carries `filters.parity_short` — a
one-line trust indicator ("30 events found · SQL 30 / Graph 30 ✓") next to the
existing full parity caption (the preset header's pinned "N matching keyframes
dataset-wide — Cypher C · SQL S" wording, unchanged); shown for dynamics presets
only, since a model-result preset's verdict isn't graph-derivable.

"What we learned" callouts (`render.learned`, a bordered container, not `st.info`)
on Active Learning and Weak Supervision, each a sentence that interpolates real
table values and is skipped outright when the arm it names is absent from this
package's arm table:

- **Active Learning** — the night-inversion sentence, e.g. "Targeting night bought
  night: `graph_rate_night` (+0.0101 night mAP50-95) is the best night arm of 13,
  while its weak-supervised twin `weak_graph_rate_night` (−0.0262) is the worst —
  the night gain came from the frames, not from cheaper labels."
- **Weak Supervision** — the loss-split sentence, e.g. "Free labels kept 18.2% of
  the ground-truth gain for the `random` pair: 49.7% was lost with the frames the
  verifier dropped and 32.1% to label noise on the ones it kept." (a second clause
  is added only when a second weak/GT pair's arm is genuinely the worst night arm
  in the whole table).

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
- the package exceeds the size budget (100 MB; currently 25.55 MB).

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
`graph_rate_night` pair is carried separately); the VLM's count error per
crowding bucket is **recomputed at build** by `export_vlm_count_buckets`, which runs
the autolabel eval's own `gt_counts`/`eval_count_buckets` over the Phase-6b label
table (`data/autolabel/labels.parquet`, the `parse_status == "ok"` rows) and
`annotations.parquet` — [AUTOLABEL_EVAL.md](AUTOLABEL_EVAL.md)'s published MAEs are a
cross-check, never a source, and no rounded constant is written into the code. That
export is optional: without the Phase-6b table the build records
`validation.vlm_count_buckets: "absent"` with a warning and writes no file.

The flagship *Cypher* twin is
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

## Package layout (Phases 1-9)

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
| `vlm_count_buckets.parquet` | optional (needs the Phase-6b VLM label table on the build machine): `model, bucket, n, mae` — the VLM's mean count error against how crowded the frame actually is, recomputed at build from `data/autolabel/labels.parquet` (`parse_status == "ok"`) and `annotations.parquet`. Shipped run: one model (`Qwen/Qwen2.5-VL-7B-Instruct`), buckets `0 / 1-3 / 4-9 / 10+` → MAE 0.0767 / 0.8652 / 2.8107 / 6.6711 over n = 38,042 / 8,868 / 2,494 / 456. `n` counts **frame-class pairs**, not frames: all ten of the VLM's count fields are pooled, so n sums to 49,860 = 4,986 parsed frames × 10 |
| `sample_frames/hero.jpg` | hand-picked night exemplar crop (`hero_token` in overview_metrics; baseline misses a shadowed car AND a pedestrian — graph_rate_night recovers only the pedestrian, a low-confidence hit; the car defeats all five models) — rendered live with overlays on Overview |
| `frame_manifest.parquet` | 250 curated frames: buckets, val/train_pool split, failure stats, per-model prediction counts (5 × `n_preds_<model>`, 0 = ran-and-found-nothing), exemplar flags (20 × `fixes_fn_vs_<a>_<b>` — every ordered pair of the five models: b fixes a's misses, True where model a has an FN that model b matched), `weak_verdict` (accepted / rejected for frames in the weak arm's candidate pool, else NA) and `al_selected_by` (the AL arm that selected the frame, else NA) |
| `gt_boxes.parquet` | GT boxes (1600×900 coords) + 5 per-model `matched_<model>` flags (NA = not evaluated) + `distance_to_ego_m` + `size_bucket` (COCO 32²/96²) + `below_visibility_min` (all-False today; parity-defensive) |
| `predictions.parquet` | 6,206 predictions × 5 models (baseline, graph_rate_night and the weak pair `weak_graph_rate_night` / `weak_graph_rate_night_gt` @640, champion @960 — per-row `imgsz`), status ∈ tp/fp/low_conf matched with the AL sweep's exact semantics |
| `sample_frames/crops/` | 250 × 960×540 crops (0.6 scale of native) |
| `sample_frames/thumbs/` | 659 × 256×144 LanceDB thumbnails (250 curated + 367 for events, filmstrip neighbors, semsearch + 42 frames retrieved in the recorded chat session) |
| `scenario_events.parquet` | 126 preset-tagged keyframes (6 presets, capped 30 each): ego dynamics, per-class min distances, `preset_tags`, `preset_rank_<name>`, t−2…t+2 filmstrip neighbor tokens + readouts, `in_curated_set`, and (v0.8) the CAN bus's own per-step speed — `can_speed_kmh` for the event frame plus `can_speed_t_minus2 … can_speed_t_plus2` for the neighbours, NA at a scene edge (the existing `speed_mps`/`speed_t_*` stay: those are GT ego-pose speed in m/s, a different reading) |
| `semantic_search_results.parquet` | recorded SigLIP results: 4 canned queries × up to 8 front-camera hits (query, rank, sample_data_token, score, k — k drives the gallery's "n of k" caption) |
| `graph_subgraphs/<preset>.json` | 6 presets × per-event subgraphs from the live graph: nodes (`on_path` flag + properties), edges, the matched path (nearest matching object first), off-path observations capped at the nearest 12; plus the preset's count Cypher and `sql_count` / `cypher_count` / `parity` on the full keyframe population (model presets: `cypher_count` null — their verdict comes from predictions, not the graph) |
| `chat_replays.json` | recorded chat session (`demo chat-record`): per replay `{id, kind (eval/showcase), question, answer, model, provider, steps [{tool, input, output}], frames (7 projected columns, no thumbnail bytes), charts, checks (graded cases only), latency_s, error}` |
| `chat_replay_summary.json` | that recording's provenance: model, provider, `recorded_at`, `git_sha`, `n_eval`, `n_passed`, `n_showcase`, `search_available`, `graph_available`, `max_turns` |

## Overview, outcome-first (Phase 9a)

Phase 9a made the page outcome-first (spec §4): title + mission → the guided-tour CTA
→ the loop strip (an unlit `loop_breadcrumb` plus four one-line beats: detect
weakness, find useful data, retrain, measure impact) → exactly four flagship metrics
(best night gain; night pedestrian mAP or, absent that slice, the CAN-speed
correlation; weak-sup share of GT gain; graph = SQL flagship) → the hero image below
that. Everything else folded into three collapsed expanders:

- `st.expander("Where these numbers come from")`, directly under the headline cards
  — the flagship-parity and weak-retention captions. They are provenance about
  those four cards, not architecture, so the review round moved them out of the
  architecture drawer; the retention caption names its card ("The Weak-sup share of
  GT gain card is …") rather than saying "the card above", which it no longer is.
- `st.expander("Dataset scale & data checks")` — camera keyframes, 2D boxes, 3D
  object observations, CAN-bus rows, and the **CAN speed vs ego-motion** correlation
  card (a check on the CAN join, not a scale figure — hence "& data checks").
- `st.expander("Architecture (for technical reviewers)")` — [PROJECT.md](PROJECT.md)
  §2 restated: the two-machine topology, the component map, CI.

Nothing left the page — it just stopped competing with the loop for the first
screenful.

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
uv run nuscenes-data-engine demo infer   # 5 checkpoints over the val frames (CPU; measured 40 s, 2026-08-23)
uv run nuscenes-data-engine demo build   # package v0.2 with the curation group
```

Facts about the shipped run: 250 frames (125 val / 125 train_pool), all eight buckets
at quota; matching reuses the AL sweep's exact parameters (IoU 0.5, conf_hit 0.4,
conf floor 0.05, visibility ≥ 2) via the property-tested box-level matcher, so
tp/fp/FN here mean what `failures.parquet` means. A val frame with `n_preds_<model>=0`
means the model ran and detected nothing (3 (token, model) pairs on one frame — a
night frame, `scene-1063`, that three of the four yolov8n checkpoints missed
entirely). Without TRINITY access, `demo build` skips the curation group and still
produces the Phase-1 package (`validation.curation: "absent"`).

`configs/demo.yaml`'s `models:` block is the list `demo infer` runs, and Phase 9b
added the two weak-arm checkpoints to it — `weak_graph_rate_night` (trained on the
verified pseudo labels) and `weak_graph_rate_night_gt` (its GT-labelled twin), both
yolov8n @640 — so all five checkpoints now score the same 125 val frames and the
package carries five models everywhere (`predictions.parquet`, `matched_<model>`,
`n_preds_<model>`, and the 20 ordered `fixes_fn_vs_<a>_<b>` pairs). The weak pair's
boxes are labelled on screen as offline inference (`demo infer`, CPU, the training
run's own checkpoint), never as a live model.

## Failure Explorer (Phase 3, 9b)

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

**Phase 9b made the page image-first.** The order is now title → breadcrumb →
caption → **Detail** → the grid, so the overlay is the first thing on screen. The
**Model** radio moved out of the sidebar and sits beside the image it changes, in
the detail's right-hand column (the page reads it back before filtering, so the grid
and the detail always agree on one model); the view-mode radio (GT / Predictions /
Overlay) sits directly under the image, and the per-box table folds away under
**Per-box detail**. The sidebar keeps the six condition filters (lighting, rain,
category, size, distance to ego, failure type) and folds Curation bucket and Sort by
into an **Advanced filters** expander. With nothing selected the detail
**auto-selects** the first frame of the current sort order and says so out loud
("… auto-selected — click View on another frame below to change it"); an explicit
View click always wins. An empty filter set still shows "No frames match these
filters" and no detail.

## Scenario Search (Phase 5, 9b)

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
prediction set" caption, plus a t−2…t+2 filmstrip (strip + slider + per-step
speed/accel readout). The **semantic gallery** is recorded
(`demo semsearch`, SigLIP offline, `semsearch.oversample: 16` to survive the 5/6
non-front-camera store) and labelled as such; per-query hit counts are shown.

**Phase 9b put the viewer on one screen** and gave a still frame its missing motion
cue. The layout is: header line (scene · severity · the compact SQL/Cypher parity
line · provenance) → the frame beside **two step curves** → **six cards, three to a row**
→ the filmstrip → the graph panel. The curves are the real CAN bus: speed from
v0.8's `can_speed_kmh` / `can_speed_t_*` (km/h) and longitudinal acceleration from
`accel_long_min_mps2` / `accel_t_*` (m/s²), plotted over the five keyframe steps with
a vertical rule that follows the filmstrip slider, captioned "Keyframes are ~0.5 s
apart; speed and longitudinal acceleration from the CAN bus". On a pre-0.8 package
the speed series falls back to the GT ego pose and both the chart's axis title and
the caption say "ego speed" instead — an old package can never mislabel the series.
The six cards are the old ego/context/model panels merged: CAN speed (the event
frame's own step of the curve above, so the two always agree), min longitudinal
accel, a **preset-aware VRU distance** card (min cyclist distance under
`fast_cyclists`, min pedestrian distance everywhere else — `rain_vru` keeps the
pedestrian card because the nearer of its two VRUs is already the grid caption),
pedestrians within 10 m, lighting/rain, and the model result (or the short
"not curated" value — the full "not in the curated prediction set" sentence stays
the viewer's caption beside the frame, rendered exactly once). The filmstrip's per-step readout stays the **GT ego-pose
speed in m/s** and now names that source out loud (it reads "ego N.N m/s"). The two
readings really do differ on the same keyframe — across the 126 shipped events the
CAN reading is a median 3 % off the ego-pose one and 5 % of events are more than
30 % apart — so neither figure is left unlabelled for the page to look like it is
contradicting itself.

## Interactive graph (Phase 6, 9b)

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

**Phase 9b reveals that path one step at a time** instead of drawing the whole
subgraph at once. A **Reveal the matched path** `select_slider` walks the export's
own `path`: Scene → Keyframe → one step per matching object, **nearest first**
(labelled like "3 · Pedestrian · 2.1 m") → Ego pose, each label prefixed with its
1-based index so the options stay unique; it opens on the last step, i.e. the whole
path. Nodes up to
the selected step are **amber**; later path nodes stay drawn but **hollow** (white
fill, grey border, grey label), so the node set never changes as you slide — only the
colours do. Context nodes and their edges are hidden behind a **Show context nodes**
toggle (off by default). The detail column shows the selected step's own facts as a
two-column table (scene name / location / night / rain; timestamp; category,
distance, visibility; CAN speed, min longitudinal accel, hard braking yes/no),
captioned with where those CAN readings actually live: "CAN speed and minimum
longitudinal acceleration are properties of the EgoPose node — the graph export has
no separate CAN node." Clicking a node still takes precedence and shows its `meta`
table, and the one-line path narrative is unchanged.

## Active Learning + Weak Supervision (Phase 7, 9b)

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

**Phase 9b added the strategy comparison and the reason chips.** Above "Every arm,
one chart" sits **"Three ways to pick 1,500 frames"** — the same mining budget spent
three ways (`random` "Random sample", `mined` "Similarity mining",
`graph_rate_night` "Graph-aware mining"), as two bar charts side by side (scenes the
mined frames came from; night share of the mined set, graph-aware highlighted), one
fixed prose note per strategy saying how it *works*, and a **what we learned**
sentence computed from the table itself: the "similarity mining found
near-duplicates … graph-aware mining spread the same budget wider and took the best
night gain" reading is emitted **only** when the graph arm really does cover more
scenes AND a higher night share than the mined arm AND holds the table's best
`delta_night`; otherwise the page falls back to a neutral sentence that just states
the three (scenes, night share, Δnight) triples. The heading counts the strategies
this package actually carries, and the whole section is skipped when fewer than two
are present. On the per-frame panel (and on the tour's selection step) a **chip row**
now sits above the factor lines: night · *k* pedestrian GT box(es) · community #*c* ·
*size* frames · mass rank · quota *before* → *after* · which pass took the frame ·
degree rank · routed failures. Every chip is a fact the package carries
(`al_selection_explain` columns plus the frame's visible GT) and each is omitted when
its own value is missing — and because **no per-frame failure rate exists**, no chip
ever contains the word "rate" (a test asserts it over every branch).

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

**Phase 9b added the two pictures behind those numbers.** **"One frame, three
views"** opens the page's evidence: row 1 takes one accepted train-pool frame and
shows GT boxes | the pseudo labels the verifier kept | both layers together, with the
verdict and the VLM's count-vote table beside that third panel; row 2 takes a
**held-out val frame** and shows GT | the weak-trained
detector | its GT-trained twin, with a **Detector** radio over the pair and a table
of the GT boxes the selected arm detects that the other does not. Both frames are
chosen from the package, not hardcoded (`filters.weak_showcase_token` /
`weak_result_token`); the val frame's boxes come from the two weak checkpoints run
offline by `demo infer` and are labelled "recorded — demo infer, CPU, checkpoint of
the training run", with the overlays themselves labelled as recomputed here. What
one frame is worth is stated as exactly that ("counted on this frame only"); the
*effect* sentences still come from the arm and loss tables. Further down, **"Where
the VLM's counting breaks down"** charts `vlm_count_buckets.parquet`'s MAE per
crowding bucket (shipped run: 0.08 at 0 objects → 0.87 at 1-3 → 2.81 at 4-9 → 6.67
at 10+) with a **what we learned** sentence that only says the error "rises" when
this package's own MAEs never fall. Its caption states that `n` counts **frame-class
pairs**, not frames — all ten count fields the VLM was asked for are pooled, more
than the five detector classes in the count-vote table above — and names the VLM the
errors belong to. On a package below v0.8 the section shows
"count-bucket chart needs demo_data >= 0.8 — rerun demo build" instead. The existing
model gallery folds the weak pair behind a **Weak-arm detections on this frame**
expander.

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
uv run nuscenes-data-engine demo build                              # package v0.8
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

The README gallery's seven PNGs are produced by `scripts/demo_screenshots.py`, a
manual Playwright tool that is deliberately **not** a project dependency:

```bash
uv pip install playwright                     # temporary, into .venv
uv run streamlit run app/demo/main.py --server.headless true --server.port 8599  # another shell
.venv/bin/python scripts/demo_screenshots.py  # -> docs/img/demo-*.png
uv pip uninstall playwright
```

It drives the machine's installed Google Chrome (`channel="chrome"`, so no browser
download), shoots a 1200×900 viewport per page, and frames each one deliberately: the
Guided tour, Failure Explorer and Weak Supervision are captured at the top of the
page (their first screen leads with the content the gallery caption names — the
Failure Explorer's detail is its auto-selected first block since Phase 9b);
Scenario Search and Active Learning are steered to their Phase-9b sections first
(the flagship event's one-screen viewer with the CAN curves; the
acquisition-strategy comparison); any other page has its first frame-detail panel
opened. It **exits non-zero if any page rendered a Streamlit exception** — a broken page cannot quietly become a README screenshot. Re-run it
whenever a page changes visibly; any capture over 300 KB is quantized to a
256-colour palette PNG by the script itself, no manual compression step needed.

These PNGs contain nuScenes-derived imagery and are covered by the attribution
section below, exactly as the packaged frames are.

## Success-criteria walk

[DEMO_PLAN.md](DEMO_PLAN.md)'s ten questions, and the page and on-screen section that
answers each. Status is filled in when the walk is actually performed: the local
column after a browser pass over the committed package, the live column against the
deployed URL.

| # | Question (DEMO_PLAN.md) | Page | Section / element that answers it | Status |
|---|---|---|---|---|
| 1 | What problem does the project solve? | Overview + Guided tour | the loop strip ("Detect weakness → Find useful data → Retrain → Measure impact") and tour step 1, **We found a blind spot** | local: 2026-08-23 · live: pending |
| 2 | Where does the baseline perception model fail? | Guided tour + Failure Explorer | tour step 2, **What the failure looks like** (the hero frame, per-model claims, badge legend), and Failure Explorer's **Detail** — now the first block on the page, auto-selected, with the **Model** radio and the GT / Predictions / Overlay toggle beside the image | local: 2026-08-23 · live: pending |
| 3 | How does the system find difficult data? | Guided tour + Scenario Search | tour step 3, **Where else does this happen?** (the flagship event, and the sentence saying the search matches driving context — not similar-looking images), and the six **preset** buttons + ranked card grid | local: 2026-08-23 · live: pending |
| 4 | Why is the graph useful? | Scenario Search | the compact `parity_short` line in the event viewer header (visible once an event is opened — flagship 30 events found · SQL 30 / Graph 30 ✓), and the same viewer's **Interactive graph** panel, whose **Reveal the matched path** slider walks Scene → Keyframe → nearest matching objects → Ego pose one step at a time (amber = revealed, hollow = still to come) | local: 2026-08-23 · live: pending |
| 5 | What does CAN-bus data add? | Scenario Search (+ Overview) | the event viewer's two **CAN curves** beside the frame (speed km/h + longitudinal accel m/s² over t−2…t+2, the rule following the filmstrip slider), the **CAN speed** card in the metric grid, and the filmstrip's own ego-pose readout; Overview's **CAN speed vs ego-motion** card (r), inside the **Dataset scale & data checks** expander | local: 2026-08-23 · live: pending |
| 6 | How does active learning choose frames? | Guided tour + Active Learning | tour step 4, **What data should we add?** (the reason chips and the one-line selection path, with the `selection_factors` lines inside the **How selection works** fold), and the deep page's **Three ways to pick 1,500 frames** strategy charts plus the **Why was this frame selected?** panel, whose reason chips head the factor lines | local: 2026-08-23 · live: pending |
| 7 | Did targeted retraining improve performance? | Guided tour + Active Learning | tour steps 5-6, **We changed the training data** (the five-strategy night-Δ chart, its fairness statement and the computed best-intervention sentence) / **Did it fix the failure?** (the side-by-side before/after exemplar and the night-pedestrian gain), plus the deep page's **Every arm, one chart** (13 arms in round order) and **Before / after** exemplars | local: 2026-08-23 · live: pending |
| 8 | How well did VLM-generated supervision work? | Weak Supervision | the retention cards (GT gain retained + verifier retention), **One frame, three views** (what the VLM's counts kept on a train-pool frame, and what the weak-trained checkpoint vs its GT-labelled twin then did on a held-out val frame), and **What the VLM saw** accepted/rejected galleries | local: 2026-08-23 · live: pending |
| 9 | Why did weak supervision underperform GT? | Weak Supervision | **Where the rest of the gain went** (dropped-frame cost vs label cost), **Where the VLM's counting breaks down** (count MAE per crowding bucket) and **What the verifier's rule selects for** (the crowding bias) | local: 2026-08-23 · live: pending |
| 10 | How does the project form a closed model-improvement loop? | Guided tour + Active Learning + Weak Supervision | tour step 7, **Closed the loop** — its guarded headline and the three hero cards (frames added / night share vs the random control / night-pedestrian mAP50-95 gain) over the four bordered answers — closed by the persistent loop breadcrumb on every page | local: 2026-08-23 · live: pending |

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
| 9a | guided tour, outcome-first Overview, loop breadcrumb + provenance + lessons | **shipped** |
| 9b | Failure Explorer hook, one-screen Scenario Search (CAN curve, progressive graph path), acquisition-strategy chart + reason chips, Weak Supervision three views + crowding trend, package 0.8 | **shipped** |
| 10 | storytelling rework of the guided tour (story titles, folds, 5-strategy chart, side-by-side, hero numbers, legend component) | **shipped** |
| 11 | story site (`web/`): designed scrollytelling front-end, exporter + committed bundle + CI guard, GitHub Pages deploy; tour framing mirror | **shipped** |
