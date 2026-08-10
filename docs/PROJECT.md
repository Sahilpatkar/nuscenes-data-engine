# nuScenes Data Engine — Project Documentation

An end-to-end **MLOps data engine for autonomous-vehicle perception**, built on the
nuScenes dataset: from raw sensor metadata to a trained, evaluated, registered,
served, monitored detector — and then the "data engine" layer on top: semantic
search, VLM auto-labeling, natural-language dataset chat, and a measured
active-learning loop.

Everything is reproducible from this repo: config-driven pipelines, versioned data,
tracked experiments, CI-gated code, and a Docker demo stack.

---

## 1. Objective

Perception models are only as good as the data curation loop around them. Most of
the real work in AV/robotics ML is not the model — it is:

- turning a raw multi-sensor dataset into **queryable, validated training data**;
- knowing **where the model fails** (night? rain? crowded scenes?) rather than one
  overall metric;
- **finding more of the data that matters** without labeling everything;
- keeping the whole thing **reproducible, tracked, and deployable** by one person.

The objective of this project is to build that loop end-to-end at a realistic scale
(204,894 images, ~1.1M boxes) with production-grade engineering practices — and to
*measure* each step honestly, including the experiments that produced negative
results.

## 2. Architecture

### Two-machine topology

| | GPU server ("TRINITY", multi-node) | Infra machine (Mac, this repo's demo host) |
|---|---|---|
| Role | Compute: ingest, train, evaluate, embed, label, mine | Ops: registry UI, serving, monitoring, demo, chat |
| Hardware | RTX 3080 Ti nodes (11.9 GB each) | Apple M4 Pro, 48 GB |
| Writes | Plain files (Parquet, weights, `mlruns/`, LanceDB) | Docker stack, synced artifacts |
| Sync | `scripts/gpu-run.sh` pushes the current branch, runs any CLI command remotely; results rsync back | `docker compose up` |

The GPU server needs **no services** — training logs to a local SQLite MLflow store
that is rsynced to the infra machine, whose MLflow server owns the UI and model
registry. `scripts/gpu-run.sh` (docs/GPU_SERVER.md) makes the remote execution
one-command: it pushes the local branch, syncs the remote checkout, and runs the
CLI there (foreground or `--bg` with a log file).

### Component map

```
                    ┌─ ingestion (Phase 1) ─────────────────────────────┐
raw nuScenes  ───►  │ flatten + 3D→2D projection → Parquet              │
(read-only)         │ Great Expectations validation → DVC/MinIO version │
                    │ + Phase B: ego pose/3D geometry + CAN-bus dynamics │
                    └──────────────┬────────────────────────────────────┘
                                   ▼ data/processed/{samples,annotations,availability,ego_pose,annotations_3d,instances,canbus}.parquet
      ┌────────────────────────────┼──────────────────────────────────────────┐
      ▼                            ▼                                          ▼
 training (2)                data engine (6)                            analytics
 YOLOv8 fine-tune            ├─ 6a embed: SigLIP2 → LanceDB             DuckDB views
 MLflow + W&B                ├─ 6b autolabel: VLM → labels.parquet      (query CLI)
      ▼                      ├─ 6c chat: LLM agent (SQL + vectors)
 evaluation (3)              └─ 6d active learning: mine → retrain
 sliced mAP + gate
      ▼
 registry (MLflow) ──► serving (4): FastAPI /predict /search /chat ──► Streamlit demo
                            │
                            └─► monitoring (5): request capture → Evidently drift
```

CI (GitHub Actions) runs two jobs on every PR: **quality** (ruff + mypy strict +
the torch-free test suite — 137 tests) and **smoke-train** (a 1-epoch CPU training
run on a tiny fixture dataset). Modules that need heavy extras guard themselves
with `importorskip`, so the same suite runs everywhere.

## 3. The build: phases, steps, and decisions

### Phase 1 — Data engineering

**What:** Walk `scene → sample → sample_data` for all 6 cameras (850 scenes, 34,149
keyframes → **204,894 images**), project each 3D box to a 2D axis-aligned box
(**~1.1M boxes**), and write two columnar Parquet tables (`samples`,
`annotations`) plus a file-integrity manifest (`availability`). Validate with
Great Expectations; version with DVC against a MinIO remote.

**Key decisions**
- **Plain Parquet, no warehouse.** DuckDB queries the files directly; Iceberg/warehouse
  migration was considered and skipped — no scale pressure justified it, and every
  later phase benefited from "the dataset is just two files".
- **5-class detector taxonomy** (`car, truck, bus, pedestrian, bicycle`) mapped from
  nuScenes' 23 categories; everything else keeps its raw `category_name` with
  `category_group = NULL`, so nothing is thrown away.
- **Scene-level condition flags** (`is_night`, `is_rain`) derived from scene
  descriptions — coarse, but they power all later slicing, monitoring, and
  stratification. Night exists only in Singapore; all rain is in Boston.
- **Visibility filtering at ingestion** (≥ 2 of 4): projected boxes for occluded
  objects are noise for a camera-only detector.

### Phase 2 — Training pipeline

**What:** Config-driven YOLOv8 fine-tuning (`configs/train.yaml`): dataset builder
that symlinks images and emits normalized labels with an idempotent build manifest,
MLflow logging (params, data-version content hash, metrics, weights artifact), W&B
streaming, sweep knobs (model size / imgsz / augmentation), DDP-safe logging.

**Key decisions**
- **Data-version hash in every run.** The processed Parquet is content-hashed and
  logged, so any model can be traced to the exact dataset build.
- **Idempotent dataset builds.** A ~400k-symlink build is skipped when the manifest
  matches — which also keeps Ultralytics' label cache valid.
- **SQLite MLflow on the GPU server** + rsync, instead of a remote tracking server:
  no ports, no auth, no coupling; the infra machine owns the UI.

### Phase 3 — Condition-sliced evaluation + registry

**What:** Evaluate any checkpoint overall **and per condition slice** (day/night,
clear/rain) by materializing sliced val datasets; log to MLflow; apply a promotion
gate (minimum overall *and* minimum night mAP) before registering
`nuscenes-yolo-detector@production`.

**Key decision:** the gate exists because overall mAP hides the failure mode that
matters — see the results table below: night is 20 points worse than day.

### Phase 4 — Serving

**What:** FastAPI service loading the registry's `@production` model
(`/predict`, `/predict/annotated`, `/health`), a Streamlit demo UI, Docker Compose
stack (api, streamlit, mlflow, minio). Degrades gracefully: a missing model or
index produces 503s on the affected endpoints, never a crash.

### Phase 5 — Monitoring + CI/CD

**What:** Every `/predict` request appends a feature row (brightness, detection
count, latency). An Evidently job compares production windows against a training
reference — demonstrated with a day-vs-night drift report (brightness and
`n_boxes` shift, exactly what the night slice predicts). CI gates merges on the
quality + smoke-train jobs; branch protection on `main`.

### Phase 6a — Semantic scene search

**What:** Embed all 204,894 frames with **SigLIP2** (`google/siglip2-base-patch16-256`)
on the GPU server; store vectors + metadata + JPEG thumbnails in **LanceDB**;
serve text-to-image, image-to-image, and similar-to-frame search in the API and a
Streamlit tab. Thumbnails live in the store, so search works on machines without
the 400 GB of images.

**Why SigLIP2:** a joint text-image embedding space gives text search for free
(CLIP-family), the base model is small enough to encode queries on CPU at request
time, and it later powered 6d's failure-similarity mining unchanged.

### Phase 6b — VLM auto-labeling (with honest evaluation)

**What:** Label a **stratified 5,000-frame sample** (by location × night × rain,
with floor allocation so small night strata survive) using a vision-language model
producing schema-validated structured JSON (time of day, weather, hazards,
per-class object counts). Evaluate against nuScenes ground truth. Results: §5.

**Key decisions**
- **Two labeling providers behind one seam:** the Claude Batch API path
  (haiku-4-5 primary + opus-4-8 comparison subset, ~$15 for 5K) and a **$0
  self-hosted path** — Qwen2.5-VL-7B-Instruct on vLLM across 4× RTX 3080 Ti
  (tensor-parallel 4). The actual run used the local path: **5,000 frames for $0**,
  99.7% parse rate.
- **Enums only where GT exists** (night/rain flags), free text elsewhere — so the
  evaluation is honest about what can actually be scored.

### Phase 6c — Chat with the dataset

**What:** A tool-calling LLM agent that answers natural-language questions by
writing **guarded DuckDB SQL** over the Parquet tables (plus the 6b labels) and
running **vector search** over the 6a embeddings — returning numbers *with example
frames*. Surfaced as a CLI (one-shot + REPL), `POST /chat`, and a Streamlit chat
tab that shows the agent's SQL per answer. Every query is logged to JSONL.

**Key decisions**
- **Local-first, deploy-flippable:** default provider is Ollama on the infra
  machine (`qwen2.5:14b`, $0/query); `CHAT_PROVIDER=anthropic` swaps in the Claude
  API (`claude-opus-4-8`) with no code change. One OpenAI-compatible transport
  covers Ollama, vLLM, and LM Studio.
- **Defense-in-depth SQL guard**, built after live probing: exactly one statement
  of parsed type SELECT **plus** a denylist — because DuckDB parses `PRAGMA` as a
  SELECT-typed statement and lets a bare SELECT read arbitrary files
  (`read_parquet('/any/path')`, `FROM 'x.parquet'`). Row caps + errors returned to
  the model as data so it repairs its own SQL.
- **Stateless server:** the client holds conversation history; the server holds
  only caches (DuckDB catalog, search engine).

### Phase 6d — Embedding-based active learning (a controlled experiment)

**What:** The loop that justifies the "data engine" name: find where the model
fails, mine visually similar unused frames, retrain, measure. Because the
production model had already seen all training data, the credible design was a
**reduced-baseline controlled experiment**: train a baseline on a night-stratified
25% of train scenes, sweep its failures on val (pure-numpy IoU matcher,
`failure_score = FN + 0.5·low-confidence`), KMeans-cluster the failure embeddings,
mine 1,500 similar frames from the held-back 75% pool via prefiltered LanceDB
search — against a **1,500-frame random control**. Leakage guards assert mined ⊆
pool and an identical val set across arms. Results (including the negative
headline) in §5.

## 4. Models: what we use and why

| Role | Model | Why this one | Alternatives considered |
|---|---|---|---|
| **Production detector** | **YOLOv8m @ 960px** (fine-tuned, registry v2) | Best sweep result (+13 mAP50 over the yolov8n@640 baseline); single-GPU trainable; mature training/val tooling; real-time capable | yolov8n/s (kept as baselines — n is the AL experiment model for cheap arms); transformer detectors (DETR-family) rejected for training cost vs. benefit at this scale |
| **Embeddings** | **SigLIP2-base-patch16-256** | Joint text-image space → text search free; CPU-encodable queries; one embedding store serves search *and* active-learning mining | Larger SigLIP/CLIP variants (unneeded quality at 4× cost); pure image embeddings (no text search) |
| **Auto-labeler** | **Qwen2.5-VL-7B-Instruct** on vLLM (chosen), Claude Batch API (built, optional) | $0 on existing GPUs; 99.7% valid structured JSON; night flag F1 0.989 vs GT | Claude haiku/opus via Batch API — better counts expected, ~$15/5K frames; the seam keeps both live |
| **Chat agent** | **qwen2.5:14b** via Ollama (default), **claude-opus-4-8** for deployment | $0 local demo on an M4 Pro; solid tool calling for SQL agents; the Claude flip exists precisely because local reliability has measured limits (language drift, occasional misread stats) | qwen2.5:7b (faster, weaker SQL), qwen3:8b (less battle-tested tool calling); vLLM-on-GPU via the same transport |

The recurring pattern: **every LLM/VLM dependency sits behind a provider seam**
(batch labeling: `BatchTransport`; chat: `ChatTransport`), so cost, quality, and
deployment environment are config choices, not rewrites.

## 5. Results and evaluations

### Detector (Phase 2–3) — production model `yolov8m@960`

| Metric | Overall | Day | **Night** | Clear | Rain |
|---|---|---|---|---|---|
| mAP50 | 0.740 | 0.743 | **0.542** | 0.739 | 0.703 |
| mAP50-95 | 0.463 | 0.465 | **0.329** | 0.465 | 0.431 |

- Sweep: baseline yolov8n@640 scored 0.608 overall mAP50 → the promoted
  yolov8m@960 gains **+0.132**, with the same relative slice pattern.
- **Headline finding: night costs ~20 mAP50 points** (0.74 → 0.54) while rain
  costs ~4 — night is the slice a real release would gate on, and it drives the
  promotion gate, the drift demo, and the 6d experiment design.

### VLM auto-labeling (Phase 6b) — Qwen2.5-VL-7B, 5,000 frames, $0

| Attribute vs GT | Accuracy | F1 |
|---|---|---|
| night | **0.997** | **0.989** |
| rain | 0.945 | 0.865 |

Object counts: excellent on sparse classes (bus MAE 0.07, motorcycle 0.05), but
**accuracy degrades sharply with crowding** — MAE rises 0.08 → 0.87 → 2.81 →
**6.67** as GT count goes 0 → 1–3 → 4–9 → 10+; pedestrians are systematically
undercounted (recall 0.58). Conclusion: VLM labels are production-useful for
*scene-level condition tagging* and *presence*, not for dense counting.

### Active learning (Phase 6d) — mined vs random, identical val set

| Arm | Train imgs | Overall mAP50-95 | Night mAP50-95 |
|---|---|---|---|
| baseline (25% of scenes) | 7,035 | 0.2477 | 0.1667 |
| + 1,500 similarity-mined | 8,535 | 0.2637 (+0.016) | 0.1739 (+0.007) |
| + 1,500 random (control) | 8,535 | **0.2817 (+0.034)** | 0.1619 (−0.005) |
| + 958 weak (pseudo, VLM-verified) | 7,993 | 0.2539 (+0.006) | 0.1483 (−0.018) |

*(Round 2 added three acquisition-score arms — best: `rate_strat` at +0.0239
overall/+0.0069 night; none beat the random gate. Round 3 added rate-weighted
graph-community budgets: `graph_rate_night` posts the project's best night gain,
**+0.0101 night mAP50-95**, at +0.0254 overall. Full nine-arm table and analysis:
docs/ACTIVE_LEARNING.md. Weak supervision — no GT on the added frames, detector
proposes + VLM verifies — retains 18% of random's GT gain (+0.0062 vs +0.0340);
three-way decomposition: dropping the 542 verifier-rejected frames costs ~50% of
the gain, losing GT on the 958 kept frames costs another ~32%. A second weak arm
on the night-champion `graph_rate_night` retains overall GT gain *more*
efficiently (39.4% vs 18.2%) but its night gain inverts, +0.0101 → **−0.0262**,
with a GT twin proving the damage is the labels, not the dropped frames. Full
breakdown: docs/ACTIVE_LEARNING.md, "Weak supervision" and "A second arm: the
night champion".)*

**The random control beat similarity mining** — a negative result worth more than
a fake win, with a quantified mechanism:

- **Diversity beats similarity:** mined frames span 219 scenes (top-10 scenes hold
  21% of them); random spans 501 scenes. Near-duplicate mining around failure
  centroids adds little new information per frame.
- **The acquisition score was crowd-biased:** absolute miss counts rank crowded
  day frames first (top-1000 failures average 9.3 GT boxes vs 4.8 corpus-wide;
  only 1.5% night) — so zero night frames were mined even though night is the
  real weakness. Per-object miss *rates* are similar day vs night; the night gap
  lives in confidence/localization, which the score couldn't see.
- The harness itself is the win: any future acquisition function is gated on
  beating an equal-budget random control.

### Dataset chat (Phase 6c) — live transcripts

7 questions answered live (10–60 s each on the M4 Pro, $0): night-scene counts
matching the analytics docs, a 98.5% VLM-vs-GT night agreement (independently
consistent with 6b's F1 0.989), semantic retrieval of construction zones and
foggy frames with attached thumbnails, and self-repair through SQL binder errors.
Documented limitations of the local model — occasional language drift, one subtly
misinterpreted statistic, hallucinated frame tokens (safely dropped) — are exactly
why the Claude flip exists. Full transcripts: docs/DATASET_CHAT.md.

## 6. What we achieved

- **A complete, reproducible AV perception pipeline** — raw metadata → validated
  Parquet → trained detector → sliced evaluation → registry promotion → serving →
  drift monitoring — operated across two machines by one command (`gpu-run.sh`).
- **A working data-engine layer**: semantic search over 204,894 frames, $0 VLM
  labeling with a real GT evaluation, natural-language dataset chat, and a
  measured active-learning loop with leakage guards and a random control.
- **Honest measurement culture**: night gap quantified (−20 mAP50); VLM counting
  limits quantified (MAE 6.67 at 10+ objects); active-learning similarity mining
  *disproven* against random at equal budget, with the mechanism identified.
- **Engineering hygiene**: 137 offline tests (torch-free CI), mypy strict, ruff,
  data versioning (DVC), experiment tracking (MLflow + W&B), branch-protected CI,
  provider seams for every paid dependency, and a security-reviewed SQL guard.

## 7. Problems this targets and solves

| Pain point | What the engine does |
|---|---|
| "Our dataset is a pile of files nobody can query" | Validated Parquet + DuckDB views + a chat agent anyone can ask in English |
| "One overall mAP hides where the model actually fails" | Condition-sliced eval, promotion gates on the worst slice, drift monitoring keyed to the same signal |
| "Finding edge cases means scrolling through images" | Text/image/similar semantic search; failure-cluster mining |
| "Labeling everything is unaffordable" | Stratified sampling + VLM auto-labels (with a measured quality envelope), $0 self-hosted option |
| "Which new data is worth adding?" | The 6d harness: mined-vs-random controlled retraining, so acquisition ideas are *tested*, not assumed |
| "Experiments aren't reproducible" | Content-hashed data versions in every run, config-driven pipelines, idempotent builds, tracked registries |
| "Cloud AI costs are unpredictable" | Every LLM/VLM behind a provider seam: local-$0 by default, paid API by config flip |

## 8. Use cases

- **AV / robotics perception teams** (the primary framing): dataset curation,
  slice-aware model release gates, edge-case retrieval, label bootstrapping.
- **Dataset operations & QA**: the availability manifest, Great Expectations
  suites, and the chat agent make "is the dataset healthy?" a one-question job.
- **ML research harness**: the reduced-baseline + random-control pattern in 6d is
  a reusable template for evaluating *any* data-acquisition strategy.
- **Applied-LLM reference**: two production-shaped LLM integrations (batch
  structured outputs; tool-calling agent with a guarded SQL tool) with provider
  seams, offline test fakes, and honest limitation write-ups.
- **Portfolio / teaching artifact**: a single repo demonstrating the full MLOps
  arc with real numbers at non-toy scale.

## 9. Future scope

Ordered roughly by value-per-effort:

1. **Active learning round 2** — the 6e graph-diversity arm delivered the first 6d
   lesson (a diversity term): GDS-Louvain community sampling over the SIMILAR_TO graph
   matched the random control's overall gain (+0.0344) *and* recovered night (+0.0036,
   where random regressed), spanning 473 scenes vs mined's 219. Was open (built in
   round 2): rate-based acquisition and night/rain-stratified
   quotas — the harness and random-control gate already exist; only the score changes.
   *Round 2 is DONE (arms `rate`, `strat`, `rate_strat` — smoothed-rate scoring and
   night 25% / rain 20% floors, spec `2026-08-02-al-round-2-design.md`). Results: the
   mechanisms compose (`rate_strat` +0.0239 overall beats both components; only arm
   besides `mined`/`graph` to lift overall AND night), `rate` posts the best night
   mAP50 of all seven arms (0.3076), quotas-without-a-night-aware-score actively hurt
   (`strat` night −0.0086) — but none beats the random control (+0.0340) or `graph`
   (+0.0344) overall: the diversity lesson survives a better score. Next: combine
   graph-community diversity with rate-weighted, night-floored budgets
   (docs/ACTIVE_LEARNING.md, "Round 2 results").*
   *Round 3 — DONE (budget ∝ embedding-routed smoothed-rate failure mass across
   Louvain communities, floor 1/community, ± night-375 floor; GDS Louvain pinned
   deterministic via `concurrency: 1`; spec `2026-08-04-al-round-3-design.md`).
   Results: rate-mass weighting alone matches size weighting (+0.0330 vs +0.0344,
   within partition noise) — but `graph_rate_night` delivers **the project's best
   night gain of all 13 arms, +0.0101 night mAP50-95** (0.1768) at ~75% of the
   best overall gain. With diversity held by the community floor, an explicit night
   floor finally moves the night slice; the overall/night trade-off is now an
   explicit, tunable choice (docs/ACTIVE_LEARNING.md, "Round 3 results").*
2. **Ego-pose / 3D geometry ingestion (Phase B) — DONE.** `ingest-geometry` now persists
   the ego pose + all 1.17M 3D GT boxes (world position, size, heading, velocity, BEV
   distance-to-ego, ego-relative coords, instance tracking) into `ego_pose`/`annotations_3d`/
   `instances` Parquet, DuckDB views, and Neo4j `EgoPose`/`ObjectObservation`/`ObjectInstance`
   nodes with point indexes. The project plan's "pedestrians within 5 m of ego at night" is
   answerable in both SQL and Cypher (see docs/GRAPH.md). **CAN-bus (steering/braking) —
   DONE.** `ingest-canbus` keyframe-aligns the nuScenes `can_bus` expansion (vehicle
   speed/steering/brake/throttle, windowed longitudinal accel, `is_hard_braking`) into
   `canbus.parquet`, a DuckDB view, and CAN properties on the `EgoPose` graph nodes:
   **34,149** keyframe-aligned rows, **94** hard-braking, an independent speed
   cross-check of **r = 0.999** against the GT-derived `ego_pose.speed_mps`, and the
   flagship "hard braking near pedestrians" query returns **30** identically in SQL and
   Cypher (docs/DATA.md, docs/GRAPH.md).
3. **Terraform cloud deployment** (the remaining roadmap item) — lift the compose
   stack to a cloud host; the chat agent's Anthropic flip means no GPU is needed
   for any serving-path component.
4. **Close the 6b→6d loop — DONE.** Use VLM labels as *weak supervision*:
   auto-label mined frames instead of relying on GT, making the engine work on
   genuinely unlabeled data (the real-world case). *The verifier (detector
   proposes, VLM confirms within ±1 count per class, frame-level accept/reject)
   retains **63.9%** of `random`'s 1,500 mined frames (958). Weak supervision
   retains **18%** of `random`'s GT gain (+0.0062 vs +0.0340 overall mAP50-95),
   and the loss splits roughly **~50% dropped frames / ~32% label quality** on the
   frames it keeps. Design, pipeline, and the full three-way decomposition:
   docs/ACTIVE_LEARNING.md, "Weak supervision", spec
   `2026-08-06-vlm-weak-supervision-design.md`. A second weak-supervision arm on
   round 3's night champion (`graph_rate_night`, 30.9% night in its mined set)
   sharpens this into a paradox: it retains **more** of the overall GT gain than
   the random arm did (39.4% vs 18.2%), because sparser night frames make
   detector/VLM agreement easier — but the night gain itself *inverts*, from GT's
   +0.0101 to the pseudo-labelled arm's **−0.0262**. The GT twin
   (`weak_graph_rate_night_gt`, the identical 1,101 accepted frames) keeps +0.0099
   of the night gain, proving — on a single seed, with an effect size (a 0.0361
   night swing between twins vs a 0.0002 frame cost) far outside the run-to-run
   noise band — that the damage is the *labels*, not the frames the verifier
   dropped. docs/ACTIVE_LEARNING.md, "A second arm: the night champion".*
5. **Chat agent upgrades** — streaming responses, chart generation from SQL
   results, a saved-questions gallery in Streamlit, and evaluation harness for
   answer correctness (the logged JSONL is already the dataset for it).
6. **Richer monitoring** — score production captures with the drift job on a
   schedule, alert on night-share/brightness shifts, and correlate drift windows
   with slice metrics.
7. **Multi-camera + temporal training** — the ingestion already carries all 6
   cameras; training currently uses them frame-independently. Scene-level
   train/val splits are in place, enabling sequence models later.
8. **Scale-out storage** — if the corpus grows past single-machine Parquet,
   the DuckDB seam makes an Iceberg/lakehouse migration localized to `catalog.py`
   and the ingestion writers.

## 10. Document index

| Doc | Covers |
|---|---|
| [DATA.md](DATA.md) | Phase 1 ingestion, schemas, validation, DVC |
| [ANALYTICS.md](ANALYTICS.md) | DuckDB views + example queries |
| [EVALUATION.md](EVALUATION.md) | Sliced metrics, sweep, promotion policy |
| [GPU_SERVER.md](GPU_SERVER.md) | Two-machine workflow, `gpu-run.sh` |
| [MONITORING.md](MONITORING.md) | Request capture + Evidently drift |
| [AUTOLABEL_EVAL.md](AUTOLABEL_EVAL.md) | 6b methodology, sampling, results, cost |
| [DATASET_CHAT.md](DATASET_CHAT.md) | 6c architecture, SQL guard, transcripts |
| [ACTIVE_LEARNING.md](ACTIVE_LEARNING.md) | 6d experiment design + results |
| [GRAPH.md](GRAPH.md) | 6e Neo4j context graph, `run_cypher`, Cypher guard |
| [PHASE4_PLAN.md](PHASE4_PLAN.md) | Serving design notes |
