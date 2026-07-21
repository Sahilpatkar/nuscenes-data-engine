# A Miniature AV Data Engine: Search, Curate, Train, Serve, and Monitor on nuScenes

An end-to-end MLOps + data engine project built on the [nuScenes](https://www.nuscenes.org/) dataset. The system ingests multimodal autonomous vehicle sensor data, validates and versions it, trains and evaluates an object detection model, serves it behind an API, and monitors it for drift — then layers a modern **data engine** on top: natural-language scene search over embeddings, VLM-based auto-labeling with rigorous evaluation, a chat interface over the dataset, and embedding-driven active learning that closes the loop back to training.

**The problem:** Autonomous vehicles generate enormous volumes of sensor data, but model quality is bottlenecked by a needle-in-a-haystack problem — the scenarios that matter (a pedestrian stepping out at night, a hard-braking event in rain) are buried in hours of uneventful driving, and no team can manually review it all.

**The goal:** Build a system that makes a large driving dataset *searchable, trustworthy, and self-improving* — where an engineer can ask for "construction zones at night" in plain language and get back frames in seconds, where every piece of data is validated, versioned, and reproducible, and where the system finds its own model's blind spots and mines the exact data needed to fix them — closing the loop from raw sensor logs to a measurably better perception model.

*(Secondary goal: in building it, demonstrate ML engineering, data infrastructure, and modern AI-stack skills — embeddings, vector search, VLM/LLM, agentic RAG — for AI/ML/data infrastructure roles.)*

---

## Why This Project

- **nuScenes is industry-real.** It is a benchmark dataset used by actual autonomous vehicle companies, with 6 cameras, LiDAR, radar, and IMU/GPS data across 1,000 driving scenes. Its relational schema and 350GB scale force real data engineering decisions.
- **The pipeline is the product.** Recruiters for MLOps and infrastructure roles care more about reproducibility, data versioning, orchestration, and deployment than leaderboard scores. The model here is deliberately simple; everything around it is deliberately production-grade.
- **Every component maps to a job-posting keyword** that can be defended in depth during an interview.
- **The data engine layer is the differentiator.** The hardest problem in AV/ML today isn't training models — it's mining huge sensor logs for the right data. Waymo, Tesla, and Motional all build "data engines" for exactly this. Phase 6 is a miniature version: semantic search, auto-labeling, dataset chat, and hard-example mining. Standard MLOps gets past the resume screen; the data engine is what interviewers remember.

## Core Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Dataset size | Full `v1.0-trainval` (850 scenes, 34,149 keyframe samples, ~212GB on a shared server); `v1.0-mini` kept as the fast dev environment | Real scale forces real data engineering; mini enables minutes-long iteration with identical code |
| Working set | **Keyframes only** (~240K camera images, ~40–50GB) for training, embeddings, and labeling | Annotations exist only on keyframes; sweeps stay in the raw zone |
| Model | Fine-tune a pretrained 2D detector (e.g., YOLOv8) on camera images | Keeps ML complexity tractable; effort goes into the pipeline |
| Annotations | Project nuScenes 3D boxes into 2D camera frames | Well-documented in the nuScenes devkit; good learning exercise |
| Storage format | Parquet tables managed as an **Apache Iceberg** (or Delta Lake) lakehouse, queried with **DuckDB**; original JPEGs for images | Lakehouse table formats + DuckDB are the current data-infra standard; images don't benefit from re-encoding |
| Analytics engine | DuckDB over the lakehouse tables | Fast local analytics, zero infrastructure, heavily used in industry |
| Embeddings | SigLIP or CLIP image embeddings for every camera frame | Powers semantic search, curation, and failure-mode clustering |
| Vector store | LanceDB (or Qdrant) | Modern, lightweight, embedded-friendly vector databases |
| Auto-labeling | VLM (local Qwen-VL/LLaVA, or Claude API) generating structured JSON scene descriptions | nuScenes ground truth lets you *measure* the VLM's labeling accuracy — a rare, valuable skill demo |
| Everything local first | docker-compose stack, MinIO instead of S3 | Zero cloud cost during development; cloud deploy is a stretch goal |

---

## Working at Full Scale (v1.0-trainval, 212GB)

The raw data lives on a **shared server** and is treated as an **immutable raw zone** — never modified, never copied wholesale. Everything derived (Iceberg tables, embeddings, YOLO exports) lives in a separate project workspace.

**Data inventory (audited):**

| Present | Missing (referenced by metadata, absent on disk) |
|---|---|
| All 13 metadata JSON tables (2.63M `sample_data` records, full annotations) | All 5 RADAR sensors — keyframes and sweeps (~940K sweeps/sensor) |
| All 6 cameras — keyframes **and** sweeps | LIDAR_TOP sweeps (297,737 referenced) |
| LIDAR_TOP keyframes (34,149 `.pcd.bin`) | |
| Maps + CAN bus (complete) | |

**Rules this imposes on the pipeline:**

1. **Availability manifest first.** Phase 1 gains a validation step that cross-checks every `sample_data` record against the filesystem and writes a per-sensor, per-scene availability manifest as a lakehouse table. All downstream stages filter on this manifest — never trust the metadata's file references. (The devkit loads metadata fine with missing blobs, but any call touching a missing file — e.g., sweep aggregation — will crash.)
2. **Keyframes are the working set.** Training, embeddings, evaluation, and labeling operate on `is_key_frame=True` camera data (~240K images, ~40–50GB). Camera sweeps and LiDAR stay in the raw zone for future extensions.
3. **DVC tracks derived data only.** The 212GB raw zone is referenced by path + a checksum manifest (computed once, early — proof of what you built against on a shared resource). DVC versions the processed outputs (a few GB).
4. **Compute goes to the data where possible.** Run ingestion and embedding jobs on the shared server if permitted; otherwise copy the keyframe camera set to local NVMe once and work from that. Avoid repeatedly streaming 240K images over NFS.
5. **Two-tier execution.** Every pipeline takes a `dataset_version` config: `mini` for fast local iteration (minutes), `trainval` for real runs (hours). Same code, different scale.
6. **Resumable by design.** Ingestion and embedding jobs are idempotent and checkpointed (process per-scene or per-blob, skip completed units on restart). At this scale, something *will* crash mid-run.

**Scale-driven adjustments per phase:** embeddings run over ~240K keyframe images (a few GPU-hours; batch + checkpoint into LanceDB incrementally); VLM auto-labeling uses a **stratified sample** (~5–10K frames across scenes/weather/time-of-day — document the sampling strategy, it's a real skill); full-set YOLO fine-tuning happens on a real GPU (local or a few dollars of rented spot compute), with the pipeline developed on mini first.

**Bonus unlocked by CAN bus data:** ego speed/steering/braking signals join the lakehouse, enabling data-engine queries like *"find scenes with hard braking near pedestrians"* — add these to the chat agent's SQL tools in Phase 6c.

---

## Architecture

```
nuScenes raw data ──► Ingest & validate ──► Lakehouse store
(cameras, LiDAR,      (Great Expectations)   (Iceberg/Parquet + DVC,
 radar, IMU/GPS)                              queried via DuckDB)
                                                   │
                              ┌────────────────────┼────────────────────┐
                              ▼                    ▼                    ▼
                        DATA ENGINE          Training pipeline    Monitoring
                  ┌─────────────────────┐   (Dagster + MLflow)   (Evidently)
                  │ Frame embeddings    │          │                  │
                  │ (SigLIP → LanceDB)  │          ▼                  │
                  │ Semantic search     │    Model registry           │
                  │ VLM auto-labeling   │    (MLflow,                 │
                  │ Dataset chat agent  │     staging→prod)           │
                  │ Hard-example mining │          │                  │
                  └──────────┬──────────┘          ▼                  │
                             │               Serving API ◄────────────┘
                             │               (FastAPI + Docker)   drift alerts
                             └──► curated training data ──► retraining loop
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Python, pandas, PyArrow/Parquet, nuScenes devkit |
| Lakehouse & analytics | Apache Iceberg (or Delta Lake) table format, DuckDB query engine |
| Data validation | Great Expectations |
| Data versioning | DVC + MinIO (local S3-compatible object store) |
| Embeddings & vector search | SigLIP/CLIP (via `open_clip` or `transformers`), LanceDB or Qdrant |
| VLM auto-labeling | Qwen-VL / LLaVA locally, or Claude API; structured JSON outputs |
| Dataset chat agent | LLM tool-use / text-to-SQL over DuckDB + vector search |
| Streaming (stretch) | Redpanda (Kafka-compatible) replaying scenes as a live sensor stream |
| Orchestration | Dagster (or Prefect) |
| Experiment tracking & registry | MLflow |
| Model | PyTorch, Ultralytics YOLOv8 (pretrained, fine-tuned) |
| Serving | FastAPI, Docker, Streamlit (demo UI) |
| Monitoring | Evidently |
| CI/CD | GitHub Actions, pre-commit, pytest |
| Infra | docker-compose (stretch: Terraform + a cloud VM) |

---

## Phased Build Plan (~16–18 weeks part-time)

Phases 1–5 build the MLOps foundation; Phase 6 builds the data engine headline. Don't skip the foundation — the data engine is only credible sitting on top of real pipelines.

### Phase 1 — Data Engineering (Weeks 1–3)

**Objective:** Turn raw nuScenes data into a validated, versioned, analytics-ready dataset.

- [ ] Download `v1.0-mini` locally for fast iteration; point pipelines at the shared-server `v1.0-trainval` via config
- [ ] Learn the schema: `scene → sample → sample_data → sample_annotation` plus `calibrated_sensor`, `ego_pose`, `category`
- [ ] **Build the data availability manifest**: cross-check all 2.63M `sample_data` records against the filesystem; write per-sensor/per-scene availability as a lakehouse table; compute a checksum manifest of the raw zone
- [ ] Build an ingestion script that flattens the relational JSON tables into Parquet metadata tables (one row per annotation, joined with scene/sensor/weather context), filtered by the availability manifest — resumable and idempotent (per-scene checkpointing)
- [ ] Project 3D annotation boxes into 2D camera-frame bounding boxes using the devkit's geometry utilities
- [ ] Add Great Expectations suites: schema conformance, null checks, bounding boxes within image bounds, valid category labels, per-scene sample counts
- [ ] Write the metadata tables as an **Iceberg (or Delta) table** on MinIO instead of loose Parquet files; query them with **DuckDB** and document 3–4 example analytical queries (annotations per category, scenes by weather, etc.)
- [ ] Set up MinIO via docker-compose; initialize DVC with MinIO as the remote
- [ ] Version the processed dataset; document how to reproduce it from raw data with one command

**Deliverable:** `make ingest` produces a validated, versioned dataset. A `DATA.md` documents the schema and validation rules.

### Phase 2 — Training Pipeline (Weeks 4–6)

**Objective:** Reproducible, orchestrated, tracked training runs.

- [ ] Convert 2D annotations into YOLO training format as a pipeline step
- [ ] Define a Dagster job: `pull_data_version → prepare_dataset → train → evaluate → log_artifacts`
- [ ] Integrate MLflow: log hyperparameters, data version hash, metrics, model weights, and sample prediction images per run
- [ ] Config-driven experiments (Hydra or plain YAML): every run reproducible from `config + data version`
- [ ] Run 3–5 experiments varying learning rate, image size, and augmentation; compare in the MLflow UI

**Deliverable:** Any experiment can be rerun exactly from its config and data version. Screenshot-worthy MLflow dashboard.

### Phase 3 — Evaluation & Model Registry (Weeks 6–7)

**Objective:** Rigorous evaluation and controlled model promotion.

- [ ] Compute per-class mAP and precision/recall on a held-out split
- [ ] **Condition-sliced evaluation** — break metrics down by night vs. day and rain vs. clear using nuScenes scene descriptions (this mirrors what AV companies actually do and is a strong differentiator)
- [ ] Define promotion criteria (e.g., minimum overall mAP and minimum night-scene mAP)
- [ ] Register passing models in the MLflow Model Registry with a `staging → production` promotion flow

**Deliverable:** An `EVALUATION.md` with sliced metrics tables and a documented promotion policy.

### Phase 4 — Serving (Weeks 8–9)

**Objective:** The production model behind a real API, demoable in a browser.

- [ ] FastAPI service: loads the current `production` model from the registry, accepts an image, returns detections (JSON + optional annotated image)
- [ ] Pydantic request/response schemas, health-check endpoint, basic request logging
- [ ] Dockerize the service; add it to docker-compose
- [ ] Streamlit demo UI: upload or pick a nuScenes image, see detections rendered
- [ ] Basic load sanity check (e.g., latency at N concurrent requests) — record numbers in the README

**Deliverable:** `docker compose up` brings up the API + demo UI. A 60–90 second screen recording for the README.

### Phase 5 — Monitoring & CI/CD (Weeks 10–12)

**Objective:** Close the loop and make the repo look professionally maintained.

- [ ] Evidently: track input image statistics (brightness, resolution, detection-count distributions) against the training reference; generate drift reports
- [ ] Simulate drift (e.g., feed only night images) and show the report catching it
- [ ] GitHub Actions: lint (ruff), tests (pytest), and a smoke-test training run on a tiny data slice for every PR
- [ ] Pre-commit hooks; branch protection on `main`
- [ ] Architecture diagram + full setup instructions in the README
- [ ] **Stretch:** deploy the serving stack to a small cloud VM with Terraform

**Deliverable:** Green CI badge, drift-detection demo, one-command local deployment.

### Phase 6 — The Data Engine (Weeks 13–18) ⭐ *the differentiator*

**Objective:** A miniature version of the "data engine" AV companies build to mine sensor logs — semantic search, auto-labeling, dataset chat, and active learning.

**6a. Natural-language scene search (weeks 13–14)**

- [ ] Embed every camera frame with SigLIP (or CLIP) as a Dagster pipeline step; store vectors + frame metadata in LanceDB (or Qdrant)
- [ ] Build text-to-image search: embed a text query, retrieve nearest frames — e.g., *"construction zone at night"*, *"pedestrian crossing in rain"*, *"partially occluded truck"*
- [ ] Add a search tab to the Streamlit app: type a sentence, see matching frames with their scene context
- [ ] Also support image-to-image search ("find frames like this one")

**6b. VLM auto-labeling with evaluation (weeks 14–15)**

- [ ] Run a VLM (local Qwen-VL/LLaVA or the Claude API) over frames to produce **structured JSON** scene descriptions: weather, time of day, hazards, object counts, notable conditions
- [ ] Validate outputs against a Pydantic schema; handle malformed responses gracefully
- [ ] **Evaluate the VLM against nuScenes ground truth** where it exists (object counts vs. annotations, weather/time vs. scene descriptions): report precision/recall per attribute
- [ ] Write up findings in `AUTOLABEL_EVAL.md` — where the VLM is reliable, where it fails, and what that implies for using LLMs as labelers

**6c. Chat with the dataset (weeks 15–16)**

- [ ] Build an LLM agent with two tools: (1) text-to-SQL over the DuckDB/Iceberg metadata tables, (2) vector search over frame embeddings
- [ ] Handle questions like *"how many scenes have pedestrians within 5 meters of the ego vehicle at night?"* — the agent writes the query, runs it, and returns results with example frames
- [ ] Add a chat tab to the Streamlit app; log every agent query for inspection
- [ ] Document 5–10 impressive example questions in the README with screenshots

**6d. Embedding-based active learning (weeks 17–18)**

- [ ] Run the production detector over held-out frames; collect false negatives / low-confidence detections
- [ ] Cluster failure cases in embedding space to identify systematic failure modes (e.g., night scenes, occlusions)
- [ ] Use vector similarity to mine additional hard examples resembling the failure clusters; add them to the training set
- [ ] Retrain via the existing Phase 2 pipeline; report the before/after metric change (e.g., night-scene mAP)
- [ ] Write up the loop in `ACTIVE_LEARNING.md`: this is the "data engine improves the model" story

**Stretch — streaming ingestion**

- [ ] Replay a nuScenes scene through Redpanda (Kafka-compatible) in timestamp order as a simulated live sensor stream; consume it into the ingestion pipeline — turns the project from batch-only into streaming-capable

**Deliverable:** A Streamlit app with search + chat tabs, two evaluation write-ups, and a measured model improvement driven by embedding-based data curation.

---

## Final Portfolio Artifacts

1. **GitHub repo** — clean structure, architecture diagram, one-command setup, green CI
2. **Demo** — hosted Streamlit app or a screen recording embedded in the README
3. **Blog post** — a design-decisions walkthrough (why Parquet, why DVC, what broke and how you fixed it); publish on Medium/dev.to/personal site
4. **MLflow screenshots** — experiment comparison and registry promotion flow

## Suggested Repo Structure

```
nuscenes-mlops/
├── README.md
├── docker-compose.yml
├── Makefile
├── configs/                # experiment + pipeline configs
├── data/                   # DVC-tracked (raw/, processed/)
├── src/
│   ├── ingestion/          # devkit parsing, Parquet export, 2D projection
│   ├── validation/         # Great Expectations suites
│   ├── training/           # Dagster job, YOLO fine-tuning
│   ├── evaluation/         # mAP + condition-sliced metrics
│   ├── serving/            # FastAPI app
│   ├── monitoring/         # Evidently reports
│   └── data_engine/        # embeddings, vector search, VLM labeling, chat agent, active learning
├── app/                    # Streamlit demo
├── tests/
└── .github/workflows/      # CI pipelines
```

## Resume Bullet (draft)

> Built a miniature autonomous-vehicle data engine on the nuScenes dataset: lakehouse ingestion with validation and versioning (Iceberg, DuckDB, Great Expectations, DVC), orchestrated training with experiment tracking (Dagster, MLflow), containerized serving and drift monitoring (FastAPI, Docker, Evidently), natural-language scene search over frame embeddings (SigLIP, LanceDB), VLM auto-labeling evaluated against ground-truth annotations, an LLM agent for querying the dataset (text-to-SQL + vector search), and an embedding-based active learning loop that measurably improved night-scene detection.

Too long for one bullet on a real resume — split it into 2–3 bullets (pipeline / data engine / measured result) when formatting.

## Skills This Demonstrates

- **Data infrastructure:** ETL on a complex relational schema, lakehouse table formats (Iceberg), DuckDB analytics, data validation, data versioning, object storage, (stretch) streaming ingestion
- **ML engineering:** transfer learning, experiment tracking, rigorous sliced evaluation, model registry workflows, active learning
- **Modern AI stack:** vision-language embeddings, vector databases and semantic search, VLM/LLM structured outputs, LLM evaluation against ground truth, agentic tool use and text-to-SQL (multimodal RAG)
- **MLOps / platform:** orchestration, containerization, API design, monitoring, CI/CD, reproducibility
- **Judgment:** scoping (mini dataset, pretrained model), designing for scale while developing on a sample, closing the data-model loop

## Key References

- nuScenes devkit: https://github.com/nutonomy/nuscenes-devkit
- nuScenes schema docs: https://www.nuscenes.org/nuscenes#data-format
- MLflow: https://mlflow.org/docs/latest/index.html
- Dagster: https://docs.dagster.io/
- DVC: https://dvc.org/doc
- Great Expectations: https://docs.greatexpectations.io/
- Evidently: https://docs.evidentlyai.com/
- DuckDB: https://duckdb.org/docs/
- Apache Iceberg (PyIceberg): https://py.iceberg.apache.org/
- LanceDB: https://lancedb.github.io/lancedb/
- Qdrant: https://qdrant.tech/documentation/
- OpenCLIP / SigLIP: https://github.com/mlfoundations/open_clip
- Qwen-VL: https://github.com/QwenLM/Qwen2.5-VL
- Redpanda: https://docs.redpanda.com/
