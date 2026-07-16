# Production ML Pipeline for Autonomous Vehicle Perception

An end-to-end MLOps project built on the [nuScenes](https://www.nuscenes.org/) dataset. The system ingests multimodal autonomous vehicle sensor data, validates and versions it, trains and evaluates an object detection model, serves it behind an API, and monitors it for drift — with CI/CD tying everything together.

**Goal:** A portfolio project demonstrating both ML engineering and data infrastructure skills for AI/ML/data infrastructure roles.

---

## Why This Project

- **nuScenes is industry-real.** It is a benchmark dataset used by actual autonomous vehicle companies, with 6 cameras, LiDAR, radar, and IMU/GPS data across 1,000 driving scenes. Its relational schema and 350GB scale force real data engineering decisions.
- **The pipeline is the product.** Recruiters for MLOps and infrastructure roles care more about reproducibility, data versioning, orchestration, and deployment than leaderboard scores. The model here is deliberately simple; everything around it is deliberately production-grade.
- **Every component maps to a job-posting keyword** that can be defended in depth during an interview.

## Core Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Dataset size | `v1.0-mini` (~4GB, 10 scenes) for development | Runs on a laptop; pipeline is designed as if the full 350GB were coming |
| Model | Fine-tune a pretrained 2D detector (e.g., YOLOv8) on camera images | Keeps ML complexity tractable; effort goes into the pipeline |
| Annotations | Project nuScenes 3D boxes into 2D camera frames | Well-documented in the nuScenes devkit; good learning exercise |
| Storage format | Parquet for metadata, original JPEGs for images | Columnar analytics on metadata; images don't benefit from re-encoding |
| Everything local first | docker-compose stack, MinIO instead of S3 | Zero cloud cost during development; cloud deploy is a stretch goal |

---

## Architecture

```
nuScenes raw data ──► Ingest & validate ──► Versioned store
(cameras, LiDAR,      (Great Expectations)   (DVC + Parquet)
 radar, IMU/GPS)                                   │
                                                   ▼
      Monitoring ◄── Serving API ◄── Model registry ◄── Training pipeline
      (Evidently)    (FastAPI +      (MLflow,            (Dagster + MLflow
          │           Docker)         staging→prod)       experiment tracking)
          └──► triggers retraining
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Python, pandas, PyArrow/Parquet, nuScenes devkit |
| Data validation | Great Expectations |
| Data versioning | DVC + MinIO (local S3-compatible object store) |
| Orchestration | Dagster (or Prefect) |
| Experiment tracking & registry | MLflow |
| Model | PyTorch, Ultralytics YOLOv8 (pretrained, fine-tuned) |
| Serving | FastAPI, Docker, Streamlit (demo UI) |
| Monitoring | Evidently |
| CI/CD | GitHub Actions, pre-commit, pytest |
| Infra | docker-compose (stretch: Terraform + a cloud VM) |

---

## Phased Build Plan (~10–12 weeks part-time)

### Phase 1 — Data Engineering (Weeks 1–3)

**Objective:** Turn raw nuScenes data into a validated, versioned, analytics-ready dataset.

- [ ] Download `v1.0-mini`; install and explore the `nuscenes-devkit`
- [ ] Learn the schema: `scene → sample → sample_data → sample_annotation` plus `calibrated_sensor`, `ego_pose`, `category`
- [ ] Build an ingestion script that flattens the relational JSON tables into Parquet metadata tables (one row per annotation, joined with scene/sensor/weather context)
- [ ] Project 3D annotation boxes into 2D camera-frame bounding boxes using the devkit's geometry utilities
- [ ] Add Great Expectations suites: schema conformance, null checks, bounding boxes within image bounds, valid category labels, per-scene sample counts
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
│   └── monitoring/         # Evidently reports
├── app/                    # Streamlit demo
├── tests/
└── .github/workflows/      # CI pipelines
```

## Resume Bullet (draft)

> Built an end-to-end ML pipeline for autonomous vehicle perception on the nuScenes dataset: data validation and versioning (Great Expectations, DVC), orchestrated training with experiment tracking (Dagster, MLflow), condition-sliced model evaluation, containerized model serving (FastAPI, Docker), drift monitoring (Evidently), and CI/CD (GitHub Actions).

## Skills This Demonstrates

- **Data infrastructure:** ETL on a complex relational schema, columnar storage, data validation, data versioning, object storage
- **ML engineering:** transfer learning, experiment tracking, rigorous sliced evaluation, model registry workflows
- **MLOps / platform:** orchestration, containerization, API design, monitoring, CI/CD, reproducibility
- **Judgment:** scoping (mini dataset, pretrained model), designing for scale while developing on a sample

## Key References

- nuScenes devkit: https://github.com/nutonomy/nuscenes-devkit
- nuScenes schema docs: https://www.nuscenes.org/nuscenes#data-format
- MLflow: https://mlflow.org/docs/latest/index.html
- Dagster: https://docs.dagster.io/
- DVC: https://dvc.org/doc
- Great Expectations: https://docs.greatexpectations.io/
- Evidently: https://docs.evidentlyai.com/
