# Phase 4 — Serving (FastAPI + Streamlit + Docker) — implementation plan

> Resume point for Phase 4. Phases 1–3 + the model-improvement pass are done and pushed;
> the promoted model is `nuscenes-yolo-detector` **@production = v2** (yolov8m @ 960,
> overall mAP50 0.740).

## Context

Phase 4 puts the promoted model behind a real API and a browsable demo — the "production
model behind an API, demoable in a browser" deliverable.

Per the compute/infra split, **serving runs on the infra machine** (loads the model from the
MLflow registry there), not the GPU server. Build + smoke-test on either; Docker build/run
happens on the infra machine. Stubs already exist:
- `src/nuscenes_data_engine/serving/app.py` (FastAPI + `/health`)
- `src/nuscenes_data_engine/serving/schemas.py` (`Detection`, `PredictResponse`, `HealthResponse`)
- `app/streamlit_app.py` (stub)
- `serve` extra: fastapi / uvicorn / streamlit / python-multipart

Verified model-loading path: `MlflowClient().get_model_version_by_alias("nuscenes-yolo-detector",
"production")` → v2, then `mlflow.artifacts.download_artifacts(mv.source)` → a dir containing
`best.pt`. Loader = resolve alias → download artifact → `YOLO(best.pt)`.

## Step 1 — Model loader (`serving/model.py`, new)
- `load_production_model()` → resolve `models:/<name>@<alias>` via
  `MlflowClient.get_model_version_by_alias`, `mlflow.artifacts.download_artifacts(mv.source)`
  → find `best.pt`, `configure_ultralytics()` (reuse `training/runtime.py`), `YOLO(best.pt)`.
  Returns `(model, version:str)`. **Fallback** to a `SERVING_WEIGHTS` path (config) when the
  registry/alias is unavailable, so the API can run from a raw checkpoint too. Cache as a
  module singleton (load once).
- Class names come from `model.names`; no hard-coded label list.

## Step 2 — Serving config (extend `config.py` `Settings`)
Add `serving_model_name` (`nuscenes-yolo-detector`), `serving_model_alias` (`production`),
`serving_imgsz` (960), `serving_conf` (0.25), `serving_device` (`cpu` — infra machine may be
CPU-only), `serving_weights` (optional fallback). Mirror in `.env.example`.

## Step 3 — FastAPI app (`serving/app.py`)
- **Lifespan** startup loads the production model into `app.state` (model + version); log the
  version/source served.
- `GET /health` → `HealthResponse(status, model_loaded, model_version)`.
- `POST /predict` → `UploadFile` (multipart) → decode (PIL/np) → `model.predict(imgsz, conf,
  device)` → build `Detection[]` (`label=model.names[cls]`, `confidence`, `bbox` xyxy) →
  `PredictResponse(detections, model_version, image_width/height)`. Basic per-request logging.
- `POST /predict/annotated` → same inference, returns the annotated image (Ultralytics
  `result.plot()` → PNG `StreamingResponse`) — the "optional annotated image".
- Extend `serving/schemas.py`: add `model_version` to `HealthResponse`; add
  `image_width/height` + `n_detections` to `PredictResponse`.
- Wire the existing CLI `serve` command to `uvicorn nuscenes_data_engine.serving.app:app`.

## Step 4 — Streamlit demo (`app/streamlit_app.py`)
Upload an image **or** pick a bundled sample (a few nuScenes CAM_FRONT frames copied into
`app/samples/` so the demo needs no dataset access) → call the FastAPI `/predict` (API URL
from env, default `http://localhost:8000`) → draw boxes + labels/confidence and show a
detections table. Degrade gracefully if the API is down.

## Step 5 — Dockerize
- `serving/Dockerfile`: `python:3.11-slim` + uv, install base + `serve` + `train` (CPU torch)
  extras, copy `src/`, `CMD uvicorn …:app --host 0.0.0.0 --port 8000`.
- Update `docker-compose.yml`: add `api` and `streamlit` services (build from the Dockerfile),
  mount `./mlruns` (registry access) + pass `MLFLOW_TRACKING_URI`; `api` depends_on `mlflow`.
  Labelled infra-machine-only.

## Step 6 — Load/latency sanity check
Small helper (script or `serve --benchmark`) timing N sequential `/predict` calls on a sample
image; print p50/p95 latency; record in the README.

## Step 7 — Tests + docs
- `tests/test_serving.py`: FastAPI `TestClient` — `/health` returns ok; `/predict` with a small
  synthetic image returns a valid `PredictResponse`. Guard the model-dependent test on a
  lightweight loader (load a tiny `yolov8n.pt` via the `SERVING_WEIGHTS` fallback so tests
  don't need the registry or the 960 model) — keeps CI fast and offline.
- README **Serving (Phase 4)** section: run the API (`make serve` / uvicorn), run Streamlit,
  `docker compose up api streamlit`, latency numbers, and the `models:/…@production` note.
  Mark roadmap Phase 4 ✅.

## Critical files
- New: `src/nuscenes_data_engine/serving/model.py`, `serving/Dockerfile`, `tests/test_serving.py`,
  `app/samples/` (a few JPEGs).
- Edit: `serving/app.py`, `serving/schemas.py`, `app/streamlit_app.py`, `config.py`,
  `.env.example`, `docker-compose.yml`, `README.md`.
- Reuse: `configure_ultralytics()` (`training/runtime.py`), `get_settings()` (`config.py`),
  the alias-resolution pattern from `evaluation/registry.py`, the existing `serve` CLI stub.

## Verification
- `uv sync --extra serve --extra train`; `uv run nuscenes-data-engine serve` (uvicorn up).
- `curl :8000/health` → `{status:ok, model_loaded:true, model_version:"2"}`.
- `curl -F file=@app/samples/<img>.jpg :8000/predict` → JSON detections; `/predict/annotated`
  returns a PNG with boxes.
- `uv run streamlit run app/streamlit_app.py` → upload/pick image → boxes rendered.
- `pytest -q` green incl. `test_serving.py`; `ruff`, `ruff format --check`, `mypy` clean.
- Latency check prints p50/p95; numbers land in README.

## Risks / notes
- Serving needs `torch`+`ultralytics` (the `train` extra) alongside `serve`; on a CPU-only
  infra machine, yolov8m@960 inference is ~1–2 s/image — fine for a demo, documented.
  `serving_device`/`serving_imgsz` are configurable.
- The Docker image is large (torch); built/run on the infra machine.
- Registry loading needs `mlruns/` present (synced) or an MLflow server reachable; the
  `SERVING_WEIGHTS` fallback covers the "just run it from a checkpoint" case.
- Streamlit calls the API (decoupled) rather than loading the model inline — the more
  production-shaped design that `docker compose up` demonstrates.
