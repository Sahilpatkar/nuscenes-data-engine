# nuscenes-data-engine

End-to-end **MLOps pipeline for autonomous-vehicle perception** on the
[nuScenes](https://www.nuscenes.org/) dataset. Raw multimodal sensor data is
ingested, validated, and versioned; a 2D object detector is trained, tracked, and
evaluated (with condition-sliced metrics); the production model is served behind an
API and monitored for drift — all tied together with CI/CD.

> The pipeline is the product. The model is deliberately simple; everything around it
> is deliberately production-grade. See [nuscenes-mlops-project-plan.md](nuscenes-mlops-project-plan.md)
> for the full design and rationale.

## Status

🚧 **Scaffolded.** Directory structure, tooling, and stub modules are in place; each
pipeline stage is a `TODO` filled in during its build phase (below).

## Toolchain

- **Python 3.11**, managed with [`uv`](https://docs.astral.sh/uv/)
- Packaging: `pyproject.toml` (hatchling), `src/` layout
- Quality: `ruff` (lint + format), `mypy` (strict), `pytest`
- Infra: `docker-compose` (MinIO + MLflow), DVC, GitHub Actions

## Quickstart

```bash
uv sync --extra dev              # create .venv, install base + dev deps
uv run nuscenes-data-engine --help
uv run nuscenes-data-engine --version
uv run pytest                    # smoke tests pass green
```

Common tasks via the Makefile:

```bash
make setup      # uv sync --extra dev
make check      # ruff + mypy + pytest
make infra-up   # start MinIO + MLflow (docker compose)
```

Heavy dependencies are opt-in extras so the base install stays light:

```bash
uv sync --extra data    # Phase 1: nuscenes-devkit, Great Expectations, DVC, Evidently
uv sync --extra train   # Phase 2–3: torch, ultralytics, mlflow, dagster
uv sync --extra serve   # Phase 4: fastapi, uvicorn, streamlit
```

## Data

The nuScenes `v1.0-trainval` dataset is read (read-only) from
`/data/ggare/datasets/nuscenes/` — it is **not** copied into the repo. Configure the
path via `.env` (see [.env.example](.env.example)). Processed, DVC-tracked outputs go
under `data/processed/`.

## Repo layout

```
configs/                    experiment + pipeline configs (data/train/eval.yaml)
data/                       DVC-tracked pipeline outputs (raw/, processed/)
src/nuscenes_data_engine/
  ingestion/                devkit parsing, Parquet export, 3D→2D projection
  validation/               Great Expectations suites
  training/                 Dagster job + YOLO fine-tuning + MLflow
  evaluation/               mAP + condition-sliced metrics
  serving/                  FastAPI app
  monitoring/               Evidently drift reports
app/                        Streamlit demo UI
tests/                      pytest suite
docs/                       DATA.md, EVALUATION.md
.github/workflows/          CI (ruff + mypy + pytest)
```

## Build roadmap

| Phase | Focus | Deliverable |
|---|---|---|
| 1 | Data engineering | `make ingest` → validated, versioned Parquet dataset |
| 2 | Training pipeline | Reproducible Dagster + MLflow runs |
| 3 | Evaluation & registry | Sliced metrics + promotion policy |
| 4 | Serving | FastAPI + Streamlit demo via `docker compose up` |
| 5 | Monitoring & CI/CD | Evidently drift reports, green CI |

## License

MIT
