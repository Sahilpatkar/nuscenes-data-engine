# EVALUATION (Phase 3)

How the trained detector is evaluated — overall and **sliced by driving condition** — and
the policy that governs promotion in the MLflow Model Registry.

Run with:

```bash
uv run nuscenes-data-engine evaluate [--weights <best.pt>] [--device 0] [--register]
```

`evaluate` runs Ultralytics validation on the official nuScenes **val** split, then on
per-condition subsets built from `scene.description` (see `configs/eval.yaml`), logs all
metrics to MLflow, and — with `--register` — registers the model and applies the
promotion aliases.

## Model under evaluation

`yolov8n` fine-tuned for 30 epochs on all 6 cameras (run `yolov8n_imgsz640_e30`,
data version `42c0d91e`). Evaluated on the full 36,114-image val split.

## Overall metrics

| mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|
| 0.608 | 0.363 | 0.740 | 0.542 |

## Condition-sliced metrics — the differentiator

Slices are derived from nuScenes scene descriptions (night/day, rain/clear). This mirrors
how AV teams actually track perception quality, where a good overall number can hide
failure modes in hard conditions.

| Slice | Images | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|
| **day** | 32,502 | 0.612 | 0.366 | 0.744 | 0.545 |
| **night** | 3,612 | **0.421** | **0.238** | 0.680 | 0.376 |
| **clear** | 29,586 | 0.605 | 0.365 | 0.737 | 0.542 |
| **rain** | 6,528 | 0.591 | 0.342 | 0.746 | 0.526 |

**Key finding:** the detector is **markedly worse at night** — mAP50 drops from **0.61
(day) → 0.42 (night)** and mAP50-95 from 0.366 → 0.238 — driven mostly by lower recall
(0.55 → 0.38): it misses more objects in the dark. Rain costs comparatively little
(mAP50 0.61 → 0.59). This is the kind of slice that would gate a real AV release even when
the aggregate number looks healthy.

## Promotion policy

Defined in `configs/eval.yaml` under `promotion`, gated on **mAP50-95**:

| Gate | Threshold | This model | Pass |
|---|---|---|---|
| overall mAP | ≥ 0.30 | 0.363 | ✅ |
| night mAP | ≥ 0.20 | 0.238 | ✅ |

The night gate is deliberate: it blocks a model that looks fine overall but degrades in the
hardest condition.

## Registry flow (MLflow)

`evaluate --register` registers the model version and applies **aliases** (the current
MLflow approach that replaces deprecated stage transitions):

- Every evaluated model → alias **`staging`**.
- Only models that clear **both** gates → alias **`production`**.

This model passed, so `nuscenes-yolo-detector` **v1** carries both `staging` and
`production`. The serving layer (Phase 4) loads `models:/nuscenes-yolo-detector@production`.

```bash
# resolve the current production model
uv run python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlruns/mlflow.db'); \
  print(mlflow.MlflowClient().get_model_version_by_alias('nuscenes-yolo-detector','production'))"
```

_Registry + `mlruns/` live on the infra machine in the two-machine topology; the GPU server
produces the run and syncs `mlruns/` over._
