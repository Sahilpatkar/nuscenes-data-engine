# EVALUATION (Phase 3)

How the trained detector is evaluated — overall and **sliced by driving condition** — and
the policy that governs promotion in the MLflow Model Registry.

Run with:

```bash
uv run nuscenes-data-engine evaluate [--weights <best.pt>] [--imgsz 960] [--device 0] [--register]
```

`evaluate` runs Ultralytics validation on the official nuScenes **val** split, then on
per-condition subsets built from `scene.description` (see `configs/eval.yaml`), logs all
metrics to MLflow, and — with `--register` — registers the model and applies the promotion
aliases.

## Current production model

`yolov8m` fine-tuned for 40 epochs @ 960px on all 6 cameras (run `yolov8m_imgsz960_e40`,
data version `3847cccf`, registry **v2 → `production`**). Evaluated on the full 36,114-image
val split.

## Overall metrics

| mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|
| 0.740 | 0.463 | 0.798 | 0.678 |

## Condition-sliced metrics — the differentiator

Slices are derived from nuScenes scene descriptions (night/day, rain/clear). This mirrors
how AV teams track perception quality, where a good overall number can hide failure modes in
hard conditions.

| Slice | Images | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|
| **day** | 32,502 | 0.743 | 0.465 | 0.798 | 0.681 |
| **night** | 3,612 | **0.542** | **0.329** | 0.762 | 0.498 |
| **clear** | 29,586 | 0.739 | 0.465 | 0.796 | 0.677 |
| **rain** | 6,528 | 0.703 | 0.431 | 0.787 | 0.648 |

**Key finding:** night is still the hardest condition — mAP50 **0.74 (day) → 0.54 (night)**,
driven by lower recall (0.68 → 0.50): the detector misses more objects in the dark. Rain
costs less (0.74 → 0.70). Night remains the slice a real AV release would gate on, even
though the aggregate looks strong.

## Improvement over the baseline

The first model (`yolov8n` @ 640, 30 ep) was deliberately minimal. A focused improvement
pass — **better 2D labels** (polygon-clip projection + keep small boxes), a **bigger model**
(n→m), and **higher resolution** (640→960) — lifted every slice, and most where it mattered:

| Slice (mAP50) | baseline yolov8n@640 | **yolov8m@960 (prod)** | Δ |
|---|---|---|---|
| overall | 0.608 | **0.740** | **+0.132** |
| day | 0.612 | 0.743 | +0.131 |
| **night** | 0.421 | **0.542** | **+0.121** |
| clear | 0.605 | 0.739 | +0.134 |
| rain | 0.591 | 0.703 | +0.112 |

Recall — the original bottleneck — rose from **0.54 → 0.68 overall** and **0.38 → 0.50 at
night**. Higher resolution was the single biggest recall lever (small/distant objects survive
downsampling), exactly the hypothesis the sliced eval pointed to.

### Experiment sweep

Three config-driven runs (40 ep, full data, tracked in W&B/MLflow) isolated each lever:

| Run | overall mAP50 | mAP50-95 | recall | lever |
|---|---|---|---|---|
| yolov8s @ 640 | 0.665 | 0.407 | 0.595 | bigger model (vs nano) |
| yolov8s @ 960 | 0.716 | 0.444 | 0.652 | + resolution |
| **yolov8m @ 960** | **0.740** | **0.463** | **0.679** | + capacity → **winner** |

## Promotion policy

Defined in `configs/eval.yaml` under `promotion`, gated on **mAP50-95**:

| Gate | Threshold | This model | Pass |
|---|---|---|---|
| overall mAP | ≥ 0.30 | 0.463 | ✅ |
| night mAP | ≥ 0.20 | 0.329 | ✅ |

The night gate is deliberate: it blocks a model that looks fine overall but degrades in the
hardest condition.

## Registry flow (MLflow)

`evaluate --register` registers the model version and applies **aliases** (the current MLflow
approach that replaces deprecated stage transitions):

- Every evaluated model → alias **`staging`**.
- Only models that clear **both** gates → alias **`production`**.

The improved model passed, so `nuscenes-yolo-detector` **v2** now carries both `staging` and
`production` (superseding the v1 baseline). The serving layer (Phase 4) loads
`models:/nuscenes-yolo-detector@production`.

```bash
# resolve the current production model
uv run python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlruns/mlflow.db'); \
  print(mlflow.MlflowClient().get_model_version_by_alias('nuscenes-yolo-detector','production'))"
```

_Registry + `mlruns/` live on the infra machine in the two-machine topology; the GPU server
produces the run and syncs `mlruns/` over._
