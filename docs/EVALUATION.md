# EVALUATION (Phase 3 deliverable — placeholder)

> Populated during Phase 3. Documents metrics, condition-sliced results, and the
> model-promotion policy.

## Metrics

_TODO(Phase 3): per-class mAP and precision/recall on the held-out nuScenes val split._

## Condition-sliced results

Metrics broken down by scene condition (the project's key differentiator):

| Slice | mAP | Precision | Recall |
|---|---|---|---|
| day | _TODO_ | | |
| night | _TODO_ | | |
| clear | _TODO_ | | |
| rain | _TODO_ | | |

Slices are derived from `scene.description` per `configs/eval.yaml`.
(Reference counts in the full trainval set: 99 night scenes, 165 rain scenes.)

## Promotion policy

A model is promoted `staging → production` in the MLflow registry only if it clears
the gates in `configs/eval.yaml`:

- `min_overall_map`
- `min_night_map`

_TODO(Phase 3): finalize thresholds and record the decision for each registered model._
