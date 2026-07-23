# Phase 6d — Embedding-Based Active Learning

Closes the data-engine loop: **find where the model fails → mine visually similar
frames from unused data → retrain → measure the delta**. The mining runs entirely on
the Phase 6a SigLIP/LanceDB embedding store; no external APIs.

## Why a reduced baseline

The production model (`yolov8m@960`) already trained on the *full* nuScenes train
split, so there is no unused data left to mine for it — any "improvement" would be
untestable. Instead we run a controlled experiment at a smaller scale:

| Arm | Train data | Question it answers |
|---|---|---|
| `baseline` | 25% of official-train scenes (night-stratified, seed 64) | Where does a data-starved model fail? |
| `mined` | baseline + 1,500 frames mined near its failure clusters | Does targeted data help? |
| `random` | baseline + 1,500 uniformly random pool frames | …more than just *any* extra data? |

All arms: `yolov8n @ 640`, 20 epochs, CAM_FRONT only. The val split is **identical
across arms by construction** (the `train_frames` filter never touches val) and
asserted at result-merge time — the three mAP numbers are directly comparable.

## The loop

1. **`al split`** — night-stratified seeded scene split: ~175 baseline scenes,
   ~525 pool scenes (`data/active_learning/split.parquet`).
2. **`al run --arm baseline`** — build the restricted YOLO dataset, train, evaluate
   (overall + condition slices), record in `results.json`.
3. **`al sweep --weights <baseline best.pt>`** — run the baseline over the official
   val split (6,019 CAM_FRONT frames) at conf 0.05 and match predictions to GT
   (IoU ≥ 0.5, greedy per class). Per-frame
   `failure_score = false_negatives + 0.5 · low_confidence_hits` → `failures.parquet`.
4. **`al mine`** — top-1,000 failure frames → SigLIP vectors → KMeans (k = 8) →
   per-cluster diagnostics (size, night share, mean failure score) → each centroid
   queries LanceDB **prefiltered to pool scenes**, quotas proportional to cluster
   size, dedupe/backfill to exactly 1,500 → `mined.parquet` + seeded
   `random.parquet` control.
5. **`al run --arm mined`**, **`al run --arm random`** — retrain + evaluate each.
6. **`al report`** — three-arm comparison table (overall + night mAP, deltas vs
   baseline) + cluster table → `data/active_learning/report.md`.

### Deployment-proxy caveat

Failures are diagnosed on the **val split**, standing in for "frames the deployed
model sees". The frames actually *mined* come strictly from the held-back
**train-scene pool** — no val frame ever enters training. Guards enforce this:

- `build_split` only ever assigns official-train scenes to baseline/pool.
- `run_mining` asserts `mined ⊆ pool` and `mined ∩ baseline = ∅`.
- `merge_results` raises if `val_images` differs between arms.

### Night stratification

Night scenes are ~12% of the corpus. The baseline split samples each `is_night`
stratum separately, so an unlucky uniform draw can't skew the baseline's night
exposure and poison the mined-vs-random comparison. The night val slice is 602
frames — small; treat night-mAP deltas with according confidence-interval humility.

## Runbook (TRINITY)

```bash
# one-time: scene split (CPU-light, deterministic)
scripts/gpu-run.sh al split

# smoke first (~2 min each)
scripts/gpu-run.sh al run --arm baseline --epochs 1
scripts/gpu-run.sh al sweep --weights runs/yolov8n_imgsz640_e1_al-baseline/weights/best.pt --limit 200

# the real thing (~45–55 min per arm on one GPU)
scripts/gpu-run.sh --bg al run --arm baseline
scripts/gpu-run.sh al sweep --weights runs/yolov8n_imgsz640_e20_al-baseline/weights/best.pt
scripts/gpu-run.sh al mine
scripts/gpu-run.sh --bg al run --arm mined
scripts/gpu-run.sh --bg al run --arm random
scripts/gpu-run.sh al report
```

Then sync results back to the infra machine (state only — the per-arm YOLO datasets
are ~10 GB of symlinks and rebuildable):

```bash
rsync -av --exclude 'arms/*/yolo' trinity-2-18:/home/mgaur/sahil/nuscenes_project/data/active_learning/ data/active_learning/
rsync -av trinity-2-18:/home/mgaur/sahil/nuscenes_project/mlruns/ mlruns/
```

Config: [configs/active_learning.yaml](../configs/active_learning.yaml). All state
lives under `data/active_learning/` (gitignored). MLflow logs one run per arm
(`register=False` — the model registry is untouched); each stage also logs a W&B
run when configured.

## Results

_To be filled after the TRINITY runs._

### Failure clusters

_To be filled: cluster table + what the clusters correspond to visually._

### Three-arm comparison

_To be filled: report.md table + interpretation (mined vs random delta, overall and
night)._
