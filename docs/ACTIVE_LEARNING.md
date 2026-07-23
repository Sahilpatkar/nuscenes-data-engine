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

All three arms trained on TRINITY (RTX 3080 Ti, ~20 min/arm), evaluated on the
identical 6,019-frame CAM_FRONT val split (602 night).

### Three-arm comparison

| arm | train imgs | overall mAP50 | overall mAP50-95 | night mAP50 | night mAP50-95 | Δ overall | Δ night |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 7,035 | 0.4351 | 0.2477 | 0.2894 | 0.1667 | — | — |
| mined | 8,535 | 0.4568 | 0.2637 | 0.2997 | **0.1739** | +0.0160 | +0.0072 |
| random | 8,535 | 0.4872 | **0.2817** | 0.2711 | 0.1619 | **+0.0340** | −0.0048 |

**The headline result is negative — and instructive: the random control beat
similarity-mining on overall mAP (+0.034 vs +0.016).** Diversity explains it:

- The 1,500 random frames span **501 scenes** (top-10 scenes hold 4.8% of frames);
  the 1,500 mined frames span only **219 scenes** (top-10 hold 21.3%, one scene
  contributes 38 frames). Embedding-similarity mining pulls near-duplicates around
  each failure centroid, so each mined frame adds less new information.
- The mined arm was nonetheless the only one to improve the **night** slice
  (+0.007 vs random's −0.005) — but on a 602-frame slice this is within noise;
  don't over-read it.

### Why the mined set contains no night frames

The sweep *did* show the expected quality gap (night mAP50-95 0.167 vs 0.248
overall), but the acquisition score didn't translate it into night mining:

- `failure_score = n_fn + 0.5·n_low_conf` counts **absolute** misses, so it ranks
  crowded frames first: the top-1,000 failure frames average **9.3 GT boxes vs 4.8
  corpus-wide**, and only 1.5% of them are night (night frames simply contain fewer
  annotated objects).
- Per-object FN *rates* are similar (day 30.5%, night 27.1% at conf 0.05) — the
  night mAP gap comes mostly from confidence/localization quality, which this
  score doesn't see.

Consequently all 8 failure clusters are day-dominated (only cluster 7 has any
night members, 16%), and the mined set is 0% night vs the pool's ~12.7%.

| cluster | size | night share | mean failure score |
|---:|---:|---:|---:|
| 0 | 232 | 0.00 | 4.88 |
| 1 | 141 | 0.00 | 5.20 |
| 2 | 147 | 0.00 | 4.94 |
| 3 | 132 | 0.00 | 4.74 |
| 4 | 109 | 0.00 | 6.92 |
| 5 | 38 | 0.00 | 5.16 |
| 6 | 109 | 0.00 | 4.29 |
| 7 | 92 | 0.16 | 4.93 |

### Takeaways for the next iteration

1. **Add a diversity term** — cap frames per scene (or per near-duplicate cluster)
   during mining; the 21% top-10-scene concentration is the main reason mined lost.
2. **Rate-based acquisition** — score `n_fn / n_gt` (or calibrated-confidence
   error) instead of absolute counts, so sparse night/edge-case frames can rank.
3. **Stratified quotas** — reserve part of the mining budget for underrepresented
   slices (night, rain) regardless of failure rank.
4. **Random is a strong baseline** — any acquisition function should be gated on
   beating an equal-budget random control, exactly as this harness does.

Runs: MLflow `nuscenes-yolo` (three `*_al-*` runs, registry untouched) and W&B
[`al-baseline` / `al-mined` / `al-random` + sweep/mine runs](https://wandb.ai/sahil-patkar88-x/nuscenes-data-engine).
