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
| `graph` | baseline + 1,500 frames from GDS-Louvain communities of the SIMILAR_TO graph (Phase 6e) | Does graph-structured diversity beat both? |

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
4. **`al mine --arm <mined|rate|strat|rate_strat>`** — (round 2 adds per-arm
   scoring/quotas; see the Round 2 section) top-1,000 failure frames → SigLIP
   vectors → KMeans (k = 8) →
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

Graph-diversity arm (Phase 6e) — mining needs Neo4j, so it runs on the **infra
machine**, then `graph.parquet` ships to the GPU server for training:

```bash
# infra machine (Neo4j on :7687 with the SIMILAR_TO edges built)
docker compose up -d neo4j && make graph-build
uv run nuscenes-data-engine al graph-mine
rsync -a data/active_learning/graph.parquet trinity-2-18:/home/mgaur/sahil/nuscenes_project/data/active_learning/
# GPU server
GPU_DEVICES=0 scripts/gpu-run.sh --bg al run --arm graph
# back on the infra machine
rsync -a trinity-2-18:/home/mgaur/sahil/nuscenes_project/data/active_learning/results.json data/active_learning/
uv run nuscenes-data-engine al report
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

All four arms trained on TRINITY (RTX 3080 Ti, ~20 min/arm), evaluated on the
identical 6,019-frame CAM_FRONT val split (602 night).

### Four-arm comparison

| arm | train imgs | overall mAP50 | overall mAP50-95 | night mAP50 | night mAP50-95 | Δ overall | Δ night |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 7,035 | 0.4351 | 0.2477 | 0.2894 | 0.1667 | — | — |
| mined | 8,535 | 0.4568 | 0.2637 | 0.2997 | **0.1739** | +0.0160 | +0.0072 |
| random | 8,535 | 0.4872 | 0.2817 | 0.2711 | 0.1619 | +0.0340 | −0.0048 |
| graph | 8,535 | 0.4861 | **0.2821** | 0.2839 | 0.1703 | **+0.0344** | **+0.0036** |

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

### Graph-diversity arm (Phase 6e) — delivering on takeaway #1

The negative result above points straight at *diversity*, so Phase 6e adds a fourth
arm that acquires frames from the **structure of the SIMILAR_TO graph** instead of raw
embedding proximity. `al graph-mine` (`active_learning/graph_mining.py`) runs **GDS
Louvain** over the pool's SIMILAR_TO subgraph (21,094 connected CAM_FRONT frames → 98
communities), allocates the 1,500-frame budget across communities with the shared
`allocate` helper, and takes each community's most-connected (representative) frames.
It reuses the same leakage guards and the *same* `random.parquet` control; only the
acquisition function changes. Neo4j lives on the infra machine, so graph-mining runs
there and ships `graph.parquet` to the GPU server for `al run --arm graph`.

**It gets the best of both.** The graph arm matches the random control's best-in-class
overall gain (**+0.0344** vs +0.0340; overall mAP50-95 0.2821, the highest of any arm)
*and* improves the night slice (**+0.0036**, night mAP50-95 0.1703) — where random
actually regressed (−0.0048). It is the only arm to lift **both** overall and night vs
baseline. The mechanism is coverage: the graph-mined 1,500 frames span **473 scenes with
12.1% night**, versus similarity-mining's **219 scenes, 0% night** — structured diversity
recovers random's coverage benefit without discarding the night stratum that pure
failure-centroid mining dropped. Design + Cypher/GDS details: [GRAPH.md](GRAPH.md).

### Takeaways for the next iteration

1. **Add a diversity term** ✅ *(done in 6e)* — the graph arm caps concentration by
   sampling across Louvain communities of the similarity graph; it matched random's
   overall gain and recovered night, confirming the 21% top-10-scene concentration was
   the main reason `mined` lost.
2. **Rate-based acquisition** — score `n_fn / n_gt` (or calibrated-confidence
   error) instead of absolute counts, so sparse night/edge-case frames can rank.
3. **Stratified quotas** — reserve part of the mining budget for underrepresented
   slices (night, rain) regardless of failure rank.
4. **Random is a strong baseline** — any acquisition function should be gated on
   beating an equal-budget random control, exactly as this harness does.

## Round 2 — rate-based scores + stratified quotas

Delivers takeaways #2 and #3 as a **2×2 ablation** against round 1's `mined` arm
(absolute score, no quotas). Only the acquisition function changes — same baseline,
sweep output, random control, training config, and val split. Design:
[the 2026-08-02 spec](superpowers/specs/2026-08-02-al-round-2-design.md).

| arm | acquisition score | quota floors (of 1,500) |
|---|---|---|
| `rate` | smoothed rate `(n_fn + 0.5·n_low_conf)/(n_gt + 0.5)` | none |
| `strat` | absolute (round 1) | night ≥ 375 (25%), rain ≥ 300 (20%) |
| `rate_strat` | smoothed rate | night ≥ 375, rain ≥ 300 |

Why: the round-1 score ranked crowded day frames (top-1,000 night share **1.5%**);
the smoothed rate restores night to **8.7%** (one all-night failure cluster of 87)
and removes the 331-frame tie at rate 1.0, while the floors test whether an explicit
boost (2× the pool's 12% night) moves night mAP where parity representation didn't.
Quota passes run night-then-rain with stratum-prefiltered centroid search and seeded
random backfill; night∧rain frames count toward both floors. The round-1 control is
write-protected: `run_mining` never regenerates an existing `random.parquet` and
fails loudly if its size no longer matches `n_mine`.

Mined-set composition (full pool, this repo's LanceDB store):

| arm | night share | rain share | scenes | top-10-scene concentration |
|---|---:|---:|---:|---:|
| `rate` | 0.232 | 0.199 | 214 | 18.9% |
| `strat` | 0.270 | 0.307 | 253 | 18.1% |
| `rate_strat` | 0.366 | 0.275 | 224 | 16.6% |
| round-1 `mined` | 0.000 | 0.403 | 219 | 21.3% |
| round-1 `random` | 0.127 | 0.191 | 501 | 4.8% |

The rate score alone (no quota) already lifts the mined set from 0% to 23% night —
the score change, not just the floors, drives night recovery. Scene diversity stays
near round-1 `mined` levels (all three are centroid-mining arms); whether the night
boost outweighs the diversity gap vs `random`/`graph` is exactly what training will
measure. Reruns reproduce identical frame sets; only stored float32 `_distance`
values jitter by machine epsilon (BLAS thread ordering), so parquet bytes differ
while every token, flag, and metric is reproducible.

```bash
# mining (infra machine or TRINITY — needs the LanceDB store + round-1 state)
uv run nuscenes-data-engine al mine --arm rate
uv run nuscenes-data-engine al mine --arm strat
uv run nuscenes-data-engine al mine --arm rate_strat
# training + report (TRINITY)
scripts/gpu-run.sh --bg al run --arm rate
scripts/gpu-run.sh --bg al run --arm strat
scripts/gpu-run.sh --bg al run --arm rate_strat
scripts/gpu-run.sh al report
```

Results: pending the TRINITY training runs.

Runs: MLflow `nuscenes-yolo` (four `*_al-*` runs, registry untouched) and W&B
[`al-baseline` / `al-mined` / `al-random` / `al-graph` + sweep/mine runs](https://wandb.ai/sahil-patkar88-x/nuscenes-data-engine).
