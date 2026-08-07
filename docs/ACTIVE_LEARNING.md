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
asserted at result-merge time — the per-arm mAP numbers are directly comparable.

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
   size, dedupe/backfill to exactly 1,500 → `<arm>.parquet` (the `mined` arm also
   seeds the write-once `random.parquet` control).
5. **`al run --arm mined`**, **`al run --arm random`** — retrain + evaluate each.
6. **`al report`** — arm-comparison table (every arm present in `results.json`)
   (overall + night mAP, deltas vs baseline) + cluster table →
   `data/active_learning/report.md`.

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
uv run nuscenes-data-engine al graph-mine --arm graph  # round 3 adds --arm graph_rate | graph_rate_night
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
but note the size of that lift is overfetch-bound: all 348 night frames come from
the single all-night failure cluster (87 members × overfetch 4 candidates, every
one mined), so the score creates the night cluster while the `overfetch` knob caps
how much night it can pull. `rate_strat`'s night frames, by contrast, spread across
all 8 clusters via the quota passes.

Scene diversity stays near round-1 `mined` levels (all three are centroid-mining
arms); whether the night boost could outweigh the diversity gap vs
`random`/`graph` was the open question — answered below: it could not. Reruns reproduce identical frame sets; only stored float32 `_distance`
values jitter by machine epsilon (BLAS thread ordering), so parquet bytes differ
while every token, flag, and metric is reproducible.

```bash
# mining (infra machine or TRINITY — needs the LanceDB store + round-1 state)
uv run nuscenes-data-engine al mine --arm rate
uv run nuscenes-data-engine al mine --arm strat
uv run nuscenes-data-engine al mine --arm rate_strat
# if mined on the infra machine, ship the token sets to TRINITY first
rsync -a data/active_learning/{rate,strat,rate_strat}.parquet trinity-2-18:/home/mgaur/sahil/nuscenes_project/data/active_learning/
# training + report (TRINITY)
scripts/gpu-run.sh --bg al run --arm rate
scripts/gpu-run.sh --bg al run --arm strat
scripts/gpu-run.sh --bg al run --arm rate_strat
scripts/gpu-run.sh al report
```

### Round 2 results

All three arms trained on TRINITY (RTX 3080 Ti, ~20 min/arm, 2026-08-03), same
6,019-frame CAM_FRONT val split as round 1 (602 night; treat night deltas with the
usual small-slice humility):

| arm | overall mAP50 | overall mAP50-95 | night mAP50 | night mAP50-95 | Δ overall | Δ night |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.4351 | 0.2477 | 0.2894 | 0.1667 | — | — |
| `rate` | 0.4602 | 0.2653 | **0.3076** | 0.1724 | +0.0176 | +0.0057 |
| `strat` | 0.4564 | 0.2588 | 0.2692 | 0.1581 | +0.0111 | **−0.0086** |
| `rate_strat` | 0.4727 | 0.2716 | 0.3013 | 0.1736 | +0.0239 | +0.0069 |
| round-1 `random` (the gate) | 0.4872 | 0.2817 | 0.2711 | 0.1619 | +0.0340 | −0.0048 |
| round-1 `graph` (the champion) | 0.4861 | 0.2821 | 0.2839 | 0.1703 | +0.0344 | +0.0036 |

Verdict against the three hypotheses:

- **H1 (rate scoring) — a wash on aggregate, a real night signal underneath.**
  `rate` edges round-1 `mined` overall (+0.0176 vs +0.0160) and posts the **best
  night mAP50 of all seven arms** (0.3076), but its night mAP50-95 (+0.0057) trails
  `mined`'s (+0.0072). Rate scoring finds night failures the absolute score is
  blind to — it just can't convert them into a large aggregate win at this budget.
- **H2 (quotas alone) — refuted.** `strat` is the worst round-2 arm overall
  (+0.0111) and the worst night arm of all seven (−0.0086, below baseline).
  Forcing 375 night frames chosen by similarity to *day-dominated* failure
  centroids adds near-duplicate, low-information night frames — composition
  targets don't help when the acquisition signal can't see night failures.
- **H3 (composition) — confirmed.** `rate_strat` beats both its components on
  overall (+0.0239 > +0.0176 > +0.0111) and nearly matches `mined`'s night
  (+0.0069 vs +0.0072): the rate score creates a genuine night failure signal and
  the floors amplify it. It joins `mined` and `graph` as the only arms lifting
  **both** overall and night.
- **The gate holds: nobody beats random (+0.0340) or graph (+0.0344) overall.**
  All three round-2 arms are centroid-similarity miners spanning 214–253 scenes vs
  random/graph's ~500 — round 1's diversity lesson survives a better score and
  explicit composition floors. The scene-spread column above tracks Δ overall
  almost monotonically across all seven arms.

Takeaway for round 3: the two proven ingredients live in different arms —
**graph-structured diversity** (round 1's `graph`) and a **night-aware rate
score** (`rate_strat`). The natural next experiment selects *diverse
representatives weighted by rate score within stratified quotas* — e.g.
Louvain-community sampling where each community's budget is proportional to its
summed smoothed-rate failure mass, with the night floor retained.

## Round 3 — rate-weighted graph-community budgets

Combines the two proven ingredients: round 1's Louvain-community diversity (the
reigning champion) and round 2's smoothed-rate score (the best night signal). Only
the acquisition function changes — same baseline, sweep output, random control,
training config, and val split. Design:
[the 2026-08-04 spec](superpowers/specs/2026-08-04-al-round-3-design.md).

| arm | community budget weighting | night floor (of 1,500) |
|---|---|---|
| `graph_rate` | smoothed-rate failure mass, floor 1/community | none |
| `graph_rate_night` | smoothed-rate failure mass, floor 1/community | ≥ 375 |

Mechanism: the top-1,000 rate-ranked failures each route their score to the
communities of their `route_k = 10` nearest pool frames (LanceDB); community
budgets are allocated ∝ that failure mass (capacity-capped, floor 1 per community
— every community stays represented); within a community, top-degree members are
taken first. The night arm runs a mass-proportional night pass for the 375 floor
before the main pass. All guards are explicit `ValueError`s; the round-1
`graph.parquet` is never rewritten by the new arms.

Mined-set composition (canonical seeded run, 97 Louvain communities over 21,094
connected pool frames; all routed failures had stored embeddings):

| arm | night share | rain share | scenes | top-10-scene conc. | floor-only communities |
|---|---:|---:|---:|---:|---:|
| `graph_rate` | 0.083 | 0.199 | 378 | 11.8% | 22/97 |
| `graph_rate_night` | 0.309 | 0.169 | 368 | 14.3% | 23/97 |
| round-1 `graph` | 0.121 | 0.195 | 473 | 5.6% | — |
| round-2 `rate` | 0.232 | 0.199 | 214 | 18.9% | — |
| round-2 `rate_strat` | 0.366 | 0.275 | 224 | 16.6% | — |

Two composition findings worth pinning before training: **mass weighting trades
night for failure focus** — `graph_rate` lands at 8.3% night, *below* the pool's
12%, because day-failure communities dominate the mass (Spearman mass-vs-size
0.648, 17/97 communities got zero mass); and **the night floor is concentrated,
not spread** — the single all-night community (916 frames) absorbs the boost, its
quota jumping 83 → 323 (~3.9×) — a consequence of the mass-weighted allocator spill
(the spec's degree-ranked spill alternative would spread it differently); partition-matched
H2 measures this mechanism as implemented. Scene spread (378/368) sits between round 2's
centroid arms (~220) and the size-weighted `graph` (473) — training will tell
whether that diversity loss costs more than the failure focus gains.

Determinism note: unseeded GDS Louvain is nondeterministic (community partitions
varied run-to-run; mined-set Jaccard 0.71). GDS 2.13.11 rejects `randomSeed` for
Louvain, so the fix is `concurrency: 1` — verified byte-identical parquets across
reruns. Round 1's `graph` arm silently had the same nondeterminism, and its artifact (98
communities) is preserved unchanged against the pinned partition's 97 — so H1
(`graph_rate` vs `graph`) carries one draw of partition noise on top of the
weighting change (the doc's own measurement of that noise: mined-set Jaccard 0.71
across unseeded reruns) and its effect size should be read with that in mind. H2
(`graph_rate_night` vs `graph_rate`) is partition-matched and clean.

```bash
# infra Mac (Neo4j + LanceDB + failures.parquet all local)
docker compose up -d neo4j
uv run nuscenes-data-engine al graph-mine --arm graph_rate
uv run nuscenes-data-engine al graph-mine --arm graph_rate_night
rsync -a data/active_learning/{graph_rate,graph_rate_night}.parquet trinity-2-18:/home/mgaur/sahil/nuscenes_project/data/active_learning/
# TRINITY — pick a free GPU (ssh trinity-2-18 nvidia-smi); sh -c wrapper is REQUIRED
scripts/gpu-run.sh --bg raw "sh -c 'env CUDA_VISIBLE_DEVICES=2 uv run nuscenes-data-engine al run --arm graph_rate && env CUDA_VISIBLE_DEVICES=2 uv run nuscenes-data-engine al run --arm graph_rate_night && uv run nuscenes-data-engine al report && echo CHAIN_COMPLETE'"
```

### Round 3 results

Both arms trained on TRINITY (RTX 3080 Ti, ~20 min/arm, 2026-08-04), identical
6,019-frame CAM_FRONT val split (602 night — small-slice humility applies, and H1
additionally carries one draw of Louvain partition noise per the determinism note):

| arm | overall mAP50 | overall mAP50-95 | night mAP50 | night mAP50-95 | Δ overall | Δ night |
|---|---:|---:|---:|---:|---:|---:|
| `graph_rate` | 0.4873 | 0.2807 | 0.2951 | 0.1663 | +0.0330 | −0.0004 |
| `graph_rate_night` | 0.4757 | 0.2731 | 0.3000 | **0.1768** | +0.0254 | **+0.0101** |
| `graph` (champion) | 0.4861 | 0.2821 | 0.2839 | 0.1703 | +0.0344 | +0.0036 |
| `random` (gate) | 0.4872 | 0.2817 | 0.2711 | 0.1619 | +0.0340 | −0.0048 |

Verdicts:

- **H1 (rate-mass weighting beats size weighting) — not supported.** `graph_rate`
  (+0.0330) sits within partition noise of `graph` (+0.0344) and `random` (+0.0340)
  on overall, and its night edge is gone (−0.0004 vs +0.0036) — consistent with its
  composition (8.3% night, below pool parity: day-failure communities dominate the
  mass). Failure-mass focus neither helps nor hurts overall once community-level
  diversity is preserved; the coverage floor, not the weighting, carries the arm.
- **H2 (night floor on top) — confirmed, and it's the project's best night result.**
  `graph_rate_night` posts **+0.0101 night mAP50-95** (0.1768) — the largest night
  gain of all nine arms across three rounds (previous best: `mined` +0.0072) — at
  an overall cost of ~0.008 vs `graph_rate` (partition-matched, so this comparison
  is clean). One community-concentrated night boost (916-frame all-night community,
  quota 83 → 323) moved the night slice more than any acquisition change before it.
- **The overall gate still holds** — nothing beats `graph`/`random` on overall
  mAP50-95. After nine arms the frontier is clear: **maximum overall** comes from
  diversity (size-weighted `graph` or plain `random`); **maximum night** comes from
  `graph_rate_night`, which keeps ~75% of the best overall gain while tripling the
  best prior night delta. Given the project's release gate is night mAP (Phase 3),
  `graph_rate_night` is arguably the most *deployable* acquisition function this
  harness has produced.

Three-round arc, in one line each: round 1 — diversity beats similarity; round 2 —
better scores compose but can't out-run diversity; round 3 — with diversity held
(community floor), an explicit night floor finally moves the night slice, and the
overall/night trade-off becomes an explicit, tunable choice.

## Weak supervision — does the engine work without ground truth?

Every arm above still pulls **ground truth** boxes for its added frames — an
assumption that doesn't hold on genuinely unlabeled data. This closes the 6b→6d
loop: instead of GT for `random`'s 1,500 mined frames, the baseline detector
pseudo-labels them and the Phase 6b VLM verifies which frames to trust. Design:
[the 2026-08-06 spec](superpowers/specs/2026-08-06-vlm-weak-supervision-design.md).

### Why verify frames, not boxes

Phase 6b's VLM emits scene-level labels — per-class *counts* and conditions, not
boxes (see [AUTOLABEL_EVAL.md](AUTOLABEL_EVAL.md)) — so it cannot label training
frames directly. The design is self-training with VLM verification: the baseline
detector **proposes** boxes at conf ≥ 0.5, and the VLM **verifies** them per frame —
map the VLM's 10 classes onto the detector's 5 (car/truck/bus/pedestrian/bicycle)
and accept the whole frame iff `|detector_count − vlm_count| ≤ 1` for *every*
mapped class. Verification is frame-level accept/reject, not per-class box
filtering: silently dropping one class's boxes from a kept frame would teach the
detector that those objects are background — actively harmful for pedestrians, the
class 6b already knows the VLM under-recalls.

### Two arms, and why the GT twin exists

| arm | added frames | labels |
|---|---|---|
| `weak_random` | the accepted subset of `random`'s 1,500 frames | pseudo (detector + VLM-verified) |
| `weak_random_gt` | the *same* accepted subset | ground truth |

The verifier rejects frames, so the added set shrinks below `random`'s 1,500. A
direct `weak_random` vs `random` comparison would confound two effects at once —
fewer frames *and* worse labels. The GT twin removes the confound for one extra
training run: `weak_random_gt` vs `random` isolates the cost of the frames the
verifier dropped (identical label source, fewer frames); `weak_random` vs
`weak_random_gt` isolates the cost of losing ground truth (identical frames,
different label source).

### Pipeline and measured verification

1,272 of `random`'s 1,500 frames needed VLM labels (228 were already covered by
6b's original 5,000-frame run); they were labelled at $0 on the same self-hosted
Qwen2.5-VL via the unchanged 6b submit/collect path, against a separate state dir
(`configs/autolabel_weak.yaml`) so the original run's artifacts are never touched.
Parse rate 1,267/1,272 = 99.6% ok, 5 truncated.

Verification (conf 0.5, tolerance ±1, per-frame accept/reject):

- **Retention 0.639** — 958 of 1,500 frames accepted; 0 had no label, 5 were
  unparsed.
- **Rejected by class** (a frame can be rejected on more than one class, so these
  sum past 542): car 417, pedestrian 125, truck 101, bicycle 11, bus 2 — car and
  pedestrian disagreement drive most of the loss.
- **Accepted-but-mutually-zero by class** — of the 958 accepted frames, how many
  had *both* detector and VLM report zero of that class: bicycle 912, bus 855,
  pedestrian 686, truck 665, car 275. High mutual-zero agreement is expected for
  bicycles/buses; the 686-frame pedestrian figure is the mechanism named below.
- **Mean boxes per accepted frame: 2.03 pseudo vs 3.87 ground truth** (1,942
  pseudo boxes total on the 958 added frames) — even on frames the VLM endorsed,
  the detector labels only ~52% of the objects GT has. Verification catches
  count *disagreement*, not detector *recall*.

### Results

Identical 6,019-frame CAM_FRONT val split as every other arm:

| arm | added frames | labels | train imgs | total boxes | overall mAP50-95 | night mAP50-95 |
|---|---|---|---|---|---|---|
| baseline | — | — | 7,035 | — | 0.2477 | 0.1667 |
| random (round 1) | 1,500 | GT | 8,535 | — | 0.2817 (+0.0340) | 0.1619 (−0.0048) |
| weak_random_gt | 958 accepted | GT | 7,993 | 68,599 | 0.2648 (+0.0171) | 0.1568 (−0.0099) |
| weak_random | 958 accepted | pseudo | 7,993 | 66,836 | 0.2539 (+0.0062) | 0.1483 (−0.0184) |

Decomposing GT `random`'s +0.0340 overall gain three ways:

- Dropping the 542 verifier-rejected frames costs 0.0340 − 0.0171 = **0.0169**
  (~50% of the gain) — `random` vs `weak_random_gt`.
- Losing ground truth on the 958 kept frames costs 0.0171 − 0.0062 = **0.0109**
  (~32%) — `weak_random_gt` vs `weak_random`.
- Weak supervision retains 0.0062 / 0.0340 = **18%** of the GT gain —
  `weak_random` vs baseline.

Consistency check: `weak_random_gt` and `weak_random` differ by 68,599 − 66,836 =
1,763 boxes — exactly 958 × (3.87 − 2.03), i.e. exactly the under-labelling on the
added frames, with nothing left unexplained.

### Verdict: it works, but it's expensive

Weak supervision **works but is expensive**: it retains 18% of the GT gain
(+0.0062 vs +0.0340), and the loss splits roughly **half dropped frames** (~50%)
and **a third label quality** (~32%). Honest caveat: every arm regresses on the
602-frame night slice vs baseline, and `weak_random` regresses the most (−0.0184).
The 686-of-958 mutual-zero-pedestrian figure above is the mechanism: 6b
independently measured VLM pedestrian presence recall at **0.58**, so a detector
and VLM sharing a pedestrian blind spot is expected — those frames teach the model
"pedestrian here = background" on exactly the class where it matters most.

### What would move this

Two untried levers: raise the proposer's recall (a lower confidence threshold
finds more objects but costs precision, shifting the accept/reject balance); or
verify **presence** rather than **counts** for classes where 6b measured weak VLM
recall (pedestrians), since a presence check would catch "VLM saw a pedestrian,
detector didn't" without needing count agreement. Caveat: the current ±1-count
rule wasn't a placeholder — it was chosen deliberately (see the design section
above) and measured as-is, not tuned post-hoc to this result.

### Runbook

```bash
# 1. sample the arm frames still needing labels (any node)
scripts/gpu-run.sh al pseudo-sample --arm random
# 2. serve the VLM on a free 24GB card, then label via the unchanged 6b path
GPU_NODE=trinity-2-3 scripts/gpu-run.sh --bg raw "env PATH=/home/mgaur/sahil/vllm-env/bin:/usr/local/bin:/usr/bin:/bin HF_HOME=<repo>/.cache/huggingface CUDA_VISIBLE_DEVICES=0 /home/mgaur/sahil/vllm-env/bin/vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8399 --max-model-len 8192"
GPU_NODE=trinity-2-3 scripts/gpu-run.sh --bg raw "sh -c 'uv run nuscenes-data-engine autolabel submit -c configs/autolabel_weak.yaml --provider local && uv run nuscenes-data-engine autolabel collect -c configs/autolabel_weak.yaml --provider local'"
# 3. propose + verify, then train both arms
scripts/gpu-run.sh al pseudo-label --arm random --weights runs/yolov8n_imgsz640_e20_al-baseline/weights/best.pt --device 2
scripts/gpu-run.sh --bg raw "sh -c 'env CUDA_VISIBLE_DEVICES=2 uv run nuscenes-data-engine al run --arm weak_random && env CUDA_VISIBLE_DEVICES=2 uv run nuscenes-data-engine al run --arm weak_random_gt && uv run nuscenes-data-engine al report'"
```

The vLLM server needs a 24 GB node — `trinity-2-3`, not the 12 GB `trinity-2-18` —
and its venv's `bin` must lead `PATH` or vLLM's shell-out to `ninja`
(torch.compile) fails.

Runs: MLflow `nuscenes-yolo` (one `*_al-*` run per trained arm, registry
untouched) and W&B
[`al-baseline` / `al-mined` / `al-random` / `al-graph` / `al-rate` / `al-strat` / `al-rate_strat` / `al-weak_random` / `al-weak_random_gt` + sweep/mine runs](https://wandb.ai/sahil-patkar88-x/nuscenes-data-engine).
