# AL Round 2 — rate-based acquisition + stratified quotas

**Status:** approved 2026-08-02. Extends the Phase 6d/6e active-learning harness with
the two open takeaways from `docs/ACTIVE_LEARNING.md`: rate-based acquisition scoring
and night/rain-stratified mining quotas. Only the acquisition function changes — the
split, baseline, sweep output, random control, training config, and val set are all
reused from round 1.

## Goal & scope (approved)

A **2×2 ablation** against round 1's `mined` arm (absolute score, no quotas), three new
arms, each one ~20-min yolov8n@640 training run on TRINITY:

| arm | acquisition score | quota floors (of 1,500) |
|---|---|---|
| `rate` | smoothed rate `(n_fn + 0.5·n_low_conf) / (n_gt + 0.5)` | none |
| `strat` | absolute (round-1 `failure_score`) | night ≥ 375, rain ≥ 300 |
| `rate_strat` | smoothed rate | night ≥ 375, rain ≥ 300 |

Decisions the user pinned:
- **Full ablation (3 arms)** over a single combined arm — clean attribution.
- **Smoothed rate with k = 0.5** — measured on the real `failures.parquet`: top-1,000
  night share 8.7% (vs 1.5% absolute), the 331-frame tie block at rate 1.0 collapses to
  1, the low-confidence signal (where the 6d analysis located the night gap) is kept,
  and it recomputes from the existing `failures.parquet` — **no re-sweep**.
- **Quota floors: night 375 (25%, ≈2× the pool's 12.0%), rain 300 (20%, ≈parity with
  19.5%)** — round 1 showed parity night share alone didn't lift night mAP (random
  regressed at ~12.7%), so the quota mechanism gets a real boost. Floors, not caps:
  the unconstrained remainder may add more of either. Pool has 2,541 night / 4,122
  rain frames, so both floors are comfortably satisfiable.

Hypotheses: **H1** rate scoring alone lifts night mAP over `mined` without losing the
overall gain; **H2** explicit quotas lift night where organic representation didn't;
**H3** the mechanisms compose. Gate (unchanged harness ethos): beat the equal-budget
random control (+0.0340 overall mAP50-95); the round-1 winner to beat is `graph`
(+0.0344 overall, +0.0036 night). The night val slice is 602 frames — night deltas are
reported with confidence-interval humility.

## Implementation: parameterize `mining.py` (approved approach)

Chosen over a separate `mining_v2.py` (would duplicate ~150 lines that drift) and over
an acquisition-function registry (YAGNI for three arms).

### Config + CLI surface

`configs/active_learning.yaml` `mining:` gains a per-arm map; existing keys unchanged:

```yaml
mining:
  n_clusters: 8
  n_mine: 1500
  seed: 64
  overfetch: 4
  arms:
    mined:      {scoring: absolute}
    rate:       {scoring: smoothed_rate}
    strat:      {scoring: absolute,      quotas: {is_night: 375, is_rain: 300}}
    rate_strat: {scoring: smoothed_rate, quotas: {is_night: 375, is_rain: 300}}
```

- `al mine` gains `--arm` (default `mined` — fully back-compatible).
- `experiment.ARMS` grows to seven; `resolve_arm_frames` maps each new arm to
  `<arm>.parquet`. `al run --arm rate|strat|rate_strat` works unchanged.

### Mining pipeline changes

New pure helpers (TDD'd, torch-free):

- **`score_failures(failures, scoring) -> Series`** — `absolute` returns the existing
  `failure_score`; `smoothed_rate` computes `(n_fn + 0.5·n_low_conf) / (n_gt + 0.5)`
  (safe at `n_gt = 0`). Unknown scoring → `ValueError`.
- **Quota accounting** — given quota floors and the already-selected set, compute each
  pass's remaining need; a frame that is both night and rain counts toward **both** floors.

Generalized selection flow per arm (rank by score → top-1,000 → KMeans k=8 unchanged);
quota passes run in the config's key order (night first, then rain):

1. **Night pass** — allocate the 375 floor across clusters proportionally (existing
   `allocate` helper); each centroid queries LanceDB with `is_night = true` added to the
   pool/channel prefilter.
2. **Rain pass** — count rain frames already selected; allocate only the remainder with
   `is_rain = true` prefilter.
3. **Unconstrained pass** — the existing proportional flow fills to exactly 1,500.

If a stratum's centroid candidates run dry after dedupe/overfetch, backfill with a
seeded random draw from that stratum's remaining pool frames (logged). `mine_candidates`
gains an optional extra-filter clause and `is_rain` in its select columns.

### Artifacts + guards

- Each arm writes `<arm>.parquet` and `clusters_<arm>.parquet` (+ per-arm cluster
  summary JSON). The `mined` arm keeps its legacy filenames (`mined.parquet`,
  `clusters.parquet`, `cluster_summary.json`).
- **`random.parquet` is only written by the `mined` arm** — the round-1 control is never
  regenerated; all new arms compare against the identical control.
- Guards: existing leakage asserts (mined ⊆ pool, ∩ baseline = ∅, `merge_results`
  val-set identity) plus new asserts — quota floors met; exactly `n_mine` frames.
- Errors: unknown arm or scoring → `ValueError` listing valid options; a floor larger
  than its stratum pool → explicit error.

### Reporting + docs

- `report.py` iterates all seven arms and adds composition-diagnostic columns computed
  from each arm's parquet joined to `samples.parquet`: `n_scenes`, `night_share`,
  `rain_share` (these told round 1's story; now they're first-class).
- Results land in a "Round 2" section of `docs/ACTIVE_LEARNING.md` — tables plus honest
  mechanism analysis, win or lose. `PROJECT.md` §9 item 1 updated.
- W&B: existing per-stage runs (`al-mine` config gains the arm name; `al-<arm>` runs).

### Testing

Torch-free (runs in CI): hand-computed `score_failures` values for both modes including
`n_gt = 0` and the saturation-tie case; quota accounting with night∧rain overlap;
dry-stratum backfill; floor-violation, unknown-arm, unknown-scoring errors; back-compat
(default arm reproduces round-1 behavior). Mining integration tests follow the existing
fake-table patterns in `tests/`. On TRINITY: smoke `al mine --arm <x>` before each
`--bg al run --arm <x>`.

## Runbook (round 2)

Mining needs the LanceDB store plus round-1 state (`failures.parquet`,
`split.parquet`) — both live on TRINITY, where round 1 ran `al mine`. Run it there;
if mined on the infra machine instead, rsync the three `<arm>.parquet` files up first
(as the graph arm's runbook did).

```bash
# TRINITY
scripts/gpu-run.sh al mine --arm rate
scripts/gpu-run.sh al mine --arm strat
scripts/gpu-run.sh al mine --arm rate_strat
# TRINITY, sequential (~20 min each)
scripts/gpu-run.sh --bg al run --arm rate
scripts/gpu-run.sh --bg al run --arm strat
scripts/gpu-run.sh --bg al run --arm rate_strat
scripts/gpu-run.sh al report
# then sync state + mlruns back to the infra machine (round-1 rsync commands)
```

## Explicitly out of scope

- **No re-sweep** — the confidence-aware score (mean matched confidence) stays a future
  option; round 2 recomputes scores from the existing `failures.parquet`.
- **No graph+rate hybrid arm** — natural round 3 if `rate_strat` wins.
- **No changes to any round-1 artifact or result** (`failures.parquet`,
  `random.parquet`, `split.parquet`, existing `results.json` entries).
