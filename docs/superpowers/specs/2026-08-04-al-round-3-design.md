# AL Round 3 — rate-weighted graph-community acquisition

**Status:** approved 2026-08-04. Combines the two proven ingredients from rounds 1–2:
the `graph` arm's Louvain-community diversity (the reigning champion, +0.0344 overall /
+0.0036 night) and the round-2 smoothed-rate score (best night signal of all seven arms).
Only the acquisition function changes — the split, baseline, sweep output, random
control, training config, and val set are untouched.

## Goal & scope (approved)

A **2-arm ablation**, each one ~20-min yolov8n@640 run on TRINITY:

| arm | community budget weighting | night floor (of 1,500) |
|---|---|---|
| `graph_rate` | smoothed-rate failure mass, floor 1 per community | none |
| `graph_rate_night` | smoothed-rate failure mass, floor 1 per community | ≥ 375 |

Decisions the user pinned:
- **2-arm ablation** (not a single combined arm; no extra seed-variance runs this round) —
  attributes any gain to rate-weighting vs the night floor separately.
- **Embedding route** for failure→community attribution: each top-1,000 rate-ranked
  failure queries LanceDB for its `route_k = 10` nearest pool frames (reusing round-2's
  `mine_candidates` with a per-failure vector) and adds its smoothed-rate score to the
  community of every routed neighbor. Deterministic; no dependence on val→pool
  `SIMILAR_TO` edge coverage; neighbors outside any community (disconnected pool
  frames) are skipped.
- **Mass-proportional allocation with floor 1**: quota ∝ community rate-mass via the
  shared `allocate` helper, every one of the ~98 communities keeps ≥ 1 frame — the
  round-1 coverage guarantee that beat everything so far. Zero-mass communities get
  exactly the floor. No blend exponents, no new hyperparameters.
- **No rain floor.** Round 2 showed rain composition doesn't move results (`strat`'s
  30.7% rain didn't help); night-only keeps the ablation clean.

Hypotheses: **H1** `graph_rate` beats `graph` — budget routed toward failure-heavy
communities outperforms size-proportional budgets at equal diversity floor. **H2**
`graph_rate_night` adds night mAP over `graph_rate` without costing overall. Gate
(unchanged): beat `random` (+0.0340 overall mAP50-95); champion to beat: `graph`
(+0.0344 / +0.0036). Night val slice is 602 frames — night deltas reported with the
usual humility.

## Acquisition pipeline (extends `active_learning/graph_mining.py`)

1. **Communities** — unchanged: mark pool → project `SIMILAR_TO` subgraph → GDS
   Louvain + degree streams (Neo4j on the infra Mac; ~98 communities / 21,094
   connected pool frames).
2. **Failure routing (new)** — read `failures.parquet`; rank by
   `score_failures(failures, "smoothed_rate")` (imported from `mining.py`), ties by
   token; take the top-1,000; `read_vectors` their SigLIP embeddings from the LanceDB
   store; for each failure vector call `mine_candidates(tbl, vector, pool_scenes,
   channel, k=route_k)`; for each returned pool neighbor with a community,
   `mass[community] += failure_score`. ~1–2 min locally.
3. **Allocation** — `allocate({c: max(int(mass*1000), 0)}, n_mine, floor=1)` over ALL
   communities (zero-mass ones included so the floor applies). If the shared helper
   cannot guarantee floors for zero-count keys, clamp counts to ≥ 1 before calling —
   behavior, not implementation, is the contract: every community ≥ 1, remainder ∝ mass.
4. **Selection** — within each community, members ranked by (degree desc, token); take
   the quota from the top (round-1 mechanism, unchanged).
5. **Night pass (`graph_rate_night` only)** — before the main pass, mirror round 2's
   quota mechanics: allocate the 375 floor across communities ∝ the same rate-mass;
   fill each community's night quota with its top-degree **night** members; if a
   community runs dry, spill to other communities' night members by (degree desc,
   token); final shortfall → seeded (`seed: 64`) random draw from unselected night
   pool frames. The main pass then fills to exactly 1,500, skipping already-selected.
6. **Guards** (round-2 convention — explicit `ValueError`, never `assert`): exactly
   `n_mine` selected; ⊆ pool; ∩ baseline = ∅; night floor met (night arm); night floor
   ≤ n_mine and ≤ night-stratum size validated up front.

The selection core (mass allocation + night pass + representative picking) is pure and
DB-free — unit-testable without Neo4j, like `select_representatives` today. Neo4j stays
confined to `run_graph_mining`; LanceDB routing is integration-tested against the tiny
store fixture from round 2.

## Wiring + artifacts

- **Config** (`configs/active_learning.yaml`):

  ```yaml
  graph_mining:
    n_mine: 1500
    floor: 1
    route_k: 10              # pool neighbors per routed failure
    arms:
      graph:            {weighting: size}
      graph_rate:       {weighting: rate_mass}
      graph_rate_night: {weighting: rate_mass, quotas: {is_night: 375}}
  ```

- **CLI**: `al graph-mine --arm <graph|graph_rate|graph_rate_night>` (default `graph`,
  fully back-compatible; unknown arm → `ValueError` listing valid arms).
- **Artifacts**: `graph_rate.parquet` / `graph_rate_night.parquet` (token column, like
  `graph.parquet`), plus per-arm community diagnostics
  `communities_<arm>.json`: community id, size, rate-mass, night members, quota — the
  table the analysis will be written from. The `graph` arm keeps its legacy artifact
  names and behavior.
- **Registration**: `experiment.ARMS` grows to nine (`graph_rate`, `graph_rate_night`
  appended); `resolve_arm_frames` maps them to their parquets; the report inherits via
  the shared `ARM_ORDER = ARMS` import; composition columns come free from
  `arm_composition`'s `f"{arm}.parquet"` pattern.
- **W&B/MLflow**: existing per-stage patterns (`al-graph-mine-<arm>` mining runs,
  `al-<arm>` training runs).

## Runbook

```bash
# infra Mac (Neo4j w/ SIMILAR_TO graph + LanceDB store + failures.parquet all local)
docker compose up -d neo4j
uv run nuscenes-data-engine al graph-mine --arm graph_rate
uv run nuscenes-data-engine al graph-mine --arm graph_rate_night
rsync -a data/active_learning/{graph_rate,graph_rate_night}.parquet trinity-2-18:/home/mgaur/sahil/nuscenes_project/data/active_learning/
# TRINITY — pick a free GPU first (ssh trinity-2-18 nvidia-smi); GPU 2 shown.
# Chain wrapped in sh -c so nohup + the log cover ALL commands (a bare && chain
# leaves arms 2+ unprotected and their output unlogged).
scripts/gpu-run.sh --bg raw "sh -c 'env CUDA_VISIBLE_DEVICES=2 uv run nuscenes-data-engine al run --arm graph_rate && env CUDA_VISIBLE_DEVICES=2 uv run nuscenes-data-engine al run --arm graph_rate_night && uv run nuscenes-data-engine al report && echo CHAIN_COMPLETE'"
# back on the Mac: sync results.json + mlruns, al report, results docs
```

## Explicitly out of scope

- No changes to round-1/round-2 artifacts, arms, or the random control.
- No rain floor, no seed-variance study (candidate for a later methods round).
- No changes to `mining.py` beyond importing its existing helpers.
