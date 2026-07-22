# Auto-labeling with a VLM — methodology and evaluation (Phase 6b)

Can an LLM be trusted as a labeler? nuScenes is one of the few settings where that
question is *measurable*: every frame already has human-quality ground truth, so VLM
labels can be scored instead of eyeballed. This phase labels a stratified sample of
front-camera frames with Claude and evaluates the labels against GT.

## Setup

- **Labelers:** `claude-haiku-4-5` over the full sample; `claude-opus-4-8` over a
  500-frame subset of the same sample (so Haiku, Opus, and GT are three-way
  comparable — including whether the 5× price gap buys labeling accuracy).
- **Transport:** the Claude **Batch API** (50% price discount, ~1h turnaround,
  submit → poll → collect with on-disk state under `data/autolabel/`).
- **Structured outputs:** each request carries a JSON schema
  (`output_config.format`), so on success the response is guaranteed schema-valid
  JSON; client-side Pydantic validation is a second net. Refusals and truncations are
  recorded per-frame as `parse_status`, never crash the pipeline.
- **Estimated cost:** ≈$10 (Haiku, 5,000 frames) + ≈$5 (Opus, 500) ≈ **$15**, printed
  by `autolabel submit` which refuses to run without `--yes`.

## Sampling strategy (part of the deliverable)

Population: CAM_FRONT keyframes ∩ availability-manifest-present = **34,149 frames**,
stratified by location × night × rain — 8 non-empty strata. Notably there is **no
Boston night data**, and night+rain exists only in singapore-hollandvillage (642
frames).

Allocation is **proportional with a floor of 250 per stratum** (seed 6, deterministic
sorted-token sampling). Pure proportional would leave ~100 frames in the small night
strata — too thin for per-condition precision/recall; a balanced design would consume
97% of the night+rain stratum and skew overall metrics away from the corpus
distribution. The floor keeps every stratum evaluable (±~6% CI) while staying
approximately representative:

| stratum | population | sampled | opus subset |
|---|---|---|---|
| boston-seaport day/dry | 12,757 | 1,726 | 172 |
| boston-seaport day/rain | 6,028 | 816 | 82 |
| singapore-hollandvillage day/dry | 770 | 250 | 25 |
| singapore-hollandvillage night/dry | 2,015 | 273 | 27 |
| singapore-hollandvillage night/rain | 642 | 250 | 25 |
| singapore-onenorth day/dry | 7,308 | 989 | 99 |
| singapore-queenstown day/dry | 3,299 | 446 | 45 |
| singapore-queenstown night/dry | 1,330 | 250 | 25 |
| **total** | 34,149 | **5,000** | **500** |

The Opus subset is drawn with the same allocator (total 500, floor 25) from within the
sampled 5K, so subset ⊆ sample by construction.

## Label schema

`SceneLabel` (see `data_engine/autolabel/schema.py`): `time_of_day`
(day/dusk_dawn/night), `weather` (clear/overcast/rain/fog), per-class `object_counts`
(10 explicit int fields), free-text `hazards` and `notable_conditions`,
`label_confidence`. Design choices:

- **Enums only where GT exists** (`is_night`, `is_rain`); free text where the signal
  is exploratory. `dusk_dawn`/`overcast`/`fog` exist because forcing binary choices on
  twilight/marginal frames would manufacture eval noise — they are reported
  descriptively, not scored.
- **Counts use a 10-class eval-only taxonomy** mapped from the fine-grained nuScenes
  `category_name` (the trainer's 5-class `category_group` is untouched). Prompt and
  schema descriptions instruct counting only clearly visible objects.
- The emitted JSON schema is sanitized for the structured-outputs endpoint (no
  numeric constraints; `additionalProperties: false` everywhere, including `$defs`).

Prompt (system, ~80 tokens): *"You label single front-camera images from a driving
dataset. Report only what is clearly visible in this image. Count object instances you
can positively identify; do not guess at heavily occluded, cut-off, or very distant
objects. Counts are for this single frame only."*

## Evaluation design

- **Condition flags:** `time_of_day` vs `is_night`, `weather` vs `is_rain` — accuracy
  and positive-class precision/recall/F1, overall and per stratum. *Caveat:* GT flags
  are derived from scene descriptions (binary, scene-level), so some disagreement is
  GT noise, not model error.
- **Counts:** per class — MAE, exact-match rate, within-±1 rate, and presence
  precision/recall (pred>0 vs GT>0) — computed **twice**: against all GT boxes and
  against boxes with `visibility_token ≥ 2` (≥40% visible), since a single camera
  frame cannot show heavily occluded annotations. Plus MAE by GT-count bucket
  (0 / 1–3 / 4–9 / 10+) — the "VLMs can't count crowds" check.
- **Haiku vs Opus:** per-attribute agreement and inter-model count MAE on the shared
  500 frames, alongside each model's own GT scores.

## Providers

Two interchangeable labelers behind one `BatchTransport` seam:

| provider | model(s) | cost | where it runs |
|---|---|---|---|
| `local` (default) | Qwen2.5-VL-7B-Instruct via a self-hosted vLLM server | **$0** | a 24 GB GPU node (the 7B model doesn't fit 12 GB unquantized) |
| `anthropic` | Haiku 4.5 full sample + Opus 4.8 subset via the Batch API | ~$15 | head node (network-bound); needs `ANTHROPIC_API_KEY` |

The local transport translates the same requests (same prompt, same sanitized JSON
schema — enforced by vLLM's structured outputs) and executes them immediately against
the server, persisting results into the identical on-disk state, so status/collect/
retry/eval are provider-agnostic.

## Runbook — local provider (free, default)

```bash
# One-time on the GPU node (own venv; vLLM has no macOS wheels so it stays out of uv.lock):
scripts/gpu-run.sh raw "cd /home/mgaur/sahil && uv venv vllm-env && VIRTUAL_ENV=vllm-env uv pip install vllm"

# Serve the VLM on a free 24GB 3090 (HF cache kept inside the repo workspace):
GPU_NODE=trinity-2-3 scripts/gpu-run.sh --bg raw \
  "HF_HOME=/home/mgaur/sahil/nuscenes_project/.cache/huggingface CUDA_VISIBLE_DEVICES=0 \
   /home/mgaur/sahil/vllm-env/bin/vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8399 --max-model-len 8192"

GPU_NODE=trinity-2-3 scripts/gpu-run.sh autolabel sample
GPU_NODE=trinity-2-3 scripts/gpu-run.sh --bg autolabel submit    # provider: local from config; free, no --yes
GPU_NODE=trinity-2-3 scripts/gpu-run.sh autolabel collect        # local results persist at submit time
rsync -a --partial --timeout=120 TRINITY:<repo>/data/autolabel/ data/autolabel/
uv run nuscenes-data-engine autolabel eval                       # tables + eval_summary.md
```

## Runbook — anthropic provider (paid alternative)

```bash
GPU_NODE=TRINITY scripts/gpu-run.sh autolabel sample
GPU_NODE=TRINITY scripts/gpu-run.sh autolabel submit --provider anthropic --dry-run
# Put the key on the server once (never in git):  ssh TRINITY 'echo "ANTHROPIC_API_KEY=sk-ant-..." >> <repo>/.env'
GPU_NODE=TRINITY scripts/gpu-run.sh autolabel submit --provider anthropic --yes   # PAID (~$15)
GPU_NODE=TRINITY scripts/gpu-run.sh autolabel status --provider anthropic         # ~1h typical
GPU_NODE=TRINITY scripts/gpu-run.sh autolabel collect --provider anthropic
rsync -a --partial --timeout=120 TRINITY:<repo>/data/autolabel/ data/autolabel/
uv run nuscenes-data-engine autolabel eval
```

Both providers append rows to the same `labels.parquet` keyed by (frame, model), so
running the paid tier later simply adds Haiku/Opus columns to the comparison.

## Results

> **TODO:** pending the paid labeling run (blocked on an `ANTHROPIC_API_KEY`).
> After `autolabel eval`, paste `data/autolabel/eval/eval_summary.md` here, flip the
> README roadmap row to ✅, and write up the findings below.

### Findings — where the VLM is reliable, where it fails

> **TODO** after the run: condition flags vs counts; small-count vs crowded frames;
> visibility sensitivity; whether Opus beats Haiku enough to justify 5× the price;
> implications for using LLMs as labelers in an AV data engine.
