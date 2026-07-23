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

## Results — Qwen2.5-VL-7B-Instruct (local, $0)

Run: 5,000 frames on 4× RTX 3080 Ti (tensor-parallel 4, eager mode), 54 frames/min,
~93 minutes end to end. Parse success **4,986/5,000 (99.7%)** — grammar-constrained
decoding produced schema-valid JSON on every non-truncated response; the 14 failures
were `max_tokens` truncations.

| attribute | accuracy | F1 (positive class) |
|---|---|---|
| night (vs `is_night`) | **0.997** | **0.989** (dusk_dawn share 0.3%) |
| rain (vs `is_rain`) | 0.945 | 0.865 |

Counts vs GT (all 4,986 ok-parsed frames):

| class | GT total | pred total | MAE | exact | ±1 | presence P / R |
|---|---|---|---|---|---|---|
| cars | 14,444 | 15,655 | 1.33 | 0.40 | 0.70 | 0.94 / 0.93 |
| trucks | 2,763 | 2,404 | 0.37 | 0.70 | 0.95 | 0.75 / 0.74 |
| buses | 707 | 526 | 0.07 | 0.93 | 1.00 | 0.84 / 0.66 |
| trailers | 635 | 274 | 0.11 | 0.91 | 0.99 | 0.62 / 0.32 |
| construction veh. | 450 | 341 | 0.07 | 0.94 | 0.99 | 0.65 / 0.59 |
| motorcycles | 388 | 230 | 0.05 | 0.96 | 1.00 | 0.84 / 0.50 |
| bicycles | 342 | 281 | 0.06 | 0.96 | 0.99 | 0.76 / 0.50 |
| pedestrians | 6,284 | 2,853 | 0.76 | 0.64 | 0.84 | 0.97 / 0.58 |
| traffic cones | 2,832 | 3,160 | 0.39 | 0.83 | 0.92 | 0.72 / 0.75 |
| barriers | 4,582 | 2,280 | 0.93 | 0.73 | 0.85 | 0.45 / 0.57 |

MAE by GT-count bucket (all classes pooled) — the crowding effect:

| GT count | n | MAE |
|---|---|---|
| 0 | 38,042 | 0.08 |
| 1–3 | 8,868 | 0.87 |
| 4–9 | 2,494 | 2.81 |
| 10+ | 456 | **6.67** |

*Note:* the planned visibility-≥2 eval variant turned out to be identical to the
all-boxes variant — Phase 1's projection already drops visibility-level-1 boxes, so
`annotations.parquet` contains only levels 2–4. The GT the VLM is scored against is
therefore already "at least partially visible" boxes.

### Findings — where the VLM is reliable, where it fails

1. **Scene-level conditions are essentially solved.** A free 7B model separates
   night/day at F1 0.99 and detects rain at F1 0.87 (some of the residual gap is GT
   noise: `is_rain` is a scene-level description flag, while the model sees a single
   frame that may not look rainy). For condition-slicing, curation, and search-index
   metadata, LLM labels are trustworthy as-is.
2. **Counting degrades sharply with crowding.** Near-perfect on empty/low-count
   classes (MAE 0.08 at GT=0), usable at 1–3 objects (0.87), unreliable at 10+
   (6.7). Use VLM counts as presence/low-cardinality signals, not as measurements.
3. **Systematic misses are interpretable.** Pedestrians are undercounted ~2.2×
   (high precision 0.97, recall 0.58 — small/distant people missed); trailers are
   mostly absorbed into "trucks" (recall 0.32); barriers show the worst precision
   (0.45) — long barrier rows get counted as few objects and non-barrier
   street furniture gets called a barrier.
4. **Structured outputs remove the malformed-JSON problem entirely.** 99.7% parse
   success with zero schema violations; the only failure mode left is truncation
   (fix: `max_tokens` headroom).
5. **Implication for the data engine:** cheap VLM labels are production-usable for
   *scene attributes* (night/rain/hazard flags feeding search and curation) and for
   *presence* of rare classes, but object counts should come from the detector, not
   the VLM. A frontier-model comparison (Haiku/Opus via `--provider anthropic`)
   remains a one-command follow-up on the same 500-frame subset.

### Serving configuration that worked (12 GB cards)

Four attempts documented for posterity: tp=2 OOMs (weights+vision tower don't fit
2×12 GB with KV cache), `ninja` must be installed in the vLLM venv (torch.compile),
and at 0.85 GPU utilization CUDA-graph capture OOMs. Final working flags:

```bash
PATH=<vllm-env>/bin:$PATH HF_HOME=<repo>/.cache/huggingface \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8399 --tensor-parallel-size 4 \
  --max-model-len 4096 --gpu-memory-utilization 0.75 --enforce-eager --max-num-seqs 16
```
