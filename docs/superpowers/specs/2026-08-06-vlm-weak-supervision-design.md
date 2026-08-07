# VLM weak supervision — closing the 6b → 6d loop

**Status:** approved 2026-08-06. Makes the data engine work on **genuinely unlabeled
data**: instead of pulling ground-truth boxes for mined frames, the baseline detector
pseudo-labels them and the Phase 6b VLM verifies those labels. The honest test is
whether a weakly-supervised arm still improves the model — and how much of the
GT-labelled gain survives.

## The constraint that shapes everything

Phase 6b's VLM emits **scene-level labels — counts and conditions, not boxes**
(`ObjectCounts`, `time_of_day`, `weather`, hazards). A detector trains on boxes, so
the VLM cannot label frames for training directly. The approved design therefore uses
**self-training with VLM verification**: the detector proposes boxes, the VLM decides
which frames to trust. This exploits 6b's measured strength (presence/condition
tagging — night F1 0.989, pedestrian presence precision 0.97) rather than its measured
weakness (exact counts when crowded — MAE 6.67 at 10+ objects).

## Experiment design (approved)

Two new training arms (~20 min each on TRINITY), read three ways against the existing
round-1 `random` result (+0.0340 overall mAP50-95, GT labels, all 1,500 frames):

| arm | added frames | labels |
|---|---|---|
| `weak_random` | accepted subset of `random.parquet` | **pseudo** (detector + VLM-verified) |
| `weak_random_gt` | the **same** accepted subset | **ground truth** |

- **`weak_random` vs `weak_random_gt`** — identical frames, so the delta is purely the
  cost of losing ground truth. This is the headline number.
- **`weak_random_gt` vs `random`** — identical label source, so the delta is purely the
  cost of dropping frames the verifier rejected.
- **`weak_random` vs baseline** — does the engine improve *at all* with no GT?

Without the GT twin, a loss could be blamed on either fewer frames or worse labels; the
twin removes that confound for one extra training run. The baseline seed set keeps its
GT labels throughout — this models the realistic case (a labelled seed, then unlabelled
data arriving).

## Pipeline: `active_learning/pseudo_label.py` + `al pseudo-label --arm <arm>`

1. **Propose.** Run the baseline detector (`al run --arm baseline` weights) over the
   arm's frames at `conf >= pseudo.conf` (default 0.5), emitting rows in the
   *annotations* schema (`sample_data_token, category_group, x_min, y_min, x_max,
   y_max`) so downstream code needs no new format.
2. **Label.** VLM-label the arm's frames not already covered by 6b's 5,000-frame run
   (~1,271 of `random`'s 1,500), reusing the existing batch pipeline and the $0 local
   Qwen2.5-VL provider. Results append to the 6b labels table.
3. **Verify.** Per frame, map the VLM's 10 classes onto the detector's 5
   (`cars→car, trucks→truck, buses→bus, pedestrians→pedestrian, bicycles→bicycle`;
   the other five VLM classes have no detector counterpart and are ignored). Accept the
   frame iff `|detector_count − vlm_count| <= tolerance` (default 1) for **every**
   mapped class. Frames whose VLM label is missing or unparsed are rejected.
4. **Write.** `pseudo_labels.parquet` (accepted frames' boxes), `accepted.parquet`
   (token list, consumed by both new arms), and a summary: retention rate, per-class
   agreement, mean boxes/frame vs GT.

Frame-level accept/reject (not per-class box filtering) is deliberate: silently
dropping one class's boxes from a kept frame teaches the detector that those objects
are background — actively harmful for pedestrians, the class that matters most.

## Training seam

`build_yolo_dataset` gains `pseudo_labels: pd.DataFrame | None`. For tokens present in
it, GT annotation rows are dropped and pseudo rows substituted **before** the existing
normalization, so there is exactly one label-line code path. `_build_key` must include a
content hash of the pseudo set — otherwise the idempotent builder would reuse a
GT-labelled dataset for a pseudo arm and silently invalidate the experiment.

`experiment.py` registers `weak_random` / `weak_random_gt`, resolving frames from
`accepted.parquet` and passing the pseudo table only for `weak_random`.

## Guards (explicit `ValueError`, per project convention)

- pseudo-labelled tokens ⊆ the arm's mined frames ⊆ pool; **no val frame is ever
  pseudo-labelled** (asserted against the official val split).
- both new arms resolve the identical accepted-token set (asserted at build time).
- retention rate is reported in the summary, the report table, and the docs; a
  retention below 50% is called out prominently rather than buried.
- an empty accepted set fails loudly instead of training a baseline-equivalent arm.

## Testing

Pure and torch-free: the verification function (class mapping, tolerance boundaries,
missing/unparsed VLM label, a frame with zero detections vs a VLM count of 0), the
pseudo-label row projection, and the `_build_key` hash sensitivity. The dataset-builder
override is tested with small fixture frames (GT rows replaced, non-pseudo tokens
untouched). Integration follows the existing fake-devkit patterns.

## Acceptance

1. Test suite green in CI.
2. TRINITY: `al pseudo-label --arm random` reports a retention rate and per-class
   agreement; both arms train; `al report` renders eleven arms.
3. The three comparisons above are reported with measured numbers — including a
   negative result, if that is what the data says.

## Runbook

Every stage needs the images, the detector, and the vLLM provider, so the whole
pipeline runs on TRINITY (unlike the graph arms, which needed the infra Mac's Neo4j).

```bash
# TRINITY — propose + label + verify, then train both arms
scripts/gpu-run.sh al pseudo-label --arm random
scripts/gpu-run.sh --bg raw "sh -c 'env CUDA_VISIBLE_DEVICES=<free-gpu> uv run nuscenes-data-engine al run --arm weak_random && env CUDA_VISIBLE_DEVICES=<free-gpu> uv run nuscenes-data-engine al run --arm weak_random_gt && uv run nuscenes-data-engine al report && echo CHAIN_COMPLETE'"
# infra Mac — sync results + mlruns back, regenerate the report, write up
```

The `sh -c` wrapper is required: a bare `&&` chain leaves later commands outside
`nohup` and their output unlogged.

*Superseded by the measured runbook in [ACTIVE_LEARNING.md](../../ACTIVE_LEARNING.md)
— the shipped flow is `al pseudo-sample` → `autolabel submit/collect` →
`al pseudo-label --weights ...` (the sketch above omits the now-required
`--weights` and collapses three separate stages into one).*

## Explicitly out of scope

- No open-vocabulary detector (Grounding DINO / OWL-ViT) — a much larger project with
  its own unvalidated error profile.
- No weak-supervision of other arms this round (`graph_rate_night` is the natural
  follow-up if `random` holds up).
- No changes to round 1–3 artifacts, the random control, or existing arms.
