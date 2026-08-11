# Weak-supervising the night champion (`graph_rate_night`)

**Status:** approved 2026-08-10. Generalizes the weak-supervision wiring beyond the
hardcoded `random_*` names (a follow-up the 2026-08-06 review flagged) and runs the
round-3 night champion through the existing pseudo-label harness.

## Why this arm

The first weak-supervision run used `random` — 12.7% night. `graph_rate_night` is
**30.9% night** (1,500 frames, 328 already VLM-labelled, **1,172 needing labels**), and
Phase 6b measured the VLM missing the lone pedestrian in **60.6%** of single-pedestrian
frames. That gives a pre-registered prediction rather than a rerun:

- retention should fall below the `random` arm's **63.9%**;
- `accepted_mutual_zero_by_class["pedestrian"]` should rise above its **686/958** share;
- the night gain (**+0.0101**, the project's best) is the most likely casualty.

Comparing weak-supervision cost across two arms of different composition is the finding,
whichever way it lands.

## Code change: one map replaces three hardcoded sites

`experiment.py` currently hardcodes `random` in `ARM_EXTRA_FILE`'s two weak entries and
in `run_arm`'s `if arm == "weak_random"` block (which reads
`random_pseudo_labels.parquet` and `random_accepted.parquet`). Replace with a
module-level map:

```python
WEAK_ARMS: dict[str, tuple[str, bool]] = {   # arm -> (base arm, uses pseudo labels)
    "weak_random": ("random", True),
    "weak_random_gt": ("random", False),
    "weak_graph_rate_night": ("graph_rate_night", True),
    "weak_graph_rate_night_gt": ("graph_rate_night", False),
}
```

- `ARM_EXTRA_FILE`'s weak entries derive from it (`f"{base}_accepted.parquet"`).
- `run_arm` resolves `(base, uses_pseudo)` from it instead of string-matching, and reads
  `f"{base}_pseudo_labels.parquet"` / `f"{base}_accepted.parquet"`. Its existing
  "run `al pseudo-label --arm <base>` first" `ValueError` names the resolved base.
- `report.py` already resolves through `ARM_EXTRA_FILE`, so both new arms inherit
  composition **and** the retention column with no change there.
- `ARMS` grows to 13.

`pseudo_label.py` needs **no change** — it is already arm-parameterized
(`al pseudo-sample --arm <arm>`, `al pseudo-label --arm <arm>`).

## Experiment

Same three-way read as the `random` round, against the existing `graph_rate_night`
result (+0.0254 overall mAP50-95 / +0.0101 night):

| arm | added frames | labels |
|---|---|---|
| `weak_graph_rate_night` | accepted subset | **pseudo** (detector + VLM-verified) |
| `weak_graph_rate_night_gt` | the **same** accepted subset | **ground truth** |

- pseudo vs GT twin → the cost of losing ground truth on identical frames;
- GT twin vs `graph_rate_night` → the cost of the frames the verifier dropped;
- pseudo vs baseline → does the engine improve at all here with no GT?

Report the same decomposition as the `random` round (which retained **18%** of the GT
gain: ~50% of the loss from dropped frames, ~32% from label quality) so the two arms are
directly comparable, and note whether the verifier again discards the box-dense frames
(the `random` round: 7.61 GT boxes/frame rejected vs 3.87 accepted).

## Guards, testing, acceptance

All existing guards apply unchanged and unmodified: label coverage before the GPU
proposal, val-split disjointness in both `run_pseudo_label` and `build_yolo_dataset`,
`accepted ⊆ arm frames`, exact-`n_mine` and cache-key hashing.

New tests: `WEAK_ARMS` resolves both pairs to the right base and pseudo flag; both new
arms resolve identical frame sets to each other; `run_arm` raises naming the correct base
arm when its pseudo table is missing; `ARM_EXTRA_FILE` stays consistent with `WEAK_ARMS`.

Acceptance: suite green in CI; the TRINITY run reports retention and per-class
diagnostics; both arms train; `al report` renders **13** arms with composition and
retention populated for all four weak arms; the decomposition is documented with
measured numbers, including a negative result if that is what the data says.

## Runbook

The vLLM server needs a 24 GB node and the venv's bin on `PATH` (vLLM shells out to
`ninja` by name); see the measured runbook in `docs/ACTIVE_LEARNING.md`.

```bash
scripts/gpu-run.sh al pseudo-sample --arm graph_rate_night     # ~1,172 frames need labels
# serve the VLM on trinity-2-3, then label via the unchanged 6b path (see ACTIVE_LEARNING.md)
GPU_DEVICES=2 scripts/gpu-run.sh al pseudo-label --arm graph_rate_night \
  --weights runs/yolov8n_imgsz640_e20_al-baseline/weights/best.pt
scripts/gpu-run.sh --bg raw "sh -c 'env CUDA_VISIBLE_DEVICES=2 uv run nuscenes-data-engine al run --arm weak_graph_rate_night && env CUDA_VISIBLE_DEVICES=2 uv run nuscenes-data-engine al run --arm weak_graph_rate_night_gt && uv run nuscenes-data-engine al report && echo CHAIN_COMPLETE'"
```

## Explicitly out of scope

- No new verification rule. The stricter pedestrian-presence check the 2026-08-06 review
  suggested is a different acquisition function and deserves its own experiment.
- No changes to `pseudo_label.py`, existing arms, or any round 1–3 artifact.
