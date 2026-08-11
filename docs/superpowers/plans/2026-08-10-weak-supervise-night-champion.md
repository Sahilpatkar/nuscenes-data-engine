# Weak-Supervising the Night Champion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run `graph_rate_night` (the round-3 night champion) through the existing pseudo-label harness, after replacing three hardcoded `random_*` sites with one `WEAK_ARMS` map — per the approved spec `docs/superpowers/specs/2026-08-10-weak-supervise-night-champion-design.md`.

**Architecture:** `pseudo_label.py` needs **no change** — it is already arm-parameterized. The only code work is in `experiment.py`: a `WEAK_ARMS` map that names each weak arm's base arm and whether it uses pseudo labels, from which `ARM_EXTRA_FILE`'s weak entries and `run_arm`'s pseudo-loading block both derive. `report.py` already recovers the base arm from `ARM_EXTRA_FILE`'s `<base>_accepted.parquet` naming, so both new arms inherit composition and the retention column for free.

**Tech Stack:** Python 3.11, pandas, Typer CLI, pytest (torch-free core). Strict mypy + ruff; guards raise `ValueError`, never `assert`.

**Working branch:** `weak-sup-night` (exists; spec committed as 1259b30).

**Conventions:** Repo root `/Users/sahilpatkar/Curosr_repos/nuscenes-data-engine`. Tests: `uv run pytest tests/test_weak_supervision.py tests/test_active_learning.py -q`. Lint/type: `uv run ruff check . && uv run mypy`. Do NOT bulk-`ruff format` (CI gates `ruff check` only; pre-existing repo-wide drift).

**Verified facts (do not re-derive):**
- `ARMS` currently has 11 entries, ending `"weak_random", "weak_random_gt",` (`experiment.py:28-33`).
- `ARM_EXTRA_FILE` (`experiment.py:38-49`) maps every non-baseline arm to its extra-frames parquet; the two weak entries both point at `random_accepted.parquet`.
- `run_arm`'s pseudo block (`experiment.py:125-139`) is `if arm == "weak_random":` and reads the literal `random_pseudo_labels.parquet` / `random_accepted.parquet`.
- `report._weak_arm_retention` (`report.py:18-40`) recovers `<base>` by stripping `_accepted.parquet` off `ARM_EXTRA_FILE[arm]` — it generalizes with no edit.
- `graph_rate_night.parquet` holds 1,500 tokens; 328 already carry an `ok` VLM label, so **1,172 need labelling**; the arm is **30.9% night**.

---

### Task 1: `WEAK_ARMS` map replaces the hardcoded `random_*` sites

**Files:**
- Modify: `src/nuscenes_data_engine/active_learning/experiment.py` (module docstring; `ARMS` at :28-33; `ARM_EXTRA_FILE` at :38-49; `run_arm`'s pseudo block at :125-139)
- Test: `tests/test_weak_supervision.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_weak_supervision.py`:

```python
def test_weak_arms_map_resolves_base_and_pseudo_flag() -> None:
    from nuscenes_data_engine.active_learning.experiment import (
        ARM_EXTRA_FILE,
        ARMS,
        WEAK_ARMS,
    )

    assert WEAK_ARMS["weak_random"] == ("random", True)
    assert WEAK_ARMS["weak_random_gt"] == ("random", False)
    assert WEAK_ARMS["weak_graph_rate_night"] == ("graph_rate_night", True)
    assert WEAK_ARMS["weak_graph_rate_night_gt"] == ("graph_rate_night", False)

    # Every weak arm is registered and resolves to its base arm's accepted set.
    for arm, (base, _uses_pseudo) in WEAK_ARMS.items():
        assert arm in ARMS
        assert ARM_EXTRA_FILE[arm] == f"{base}_accepted.parquet"

    # Exactly one pseudo arm and one GT twin per base.
    for base in {b for b, _ in WEAK_ARMS.values()}:
        flags = sorted(uses for b, uses in WEAK_ARMS.values() if b == base)
        assert flags == [False, True], f"base {base} needs exactly one pseudo + one GT arm"


def test_night_weak_arms_share_frames_and_resolve(tmp_path: Path) -> None:
    from nuscenes_data_engine.active_learning.experiment import resolve_arm_frames

    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame(
        {
            "sample_data_token": ["bl-1", "bl-2", "p-1", "p-2"],
            "scene_name": ["bl", "bl", "pool", "pool"],
            "channel": ["CAM_FRONT"] * 4,
            "filename": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "is_night": [False] * 4,
        }
    ).to_parquet(processed / "samples.parquet", index=False)
    state = tmp_path / "state"
    state.mkdir()
    pd.DataFrame({"scene_name": ["bl", "pool"], "role": ["baseline", "pool"]}).to_parquet(
        state / "split.parquet", index=False
    )
    pd.DataFrame({"sample_data_token": ["p-2"]}).to_parquet(
        state / "graph_rate_night_accepted.parquet", index=False
    )

    cfg = {"split": {"channel": "CAM_FRONT"}}
    weak = resolve_arm_frames(state, processed, cfg, "weak_graph_rate_night")
    weak_gt = resolve_arm_frames(state, processed, cfg, "weak_graph_rate_night_gt")
    assert weak == weak_gt == {"bl-1", "bl-2", "p-2"}


def test_run_arm_missing_pseudo_table_names_the_right_base(tmp_path: Path) -> None:
    """The error must name the arm's own base, not a hardcoded 'random'."""
    import yaml

    from nuscenes_data_engine.active_learning.experiment import run_arm

    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame(
        {
            "sample_data_token": ["bl-1", "p-1"],
            "scene_name": ["bl", "pool"],
            "channel": ["CAM_FRONT"] * 2,
            "filename": ["a.jpg", "b.jpg"],
            "is_night": [False] * 2,
        }
    ).to_parquet(processed / "samples.parquet", index=False)
    state = tmp_path / "state"
    state.mkdir()
    pd.DataFrame({"scene_name": ["bl", "pool"], "role": ["baseline", "pool"]}).to_parquet(
        state / "split.parquet", index=False
    )
    pd.DataFrame({"sample_data_token": ["p-1"]}).to_parquet(
        state / "graph_rate_night_accepted.parquet", index=False
    )
    config = tmp_path / "al.yaml"
    config.write_text(
        yaml.safe_dump({"state": {"dir": str(state)}, "split": {"channel": "CAM_FRONT"}})
    )

    with pytest.raises(ValueError, match="graph_rate_night_pseudo_labels.parquet"):
        run_arm(config, arm="weak_graph_rate_night", processed_dir=processed)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_weak_supervision.py -k "weak_arms_map or night_weak_arms or names_the_right_base" -v`
Expected: FAIL — `ImportError: cannot import name 'WEAK_ARMS'` for the first two; the third fails on the unknown arm.

- [ ] **Step 3: Implement**

In `src/nuscenes_data_engine/active_learning/experiment.py`:

(a) `ARMS` becomes:

```python
ARMS = (
    "baseline", "mined", "random", "graph",
    "rate", "strat", "rate_strat",
    "graph_rate", "graph_rate_night",
    "weak_random", "weak_random_gt",
    "weak_graph_rate_night", "weak_graph_rate_night_gt",
)

# Weak-supervision arms -> (base arm whose accepted set they train on, uses pseudo
# labels). Each base gets exactly two arms: the pseudo one and its GT twin, which
# train on the SAME frames so a delta isolates label quality from frame count.
WEAK_ARMS: dict[str, tuple[str, bool]] = {
    "weak_random": ("random", True),
    "weak_random_gt": ("random", False),
    "weak_graph_rate_night": ("graph_rate_night", True),
    "weak_graph_rate_night_gt": ("graph_rate_night", False),
}
```

(b) `ARM_EXTRA_FILE` — replace the two hardcoded weak entries with a derivation, keeping the non-weak entries literal:

```python
# Per-arm extra-frames parquet, relative to the AL state dir. Module-level (not just a
# local in resolve_arm_frames) so report.arm_composition can resolve the same file —
# the weak arms share their base arm's <base>_accepted.parquet, not f"{arm}.parquet".
ARM_EXTRA_FILE: dict[str, str] = {
    "mined": "mined.parquet",
    "random": "random.parquet",
    "graph": "graph.parquet",
    "rate": "rate.parquet",
    "strat": "strat.parquet",
    "rate_strat": "rate_strat.parquet",
    "graph_rate": "graph_rate.parquet",
    "graph_rate_night": "graph_rate_night.parquet",
    **{arm: f"{base}_accepted.parquet" for arm, (base, _) in WEAK_ARMS.items()},
}
```

(Note `WEAK_ARMS` must be defined **above** `ARM_EXTRA_FILE` for this comprehension.)

(c) `run_arm`'s pseudo block — replace the `if arm == "weak_random":` block with:

```python
    pseudo_labels = None
    pseudo_tokens = None
    base, uses_pseudo = WEAK_ARMS.get(arm, ("", False))
    if uses_pseudo:
        pseudo_path = state_dir / f"{base}_pseudo_labels.parquet"
        if not pseudo_path.is_file():
            raise ValueError(
                f"Arm {arm!r} needs {pseudo_path} — run `al pseudo-label --arm {base}` first"
            )
        pseudo_labels = pd.read_parquet(pseudo_path)
        # Every accepted frame is pseudo-labelled, including those the detector found
        # nothing in — those have no rows in the table, so the token set comes from
        # accepted.parquet or their ground truth would survive into a "no GT" arm.
        pseudo_tokens = set(
            pd.read_parquet(state_dir / f"{base}_accepted.parquet")["sample_data_token"]
        )
```

(d) Update the module docstring's arm paragraph to:

```python
"""Experiment arms: build the arm's dataset, train, evaluate, record results.

Arms: ``baseline`` (25% train scenes) plus baseline-and-extra-frames arms —
``mined``/``random``/``graph`` from round 1, ``rate``/``strat``/``rate_strat`` from
round 2 (docs/superpowers/specs/2026-08-02-al-round-2-design.md), and
``graph_rate``/``graph_rate_night`` from round 3
(docs/superpowers/specs/2026-08-04-al-round-3-design.md). Plus weak-supervision pairs
(see ``WEAK_ARMS``): each base arm gets a pseudo-labelled arm and a GT twin trained on
the SAME accepted frame set, isolating label quality from frame count
(docs/superpowers/specs/2026-08-06-vlm-weak-supervision-design.md and
docs/superpowers/specs/2026-08-10-weak-supervise-night-champion-design.md). Each arm
gets its own YOLO dataset dir and run-name suffix; the val split is identical across
arms by construction and asserted at result-merge time.
"""
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_weak_supervision.py tests/test_active_learning.py -q`
Expected: all pass, including the pre-existing `weak_random` tests (the map must not change their behaviour).

- [ ] **Step 5: Commit**

```bash
git add src/nuscenes_data_engine/active_learning/experiment.py tests/test_weak_supervision.py
git commit -m "weak-sup night: WEAK_ARMS map replaces hardcoded random_* wiring"
```

---

### Task 2: CI-parity sweep

- [ ] **Step 1:** `uv run pytest -q` — report exact counts (expect ~264 passed, 2 skipped, 2 deselected). `uv run ruff check .` clean. Bare `uv run mypy` clean.
- [ ] **Step 2:** Confirm the report still renders and now knows about 13 arms: `uv run python -c "from nuscenes_data_engine.active_learning.report import ARM_ORDER; print(len(ARM_ORDER), ARM_ORDER[-4:])"` → `13` and the four weak arms.
- [ ] **Step 3:** Commit only if fixes were needed: `git commit -am "weak-sup night: lint/type fixes"`.

---

### Task 3: TRINITY run (operational; no commits)

Push first: `git push -u origin weak-sup-night` (gpu-run.sh runs the pushed branch).

- [ ] **Step 1: Write the weak sample.** `scripts/gpu-run.sh al pseudo-sample --arm graph_rate_night` — expect **1,172 of 1,500** frames needing labels. Record the number; if it differs materially, STOP and report (it means the label tables moved).
- [ ] **Step 2: Serve the VLM.** Needs a 24 GB node and the venv bin on `PATH` (vLLM shells out to `ninja` by name; `--enforce-eager` does NOT avoid this):

```bash
GPU_NODE=trinity-2-3 scripts/gpu-run.sh --bg raw "env PATH=/home/mgaur/sahil/vllm-env/bin:/usr/local/bin:/usr/bin:/bin HF_HOME=/home/mgaur/sahil/nuscenes_project/.cache/huggingface CUDA_VISIBLE_DEVICES=0 /home/mgaur/sahil/vllm-env/bin/vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8399 --max-model-len 8192"
```

Poll readiness (~2 min): `ssh trinity-2-3 "curl -s -o /dev/null -w '%{http_code}' http://localhost:8399/v1/models"` → `200`.

- [ ] **Step 3: Label.** The weak state dir already holds the `random` round's labels; `build_weak_sample` skips tokens already `ok`, so only the 1,172 new ones are sent:

```bash
GPU_NODE=trinity-2-3 scripts/gpu-run.sh --bg raw "sh -c 'uv run nuscenes-data-engine autolabel submit -c configs/autolabel_weak.yaml --provider local && uv run nuscenes-data-engine autolabel collect -c configs/autolabel_weak.yaml --provider local && echo LABEL_CHAIN_COMPLETE'"
```

Watch the job's own log (`ls -t gpu-run-*.log` on trinity-2-3 — pick the labelling job's, not the server's). Record rows written and the parse rate.

- [ ] **Step 4: Stop the VLM server** once labelling completes — it holds a 24 GB card on a shared cluster: `ssh trinity-2-3 "pkill -f 'vllm serve'"`, then confirm with `ssh trinity-2-3 "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"` (empty). Note `pgrep -f 'vllm serve'` over ssh matches the ssh command string itself, so use the nvidia-smi check.
- [ ] **Step 5: Propose + verify.**

```bash
GPU_DEVICES=2 scripts/gpu-run.sh al pseudo-label --arm graph_rate_night \
  --weights runs/yolov8n_imgsz640_e20_al-baseline/weights/best.pt
```

Record: retention, n_accepted, n_no_label, n_unparsed, rejected_by_class, accepted_mutual_zero_by_class, mean_boxes_per_accepted_frame, mean_gt_boxes_per_accepted_frame. **Compare against the `random` round's pre-registered predictions** (retention 0.639, mutual-zero pedestrian 686/958): note whether retention fell and whether the pedestrian mutual-zero share rose. A retention below 0.5 is a finding, not a stop condition.

- [ ] **Step 6: Train both arms** (`sh -c` wrapper required — a bare `&&` chain leaves later commands outside `nohup` and unlogged). Pick a free GPU with `ssh trinity-2-18 nvidia-smi`:

```bash
scripts/gpu-run.sh --bg raw "sh -c 'env CUDA_VISIBLE_DEVICES=2 uv run nuscenes-data-engine al run --arm weak_graph_rate_night && env CUDA_VISIBLE_DEVICES=2 uv run nuscenes-data-engine al run --arm weak_graph_rate_night_gt && uv run nuscenes-data-engine al report && echo CHAIN_COMPLETE'"
```

- [ ] **Step 7: Sync back** — results plus the artifacts the report's composition/retention columns need:

```bash
rsync -a trinity-2-18:/home/mgaur/sahil/nuscenes_project/data/active_learning/{results.json,graph_rate_night_accepted.parquet,graph_rate_night_pseudo_labels.parquet,graph_rate_night_pseudo_summary.json} data/active_learning/
rsync -a trinity-2-18:/home/mgaur/sahil/nuscenes_project/mlruns/ mlruns/
uv run nuscenes-data-engine al report
```

Confirm the report renders **13 arms** with composition and retention populated for all four weak arms.

- [ ] **Step 8: Compute the decomposition** for the docs (mirroring the `random` round's):

```bash
uv run python -c "
import json
r = json.load(open('data/active_learning/results.json'))
b = r['baseline']['overall']['mAP50-95']; bn = r['baseline']['night']['mAP50-95']
for a in ('baseline','graph_rate_night','weak_graph_rate_night_gt','weak_graph_rate_night'):
    rec = r[a]; o = rec['overall']['mAP50-95']; n = rec['night']['mAP50-95']
    print(f\"{a:26s} imgs {rec.get('n_train_images')} boxes {rec.get('n_boxes')} overall {o:.4f} ({o-b:+.4f}) night {n:.4f} ({n-bn:+.4f})\")
g = r['graph_rate_night']['overall']['mAP50-95'] - b
gt = r['weak_graph_rate_night_gt']['overall']['mAP50-95'] - b
w = r['weak_graph_rate_night']['overall']['mAP50-95'] - b
print(f'dropped-frame cost {g-gt:+.4f} | label cost {gt-w:+.4f} | retained {w/g:.1%} of the GT gain')
"
```

Also compute the crowding check (the `random` round found rejected frames held 7.61 GT boxes/frame vs 3.87 accepted):

```bash
uv run python -c "
import pandas as pd
acc = set(pd.read_parquet('data/active_learning/graph_rate_night_accepted.parquet').sample_data_token)
arm = set(pd.read_parquet('data/active_learning/graph_rate_night.parquet').sample_data_token)
ann = pd.read_parquet('data/processed/annotations.parquet', columns=['sample_data_token','category_group'])
ann = ann[ann.category_group.notna()]
rej = arm - acc
print(f'accepted {len(acc)}: {len(ann[ann.sample_data_token.isin(acc)])/len(acc):.2f} GT boxes/frame')
print(f'rejected {len(rej)}: {len(ann[ann.sample_data_token.isin(rej)])/len(rej):.2f} GT boxes/frame')
s = pd.read_parquet('data/processed/samples.parquet', columns=['sample_data_token','is_night']).set_index('sample_data_token')
print('accepted night share:', round(s.loc[sorted(acc)].is_night.mean(), 3))
"
```

---

### Task 4: Docs

**Files:** `docs/ACTIVE_LEARNING.md`, `docs/PROJECT.md`

- [ ] **Step 1:** In `docs/ACTIVE_LEARNING.md`'s weak-supervision section, add a subsection comparing the two weak-supervised arms. Use the Task 3 numbers verbatim. Cover: the pre-registered predictions and whether they held (retention vs 0.639; pedestrian mutual-zero share vs 686/958; whether the night gain survived); the three-way decomposition against `graph_rate_night` (+0.0254 overall / +0.0101 night) beside `random`'s (18% retained, ~50% dropped-frame / ~32% label cost); the crowding check; and a verdict on whether weak supervision costs *more* on a night-heavy arm. Include a single-seed noise caveat, as that section already does.
- [ ] **Step 2:** In `docs/PROJECT.md` §9, extend the weak-supervision item with the night-arm result in one or two sentences; if §5's results table carries a weak-supervision line, add the night pair alongside it.
- [ ] **Step 3:** Verify: `grep -n "weak_graph_rate_night" docs/*.md | head`, and `uv run pytest tests/test_weak_supervision.py -q` still green (docs-only sanity).
- [ ] **Step 4:** Commit: `git add docs && git commit -m "weak-sup night: docs (night-arm results + cross-arm comparison)"`

---

### After the plan (not plan tasks)

1. Final whole-branch review, then superpowers:finishing-a-development-branch (push, PR, CI).
2. If weak supervision proves systematically costlier on night-heavy arms, the stricter pedestrian-presence rule (zero tolerance when either side reports zero) becomes the natural next experiment — it is deliberately out of scope here.
