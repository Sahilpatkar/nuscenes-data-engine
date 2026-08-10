# VLM Weak Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train an active-learning arm on frames the baseline detector pseudo-labelled and the Phase 6b VLM verified — no ground truth for the added frames — per the approved spec `docs/superpowers/specs/2026-08-06-vlm-weak-supervision-design.md`.

**Architecture:** A new `active_learning/pseudo_label.py` holds a pure verification core (map the VLM's 10 counts onto the detector's 5 classes; accept a frame iff every mapped class agrees within ±1) plus GPU-side box proposal. VLM labelling of the not-yet-labelled arm frames reuses the existing Phase 6b `autolabel submit/collect` commands pointed at a separate state dir via `configs/autolabel_weak.yaml` — no new labelling code. `build_yolo_dataset` gains a `pseudo_labels` override (with the cache key hashed) so a pseudo arm can never reuse a GT-labelled build.

**Tech Stack:** Python 3.11, pandas/pyarrow, Ultralytics YOLO (proposal only), the existing vLLM/Qwen2.5-VL local provider, Typer CLI, pytest (torch-free core). Strict mypy + ruff; guards raise `ValueError`, never `assert`.

**Working branch:** `vlm-weak-supervision` (exists; spec committed as d4f4268).

**Conventions:** Repo root `/Users/sahilpatkar/Curosr_repos/nuscenes-data-engine`. Tests: `uv run pytest tests/test_weak_supervision.py -v` (new file). Lint/type: `uv run ruff check src tests && uv run mypy src`. Mirror `active_learning/mining.py` (module layout, config resolution, explicit `ValueError` guards) and `tests/test_active_learning.py` (test style).

**Key facts already verified:**
- `DETECTION_CLASSES = ("car", "truck", "bus", "pedestrian", "bicycle")`, `CLASS_TO_INDEX` in `ingestion/categories.py`.
- VLM `ObjectCounts` fields: `cars, trucks, buses, trailers, construction_vehicles, motorcycles, bicycles, pedestrians, traffic_cones, barriers` — flattened as columns in `data/autolabel/labels.parquet` (5,000 rows, also has `sample_data_token`, `parse_status`).
- `random.parquet` has 1,500 tokens; 229 already appear in `labels.parquet`, so ~1,271 need labelling.
- `build_yolo_dataset(processed_dir, dataroot, out_dir, *, cameras, limit_scenes, train_frames, force)` builds label lines at `training/dataset.py:156-177` from `annotations` columns `sample_data_token, category_group, x_min, y_min, x_max, y_max`; `_build_key` at `:48-67`.

---

### Task 1: Pure verification core

**Files:**
- Create: `src/nuscenes_data_engine/active_learning/pseudo_label.py`
- Create: `tests/test_weak_supervision.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_weak_supervision.py`:

```python
"""Tests for VLM weak supervision: pure verification core + dataset seam."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nuscenes_data_engine.active_learning.pseudo_label import (
    VLM_TO_DETECTOR,
    detection_counts,
    verify_frames,
)


def test_vlm_to_detector_covers_every_detector_class() -> None:
    from nuscenes_data_engine.ingestion.categories import DETECTION_CLASSES

    assert set(VLM_TO_DETECTOR.values()) == set(DETECTION_CLASSES)
    # The five VLM classes with no detector counterpart are deliberately absent.
    assert "traffic_cones" not in VLM_TO_DETECTOR
    assert "construction_vehicles" not in VLM_TO_DETECTOR


def test_detection_counts_per_frame_and_class() -> None:
    boxes = pd.DataFrame(
        {
            "sample_data_token": ["f1", "f1", "f1", "f2"],
            "category_group": ["car", "car", "pedestrian", "bus"],
        }
    )
    counts = detection_counts(boxes)
    assert counts["f1"]["car"] == 2
    assert counts["f1"]["pedestrian"] == 1
    assert counts["f1"].get("bus", 0) == 0
    assert counts["f2"]["bus"] == 1


def _labels(**counts: int) -> dict[str, Any]:
    """A parsed VLM label row with all ten count fields (unspecified ones zero)."""
    row = {
        "cars": 0, "trucks": 0, "buses": 0, "trailers": 0, "construction_vehicles": 0,
        "motorcycles": 0, "bicycles": 0, "pedestrians": 0, "traffic_cones": 0,
        "barriers": 0, "parse_status": "ok",
    }
    row.update(counts)
    return row


def test_verify_frames_accepts_within_tolerance() -> None:
    det = {"f1": {"car": 3, "pedestrian": 1}}
    vlm = {"f1": _labels(cars=4, pedestrians=1)}  # car off by 1 -> still accepted
    accepted, diagnostics = verify_frames(det, vlm, tolerance=1)
    assert accepted == ["f1"]
    assert diagnostics["n_candidates"] == 1 and diagnostics["n_accepted"] == 1


def test_verify_frames_rejects_when_any_class_disagrees() -> None:
    det = {"f1": {"car": 3, "pedestrian": 1}}
    vlm = {"f1": _labels(cars=3, pedestrians=5)}  # pedestrians off by 4
    accepted, diagnostics = verify_frames(det, vlm, tolerance=1)
    assert accepted == []
    assert diagnostics["rejected_by_class"]["pedestrian"] == 1


def test_verify_frames_tolerance_boundary_is_inclusive() -> None:
    det = {"f1": {"car": 2}}
    assert verify_frames(det, {"f1": _labels(cars=3)}, tolerance=1)[0] == ["f1"]
    assert verify_frames(det, {"f1": _labels(cars=4)}, tolerance=1)[0] == []


def test_verify_frames_zero_detections_matches_zero_counts() -> None:
    # A genuine empty frame: detector found nothing, VLM saw nothing -> accept
    # (an empty label file is a valid background training example).
    accepted, _ = verify_frames({"f1": {}}, {"f1": _labels()}, tolerance=1)
    assert accepted == ["f1"]


def test_verify_frames_rejects_missing_or_unparsed_labels() -> None:
    det = {"f1": {"car": 1}, "f2": {"car": 1}}
    vlm = {"f2": _labels(cars=1, parse_status="error")}  # f1 absent, f2 unparsed
    accepted, diagnostics = verify_frames(det, vlm, tolerance=1)
    assert accepted == []
    assert diagnostics["n_no_label"] == 1
    assert diagnostics["n_unparsed"] == 1


def test_verify_frames_is_deterministic_and_sorted() -> None:
    det = {t: {"car": 1} for t in ("f3", "f1", "f2")}
    vlm = {t: _labels(cars=1) for t in ("f3", "f1", "f2")}
    accepted, _ = verify_frames(det, vlm, tolerance=1)
    assert accepted == ["f1", "f2", "f3"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_weak_supervision.py -v`
Expected: FAIL at import — `ModuleNotFoundError: nuscenes_data_engine.active_learning.pseudo_label`.

- [ ] **Step 3: Implement the core**

Create `src/nuscenes_data_engine/active_learning/pseudo_label.py`:

```python
"""VLM-verified self-training: pseudo-label mined frames without ground truth.

Phase 6b's VLM emits scene-level *counts*, not boxes, so it cannot label frames for a
detector directly. Instead the baseline detector proposes boxes and the VLM verifies
them: a frame is kept only when every detector class the VLM can see agrees within a
tolerance. This leans on 6b's measured strength (presence tagging) rather than its
weakness (exact counts when crowded). Design:
docs/superpowers/specs/2026-08-06-vlm-weak-supervision-design.md
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("nuscenes_data_engine")

# VLM ObjectCounts field -> detector class. The five VLM classes with no detector
# counterpart (trailers, construction_vehicles, motorcycles, traffic_cones, barriers)
# are intentionally unmapped: the detector cannot be wrong about what it never predicts.
VLM_TO_DETECTOR: dict[str, str] = {
    "cars": "car",
    "trucks": "truck",
    "buses": "bus",
    "pedestrians": "pedestrian",
    "bicycles": "bicycle",
}


def detection_counts(boxes: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Per-frame, per-detector-class box counts from an annotations-schema frame."""
    counts: dict[str, dict[str, int]] = {}
    for token, group in boxes.groupby("sample_data_token")["category_group"]:
        counts[str(token)] = {str(k): int(v) for k, v in Counter(group).items()}
    return counts


def verify_frames(
    det_counts: dict[str, dict[str, int]],
    vlm_labels: dict[str, dict[str, Any]],
    tolerance: int,
) -> tuple[list[str], dict[str, Any]]:
    """Frames whose pseudo-labels the VLM corroborates, plus rejection diagnostics.

    A frame is accepted iff, for every mapped class, ``|detector - vlm| <= tolerance``.
    Frames with no VLM label, or whose label failed to parse, are rejected — the
    verifier can only vouch for what it actually saw.
    """
    accepted: list[str] = []
    rejected_by_class: Counter[str] = Counter()
    n_no_label = 0
    n_unparsed = 0

    for token in sorted(det_counts):
        label = vlm_labels.get(token)
        if label is None:
            n_no_label += 1
            continue
        if label.get("parse_status") != "ok":
            n_unparsed += 1
            continue
        disagreeing = [
            det_class
            for vlm_field, det_class in VLM_TO_DETECTOR.items()
            if abs(int(det_counts[token].get(det_class, 0)) - int(label[vlm_field]))
            > tolerance
        ]
        if disagreeing:
            rejected_by_class.update(disagreeing)
            continue
        accepted.append(token)

    diagnostics = {
        "n_candidates": len(det_counts),
        "n_accepted": len(accepted),
        "n_no_label": n_no_label,
        "n_unparsed": n_unparsed,
        "retention": len(accepted) / len(det_counts) if det_counts else 0.0,
        "rejected_by_class": dict(rejected_by_class),
    }
    return accepted, diagnostics
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_weak_supervision.py -v`
Expected: 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuscenes_data_engine/active_learning/pseudo_label.py tests/test_weak_supervision.py
git commit -m "weak-sup: pure verification core (VLM count check over detector boxes)"
```

---

### Task 2: Box proposal (detector → annotations schema)

**Files:**
- Modify: `src/nuscenes_data_engine/active_learning/pseudo_label.py`
- Test: `tests/test_weak_supervision.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_weak_supervision.py`:

```python
def test_boxes_to_rows_projects_to_annotations_schema() -> None:
    import numpy as np

    from nuscenes_data_engine.active_learning.pseudo_label import boxes_to_rows

    rows = boxes_to_rows(
        "f1",
        np.array([[10.0, 20.0, 110.0, 220.0], [0.0, 0.0, 50.0, 50.0]]),
        np.array([0, 3]),  # car, pedestrian (CLASS_TO_INDEX order)
        np.array([0.9, 0.7]),
    )
    assert [r["category_group"] for r in rows] == ["car", "pedestrian"]
    assert rows[0]["sample_data_token"] == "f1"
    assert (rows[0]["x_min"], rows[0]["y_min"]) == (10.0, 20.0)
    assert (rows[0]["x_max"], rows[0]["y_max"]) == (110.0, 220.0)
    assert rows[0]["score"] == pytest.approx(0.9)
    assert set(rows[0]) == {
        "sample_data_token", "category_group", "x_min", "y_min", "x_max", "y_max", "score",
    }


def test_boxes_to_rows_empty_frame_yields_no_rows() -> None:
    import numpy as np

    from nuscenes_data_engine.active_learning.pseudo_label import boxes_to_rows

    assert boxes_to_rows("f1", np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_weak_supervision.py -k boxes_to_rows -v`
Expected: FAIL — `ImportError: cannot import name 'boxes_to_rows'`.

- [ ] **Step 3: Implement**

Append to `pseudo_label.py`:

```python
def boxes_to_rows(
    token: str,
    xyxy: Any,
    classes: Any,
    scores: Any,
) -> list[dict[str, Any]]:
    """One detector frame's boxes as annotations-schema rows (pixel xyxy + score)."""
    from nuscenes_data_engine.ingestion.categories import DETECTION_CLASSES

    return [
        {
            "sample_data_token": token,
            "category_group": DETECTION_CLASSES[int(cls)],
            "x_min": float(box[0]),
            "y_min": float(box[1]),
            "x_max": float(box[2]),
            "y_max": float(box[3]),
            "score": float(score),
        }
        for box, cls, score in zip(xyxy, classes, scores, strict=True)
    ]


def propose_boxes(
    weights: Path,
    frames: pd.DataFrame,
    dataroot: Path,
    *,
    conf: float,
    imgsz: int,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    """Run the baseline detector over ``frames`` (needs GPU + ultralytics + images).

    ``frames`` needs ``sample_data_token`` and ``filename``. Returns annotations-schema
    rows for every detection at or above ``conf`` — frames with no detection simply
    contribute no rows (a valid empty/background label file downstream).
    """
    from nuscenes_data_engine.training.runtime import configure_ultralytics

    configure_ultralytics()  # before importing ultralytics
    from ultralytics import YOLO

    model = YOLO(str(weights))
    rows: list[dict[str, Any]] = []
    records = frames.to_dict("records")
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        paths = [str(dataroot / r["filename"]) for r in batch]
        results = model.predict(paths, imgsz=imgsz, conf=conf, device=device, verbose=False)
        for record, result in zip(batch, results, strict=True):
            boxes = result.boxes
            if not len(boxes):
                continue
            rows.extend(
                boxes_to_rows(
                    str(record["sample_data_token"]),
                    boxes.xyxy.cpu().numpy(),
                    boxes.cls.cpu().numpy().astype(int),
                    boxes.conf.cpu().numpy(),
                )
            )
        if (start // batch_size) % 10 == 0:
            logger.info("Proposing: %d/%d frames", min(start + batch_size, len(records)), len(records))
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_weak_supervision.py -v`
Expected: 10 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuscenes_data_engine/active_learning/pseudo_label.py tests/test_weak_supervision.py
git commit -m "weak-sup: detector box proposal into the annotations schema"
```

---

### Task 3: The weak-labelling sample + `configs/autolabel_weak.yaml`

The arm's frames that Phase 6b never labelled must go through the VLM. This reuses the existing `autolabel submit/collect` commands unchanged, pointed at a separate state dir so 6b's 5,000-frame run is never touched.

**Files:**
- Modify: `src/nuscenes_data_engine/active_learning/pseudo_label.py`
- Create: `configs/autolabel_weak.yaml`
- Modify: `src/nuscenes_data_engine/cli.py` (new `al pseudo-sample` command)
- Test: `tests/test_weak_supervision.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_weak_supervision.py`:

```python
def test_build_weak_sample_selects_only_unlabelled_arm_frames(tmp_path: Path) -> None:
    from nuscenes_data_engine.active_learning.pseudo_label import build_weak_sample

    samples = pd.DataFrame(
        {
            "sample_data_token": ["a", "b", "c"],
            "sample_token": ["sa", "sb", "sc"],
            "channel": ["CAM_FRONT"] * 3,
            "filename": ["a.jpg", "b.jpg", "c.jpg"],
            "width": [1600] * 3,
            "height": [900] * 3,
            "timestamp": [1, 2, 3],
            "n_boxes": [1, 2, 3],
            "scene_token": ["s1"] * 3,
            "scene_name": ["scene-0001"] * 3,
            "scene_description": ["x"] * 3,
            "log_token": ["l1"] * 3,
            "location": ["boston-seaport"] * 3,
            "is_night": [False] * 3,
            "is_rain": [False] * 3,
        }
    )
    arm_tokens = ["a", "b", "c"]
    already = pd.DataFrame({"sample_data_token": ["b"], "parse_status": ["ok"]})

    weak = build_weak_sample(samples, arm_tokens, already)
    assert list(weak["sample_data_token"]) == ["a", "c"]  # 'b' already labelled
    # Columns the 6b submit path reads must all be present.
    for column in ("filename", "sample_token", "scene_name", "present", "in_opus_subset"):
        assert column in weak.columns
    assert weak["present"].all() and not weak["in_opus_subset"].any()


def test_build_weak_sample_relabels_unparsed_frames(tmp_path: Path) -> None:
    from nuscenes_data_engine.active_learning.pseudo_label import build_weak_sample

    samples = pd.DataFrame(
        {
            "sample_data_token": ["a"], "sample_token": ["sa"], "channel": ["CAM_FRONT"],
            "filename": ["a.jpg"], "width": [1600], "height": [900], "timestamp": [1],
            "n_boxes": [1], "scene_token": ["s1"], "scene_name": ["scene-0001"],
            "scene_description": ["x"], "log_token": ["l1"], "location": ["boston-seaport"],
            "is_night": [False], "is_rain": [False],
        }
    )
    already = pd.DataFrame({"sample_data_token": ["a"], "parse_status": ["error"]})
    weak = build_weak_sample(samples, ["a"], already)
    assert list(weak["sample_data_token"]) == ["a"]  # unparsed -> label again
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_weak_supervision.py -k weak_sample -v`
Expected: FAIL — `ImportError: cannot import name 'build_weak_sample'`.

- [ ] **Step 3: Implement**

Append to `pseudo_label.py`:

```python
def build_weak_sample(
    samples: pd.DataFrame, arm_tokens: list[str], existing_labels: pd.DataFrame | None
) -> pd.DataFrame:
    """The arm frames still needing a VLM label, in Phase 6b's ``sample.parquet`` shape.

    Frames already labelled with ``parse_status == 'ok'`` are skipped; unparsed ones are
    re-labelled. The extra columns (``present``/``in_opus_subset``/``stratum``) exist so
    the unchanged 6b submit/collect path can consume this table.
    """
    labelled: set[str] = set()
    if existing_labels is not None and len(existing_labels):
        ok = existing_labels[existing_labels["parse_status"] == "ok"]
        labelled = set(ok["sample_data_token"])
    wanted = [t for t in arm_tokens if t not in labelled]
    weak = samples[samples["sample_data_token"].isin(wanted)].copy()
    weak["present"] = True
    weak["in_opus_subset"] = False
    weak["stratum"] = "weak-supervision"
    return weak.sort_values("sample_data_token", ignore_index=True)
```

- [ ] **Step 4: Create `configs/autolabel_weak.yaml`**

This mirrors `configs/autolabel.yaml` exactly except for `state.dir` (verified against
that file and `_state_paths` at `batch.py:296-304`, which derives sample/batches/results/labels
from it). The `sample:` block is intentionally omitted — `al pseudo-sample` writes the
sample table, so `autolabel sample` is never run against this config.

```yaml
# Phase 6b config reused for weak supervision: identical labelling, separate state dir
# so the original 5,000-frame run's sample/batches/results/labels are never touched.
# The sample is written by `al pseudo-sample --arm <arm>`, not `autolabel sample`.
provider: local

models:
  primary: claude-haiku-4-5
  comparison: claude-opus-4-8

local:
  base_url: http://localhost:8399/v1
  model: Qwen/Qwen2.5-VL-7B-Instruct
  concurrency: 8

batch:
  chunk_size: 500
  max_batch_bytes: 190000000
  max_tokens: 800

state:
  dir: data/active_learning/autolabel_weak
```

- [ ] **Step 5: Add the `al pseudo-sample` CLI command**

In `src/nuscenes_data_engine/cli.py`, after the `al_mine` command:

```python
@al_app.command("pseudo-sample")
def al_pseudo_sample(
    arm: str = typer.Option(..., "--arm", help="Arm whose frames need VLM labels."),
    config: Path = typer.Option(Path("configs/active_learning.yaml"), "--config", "-c"),
    weak_config: Path = typer.Option(Path("configs/autolabel_weak.yaml"), "--weak-config"),
) -> None:
    """Write the VLM sample for an arm's not-yet-labelled frames (step 1 of weak sup)."""
    from nuscenes_data_engine.active_learning.pseudo_label import run_pseudo_sample

    summary = run_pseudo_sample(config, weak_config, arm=arm)
    logger.info("Pseudo-sample: %s", summary)
```

- [ ] **Step 6: Implement `run_pseudo_sample`**

Append to `pseudo_label.py`:

```python
def run_pseudo_sample(
    config_path: Path, weak_config_path: Path, *, arm: str, processed_dir: Path | None = None
) -> dict[str, Any]:
    """Write ``sample.parquet`` (6b shape) for the arm frames still needing VLM labels."""
    from nuscenes_data_engine.config import get_settings, load_yaml

    cfg = load_yaml(config_path)
    weak_cfg = load_yaml(weak_config_path)
    settings = get_settings()
    state_dir = Path(cfg.get("state", {}).get("dir", "data/active_learning"))
    processed = processed_dir or Path("data/processed")
    weak_state = Path(weak_cfg.get("state", {}).get("dir", "data/active_learning/autolabel_weak"))

    arm_path = state_dir / f"{arm}.parquet"
    if not arm_path.is_file():
        raise ValueError(f"No mined frames for arm {arm!r} at {arm_path}")
    arm_tokens = list(pd.read_parquet(arm_path)["sample_data_token"])

    samples = pd.read_parquet(processed / "samples.parquet")
    labels_path = Path(settings.data_dir) / "autolabel" / "labels.parquet"
    existing = pd.read_parquet(labels_path) if labels_path.is_file() else None
    weak_labels_path = weak_state / "labels.parquet"
    if weak_labels_path.is_file():  # a previous weak run already labelled some
        weak_existing = pd.read_parquet(weak_labels_path)
        existing = weak_existing if existing is None else pd.concat([existing, weak_existing])

    weak = build_weak_sample(samples, arm_tokens, existing)
    weak_state.mkdir(parents=True, exist_ok=True)
    weak.to_parquet(weak_state / "sample.parquet", index=False)
    summary = {
        "arm": arm,
        "n_arm_frames": len(arm_tokens),
        "n_needing_labels": len(weak),
        "sample_parquet": str(weak_state / "sample.parquet"),
    }
    logger.info(
        "Arm %s: %d/%d frames need VLM labels -> %s",
        arm, len(weak), len(arm_tokens), weak_state / "sample.parquet",
    )
    return summary
```

- [ ] **Step 7: Verify**

Run: `uv run pytest tests/test_weak_supervision.py -v` (12 PASS) and `uv run nuscenes-data-engine al pseudo-sample --help` (shows `--arm`, `--config`, `--weak-config`). Do NOT run it for real yet (needs the arm parquet + samples, fine locally, but the run belongs to Task 8).

- [ ] **Step 8: Commit**

```bash
git add src/nuscenes_data_engine/active_learning/pseudo_label.py configs/autolabel_weak.yaml src/nuscenes_data_engine/cli.py tests/test_weak_supervision.py
git commit -m "weak-sup: weak-labelling sample + separate autolabel state dir + CLI"
```

---

### Task 4: `run_pseudo_label` orchestration + CLI + config

**Files:**
- Modify: `src/nuscenes_data_engine/active_learning/pseudo_label.py`
- Modify: `configs/active_learning.yaml`
- Modify: `src/nuscenes_data_engine/cli.py`
- Test: `tests/test_weak_supervision.py`

- [ ] **Step 1: Write the failing test** (orchestration with a stubbed proposer, so it runs in CI)

Append to `tests/test_weak_supervision.py`:

```python
def test_run_pseudo_label_writes_accepted_and_pseudo_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    from nuscenes_data_engine.active_learning import pseudo_label as pl

    state = tmp_path / "state"
    state.mkdir()
    processed = tmp_path / "processed"
    processed.mkdir()
    weak_state = tmp_path / "weak"
    weak_state.mkdir()

    # Arm has three frames; f1 agrees, f2 disagrees on pedestrians, f3 has no label.
    pd.DataFrame({"sample_data_token": ["f1", "f2", "f3"]}).to_parquet(
        state / "random.parquet", index=False
    )
    pd.DataFrame(
        {
            "sample_data_token": ["f1", "f2", "f3"],
            "filename": ["f1.jpg", "f2.jpg", "f3.jpg"],
            "channel": ["CAM_FRONT"] * 3,
            "scene_name": ["scene-0001"] * 3,
        }
    ).to_parquet(processed / "samples.parquet", index=False)
    base = {
        "cars": 0, "trucks": 0, "buses": 0, "trailers": 0, "construction_vehicles": 0,
        "motorcycles": 0, "bicycles": 0, "pedestrians": 0, "traffic_cones": 0,
        "barriers": 0, "parse_status": "ok",
    }
    pd.DataFrame(
        [
            {**base, "sample_data_token": "f1", "cars": 2, "pedestrians": 1},
            {**base, "sample_data_token": "f2", "cars": 2, "pedestrians": 9},
        ]
    ).to_parquet(weak_state / "labels.parquet", index=False)

    def _fake_propose(weights, frames, dataroot, **kwargs):  # noqa: ANN001, ANN003
        return pd.DataFrame(
            [
                {"sample_data_token": "f1", "category_group": "car",
                 "x_min": 0.0, "y_min": 0.0, "x_max": 10.0, "y_max": 10.0, "score": 0.9},
                {"sample_data_token": "f1", "category_group": "car",
                 "x_min": 20.0, "y_min": 0.0, "x_max": 30.0, "y_max": 10.0, "score": 0.8},
                {"sample_data_token": "f1", "category_group": "pedestrian",
                 "x_min": 5.0, "y_min": 5.0, "x_max": 9.0, "y_max": 25.0, "score": 0.7},
                {"sample_data_token": "f2", "category_group": "car",
                 "x_min": 0.0, "y_min": 0.0, "x_max": 10.0, "y_max": 10.0, "score": 0.9},
                {"sample_data_token": "f2", "category_group": "car",
                 "x_min": 20.0, "y_min": 0.0, "x_max": 30.0, "y_max": 10.0, "score": 0.8},
                {"sample_data_token": "f3", "category_group": "car",
                 "x_min": 0.0, "y_min": 0.0, "x_max": 10.0, "y_max": 10.0, "score": 0.9},
            ]
        )

    monkeypatch.setattr(pl, "propose_boxes", _fake_propose)
    monkeypatch.setattr(pl, "_official_val_scenes", lambda: {"scene-9999"})  # devkit-free

    config = tmp_path / "al.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "state": {"dir": str(state)},
                "pseudo": {"conf": 0.5, "tolerance": 1, "imgsz": 640, "batch": 32},
                "train": {"imgsz": 640},
            }
        )
    )
    weak_config = tmp_path / "weak.yaml"
    weak_config.write_text(yaml.safe_dump({"state": {"dir": str(weak_state)}}))

    summary = pl.run_pseudo_label(
        config, weak_config, arm="random", weights=tmp_path / "best.pt",
        processed_dir=processed, device="cpu",
    )

    assert summary["n_accepted"] == 1  # only f1
    assert summary["retention"] == pytest.approx(1 / 3)
    accepted = pd.read_parquet(state / "random_accepted.parquet")
    assert list(accepted["sample_data_token"]) == ["f1"]
    pseudo = pd.read_parquet(state / "random_pseudo_labels.parquet")
    assert set(pseudo["sample_data_token"]) == {"f1"}  # rejected frames' boxes dropped
    assert len(pseudo) == 3
    assert "score" not in pseudo.columns or pseudo["score"].notna().all()


def test_run_pseudo_label_empty_acceptance_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    from nuscenes_data_engine.active_learning import pseudo_label as pl

    state = tmp_path / "state"
    state.mkdir()
    processed = tmp_path / "processed"
    processed.mkdir()
    weak_state = tmp_path / "weak"
    weak_state.mkdir()
    pd.DataFrame({"sample_data_token": ["f1"]}).to_parquet(state / "random.parquet", index=False)
    pd.DataFrame(
        {"sample_data_token": ["f1"], "filename": ["f1.jpg"], "channel": ["CAM_FRONT"],
         "scene_name": ["scene-0001"]}
    ).to_parquet(processed / "samples.parquet", index=False)
    pd.DataFrame(
        [{"sample_data_token": "f1", "parse_status": "error"}]
    ).to_parquet(weak_state / "labels.parquet", index=False)

    monkeypatch.setattr(
        pl, "propose_boxes",
        lambda *a, **k: pd.DataFrame(
            [{"sample_data_token": "f1", "category_group": "car",
              "x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0, "score": 0.9}]
        ),
    )
    monkeypatch.setattr(pl, "_official_val_scenes", lambda: {"scene-9999"})  # devkit-free
    config = tmp_path / "al.yaml"
    config.write_text(
        yaml.safe_dump({"state": {"dir": str(state)}, "pseudo": {"conf": 0.5, "tolerance": 1}})
    )
    weak_config = tmp_path / "weak.yaml"
    weak_config.write_text(yaml.safe_dump({"state": {"dir": str(weak_state)}}))

    with pytest.raises(ValueError, match="no frames survived"):
        pl.run_pseudo_label(
            config, weak_config, arm="random", weights=tmp_path / "best.pt",
            processed_dir=processed, device="cpu",
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_weak_supervision.py -k run_pseudo_label -v`
Expected: FAIL — `AttributeError`/`ImportError` for `run_pseudo_label`.

- [ ] **Step 3: Implement**

Append to `pseudo_label.py`:

```python
def _official_val_scenes() -> set[str]:
    """The nuScenes val scene names (devkit); isolated so tests can monkeypatch it."""
    from nuscenes.utils.splits import create_splits_scenes

    return set(create_splits_scenes()["val"])


def run_pseudo_label(
    config_path: Path,
    weak_config_path: Path,
    *,
    arm: str,
    weights: Path,
    processed_dir: Path | None = None,
    device: str = "0",
) -> dict[str, Any]:
    """Propose boxes for an arm's frames, verify them against VLM counts, write both.

    Writes ``<arm>_pseudo_labels.parquet`` (accepted frames' boxes) and
    ``<arm>_accepted.parquet`` (the token list both weak arms train on).
    """
    from nuscenes_data_engine.config import get_settings, load_yaml

    cfg = load_yaml(config_path)
    weak_cfg = load_yaml(weak_config_path)
    settings = get_settings()
    state_dir = Path(cfg.get("state", {}).get("dir", "data/active_learning"))
    processed = processed_dir or Path("data/processed")
    pseudo_cfg = cfg.get("pseudo", {})
    conf = float(pseudo_cfg.get("conf", 0.5))
    tolerance = int(pseudo_cfg.get("tolerance", 1))
    imgsz = int(pseudo_cfg.get("imgsz", cfg.get("train", {}).get("imgsz", 640)))
    batch_size = int(pseudo_cfg.get("batch", 32))
    weak_state = Path(weak_cfg.get("state", {}).get("dir", "data/active_learning/autolabel_weak"))

    arm_path = state_dir / f"{arm}.parquet"
    if not arm_path.is_file():
        raise ValueError(f"No mined frames for arm {arm!r} at {arm_path}")
    arm_tokens = set(pd.read_parquet(arm_path)["sample_data_token"])

    samples = pd.read_parquet(processed / "samples.parquet")
    frames = samples[samples["sample_data_token"].isin(arm_tokens)]
    if len(frames) != len(arm_tokens):
        raise ValueError(
            f"{len(arm_tokens) - len(frames)} arm frames missing from samples.parquet"
        )

    boxes = propose_boxes(
        weights, frames, Path(settings.nuscenes_dataroot),
        conf=conf, imgsz=imgsz, device=device, batch_size=batch_size,
    )
    # Frames with no detection still count as candidates (empty label = background).
    det_counts = {token: {} for token in sorted(arm_tokens)}
    det_counts.update(detection_counts(boxes))

    labels: list[pd.DataFrame] = []
    for path in (
        Path(settings.data_dir) / "autolabel" / "labels.parquet",
        weak_state / "labels.parquet",
    ):
        if path.is_file():
            labels.append(pd.read_parquet(path))
    if not labels:
        raise ValueError("No VLM labels found — run `al pseudo-sample` then autolabel first.")
    all_labels = pd.concat(labels, ignore_index=True).drop_duplicates(
        subset="sample_data_token", keep="last"
    )
    vlm_labels = {
        str(row["sample_data_token"]): row
        for row in all_labels.to_dict("records")
        if str(row["sample_data_token"]) in arm_tokens
    }

    accepted, diagnostics = verify_frames(det_counts, vlm_labels, tolerance)
    if not accepted:
        raise ValueError(
            f"Verification kept nothing for arm {arm!r}: no frames survived "
            f"(candidates {diagnostics['n_candidates']}, no-label {diagnostics['n_no_label']}, "
            f"unparsed {diagnostics['n_unparsed']})"
        )

    accepted_set = set(accepted)
    if not accepted_set <= arm_tokens:
        raise ValueError("accepted frames must come from the arm's mined frames")
    # Defence in depth: the arm parquet is pool-only by construction (round-1 mining
    # guards), but a pseudo-labelled val frame would silently corrupt every arm's
    # comparison, so check it here too rather than trusting an upstream invariant.
    val_scenes = set(_official_val_scenes())
    val_frames = set(
        samples[samples["scene_name"].isin(val_scenes)]["sample_data_token"]
    )
    if accepted_set & val_frames:
        raise ValueError(
            f"{len(accepted_set & val_frames)} accepted frames are in the val split"
        )
    pseudo = boxes[boxes["sample_data_token"].isin(accepted_set)].reset_index(drop=True)
    pd.DataFrame({"sample_data_token": accepted}).to_parquet(
        state_dir / f"{arm}_accepted.parquet", index=False
    )
    pseudo.to_parquet(state_dir / f"{arm}_pseudo_labels.parquet", index=False)

    summary = {
        "arm": arm,
        "conf": conf,
        "tolerance": tolerance,
        "n_boxes": len(pseudo),
        "mean_boxes_per_accepted_frame": len(pseudo) / len(accepted),
        **diagnostics,
    }
    logger.info(
        "Arm %s: %d/%d frames accepted (retention %.2f), %d pseudo boxes; "
        "rejected by class: %s",
        arm, diagnostics["n_accepted"], diagnostics["n_candidates"],
        diagnostics["retention"], len(pseudo), diagnostics["rejected_by_class"],
    )
    return summary
```

- [ ] **Step 4: Add the CLI command**

In `cli.py`, after `al_pseudo_sample`:

```python
@al_app.command("pseudo-label")
def al_pseudo_label(
    arm: str = typer.Option(..., "--arm", help="Arm to pseudo-label (e.g. random)."),
    weights: Path = typer.Option(..., "--weights", help="Baseline best.pt."),
    config: Path = typer.Option(Path("configs/active_learning.yaml"), "--config", "-c"),
    weak_config: Path = typer.Option(Path("configs/autolabel_weak.yaml"), "--weak-config"),
    device: str = typer.Option("0", "--device"),
    wandb: bool | None = typer.Option(None, "--wandb/--no-wandb", help="W&B run logging."),
) -> None:
    """Propose boxes with the baseline detector and keep the VLM-verified frames."""
    from nuscenes_data_engine.active_learning.pseudo_label import run_pseudo_label
    from nuscenes_data_engine.tracking import wandb_run

    with wandb_run(
        "al-pseudo-label", name=f"al-pseudo-label-{arm}", config={"arm": arm}, enabled=wandb
    ) as run:
        summary = run_pseudo_label(
            config, weak_config, arm=arm, weights=weights, device=device
        )
        if run is not None:
            # bool is an int subclass — exclude it so flag fields aren't logged as 0/1
            run.log(
                {
                    k: v
                    for k, v in summary.items()
                    if isinstance(v, int | float) and not isinstance(v, bool)
                }
            )
    logger.info("Pseudo-label summary: %s", summary)
```

- [ ] **Step 5: Add the config block**

In `configs/active_learning.yaml`, after the `graph_mining:` block:

```yaml
pseudo:                    # weak supervision (docs/superpowers/specs/2026-08-06-vlm-weak-supervision-design.md)
  conf: 0.5                # detector confidence floor for proposed boxes
  tolerance: 1             # |detector - VLM| count tolerance, per mapped class
  imgsz: 640               # proposal inference size (matches the AL arms' training size)
  batch: 32
```

- [ ] **Step 6: Verify**

Run: `uv run pytest tests/test_weak_supervision.py -v` (14 PASS), `uv run nuscenes-data-engine al pseudo-label --help`, and
`uv run python -c "from nuscenes_data_engine.config import load_yaml; from pathlib import Path; print(load_yaml(Path('configs/active_learning.yaml'))['pseudo'])"`.

- [ ] **Step 7: Commit**

```bash
git add src/nuscenes_data_engine/active_learning/pseudo_label.py src/nuscenes_data_engine/cli.py configs/active_learning.yaml tests/test_weak_supervision.py
git commit -m "weak-sup: run_pseudo_label orchestration + al pseudo-label CLI + config"
```

---

### Task 5: The dataset seam (`pseudo_labels` override + cache key)

**Files:**
- Modify: `src/nuscenes_data_engine/training/dataset.py` (`_build_key` at :48-67; `build_yolo_dataset` signature at :88-97 and the label block at :156-158)
- Test: `tests/test_weak_supervision.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_weak_supervision.py`:

```python
def test_build_key_changes_with_pseudo_labels() -> None:
    pytest.importorskip("nuscenes")
    import nuscenes_data_engine.training.dataset as dataset_mod
    from nuscenes_data_engine.training.dataset import _build_key

    original = dataset_mod.compute_data_version
    dataset_mod.compute_data_version = lambda _: "v0"  # type: ignore[assignment]
    try:
        kwargs: dict[str, Any] = {"cameras": ["CAM_FRONT"], "limit_scenes": None}
        plain = _build_key(Path("x"), **kwargs)
        pseudo_a = _build_key(Path("x"), **kwargs, pseudo_labels=pd.DataFrame(
            {"sample_data_token": ["f1"], "category_group": ["car"],
             "x_min": [0.0], "y_min": [0.0], "x_max": [1.0], "y_max": [1.0]}
        ))
        pseudo_b = _build_key(Path("x"), **kwargs, pseudo_labels=pd.DataFrame(
            {"sample_data_token": ["f1"], "category_group": ["truck"],
             "x_min": [0.0], "y_min": [0.0], "x_max": [1.0], "y_max": [1.0]}
        ))
    finally:
        dataset_mod.compute_data_version = original  # type: ignore[assignment]

    assert "pseudo_labels" not in plain  # pre-existing manifests keep matching
    assert pseudo_a["pseudo_labels"] != pseudo_b["pseudo_labels"]  # content-sensitive


def test_apply_pseudo_labels_replaces_gt_for_those_tokens_only() -> None:
    from nuscenes_data_engine.training.dataset import _apply_pseudo_labels

    gt = pd.DataFrame(
        {
            "sample_data_token": ["f1", "f1", "f2"],
            "category_group": ["car", "pedestrian", "bus"],
            "x_min": [0.0, 1.0, 2.0], "y_min": [0.0, 1.0, 2.0],
            "x_max": [10.0, 11.0, 12.0], "y_max": [10.0, 11.0, 12.0],
        }
    )
    pseudo = pd.DataFrame(
        {
            "sample_data_token": ["f1"], "category_group": ["truck"],
            "x_min": [5.0], "y_min": [5.0], "x_max": [15.0], "y_max": [15.0],
            "score": [0.9],
        }
    )
    merged = _apply_pseudo_labels(gt, pseudo, {"f1"})
    f1 = merged[merged["sample_data_token"] == "f1"]
    assert list(f1["category_group"]) == ["truck"]  # GT rows for f1 replaced
    f2 = merged[merged["sample_data_token"] == "f2"]
    assert list(f2["category_group"]) == ["bus"]  # untouched
    assert set(gt.columns) <= set(merged.columns)  # schema preserved for the builder


def test_apply_pseudo_labels_accepted_but_empty_frame_loses_its_gt() -> None:
    """An accepted frame the detector found nothing in must train as a background.

    Its GT must NOT survive: the arm's whole claim is that these frames carry no
    ground truth. Deriving the replacement key from the pseudo table (which has no
    rows for such a frame) would silently leave the GT in place.
    """
    from nuscenes_data_engine.training.dataset import _apply_pseudo_labels

    gt = pd.DataFrame(
        {
            "sample_data_token": ["empty", "other"],
            "category_group": ["pedestrian", "car"],
            "x_min": [0.0, 2.0], "y_min": [0.0, 2.0],
            "x_max": [10.0, 12.0], "y_max": [10.0, 12.0],
        }
    )
    pseudo = pd.DataFrame(
        {
            "sample_data_token": pd.Series([], dtype=str),
            "category_group": pd.Series([], dtype=str),
            "x_min": pd.Series([], dtype=float), "y_min": pd.Series([], dtype=float),
            "x_max": pd.Series([], dtype=float), "y_max": pd.Series([], dtype=float),
        }
    )
    merged = _apply_pseudo_labels(gt, pseudo, {"empty"})
    assert "empty" not in set(merged["sample_data_token"])  # trains as a background
    assert list(merged[merged["sample_data_token"] == "other"]["category_group"]) == ["car"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_weak_supervision.py -k "build_key_changes or apply_pseudo" -v`
Expected: FAIL — `_build_key() got an unexpected keyword argument 'pseudo_labels'` / `ImportError` for `_apply_pseudo_labels`.

- [ ] **Step 3: Implement**

In `src/nuscenes_data_engine/training/dataset.py`:

(a) extend `_build_key`:

```python
def _build_key(
    processed_dir: Path,
    cameras: list[str] | None,
    limit_scenes: int | None,
    train_frames: set[str] | None = None,
    pseudo_labels: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Identity of a YOLO build — everything that determines its contents."""
    key = {
        "builder_version": BUILDER_VERSION,
        "data_version": compute_data_version(processed_dir),
        "cameras": sorted(cameras) if cameras else None,
        "limit_scenes": limit_scenes,
    }
    if train_frames is not None:
        # Added conditionally so pre-existing manifests (built without the param)
        # keep matching and don't trigger a ~400k-file rebuild.
        key["train_frames"] = hashlib.sha256(
            "\n".join(sorted(train_frames)).encode()
        ).hexdigest()[:16]
    if pseudo_labels is not None:
        # Content hash: a pseudo-labelled arm must never reuse a GT-labelled build.
        columns = ["sample_data_token", "category_group", "x_min", "y_min", "x_max", "y_max"]
        payload = (
            pseudo_labels[columns]
            .sort_values(columns, ignore_index=True)
            .to_csv(index=False)
            .encode()
        )
        key["pseudo_labels"] = hashlib.sha256(payload).hexdigest()[:16]
    return key
```

(b) add the merge helper above `build_yolo_dataset`:

```python
def _apply_pseudo_labels(
    annotations: pd.DataFrame,
    pseudo_labels: pd.DataFrame,
    pseudo_tokens: set[str],
) -> pd.DataFrame:
    """Replace GT rows with pseudo rows for every token in ``pseudo_tokens``.

    Whole-frame replacement (not a merge): a frame is labelled either by ground truth
    or by the detector, never half of each. The token set is passed explicitly rather
    than derived from ``pseudo_labels`` because an accepted frame with **zero** detected
    boxes contributes no rows — deriving the key from the table would leave that frame's
    ground truth in place and silently break the arm's "no GT" claim.
    """
    kept = annotations[~annotations["sample_data_token"].isin(pseudo_tokens)]
    columns = [c for c in annotations.columns if c in pseudo_labels.columns]
    return pd.concat([kept, pseudo_labels[columns]], ignore_index=True)
```

(c) in `build_yolo_dataset`, add the parameter and wire both call sites:

- signature gains, after `train_frames`:
  ```python
    pseudo_labels: pd.DataFrame | None = None,
    pseudo_tokens: set[str] | None = None,
  ```
- docstring gains: `pseudo_labels: Replace ground-truth boxes with these for every token in pseudo_tokens (weak supervision). The cache key includes their content hash. pseudo_tokens: The frames whose labels come from pseudo_labels — passed explicitly because an accepted frame with zero detected boxes has no rows in pseudo_labels and must still lose its ground truth.`
- the `key = _build_key(...)` call gains `pseudo_labels=pseudo_labels`;
- immediately before the existing line `ann = annotations[annotations["category_group"].notna()].copy()` insert:

```python
    if pseudo_labels is not None:
        if pseudo_tokens is None:
            raise ValueError("pseudo_labels requires pseudo_tokens (empty frames have no rows)")
        annotations = _apply_pseudo_labels(annotations, pseudo_labels, pseudo_tokens)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_weak_supervision.py tests/test_training.py -q`
Expected: all pass (the new tests plus the existing training tests, which must be unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/nuscenes_data_engine/training/dataset.py tests/test_weak_supervision.py
git commit -m "weak-sup: pseudo-label override in the YOLO dataset build (cache-key hashed)"
```

---

### Task 6: Register the two arms + report

**Files:**
- Modify: `src/nuscenes_data_engine/active_learning/experiment.py` (`ARMS` at :23-27; `resolve_arm_frames` `extra_file` dict; `run_arm`'s `build_yolo_dataset` call)
- Test: `tests/test_weak_supervision.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_weak_supervision.py`:

```python
def test_weak_arms_registered_and_share_accepted_frames(tmp_path: Path) -> None:
    from nuscenes_data_engine.active_learning.experiment import ARMS, resolve_arm_frames

    assert {"weak_random", "weak_random_gt"} <= set(ARMS)

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
    pd.DataFrame({"sample_data_token": ["p-1"]}).to_parquet(
        state / "random_accepted.parquet", index=False
    )

    cfg = {"split": {"channel": "CAM_FRONT"}}
    weak = resolve_arm_frames(state, processed, cfg, "weak_random")
    weak_gt = resolve_arm_frames(state, processed, cfg, "weak_random_gt")
    assert weak == weak_gt == {"bl-1", "bl-2", "p-1"}  # identical frame sets
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_weak_supervision.py -k weak_arms -v`
Expected: FAIL — ARMS assertion or `KeyError: 'weak_random'`.

- [ ] **Step 3: Implement**

In `experiment.py`:

(a) `ARMS` gains the two arms:

```python
ARMS = (
    "baseline", "mined", "random", "graph",
    "rate", "strat", "rate_strat",
    "graph_rate", "graph_rate_night",
    "weak_random", "weak_random_gt",
)
```

(b) `extra_file` dict gains — both arms read the SAME accepted set, which is what makes the comparison valid:

```python
        "weak_random": "random_accepted.parquet",
        "weak_random_gt": "random_accepted.parquet",
```

(c) in `run_arm`, load the pseudo table for `weak_random` only and pass it through. Immediately before the `build_yolo_dataset(` call insert:

```python
    pseudo_labels = None
    pseudo_tokens = None
    if arm == "weak_random":
        pseudo_path = state_dir / "random_pseudo_labels.parquet"
        if not pseudo_path.is_file():
            raise ValueError(
                f"Arm {arm!r} needs {pseudo_path} — run `al pseudo-label --arm random` first"
            )
        pseudo_labels = pd.read_parquet(pseudo_path)
        # Every accepted frame is pseudo-labelled, including those the detector found
        # nothing in — those have no rows in the table, so the token set comes from
        # accepted.parquet or their ground truth would survive into a "no GT" arm.
        pseudo_tokens = set(
            pd.read_parquet(state_dir / "random_accepted.parquet")["sample_data_token"]
        )
```

and add `pseudo_labels=pseudo_labels, pseudo_tokens=pseudo_tokens,` to the `build_yolo_dataset(...)` call's keyword arguments.

(d) update the module docstring's arm list to mention the weak-supervision pair and the spec path.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_weak_supervision.py tests/test_active_learning.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/nuscenes_data_engine/active_learning/experiment.py tests/test_weak_supervision.py
git commit -m "weak-sup: register weak_random / weak_random_gt arms (shared accepted set)"
```

---

### Task 7: CI-parity sweep

- [ ] **Step 1:** `uv run pytest -q` — report exact counts (expect ~240 passed, 2 skipped, 2 deselected). `uv run ruff check .` clean. Bare `uv run mypy` clean. Do NOT bulk-`ruff format` (CI doesn't gate it; pre-existing repo-wide drift).
- [ ] **Step 2:** Commit only if fixes were needed: `git commit -am "weak-sup: lint/type fixes"`.

---

### Task 8: TRINITY run (operational; no commits)

Needs images, the detector, and the vLLM server — all on TRINITY. Push first: `git push -u origin vlm-weak-supervision`.

- [ ] **Step 1: Locate the baseline weights.** `ssh trinity-2-18 ls /home/mgaur/sahil/nuscenes_project/runs/ | grep al-baseline` — the arms trained from `runs/yolov8n_imgsz640_e20_al-baseline/weights/best.pt`. Record the exact path.
- [ ] **Step 2: Write the weak sample.** `scripts/gpu-run.sh al pseudo-sample --arm random` — expect ~1,271 of 1,500 frames needing labels. Record the number.
- [ ] **Step 3: Start the vLLM server** the same way Phase 6b did (see docs/AUTOLABEL_EVAL.md's runbook for the exact command and the model id), then label:
  ```bash
  scripts/gpu-run.sh autolabel submit -c configs/autolabel_weak.yaml --provider local --yes
  scripts/gpu-run.sh autolabel collect -c configs/autolabel_weak.yaml --provider local
  ```
  Expect a parse rate near 6b's 99.7%. Record rows written and parse failures. If the vLLM startup differs from the documented 6b command, STOP and report rather than improvising.
- [ ] **Step 4: Propose + verify.** `scripts/gpu-run.sh al pseudo-label --arm random --weights <path from Step 1>` — record retention, n_accepted, rejected_by_class, mean boxes/frame. **If retention < 0.5, do not stop — that is a finding**; record it and continue (the docs will report it).
- [ ] **Step 5: Train both arms** (wrapped in `sh -c` — a bare `&&` chain leaves later commands outside `nohup` and unlogged):
  ```bash
  scripts/gpu-run.sh --bg raw "sh -c 'env CUDA_VISIBLE_DEVICES=<free-gpu> uv run nuscenes-data-engine al run --arm weak_random && env CUDA_VISIBLE_DEVICES=<free-gpu> uv run nuscenes-data-engine al run --arm weak_random_gt && uv run nuscenes-data-engine al report && echo CHAIN_COMPLETE'"
  ```
  Pick a free GPU with `ssh trinity-2-18 nvidia-smi`. Watch for `Arm weak_random: overall mAP50-95 ...` lines.
- [ ] **Step 6: Sync back.** `rsync -a trinity-2-18:/home/mgaur/sahil/nuscenes_project/data/active_learning/results.json data/active_learning/` and the same for `mlruns/`. Then `uv run nuscenes-data-engine al report` locally.
- [ ] **Step 7: Record** for the docs: both arms' overall/night mAP50-95 and deltas vs baseline, plus the three comparisons (weak vs weak_gt; weak_gt vs random; weak vs baseline).

### Task 9: Docs

**Files:** `docs/ACTIVE_LEARNING.md`, `docs/AUTOLABEL_EVAL.md`, `docs/PROJECT.md`

- [ ] **Step 1:** `docs/ACTIVE_LEARNING.md` — a "Weak supervision" section before the final "Runs:" paragraph: why the VLM can't label directly (counts, not boxes), the propose→verify→train pipeline, the ±1 rule and why frame-level rather than per-class, the measured retention and rejection breakdown, the three-way results table, and the verdict on whether the engine works without GT. Use the Task 8 numbers verbatim.
- [ ] **Step 2:** `docs/AUTOLABEL_EVAL.md` — a short "Reuse for weak supervision" note: the same labeller, a separate state dir (`configs/autolabel_weak.yaml`), and what the count quality means for verification (presence is reliable; exact counts under crowding are not — which is why the tolerance exists).
- [ ] **Step 3:** `docs/PROJECT.md` — §9 item 4 ("Close the 6b→6d loop") marked DONE with the headline numbers; §5 results section gains the weak-supervision line.
- [ ] **Step 4:** Commit: `git add docs && git commit -m "weak-sup: docs (method, retention, three-way results)"`

---

### After the plan (not plan tasks)

1. Final whole-branch review, then superpowers:finishing-a-development-branch (push, PR, CI).
2. If `weak_random` holds up, the natural follow-up is weak-supervising `graph_rate_night` (the night champion).
