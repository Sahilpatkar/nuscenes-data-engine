"""Tests for Phase 6d active learning (offline, CI-safe; no training, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from nuscenes_data_engine.active_learning.graph_mining import select_representatives
from nuscenes_data_engine.active_learning.matching import FrameFailure, iou_matrix, match_frame
from nuscenes_data_engine.active_learning.report import render_report
from nuscenes_data_engine.active_learning.sweep import summarize_failures
from nuscenes_data_engine.training.train import _run_name

# ---------------------------------------------------------------------------
# graph_mining.py — Phase 6e graph-diversity selection (pure; no Neo4j)
# ---------------------------------------------------------------------------


def test_select_representatives_spreads_budget_and_ranks_by_degree() -> None:
    communities = {"a": 0, "b": 0, "c": 0, "d": 1, "e": 1, "f": 2}
    degrees = {"a": 3.0, "b": 1.0, "c": 2.0, "d": 5.0, "e": 1.0, "f": 9.0}
    # budget 4, floor 1 -> allocate(sizes {0:3,1:2,2:1}) = {0:2, 1:1, 2:1};
    # within each community the top-degree frames are taken (ties by token).
    selected = select_representatives(communities, degrees, n_mine=4, floor=1)
    assert selected == ["a", "c", "d", "f"]


def test_select_representatives_is_deterministic_and_bounded() -> None:
    communities = {f"t{i}": i % 3 for i in range(30)}
    degrees = {f"t{i}": float(i) for i in range(30)}
    first = select_representatives(communities, degrees, n_mine=9, floor=1)
    assert first == select_representatives(communities, degrees, n_mine=9, floor=1)
    assert len(first) == 9 and len(set(first)) == 9  # no duplicates


# ---------------------------------------------------------------------------
# matching.py — pure numpy, runs in torch-free CI
# ---------------------------------------------------------------------------


def test_iou_matrix_exact_values() -> None:
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    ious = iou_matrix(a, b)
    assert ious.shape == (1, 3)
    assert ious[0] == pytest.approx([1.0, 50.0 / 150.0, 0.0])


def test_iou_matrix_empty() -> None:
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)
    assert iou_matrix(np.zeros((2, 4)), np.zeros((0, 4))).shape == (2, 0)


def test_match_frame_perfect_match() -> None:
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]])
    classes = np.array([0, 1])
    failure = match_frame(boxes, classes, np.array([0.9, 0.8]), boxes, classes)
    assert (failure.n_gt, failure.n_matched, failure.n_fn, failure.n_low_conf) == (2, 2, 0, 0)
    assert failure.failure_score == 0.0


def test_match_frame_false_negative_and_low_conf() -> None:
    gt_boxes = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]])
    gt_classes = np.array([0, 0])
    # One GT matched but below conf_hit, the other missed entirely.
    failure = match_frame(
        np.array([[0.0, 0.0, 10.0, 10.0]]),
        np.array([0]),
        np.array([0.2]),
        gt_boxes,
        gt_classes,
        conf_hit=0.4,
    )
    assert (failure.n_matched, failure.n_fn, failure.n_low_conf) == (1, 1, 1)
    assert failure.failure_score == 1.5


def test_match_frame_class_mismatch_is_fn() -> None:
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    failure = match_frame(box, np.array([1]), np.array([0.9]), box, np.array([0]))
    assert failure.n_fn == 1 and failure.n_matched == 0


def test_match_frame_iou_threshold() -> None:
    gt = np.array([[0.0, 0.0, 10.0, 10.0]])
    barely_off = np.array([[6.0, 0.0, 16.0, 10.0]])  # IoU = 4/16 = 0.25 < 0.5
    failure = match_frame(barely_off, np.array([0]), np.array([0.9]), gt, np.array([0]))
    assert failure.n_fn == 1


def test_match_frame_greedy_prefers_high_conf() -> None:
    gt = np.array([[0.0, 0.0, 10.0, 10.0]])
    preds = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0]])
    # High-conf pred claims the GT; the duplicate doesn't double-match.
    failure = match_frame(preds, np.array([0, 0]), np.array([0.3, 0.9]), gt, np.array([0]))
    assert (failure.n_matched, failure.n_fn, failure.n_low_conf) == (1, 0, 0)


def test_match_frame_empty_edges() -> None:
    no_boxes, no_classes = np.zeros((0, 4)), np.zeros(0, int)
    gt = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert match_frame(no_boxes, no_classes, np.zeros(0), gt, np.array([0])).n_fn == 1
    empty = match_frame(no_boxes, no_classes, np.zeros(0), np.zeros((0, 4)), no_classes)
    assert empty == FrameFailure(n_gt=0, n_matched=0, n_fn=0, n_low_conf=0)


def test_summarize_failures_day_night() -> None:
    failures = pd.DataFrame(
        {
            "is_night": [False, False, True],
            "failure_score": [1.0, 2.0, 6.0],
            "n_fn": [1, 2, 5],
            "n_low_conf": [0, 0, 2],
        }
    )
    stats = summarize_failures(failures)
    assert stats["mean_failure_score_day"] == pytest.approx(1.5)
    assert stats["mean_failure_score_night"] == pytest.approx(6.0)
    assert stats["total_fn"] == 8.0


# ---------------------------------------------------------------------------
# train.py run naming
# ---------------------------------------------------------------------------


def test_run_name_suffix() -> None:
    assert _run_name("yolov8n.pt", 640, 20) == "yolov8n_imgsz640_e20"
    assert _run_name("yolov8n.pt", 640, 20, "al-mined") == "yolov8n_imgsz640_e20_al-mined"


# ---------------------------------------------------------------------------
# split.py — needs the nuScenes devkit for the official scene split
# ---------------------------------------------------------------------------


def _write_samples(
    processed_dir: Path, scenes: dict[str, bool], frames_per_scene: int = 3
) -> None:
    """samples.parquet with the given {scene_name: is_night} for CAM_FRONT (+ noise channel)."""
    rows = []
    for scene, is_night in scenes.items():
        for i in range(frames_per_scene):
            for channel in ("CAM_FRONT", "CAM_BACK"):
                rows.append(
                    {
                        "sample_data_token": f"{scene}-{channel}-{i}",
                        "scene_name": scene,
                        "channel": channel,
                        "filename": f"samples/{channel}/{scene}-{i}.jpg",
                        "is_night": is_night,
                    }
                )
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(processed_dir / "samples.parquet")


@pytest.fixture(scope="module")
def train_scene_names() -> list[str]:
    splits = pytest.importorskip("nuscenes.utils.splits")
    return sorted(splits.create_splits_scenes()["train"])[:8]


def test_build_split_stratified_and_deterministic(
    tmp_path: Path, train_scene_names: list[str]
) -> None:
    from nuscenes_data_engine.active_learning.split import build_split

    # 8 official-train scenes, 2 of them night.
    scenes = {name: i < 2 for i, name in enumerate(train_scene_names)}
    _write_samples(tmp_path, scenes)

    cfg = {"baseline_frac": 0.25, "seed": 64, "channel": "CAM_FRONT"}
    split = build_split(tmp_path, cfg)

    assert set(split["scene_name"]) == set(train_scene_names)
    assert set(split["role"]) <= {"baseline", "pool"}
    # Night stratification: max(1, round(2 * .25)) = 1 night scene in the baseline.
    baseline = split[split["role"] == "baseline"]
    assert int(baseline["is_night"].sum()) == 1
    assert len(baseline) == 1 + max(1, round(6 * 0.25))
    # CAM_FRONT frames only.
    assert split["n_frames"].sum() == len(scenes) * 3

    assert build_split(tmp_path, cfg).equals(split)  # deterministic


def test_split_excludes_val_scenes(tmp_path: Path, train_scene_names: list[str]) -> None:
    from nuscenes.utils.splits import create_splits_scenes

    from nuscenes_data_engine.active_learning.split import build_split

    val_scene = sorted(create_splits_scenes()["val"])[0]
    _write_samples(tmp_path, {train_scene_names[0]: False, val_scene: False})
    split = build_split(tmp_path, {"baseline_frac": 0.5, "seed": 64})
    assert val_scene not in set(split["scene_name"])


def test_frames_for_scenes_channel_filter(tmp_path: Path, train_scene_names: list[str]) -> None:
    from nuscenes_data_engine.active_learning.split import frames_for_scenes

    scene = train_scene_names[0]
    _write_samples(tmp_path, {scene: False, train_scene_names[1]: False})
    frames = frames_for_scenes(tmp_path, {scene}, "CAM_FRONT")
    assert frames == {f"{scene}-CAM_FRONT-{i}" for i in range(3)}


def test_run_split_writes_parquet(tmp_path: Path, train_scene_names: list[str]) -> None:
    from nuscenes_data_engine.active_learning.split import run_split

    _write_samples(tmp_path / "processed", {n: i < 2 for i, n in enumerate(train_scene_names)})
    config = tmp_path / "al.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "split": {"baseline_frac": 0.25, "seed": 64, "channel": "CAM_FRONT"},
                "state": {"dir": str(tmp_path / "state")},
            }
        )
    )
    summary = run_split(config, processed_dir=tmp_path / "processed")
    assert (tmp_path / "state" / "split.parquet").is_file()
    assert set(summary["scenes"]) == {"baseline", "pool"}


# ---------------------------------------------------------------------------
# dataset.py train_frames restriction — needs the devkit
# ---------------------------------------------------------------------------


def test_build_key_train_frames_back_compat() -> None:
    pytest.importorskip("nuscenes")
    from nuscenes_data_engine.training.dataset import _build_key

    kwargs: dict[str, Any] = {"cameras": ["CAM_FRONT"], "limit_scenes": None}

    # Patch out the data-version hash: key structure is what's under test.
    import nuscenes_data_engine.training.dataset as dataset_mod

    original = dataset_mod.compute_data_version
    dataset_mod.compute_data_version = lambda _: "v0"  # type: ignore[assignment]
    try:
        plain = _build_key(Path("x"), **kwargs)
        with_a = _build_key(Path("x"), **kwargs, train_frames={"t1", "t2"})
        with_b = _build_key(Path("x"), **kwargs, train_frames={"t1", "t3"})
        reordered = _build_key(Path("x"), **kwargs, train_frames={"t2", "t1"})
    finally:
        dataset_mod.compute_data_version = original  # type: ignore[assignment]

    assert "train_frames" not in plain  # pre-existing manifests keep matching
    assert with_a["train_frames"] != with_b["train_frames"]
    assert with_a == reordered  # order-free


# ---------------------------------------------------------------------------
# mining.py — tiny real LanceDB with 8-d vectors; needs lancedb + sklearn
# ---------------------------------------------------------------------------

DIM = 8


def _unit(direction: int, jitter: float, rng: np.random.Generator) -> list[float]:
    vec = rng.normal(scale=jitter, size=DIM)
    vec[direction] += 1.0
    return (vec / np.linalg.norm(vec)).astype(np.float32).tolist()


@pytest.fixture()
def mining_setup(tmp_path: Path) -> Path:
    """State + store for run_mining: 2 failure modes, pool frames near each mode."""
    lancedb = pytest.importorskip("lancedb")
    pytest.importorskip("sklearn")
    del lancedb

    from nuscenes_data_engine.data_engine import store

    rng = np.random.default_rng(7)
    processed = tmp_path / "processed"
    state = tmp_path / "state"
    state.mkdir()

    # Scenes: baseline (bl-*), pool (pool-*), val (val-*) — 4 CAM_FRONT frames each.
    scenes = {
        "bl-0": False,
        "bl-1": False,
        "pool-0": False,
        "pool-1": True,
        "pool-2": False,
        "pool-3": True,
        "val-0": False,
        "val-1": True,
    }
    _write_samples(processed, scenes, frames_per_scene=4)
    pd.DataFrame(
        {
            "scene_name": list(scenes),
            "role": ["baseline"] * 2 + ["pool"] * 4 + ["val"] * 2,
            "is_night": list(scenes.values()),
            "n_frames": 4,
        }
    ).query("role != 'val'").to_parquet(state / "split.parquet", index=False)

    # Failures: val frames, mode 0 (day, axis 0) and mode 1 (night, axis 1).
    failure_rows, vec_rows = [], []
    for i in range(4):
        for scene, axis in (("val-0", 0), ("val-1", 1)):
            token = f"{scene}-CAM_FRONT-{i}"
            failure_rows.append(
                {
                    "sample_data_token": token,
                    "scene_name": scene,
                    "is_night": scene == "val-1",
                    "n_gt": 5,
                    "n_matched": 2,
                    "n_fn": 3 if scene == "val-1" else 1,
                    "n_low_conf": 2,
                    "failure_score": 4.0 if scene == "val-1" else 2.0,
                }
            )
            vec_rows.append((token, scene, axis))
    pd.DataFrame(failure_rows).to_parquet(state / "failures.parquet", index=False)

    # Pool + baseline frames in the store: pool-0/2 near axis 0, pool-1/3 near axis 1.
    for scene, is_night in scenes.items():
        if scene.startswith("val"):
            continue
        axis = 1 if is_night else 0
        for i in range(4):
            vec_rows.append((f"{scene}-CAM_FRONT-{i}", scene, axis))

    tbl = store.open_frames_table(tmp_path / "lancedb", "frames", DIM, create=True)
    store.add_frames(
        tbl,
        [
            {
                "vector": _unit(axis, 0.1, rng),
                "sample_data_token": token,
                "sample_token": token,
                "scene_token": scene,
                "scene_name": scene,
                "scene_description": "x",
                "channel": "CAM_FRONT",
                "filename": f"{token}.jpg",
                "timestamp": 0,
                "location": "x",
                "is_night": scene in ("pool-1", "pool-3", "val-1"),
                "is_rain": False,
                "n_boxes": 1,
                "thumbnail": b"",
            }
            for token, scene, axis in vec_rows
        ],
    )

    (tmp_path / "engine.yaml").write_text(
        yaml.safe_dump({"lancedb": {"path": str(tmp_path / "lancedb"), "table": "frames"}})
    )
    config = tmp_path / "al.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "split": {"channel": "CAM_FRONT"},
                "sweep": {"top_k_failures": 8},
                "mining": {"n_clusters": 2, "n_mine": 6, "seed": 64, "overfetch": 2},
                "state": {"dir": str(state)},
                "engine_config": str(tmp_path / "engine.yaml"),
            }
        )
    )
    return config


def test_run_mining_selects_pool_frames(mining_setup: Path, tmp_path: Path) -> None:
    from nuscenes_data_engine.active_learning.mining import run_mining

    summary = run_mining(mining_setup, processed_dir=tmp_path / "processed")
    state = tmp_path / "state"

    mined = pd.read_parquet(state / "mined.parquet")
    random = pd.read_parquet(state / "random.parquet")
    clusters = pd.read_parquet(state / "clusters.parquet")

    assert len(mined) == 6 == summary["n_mined"]  # exact n_mine after dedupe/backfill
    assert len(random) == 6 == summary["n_random"]
    assert mined["sample_data_token"].is_unique

    pool_tokens = {
        f"pool-{s}-CAM_FRONT-{i}" for s in range(4) for i in range(4)
    }
    assert set(mined["sample_data_token"]) <= pool_tokens  # never baseline or val
    assert set(random["sample_data_token"]) <= pool_tokens

    # Two clusters, sizes summing to the failure count, night/day separated.
    assert len(clusters) == 2 and clusters["size"].sum() == 8
    assert set(clusters["night_share"].round(2)) == {0.0, 1.0}
    assert (state / "cluster_summary.json").is_file()

    # Seeded → rerun reproduces both token sets.
    rerun = run_mining(mining_setup, processed_dir=tmp_path / "processed")
    assert rerun["n_mined"] == 6
    assert pd.read_parquet(state / "mined.parquet")["sample_data_token"].tolist() == mined[
        "sample_data_token"
    ].tolist()
    assert pd.read_parquet(state / "random.parquet")["sample_data_token"].tolist() == random[
        "sample_data_token"
    ].tolist()


# ---------------------------------------------------------------------------
# experiment.py — arm token resolution, config overlay, results merge
# ---------------------------------------------------------------------------


def test_resolve_arm_frames(tmp_path: Path) -> None:
    from nuscenes_data_engine.active_learning.experiment import resolve_arm_frames

    _write_samples(tmp_path / "processed", {"bl-0": False, "pool-0": False})
    state = tmp_path / "state"
    state.mkdir()
    pd.DataFrame(
        {"scene_name": ["bl-0", "pool-0"], "role": ["baseline", "pool"]}
    ).to_parquet(state / "split.parquet", index=False)
    mined_tokens = ["pool-0-CAM_FRONT-0", "pool-0-CAM_FRONT-2"]
    pd.DataFrame({"sample_data_token": mined_tokens}).to_parquet(
        state / "mined.parquet", index=False
    )

    cfg = {"split": {"channel": "CAM_FRONT"}}
    baseline = resolve_arm_frames(state, tmp_path / "processed", cfg, "baseline")
    assert baseline == {f"bl-0-CAM_FRONT-{i}" for i in range(3)}
    mined = resolve_arm_frames(state, tmp_path / "processed", cfg, "mined")
    assert mined == baseline | set(mined_tokens)


def test_overlay_train_config(tmp_path: Path) -> None:
    from nuscenes_data_engine.active_learning.experiment import overlay_train_config

    base = tmp_path / "train.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "model": {"weights": "yolov8m.pt", "imgsz": 960, "classes": ["car"]},
                "train": {"epochs": 50, "batch": 8, "lr0": 0.01},
            }
        )
    )
    cfg = overlay_train_config(base, {"model": "yolov8n.pt", "imgsz": 640, "epochs": 20})
    assert cfg["model"]["weights"] == "yolov8n.pt"
    assert cfg["model"]["imgsz"] == 640
    assert cfg["train"]["epochs"] == 20
    assert cfg["train"]["batch"] == 8  # untouched keys survive
    assert cfg["model"]["classes"] == ["car"]
    assert cfg["train"]["lr0"] == 0.01


def test_merge_results_asserts_val_identity(tmp_path: Path) -> None:
    from nuscenes_data_engine.active_learning.experiment import merge_results

    record = {"val_images": 6019, "overall": {"mAP50-95": 0.5}}
    merge_results(tmp_path, "baseline", record)
    merged = merge_results(tmp_path, "mined", {"val_images": 6019, "overall": {"mAP50-95": 0.6}})
    assert set(merged) == {"baseline", "mined"}
    assert json.loads((tmp_path / "results.json").read_text())["mined"]["overall"] == {
        "mAP50-95": 0.6
    }

    with pytest.raises(AssertionError, match="val split differs"):
        merge_results(tmp_path, "random", {"val_images": 42})

    # Re-running the same arm overwrites without tripping the assertion.
    merge_results(tmp_path, "baseline", {"val_images": 6019, "overall": {"mAP50-95": 0.55}})


# ---------------------------------------------------------------------------
# report.py
# ---------------------------------------------------------------------------


def _fake_results() -> dict[str, Any]:
    def record(overall: float, night: float, n: int) -> dict[str, Any]:
        return {
            "n_train_images": n,
            "overall": {"mAP50": overall + 0.2, "mAP50-95": overall},
            "night": {"mAP50": night + 0.2, "mAP50-95": night},
        }

    return {
        "baseline": record(0.40, 0.30, 7000),
        "mined": record(0.45, 0.38, 8500),
        "random": record(0.42, 0.32, 8500),
    }


def test_render_report_deltas() -> None:
    clusters = pd.DataFrame(
        {"cluster": [0, 1], "size": [600, 400], "night_share": [0.9, 0.1],
         "mean_failure_score": [4.2, 2.0]}
    )
    markdown = render_report(_fake_results(), clusters)
    assert "| baseline" in markdown and "| mined" in markdown and "| random" in markdown
    assert "0.05" in markdown  # mined overall delta
    assert "0.08" in markdown  # mined night delta
    assert "Failure clusters" in markdown
    # Baseline row carries no delta.
    baseline_row = next(line for line in markdown.splitlines() if "baseline" in line)
    assert baseline_row.count("nan") == 0


def test_run_report_writes_markdown(tmp_path: Path) -> None:
    from nuscenes_data_engine.active_learning.report import run_report

    state = tmp_path / "state"
    state.mkdir()
    (state / "results.json").write_text(json.dumps(_fake_results()))
    config = tmp_path / "al.yaml"
    config.write_text(yaml.safe_dump({"state": {"dir": str(state)}}))
    markdown = run_report(config)
    assert (state / "report.md").read_text() == markdown
    assert "Active-learning experiment report" in markdown
