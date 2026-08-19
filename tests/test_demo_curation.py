"""Tests for the demo curation pipeline (pure; no torch, no network)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from nuscenes_data_engine.active_learning.matching import match_frame, match_frame_boxes


def test_match_frame_boxes_assigns_statuses() -> None:
    gt_boxes = np.array([[0, 0, 100, 100], [200, 200, 300, 300]], dtype=float)
    gt_classes = np.array([0, 1])
    pred_boxes = np.array(
        [[5, 5, 105, 105], [400, 400, 500, 500], [210, 210, 290, 290]], dtype=float
    )
    pred_classes = np.array([0, 0, 1])
    pred_conf = np.array([0.9, 0.8, 0.2])
    result = match_frame_boxes(
        pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes, iou=0.5, conf_hit=0.4
    )
    assert list(result.pred_status) == ["tp", "fp", "low_conf"]
    assert result.pred_matched_gt[0] == 0          # first pred matched GT row 0
    assert result.pred_matched_gt[1] == -1         # unmatched
    # IoU(pred2, gt1) = 0.64 >= iou threshold, so the low-conf pred DOES claim GT
    # row 1 — match_frame counts low-conf hits toward n_matched/matched_gt too
    # (verified directly against match_frame on this exact fixture), so gt_matched
    # must be [True, True] here for the equivalence property to hold in general.
    assert list(result.gt_matched) == [True, True]  # low-conf pred still matches GT


def test_match_frame_boxes_one_gt_consumes_one_pred() -> None:
    gt_boxes = np.array([[0, 0, 100, 100]], dtype=float)
    gt_classes = np.array([0])
    pred_boxes = np.array([[0, 0, 100, 100], [1, 1, 99, 99]], dtype=float)
    pred_classes = np.array([0, 0])
    pred_conf = np.array([0.9, 0.85])
    result = match_frame_boxes(
        pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes, iou=0.5, conf_hit=0.4
    )
    assert list(result.pred_status).count("tp") == 1
    assert list(result.pred_status).count("fp") == 1


def test_match_frame_boxes_iou_boundary() -> None:
    # IoU exactly at the threshold counts as a match iff match_frame counts it.
    gt_boxes = np.array([[0, 0, 100, 100]], dtype=float)
    gt_classes = np.array([0])
    pred_boxes = np.array([[0, 0, 100, 50]], dtype=float)  # IoU = 0.5 exactly
    pred_classes = np.array([0])
    pred_conf = np.array([0.9])
    boxes = match_frame_boxes(
        pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes, iou=0.5, conf_hit=0.4
    )
    frame = match_frame(
        pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes, iou=0.5, conf_hit=0.4
    )
    assert (boxes.pred_status[0] == "tp") == (frame.n_matched == 1)


def test_match_frame_boxes_counts_agree_with_match_frame() -> None:
    """Property: box-level results aggregate to exactly match_frame's counts."""
    rng = np.random.default_rng(7)
    for _ in range(25):
        n_gt, n_pred = rng.integers(0, 8), rng.integers(0, 8)
        gt_boxes = rng.uniform(0, 800, (n_gt, 2))
        gt_boxes = np.hstack([gt_boxes, gt_boxes + rng.uniform(20, 200, (n_gt, 2))])
        gt_classes = rng.integers(0, 3, n_gt)
        pred_boxes = rng.uniform(0, 800, (n_pred, 2))
        pred_boxes = np.hstack([pred_boxes, pred_boxes + rng.uniform(20, 200, (n_pred, 2))])
        pred_classes = rng.integers(0, 3, n_pred)
        pred_conf = rng.uniform(0.05, 1.0, n_pred)
        frame = match_frame(
            pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes, iou=0.5, conf_hit=0.4
        )
        boxes = match_frame_boxes(
            pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes, iou=0.5, conf_hit=0.4
        )
        assert boxes.gt_matched.sum() == frame.n_matched
        assert (~boxes.gt_matched).sum() == frame.n_fn
        assert list(boxes.pred_status).count("low_conf") == frame.n_low_conf


def _curation_fixture(tmp_path: Path) -> dict[str, Path]:
    processed = tmp_path / "processed"
    processed.mkdir()
    al = tmp_path / "al"
    al.mkdir()
    pd.DataFrame({
        "sample_data_token": [f"v{i}" for i in range(8)],
        "scene_name": ["s"] * 8,
        "is_night": [True, True, True, False, False, False, False, False],
        "n_gt": [5, 4, 3, 6, 2, 3, 4, 3],
        "n_matched": [1, 2, 3, 2, 2, 3, 4, 3],
        "n_fn": [4, 2, 0, 4, 0, 0, 0, 0],
        "n_low_conf": [2, 1, 0, 3, 0, 0, 0, 0],
        "failure_score": [0.9, 0.7, 0.2, 0.8, 0.1, 0.05, 0.02, 0.01],
    }).to_parquet(al / "failures.parquet")
    pd.DataFrame({
        "sample_data_token": [f"v{i}" for i in range(8)] + ["t1", "t2", "t3"],
        "sample_token": [f"st{i}" for i in range(8)] + ["st8", "st9", "st10"],
        "channel": ["CAM_FRONT"] * 11,
        "filename": [f"samples/CAM_FRONT/{i}.jpg" for i in range(11)],
        # Deliberately DIFFERENT from failures.parquet's uniform "s" scene_name below,
        # so a test that accidentally sourced scene_name from failures.parquet instead
        # of here would fail loudly rather than passing by coincidence (carried review
        # item: scene_name must be sourced from samples.parquet for every row, the
        # same treatment is_night/is_rain already get).
        "scene_name": [f"scene-{i}" for i in range(8)] + ["scene-8", "scene-9", "scene-10"],
        # Matches failures.parquet's is_night for v0..v7; arbitrary-but-fixed for the
        # pool tokens t1..t3, which have no failures.parquet row of their own — I3
        # requires is_night/is_rain to be sourced from here for every manifest row.
        "is_night": [True, True, True, False, False, False, False, False, False, True, False],
        "is_rain": [False] * 8 + [False, True, False],
    }).to_parquet(processed / "samples.parquet")
    pd.DataFrame({
        "sample_token": ["st0", "st9", "st3"],
        # is_hard_braking is `bool | null` on real data (docs/DATA.md:166) — st3's
        # None (no CAN window) must not crash the curation mask (C1).
        "is_hard_braking": [True, True, None],
    }).to_parquet(processed / "canbus.parquet")
    pd.DataFrame({"sample_data_token": ["t1", "t2", "t3"]}).to_parquet(
        al / "graph_rate_night.parquet"
    )
    pd.DataFrame({"sample_data_token": ["t1"]}).to_parquet(
        al / "graph_rate_night_accepted.parquet"
    )
    return {"processed": processed, "al": al}


def test_curate_buckets_split_and_dedup(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.curate import run_curate

    paths = _curation_fixture(tmp_path)
    quotas = {"night_failure": 2, "day_failure": 1, "hard_braking": 2, "al_selected": 3,
              "weak_accepted": 1, "weak_rejected": 2, "clean_success": 1, "semantic": 2}
    manifest = run_curate(
        processed_dir=paths["processed"], al_dir=paths["al"],
        staging_dir=tmp_path / "staging", quotas=quotas,
        al_arm="graph_rate_night", weak_arm="graph_rate_night",
        semantic_hits=lambda queries: [("v5", ["semantic"]), ("t2", ["semantic"])],
        semantic_queries=["q"],
    )
    by_token = manifest.set_index("sample_data_token")
    assert by_token.loc["v0", "split"] == "val"          # night_failure top score
    assert by_token.loc["t1", "split"] == "train_pool"    # weak_accepted
    assert "weak_rejected" in by_token.loc["t2", "curation_buckets"]
    assert "semantic" in by_token.loc["t2", "curation_buckets"]  # dedup: multi-bucket
    assert len(manifest) == len(set(manifest["sample_data_token"]))
    # hard_braking maps sample_token -> CAM_FRONT sample_data_token
    assert "hard_braking" in by_token.loc["v0", "curation_buckets"]
    # filelist written, relative, unique
    filelist = (tmp_path / "staging" / "rsync_filelist.txt").read_text().splitlines()
    assert len(filelist) == len(manifest)
    assert all(not line.startswith("/") for line in filelist)


def test_curate_is_deterministic(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.curate import run_curate

    paths = _curation_fixture(tmp_path)
    kwargs = dict(
        processed_dir=paths["processed"], al_dir=paths["al"],
        quotas={"night_failure": 2, "day_failure": 1, "hard_braking": 1, "al_selected": 2,
                "weak_accepted": 1, "weak_rejected": 1, "clean_success": 1, "semantic": 0},
        al_arm="graph_rate_night", weak_arm="graph_rate_night",
        semantic_hits=lambda queries: [], semantic_queries=[],
    )
    first = run_curate(staging_dir=tmp_path / "s1", **kwargs)
    second = run_curate(staging_dir=tmp_path / "s2", **kwargs)
    pd.testing.assert_frame_equal(first, second)


def test_curate_clean_success_requires_no_failures(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.curate import run_curate

    paths = _curation_fixture(tmp_path)
    manifest = run_curate(
        processed_dir=paths["processed"], al_dir=paths["al"],
        staging_dir=tmp_path / "staging",
        quotas={"night_failure": 0, "day_failure": 0, "hard_braking": 0, "al_selected": 0,
                "weak_accepted": 0, "weak_rejected": 0, "clean_success": 2, "semantic": 0},
        al_arm="graph_rate_night", weak_arm="graph_rate_night",
        semantic_hits=lambda queries: [], semantic_queries=[],
    )
    stats = manifest.set_index("sample_data_token")
    assert all(stats["n_fn"] == 0) and all(stats["n_gt"] >= 3)


def test_curate_null_is_hard_braking_does_not_crash(tmp_path: Path) -> None:
    """C1: is_hard_braking is bool|null on real data (docs/DATA.md:166); a plain
    boolean mask over a None-containing object column raises. st3's None row in the
    fixture (added alongside st0/st9) exercises this on every hard_braking-quota run.
    """
    from nuscenes_data_engine.demo.curate import run_curate

    paths = _curation_fixture(tmp_path)
    quotas = {"night_failure": 0, "day_failure": 0, "hard_braking": 5, "al_selected": 0,
              "weak_accepted": 0, "weak_rejected": 0, "clean_success": 0, "semantic": 0}
    manifest = run_curate(
        processed_dir=paths["processed"], al_dir=paths["al"],
        staging_dir=tmp_path / "staging", quotas=quotas,
        al_arm="graph_rate_night", weak_arm="graph_rate_night",
        semantic_hits=lambda queries: [], semantic_queries=[],
    )
    by_token = manifest.set_index("sample_data_token")
    # st0/st9 -> v0/t2 are real hard-braking hits; st3's null must not appear as one.
    assert "hard_braking" in by_token.loc["v0", "curation_buckets"]
    assert "hard_braking" in by_token.loc["t2", "curation_buckets"]
    assert "v3" not in by_token.index or "hard_braking" not in by_token.loc["v3", "curation_buckets"]


def test_curate_rejects_non_cam_front_token(tmp_path: Path) -> None:
    """I2: every curated token must be CAM_FRONT -- the semantic bucket is the one
    door not already CAM_FRONT-filtered (LanceDB spans all six channels)."""
    from nuscenes_data_engine.demo.curate import run_curate

    paths = _curation_fixture(tmp_path)
    samples_path = paths["processed"] / "samples.parquet"
    samples = pd.read_parquet(samples_path)
    samples = pd.concat(
        [
            samples,
            pd.DataFrame({
                "sample_data_token": ["b1"],
                "sample_token": ["stb1"],
                "channel": ["CAM_BACK"],
                "filename": ["samples/CAM_BACK/b1.jpg"],
                "is_night": [False],
                "is_rain": [False],
            }),
        ],
        ignore_index=True,
    )
    samples.to_parquet(samples_path)

    quotas = {"night_failure": 0, "day_failure": 0, "hard_braking": 0, "al_selected": 0,
              "weak_accepted": 0, "weak_rejected": 0, "clean_success": 0, "semantic": 1}
    with pytest.raises(ValueError, match="b1"):
        run_curate(
            processed_dir=paths["processed"], al_dir=paths["al"],
            staging_dir=tmp_path / "staging", quotas=quotas,
            al_arm="graph_rate_night", weak_arm="graph_rate_night",
            semantic_hits=lambda queries: [("b1", ["semantic"])],
            semantic_queries=["q"],
        )


def test_curate_manifest_dtypes_are_bool_and_nullable(tmp_path: Path) -> None:
    """I3: is_night/is_rain come from samples.parquet for every row (bool, no NA);
    failure-ledger stat columns use pandas nullable dtypes so train_pool rows carry
    pd.NA instead of poisoning the column to object/float64."""
    from nuscenes_data_engine.demo.curate import run_curate

    paths = _curation_fixture(tmp_path)
    quotas = {"night_failure": 2, "day_failure": 1, "hard_braking": 0, "al_selected": 0,
              "weak_accepted": 1, "weak_rejected": 1, "clean_success": 0, "semantic": 0}
    manifest = run_curate(
        processed_dir=paths["processed"], al_dir=paths["al"],
        staging_dir=tmp_path / "staging", quotas=quotas,
        al_arm="graph_rate_night", weak_arm="graph_rate_night",
        semantic_hits=lambda queries: [], semantic_queries=[],
    )
    assert (manifest["split"] == "val").any()
    assert (manifest["split"] == "train_pool").any()
    assert manifest["is_night"].dtype == np.dtype(bool)
    assert manifest["is_night"].isna().sum() == 0
    # scene_name (carried review item): sourced from samples.parquet for every row,
    # val and train_pool alike -- no NA even for tokens absent from failures.parquet.
    assert manifest["scene_name"].isna().sum() == 0
    # boolean filtering must not crash even with train_pool rows present
    filtered = manifest[manifest["is_night"]]
    assert len(filtered) >= 1
    assert str(manifest["n_gt"].dtype) == "Int64"
    assert str(manifest["failure_score"].dtype) == "Float64"
    pool_row = manifest.set_index("sample_data_token").loc["t1"]
    assert pool_row["n_gt"] is pd.NA


def test_curate_al_selected_excludes_weak_claimed_tokens(tmp_path: Path) -> None:
    """I4: al_selected must draw from tokens the weak buckets didn't already claim,
    not re-slice the same token-ascending prefix of the same arm pool."""
    from nuscenes_data_engine.demo.curate import run_curate

    processed = tmp_path / "processed"
    processed.mkdir()
    al = tmp_path / "al"
    al.mkdir()
    tokens = [f"t{i}" for i in range(1, 6)]
    pd.DataFrame({
        "sample_data_token": pd.Series([], dtype="object"),
        "scene_name": pd.Series([], dtype="object"),
        "is_night": pd.Series([], dtype="bool"),
        "n_gt": pd.Series([], dtype="int64"),
        "n_matched": pd.Series([], dtype="int64"),
        "n_fn": pd.Series([], dtype="int64"),
        "n_low_conf": pd.Series([], dtype="int64"),
        "failure_score": pd.Series([], dtype="float64"),
    }).to_parquet(al / "failures.parquet")
    pd.DataFrame({
        "sample_data_token": tokens,
        "sample_token": [f"s{i}" for i in range(1, 6)],
        "channel": ["CAM_FRONT"] * 5,
        "filename": [f"samples/CAM_FRONT/{i}.jpg" for i in range(1, 6)],
        "scene_name": [f"scene-{i}" for i in range(1, 6)],
        "is_night": [False] * 5,
        "is_rain": [False] * 5,
    }).to_parquet(processed / "samples.parquet")
    pd.DataFrame({
        "sample_token": pd.Series([], dtype="object"),
        "is_hard_braking": pd.Series([], dtype="object"),
    }).to_parquet(processed / "canbus.parquet")
    pd.DataFrame({"sample_data_token": tokens}).to_parquet(al / "graph_rate_night.parquet")
    pd.DataFrame({"sample_data_token": ["t1"]}).to_parquet(
        al / "graph_rate_night_accepted.parquet"
    )

    quotas = {"night_failure": 0, "day_failure": 0, "hard_braking": 0, "al_selected": 2,
              "weak_accepted": 1, "weak_rejected": 2, "clean_success": 0, "semantic": 0}
    manifest = run_curate(
        processed_dir=processed, al_dir=al, staging_dir=tmp_path / "staging",
        quotas=quotas, al_arm="graph_rate_night", weak_arm="graph_rate_night",
        semantic_hits=lambda queries: [], semantic_queries=[],
    )
    by_token = manifest.set_index("sample_data_token")
    # weak_accepted=[t1], weak_rejected=[t2,t3] claim the first three tokens (sorted
    # ascending); al_selected (quota 2) must draw from what's left: t4, t5.
    assert "al_selected" in by_token.loc["t4", "curation_buckets"]
    assert "al_selected" in by_token.loc["t5", "curation_buckets"]
    assert "al_selected" not in by_token.loc["t1", "curation_buckets"]
    assert "al_selected" not in by_token.loc["t2", "curation_buckets"]
    assert "al_selected" not in by_token.loc["t3", "curation_buckets"]
    assert len(set(manifest["sample_data_token"])) == 5  # t1..t5 all curated, no redundancy


def test_curate_empty_quotas_returns_empty_manifest_with_columns(tmp_path: Path) -> None:
    """M1: an all-empty quotas dict must not KeyError on column access downstream --
    the manifest is empty but still has every expected column."""
    from nuscenes_data_engine.demo.curate import run_curate

    paths = _curation_fixture(tmp_path)
    manifest = run_curate(
        processed_dir=paths["processed"], al_dir=paths["al"],
        staging_dir=tmp_path / "staging", quotas={},
        al_arm="graph_rate_night", weak_arm="graph_rate_night",
        semantic_hits=lambda queries: [], semantic_queries=[],
    )
    assert len(manifest) == 0
    assert "filename" in manifest.columns
    assert "sample_data_token" in manifest.columns
    assert (tmp_path / "staging" / "rsync_filelist.txt").read_text() == ""


def test_curate_warns_when_bucket_underfills(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """M2: requesting more than a bucket can supply logs a warning instead of
    silently shipping a short manifest."""
    from nuscenes_data_engine.demo.curate import run_curate

    paths = _curation_fixture(tmp_path)
    quotas = {"night_failure": 100, "day_failure": 0, "hard_braking": 0, "al_selected": 0,
              "weak_accepted": 0, "weak_rejected": 0, "clean_success": 0, "semantic": 0}
    with caplog.at_level(logging.WARNING, logger="nuscenes_data_engine"):
        run_curate(
            processed_dir=paths["processed"], al_dir=paths["al"],
            staging_dir=tmp_path / "staging", quotas=quotas,
            al_arm="graph_rate_night", weak_arm="graph_rate_night",
            semantic_hits=lambda queries: [], semantic_queries=[],
        )
    assert "night_failure" in caplog.text


def test_curate_rejects_unknown_quota_keys(tmp_path: Path) -> None:
    """M3: a typo'd quota key must fail loudly, not be silently ignored."""
    from nuscenes_data_engine.demo.curate import run_curate

    paths = _curation_fixture(tmp_path)
    quotas = {"night_failure": 1, "bogus_bucket": 5}
    with pytest.raises(ValueError, match="bogus_bucket"):
        run_curate(
            processed_dir=paths["processed"], al_dir=paths["al"],
            staging_dir=tmp_path / "staging", quotas=quotas,
            al_arm="graph_rate_night", weak_arm="graph_rate_night",
            semantic_hits=lambda queries: [], semantic_queries=[],
        )


def test_run_infer_writes_predictions_gt_and_exemplar_flags(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.infer import run_infer

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "images").mkdir()
    # two val frames with 1600x900 fake images
    from PIL import Image

    for token in ("v0", "v1"):
        Image.new("RGB", (1600, 900), "gray").save(staging / "images" / f"{token}.jpg")
    manifest = pd.DataFrame({
        "sample_data_token": ["v0", "v1", "t1"],
        "split": ["val", "val", "train_pool"],
        "filename": ["images/v0.jpg", "images/v1.jpg", "images/t1.jpg"],
        "curation_buckets": [["night_failure"], ["day_failure"], ["weak_accepted"]],
    })
    manifest.to_parquet(staging / "frame_manifest.parquet")
    Image.new("RGB", (1600, 900), "gray").save(staging / "images" / "t1.jpg")
    annotations = pd.DataFrame({
        "annotation_token": ["a1", "a2", "a3"],
        "sample_data_token": ["v0", "v0", "v1"],
        "category_group": ["pedestrian", "car", "car"],
        "x_min": [100.0, 500.0, 300.0], "y_min": [100.0, 300.0, 200.0],
        "x_max": [200.0, 700.0, 500.0], "y_max": [300.0, 500.0, 400.0],
        # Task 3 correction (2026-08-18): annotations must carry visibility_token so
        # run_infer can mirror sweep.py's visibility_min filter; full visibility here
        # (>= the default "2") keeps all three rows in the matched set for this test.
        "visibility_token": ["4", "4", "4"],
    })

    # baseline misses the pedestrian; night model finds it -> exemplar flag
    def fake_predict(model_name: str):
        def predict(image_path: Path):
            hits = {"baseline": [[500, 300, 700, 500, 0.9, "car"]],
                    "night": [[500, 300, 700, 500, 0.9, "car"],
                              [100, 100, 200, 300, 0.8, "pedestrian"]]}
            rows = hits[model_name] if "v0" in str(image_path) else []
            import numpy as np
            return {
                "boxes": np.array([r[:4] for r in rows], dtype=float).reshape(-1, 4),
                "conf": np.array([r[4] for r in rows], dtype=float),
                "classes": [r[5] for r in rows],
            }
        return predict

    out = run_infer(
        staging_dir=staging, annotations=annotations,
        models={"baseline": fake_predict("baseline"), "night": fake_predict("night")},
        crop_size=(960, 540), iou=0.5, conf_hit=0.4,
    )
    preds = pd.read_parquet(staging / "predictions.parquet")
    assert set(preds["model"]) == {"baseline", "night"}
    v0_base = preds[(preds.sample_data_token == "v0") & (preds.model == "baseline")]
    assert list(v0_base["status"]) == ["tp"]
    gt = pd.read_parquet(staging / "gt_boxes.parquet")
    row = gt[(gt.annotation_token == "a1")].iloc[0]
    assert row["matched_baseline"] == False and row["matched_night"] == True  # noqa: E712
    updated = pd.read_parquet(staging / "frame_manifest.parquet")
    v0 = updated.set_index("sample_data_token").loc["v0"]
    assert v0["fixes_fn_vs_baseline_night"] == True  # noqa: E712
    # crops written for every frame incl. train_pool, at 960x540
    from PIL import Image as PILImage
    crop = PILImage.open(staging / "crops" / "v0.jpg")
    assert crop.size == (960, 540)
    assert (staging / "crops" / "t1.jpg").is_file()
    assert out["n_predictions"] == len(preds)


def test_run_infer_missing_image_raises(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.infer import run_infer

    staging = tmp_path / "staging"
    staging.mkdir()
    pd.DataFrame({
        "sample_data_token": ["v0"], "split": ["val"],
        "filename": ["images/v0.jpg"], "curation_buckets": [["night_failure"]],
    }).to_parquet(staging / "frame_manifest.parquet")
    with pytest.raises(ValueError, match="v0"):
        run_infer(
            staging_dir=staging, annotations=pd.DataFrame(
                columns=["annotation_token", "sample_data_token", "category_group",
                         "x_min", "y_min", "x_max", "y_max", "visibility_token"]),
            models={"baseline": lambda p: None}, crop_size=(960, 540),
            iou=0.5, conf_hit=0.4,
        )


def test_run_infer_below_visibility_min_excluded_but_flagged(tmp_path: Path) -> None:
    """Correction (2026-08-18, review round): sweep.py filters GT to
    ``visibility_token >= visibility_min`` before matching, so run_infer must do the
    same -- otherwise the demo's FN semantics disagree with failures.parquet. a3 sits
    below the default ``visibility_min="2"`` threshold: it must be flagged in
    gt_boxes.parquet, excluded from matching (NA rather than False -- unmatched is not
    the same claim as "false negative"), and never a prediction's matched target even
    when a prediction's box perfectly overlaps it.
    """
    from nuscenes_data_engine.demo.infer import run_infer

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "images").mkdir()
    from PIL import Image

    Image.new("RGB", (1600, 900), "gray").save(staging / "images" / "v0.jpg")
    pd.DataFrame({
        "sample_data_token": ["v0"], "split": ["val"],
        "filename": ["images/v0.jpg"], "curation_buckets": [["night_failure"]],
    }).to_parquet(staging / "frame_manifest.parquet")
    annotations = pd.DataFrame({
        "annotation_token": ["a1", "a3"],
        "sample_data_token": ["v0", "v0"],
        "category_group": ["car", "pedestrian"],
        "x_min": [500.0, 100.0], "y_min": [300.0, 100.0],
        "x_max": [700.0, 200.0], "y_max": [500.0, 300.0],
        "visibility_token": ["4", "1"],  # a3 is below the default visibility_min="2"
    })

    def fake_predict(image_path: Path):
        import numpy as np
        # A perfect hit on a3's box -- must NOT be allowed to claim it (excluded).
        return {
            "boxes": np.array(
                [[500.0, 300.0, 700.0, 500.0], [100.0, 100.0, 200.0, 300.0]], dtype=float
            ),
            "conf": np.array([0.9, 0.9]),
            "classes": ["car", "pedestrian"],
        }

    run_infer(
        staging_dir=staging, annotations=annotations,
        models={"baseline": fake_predict}, crop_size=(960, 540), iou=0.5, conf_hit=0.4,
    )
    gt = pd.read_parquet(staging / "gt_boxes.parquet").set_index("annotation_token")
    assert gt.loc["a3", "below_visibility_min"] == True  # noqa: E712
    assert gt.loc["a1", "below_visibility_min"] == False  # noqa: E712
    assert pd.isna(gt.loc["a3", "matched_baseline"])  # unmatched, but NOT a false negative
    assert gt.loc["a1", "matched_baseline"] == True  # noqa: E712

    preds = pd.read_parquet(staging / "predictions.parquet")
    assert "a3" not in set(preds["matched_annotation_token"].dropna())


def test_run_infer_uses_custom_images_root(tmp_path: Path) -> None:
    """``images_root`` defaults to ``staging_dir`` (what the tests above rely on); the
    real CLI run passes ``data/raw/demo_frames``, the rsync destination, which is a
    separate directory from the curation staging dir."""
    from nuscenes_data_engine.demo.infer import run_infer

    staging = tmp_path / "staging"
    staging.mkdir()
    images_root = tmp_path / "raw_frames"
    (images_root / "samples" / "CAM_FRONT").mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (1600, 900), "gray").save(
        images_root / "samples" / "CAM_FRONT" / "v0.jpg"
    )
    pd.DataFrame({
        "sample_data_token": ["v0"], "split": ["val"],
        "filename": ["samples/CAM_FRONT/v0.jpg"], "curation_buckets": [["night_failure"]],
    }).to_parquet(staging / "frame_manifest.parquet")
    annotations = pd.DataFrame(
        columns=["annotation_token", "sample_data_token", "category_group",
                 "x_min", "y_min", "x_max", "y_max", "visibility_token"]
    )
    run_infer(
        staging_dir=staging, annotations=annotations, models={},
        crop_size=(960, 540), iou=0.5, conf_hit=0.4, images_root=images_root,
    )
    assert (staging / "crops" / "v0.jpg").is_file()


def test_run_infer_exemplar_flags_for_every_ordered_model_pair(tmp_path: Path) -> None:
    """fixes_fn_vs_<a>_<b> must exist for every ORDERED pair among 3+ models, not just
    adjacent ones -- with 3 models {A, B, C} that's 6 columns, and the True/False
    pattern for each must reflect that specific pair's (a missed it, b caught it)
    relationship, not some aggregate."""
    from nuscenes_data_engine.demo.infer import run_infer

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "images").mkdir()
    from PIL import Image

    Image.new("RGB", (1600, 900), "gray").save(staging / "images" / "v0.jpg")
    pd.DataFrame({
        "sample_data_token": ["v0"], "split": ["val"],
        "filename": ["images/v0.jpg"], "curation_buckets": [["night_failure"]],
    }).to_parquet(staging / "frame_manifest.parquet")
    annotations = pd.DataFrame({
        "annotation_token": ["a1"],
        "sample_data_token": ["v0"],
        "category_group": ["pedestrian"],
        "x_min": [100.0], "y_min": [100.0], "x_max": [200.0], "y_max": [300.0],
        "visibility_token": ["4"],
    })

    import numpy as np

    def miss(image_path: Path) -> dict:
        return {"boxes": np.zeros((0, 4)), "conf": np.zeros(0), "classes": []}

    def hit(image_path: Path) -> dict:
        return {
            "boxes": np.array([[100.0, 100.0, 200.0, 300.0]]),
            "conf": np.array([0.9]),
            "classes": ["pedestrian"],
        }

    # Only "model_b" finds the pedestrian; "model_a" and "model_c" both miss it.
    run_infer(
        staging_dir=staging, annotations=annotations,
        models={"model_a": miss, "model_b": hit, "model_c": miss},
        crop_size=(960, 540), iou=0.5, conf_hit=0.4,
    )
    manifest = pd.read_parquet(staging / "frame_manifest.parquet")
    v0 = manifest.set_index("sample_data_token").loc["v0"]
    assert v0["fixes_fn_vs_model_a_model_b"] == True  # noqa: E712
    assert v0["fixes_fn_vs_model_c_model_b"] == True  # noqa: E712
    assert v0["fixes_fn_vs_model_a_model_c"] == False  # noqa: E712
    assert v0["fixes_fn_vs_model_b_model_a"] == False  # noqa: E712
    assert v0["fixes_fn_vs_model_b_model_c"] == False  # noqa: E712
    assert v0["fixes_fn_vs_model_c_model_a"] == False  # noqa: E712


def _single_frame_fixture(tmp_path: Path, *, image_size: tuple[int, int] = (1600, 900)) -> Path:
    """One val frame (v0), one GT box, no models -- shared scaffolding for the
    fixture-only tests below (unmapped class, image-size guard, stale columns)."""
    from PIL import Image

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "images").mkdir()
    Image.new("RGB", image_size, "gray").save(staging / "images" / "v0.jpg")
    pd.DataFrame({
        "sample_data_token": ["v0"], "split": ["val"],
        "filename": ["images/v0.jpg"], "curation_buckets": [["night_failure"]],
    }).to_parquet(staging / "frame_manifest.parquet")
    return staging


def test_run_infer_unmapped_pred_class_raises(tmp_path: Path) -> None:
    """A class outside CLASS_TO_INDEX (e.g. a COCO-style "person") must fail loudly,
    naming the offending model, the bad class, and the accepted vocabulary -- not a
    bare, uninformative KeyError."""
    from nuscenes_data_engine.demo.infer import run_infer

    staging = _single_frame_fixture(tmp_path)
    annotations = pd.DataFrame({
        "annotation_token": ["a1"], "sample_data_token": ["v0"],
        "category_group": ["pedestrian"],
        "x_min": [100.0], "y_min": [100.0], "x_max": [200.0], "y_max": [300.0],
        "visibility_token": ["4"],
    })

    import numpy as np

    def bad_predict(image_path: Path) -> dict:
        return {
            "boxes": np.array([[100.0, 100.0, 200.0, 300.0]]),
            "conf": np.array([0.9]),
            "classes": ["person"],  # not one of DETECTION_CLASSES
        }

    with pytest.raises(ValueError, match=r"rogue_model.*person"):
        run_infer(
            staging_dir=staging, annotations=annotations,
            models={"rogue_model": bad_predict},
            crop_size=(960, 540), iou=0.5, conf_hit=0.4,
        )


def test_run_infer_wrong_size_image_raises(tmp_path: Path) -> None:
    """A staged image that isn't exactly IMAGE_WIDTH x IMAGE_HEIGHT would put
    predictions/GT in the wrong pixel space -- must raise naming the token, not
    silently resize a mis-projected frame."""
    from nuscenes_data_engine.demo.infer import run_infer

    staging = _single_frame_fixture(tmp_path, image_size=(800, 450))
    annotations = pd.DataFrame(
        columns=["annotation_token", "sample_data_token", "category_group",
                 "x_min", "y_min", "x_max", "y_max", "visibility_token"]
    )
    with pytest.raises(ValueError, match="v0"):
        run_infer(
            staging_dir=staging, annotations=annotations, models={},
            crop_size=(960, 540), iou=0.5, conf_hit=0.4,
        )


def test_run_infer_drops_stale_exemplar_columns_on_rerun(tmp_path: Path) -> None:
    """Rerunning demo infer with a different model roster must not ship a PRIOR
    run's fixes_fn_vs_<a>_<b> columns alongside (or instead of) the current ones."""
    from nuscenes_data_engine.demo.infer import run_infer

    staging = _single_frame_fixture(tmp_path)
    annotations = pd.DataFrame({
        "annotation_token": ["a1"], "sample_data_token": ["v0"],
        "category_group": ["pedestrian"],
        "x_min": [100.0], "y_min": [100.0], "x_max": [200.0], "y_max": [300.0],
        "visibility_token": ["4"],
    })

    import numpy as np

    def miss(image_path: Path) -> dict:
        return {"boxes": np.zeros((0, 4)), "conf": np.zeros(0), "classes": []}

    def hit(image_path: Path) -> dict:
        return {
            "boxes": np.array([[100.0, 100.0, 200.0, 300.0]]),
            "conf": np.array([0.9]),
            "classes": ["pedestrian"],
        }

    # First run: two models -> fixes_fn_vs_old_a_old_b / fixes_fn_vs_old_b_old_a.
    run_infer(
        staging_dir=staging, annotations=annotations,
        models={"old_a": miss, "old_b": hit},
        crop_size=(960, 540), iou=0.5, conf_hit=0.4,
    )
    first_manifest = pd.read_parquet(staging / "frame_manifest.parquet")
    assert "fixes_fn_vs_old_a_old_b" in first_manifest.columns
    assert "n_preds_old_a" in first_manifest.columns

    # Second run over the same staging dir, single model -> zero pair columns.
    run_infer(
        staging_dir=staging, annotations=annotations,
        models={"new_solo": hit},
        crop_size=(960, 540), iou=0.5, conf_hit=0.4,
    )
    second_manifest = pd.read_parquet(staging / "frame_manifest.parquet")
    stale = [c for c in second_manifest.columns if c.startswith("fixes_fn_vs_")]
    assert stale == []
    # a PRIOR run's n_preds_<model> columns must not survive a rerun with a
    # different model roster either -- same staleness class as fixes_fn_vs_<a>_<b>.
    stale_n_preds = [c for c in second_manifest.columns if c.startswith("n_preds_")]
    assert stale_n_preds == ["n_preds_new_solo"]


def test_run_infer_records_n_preds_including_explicit_zero(tmp_path: Path) -> None:
    """Task-5 fix: 'ran and found nothing' (0 predictions) is a legitimate finding —
    exactly the kind of total-miss frame the demo wants to show — and must be
    distinguished from 'never ran' (NA). build.py's val-coverage validation keys off
    this n_preds_<model> column, not off predictions.parquet row presence, precisely
    because a real zero-prediction model produces zero prediction.parquet rows too,
    indistinguishable from a model that was never configured."""
    from nuscenes_data_engine.demo.infer import run_infer

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "images").mkdir()
    from PIL import Image

    Image.new("RGB", (1600, 900), "gray").save(staging / "images" / "v0.jpg")
    Image.new("RGB", (1600, 900), "gray").save(staging / "images" / "t1.jpg")
    pd.DataFrame({
        "sample_data_token": ["v0", "t1"], "split": ["val", "train_pool"],
        "filename": ["images/v0.jpg", "images/t1.jpg"],
        "curation_buckets": [["night_failure"], ["weak_accepted"]],
    }).to_parquet(staging / "frame_manifest.parquet")
    annotations = pd.DataFrame({
        "annotation_token": ["a1"], "sample_data_token": ["v0"],
        "category_group": ["pedestrian"],
        "x_min": [100.0], "y_min": [100.0], "x_max": [200.0], "y_max": [300.0],
        "visibility_token": ["4"],
    })

    import numpy as np

    def miss(image_path: Path) -> dict:
        return {"boxes": np.zeros((0, 4)), "conf": np.zeros(0), "classes": []}

    def hit(image_path: Path) -> dict:
        return {
            "boxes": np.array([[100.0, 100.0, 200.0, 300.0]]),
            "conf": np.array([0.9]),
            "classes": ["pedestrian"],
        }

    run_infer(
        staging_dir=staging, annotations=annotations,
        models={"baseline": miss, "champion": hit},
        crop_size=(960, 540), iou=0.5, conf_hit=0.4,
    )
    manifest = pd.read_parquet(staging / "frame_manifest.parquet")
    by_token = manifest.set_index("sample_data_token")
    assert by_token.loc["v0", "n_preds_baseline"] == 0     # ran, found nothing
    assert by_token.loc["v0", "n_preds_champion"] == 1
    assert str(manifest["n_preds_baseline"].dtype) == "Int64"
    assert str(manifest["n_preds_champion"].dtype) == "Int64"
    # train_pool: no model ever ran over it -- NA, not zero.
    assert pd.isna(by_token.loc["t1", "n_preds_baseline"])
    assert pd.isna(by_token.loc["t1", "n_preds_champion"])


def test_run_infer_empty_predictions_has_stable_dtypes(tmp_path: Path) -> None:
    """A run with no val frames writes an empty predictions.parquet -- its schema
    must still match a non-empty run's (explicit dtypes), not infer everything as
    object/float64 from zero rows."""
    from nuscenes_data_engine.demo.infer import run_infer

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "images").mkdir()
    from PIL import Image

    Image.new("RGB", (1600, 900), "gray").save(staging / "images" / "t1.jpg")
    pd.DataFrame({
        "sample_data_token": ["t1"], "split": ["train_pool"],
        "filename": ["images/t1.jpg"], "curation_buckets": [["weak_accepted"]],
    }).to_parquet(staging / "frame_manifest.parquet")
    annotations = pd.DataFrame(
        columns=["annotation_token", "sample_data_token", "category_group",
                 "x_min", "y_min", "x_max", "y_max", "visibility_token"]
    )
    run_infer(
        staging_dir=staging, annotations=annotations, models={"baseline": lambda p: None},
        crop_size=(960, 540), iou=0.5, conf_hit=0.4,
    )
    preds = pd.read_parquet(staging / "predictions.parquet")
    assert len(preds) == 0
    assert str(preds["x_min"].dtype) == "float64"
    assert str(preds["conf"].dtype) == "float64"
    assert str(preds["imgsz"].dtype) == "Int64"
    assert str(preds["sample_data_token"].dtype) == "object"
    assert str(preds["status"].dtype) == "object"


def test_demo_infer_cli_rejects_legacy_flat_model_shape(tmp_path: Path) -> None:
    """Review round 2, item 3 (test gap): cli.py's ``demo infer`` guards each
    ``models.<name>`` entry with ``isinstance(spec, dict)`` before touching
    ``spec["imgsz"]`` -- reachable torch-free (the guard fires inside the model-spec
    loop, strictly before ``make_ultralytics_predictor`` would ever import
    ultralytics), but had zero coverage. Drives the guard through the real CLI
    command (CliRunner), not just a unit call, so a regression in the preflight
    ordering around it would also be caught.
    """
    from typer.testing import CliRunner

    from nuscenes_data_engine.cli import app

    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame({
        "sample_data_token": pd.Series([], dtype="object"),
        "channel": pd.Series([], dtype="object"),
    }).to_parquet(processed / "annotations.parquet")

    staging = tmp_path / "staging"
    staging.mkdir()
    pd.DataFrame({"sample_data_token": ["v0"]}).to_parquet(staging / "frame_manifest.parquet")

    images_root = tmp_path / "images_root"
    images_root.mkdir()
    (images_root / "dummy.jpg").write_bytes(b"x")  # only existence/non-emptiness checked

    config = {
        "paths": {
            "processed_dir": str(processed),
            "mlruns_dir": str(tmp_path / "mlruns"),
        },
        "curation": {
            "staging_dir": str(staging),
            "images_root": str(images_root),
            "crop_size": [960, 540],
        },
        "models": {"baseline": "runX"},  # legacy flat shape -- must be rejected
    }
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(yaml.safe_dump(config))
    al_config_path = tmp_path / "active_learning.yaml"
    al_config_path.write_text(yaml.safe_dump({"sweep": {}}))

    result = CliRunner().invoke(
        app,
        ["demo", "infer", "--config", str(config_path), "--al-config", str(al_config_path)],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "models.baseline" in str(result.exception)
    assert "run, imgsz" in str(result.exception)


def test_front_camera_hits_drops_non_cam_front_and_preserves_rank_order() -> None:
    """Task-5 live-run fix: real SigLIP search returned a CAM_BACK token
    (85f49850a882409ea01b2e8f3d006d05) that reached run_curate's channel guard and
    blew up the whole curate run -- filtering has to happen at the semantic source,
    before curation ever sees a non-CAM_FRONT token. Rank order (nearest-first) must
    survive filtering -- it only drops entries, never re-sorts."""
    from nuscenes_data_engine.demo.curate import _front_camera_hits

    results = [
        {"sample_data_token": "f1", "channel": "CAM_FRONT"},
        {"sample_data_token": "b1", "channel": "CAM_BACK"},
        {"sample_data_token": "f2", "channel": "CAM_FRONT"},
        {"sample_data_token": "l1", "channel": "CAM_FRONT_LEFT"},
        {"sample_data_token": "f3", "channel": "CAM_FRONT"},
    ]
    hits = _front_camera_hits(results, quota=2)
    assert hits == [("f1", ["semantic"]), ("f2", ["semantic"])]


def test_front_camera_hits_returns_all_matches_when_under_quota() -> None:
    from nuscenes_data_engine.demo.curate import _front_camera_hits

    results = [
        {"sample_data_token": "b1", "channel": "CAM_BACK"},
        {"sample_data_token": "f1", "channel": "CAM_FRONT"},
    ]
    hits = _front_camera_hits(results, quota=5)
    assert hits == [("f1", ["semantic"])]


def test_front_camera_hits_all_non_cam_front_yields_empty() -> None:
    from nuscenes_data_engine.demo.curate import _front_camera_hits

    results = [
        {"sample_data_token": "b1", "channel": "CAM_BACK"},
        {"sample_data_token": "l1", "channel": "CAM_FRONT_LEFT"},
    ]
    assert _front_camera_hits(results, quota=5) == []


def test_front_camera_hits_empty_results_and_zero_quota() -> None:
    from nuscenes_data_engine.demo.curate import _front_camera_hits

    assert _front_camera_hits([], quota=5) == []
    assert _front_camera_hits(
        [{"sample_data_token": "f1", "channel": "CAM_FRONT"}], quota=0
    ) == []
