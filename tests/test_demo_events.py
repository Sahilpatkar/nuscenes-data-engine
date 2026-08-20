"""Tests for demo/events.py — synthetic-fixture unit tests (pure; no torch, no network).

Two fixture builders:
- ``_write_processed``: writes samples/canbus/ego_pose/annotations_3d parquets from a
  list of per-frame dicts (the "dataset-wide" inputs every preset can see).
- ``_write_staging``: writes frame_manifest/gt_boxes/predictions parquets (the
  curated-val-only inputs the two model-result presets read).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nuscenes_data_engine.demo.events import build_events

PRESETS_CFG = {
    "cap_per_preset": 30,
    "near_dist_m": 10.0,
    "high_speed_mps": 10.0,
    "model_for_results": "baseline",
}


def _write_processed(processed_dir: Path, frames: list[dict[str, Any]]) -> None:
    """Build samples/canbus/ego_pose/annotations_3d parquets from ``frames``.

    Each frame dict: sdt, st, scene_token, scene_name, timestamp, is_night, is_rain,
    hard_braking (True/False/None), accel, speed, pedestrians/vehicles/cyclists
    (lists of distance_to_ego_m floats, possibly empty).

    Rows are written in the GIVEN (not necessarily timestamp-sorted) order, on
    purpose — ``build_events`` must re-sort by timestamp itself before computing
    filmstrip neighbors (test_neighbors_ordered_and_na_at_scene_edges depends on
    this: the fixture below is deliberately shuffled).
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "sample_data_token": [f["sdt"] for f in frames],
            "sample_token": [f["st"] for f in frames],
            "channel": ["CAM_FRONT"] * len(frames),
            "scene_token": [f["scene_token"] for f in frames],
            "scene_name": [f["scene_name"] for f in frames],
            "timestamp": [f["timestamp"] for f in frames],
            "is_night": [f["is_night"] for f in frames],
            "is_rain": [f["is_rain"] for f in frames],
        }
    ).to_parquet(processed_dir / "samples.parquet")

    pd.DataFrame(
        {
            "sample_token": [f["st"] for f in frames],
            "is_hard_braking": [f["hard_braking"] for f in frames],
            "accel_long_min_mps2": [f["accel"] for f in frames],
        }
    ).to_parquet(processed_dir / "canbus.parquet")

    pd.DataFrame(
        {
            "sample_token": [f["st"] for f in frames],
            "speed_mps": [f["speed"] for f in frames],
        }
    ).to_parquet(processed_dir / "ego_pose.parquet")

    ann_rows: list[dict[str, Any]] = []
    for f in frames:
        for i, dist in enumerate(f.get("pedestrians", [])):
            ann_rows.append(
                {
                    "sample_token": f["st"],
                    "category_group": "pedestrian",
                    "distance_to_ego_m": dist,
                    "annotation_token": f"{f['sdt']}-ped-{i}",
                }
            )
        for i, dist in enumerate(f.get("vehicles", [])):
            ann_rows.append(
                {
                    "sample_token": f["st"],
                    "category_group": "car",
                    "distance_to_ego_m": dist,
                    "annotation_token": f"{f['sdt']}-veh-{i}",
                }
            )
        for i, dist in enumerate(f.get("cyclists", [])):
            ann_rows.append(
                {
                    "sample_token": f["st"],
                    "category_group": "bicycle",
                    "distance_to_ego_m": dist,
                    "annotation_token": f"{f['sdt']}-cyc-{i}",
                }
            )
    pd.DataFrame(ann_rows).to_parquet(processed_dir / "annotations_3d.parquet")


def _write_staging(
    staging_dir: Path,
    *,
    val_tokens: list[str],
    fn_pedestrian_tokens: list[str],
    low_conf_tokens: list[str],
) -> None:
    """Minimal curated-val staging: only the columns build_events actually reads.

    ``fn_pedestrian_tokens`` get one pedestrian GT box with matched_baseline=False
    (a false negative); ``low_conf_tokens`` get one baseline prediction with
    status='low_conf'. A token can appear in both lists.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "sample_data_token": val_tokens,
            "split": ["val"] * len(val_tokens),
        }
    ).to_parquet(staging_dir / "frame_manifest.parquet")

    gt_rows = [
        {
            "annotation_token": f"{token}-fn-ped",
            "sample_data_token": token,
            "category_group": "pedestrian",
            "matched_baseline": False,
        }
        for token in fn_pedestrian_tokens
    ]
    # A decoy TP pedestrian box on a low_conf-only token: proves the fn predicate
    # doesn't just key off "has a pedestrian GT box" but specifically the miss.
    for token in low_conf_tokens:
        if token not in fn_pedestrian_tokens:
            gt_rows.append(
                {
                    "annotation_token": f"{token}-tp-ped",
                    "sample_data_token": token,
                    "category_group": "pedestrian",
                    "matched_baseline": True,
                }
            )
    pd.DataFrame(gt_rows).to_parquet(staging_dir / "gt_boxes.parquet")

    pred_rows = [
        {
            "sample_data_token": token,
            "model": "baseline",
            "status": "low_conf",
            "conf": 0.2,
        }
        for token in low_conf_tokens
    ]
    # Decoy: a low_conf prediction from a DIFFERENT model on an fn-only token, plus a
    # non-hard-braking low_conf baseline hit — neither should tag low_conf_braking.
    for token in fn_pedestrian_tokens:
        if token not in low_conf_tokens:
            pred_rows.append(
                {
                    "sample_data_token": token,
                    "model": "graph_rate_night",
                    "status": "low_conf",
                    "conf": 0.3,
                }
            )
    pd.DataFrame(pred_rows).to_parquet(staging_dir / "predictions.parquet")


def _base_frames() -> list[dict[str, Any]]:
    """~10 keyframes across 2 scenes covering every preset + edge/null cases.

    scene A (edge frames a0/a4 untagged, for the plain scene-edge case):
      a0 plain filler; a1 FLAGSHIP (hard braking + a pedestrian at 5m < near_dist_m);
      a2 curated FN-ped-at-night (dynamics-isolated: its own nearest pedestrian is
      50m away, so it must NOT also pick up the dynamics night_pedestrians tag);
      a3 null is_hard_braking WITH a near pedestrian (5m) — if null were ever treated
      as True this would wrongly join the flagship set; a4 plain filler.
    scene B (edges ARE tagged events, to test NA-neighbor + ordering on real rows):
      b1 (edge start) fast-cyclist; b0 curated low-conf-braking (hard braking, no
      near pedestrian, decoy low-conf pred present); b_mid plain filler; b2 rain-VRU;
      b3 (edge end) night-pedestrians.
    Rows are written in the printed order here but that is NOT timestamp order —
    b1 (earliest timestamp) is listed before b0 and b3 (latest) precedes b_mid/b2
    on the page, forcing build_events to re-sort by timestamp itself.
    """
    return [
        {
            "sdt": "a0_sdt",
            "st": "a0_st",
            "scene_token": "sceneA",
            "scene_name": "scene-A",
            "timestamp": 1000,
            "is_night": False,
            "is_rain": False,
            "hard_braking": False,
            "accel": -1.0,
            "speed": 5.0,
            "vehicles": [50.0],
        },
        {
            "sdt": "a1_sdt",
            "st": "a1_st",
            "scene_token": "sceneA",
            "scene_name": "scene-A",
            "timestamp": 1001,
            "is_night": False,
            "is_rain": False,
            "hard_braking": True,
            "accel": -7.5,
            "speed": 8.0,
            "pedestrians": [5.0],
            "vehicles": [30.0],
        },
        {
            "sdt": "a2_sdt",
            "st": "a2_st",
            "scene_token": "sceneA",
            "scene_name": "scene-A",
            "timestamp": 1002,
            "is_night": True,
            "is_rain": False,
            "hard_braking": False,
            "accel": -0.5,
            "speed": 6.0,
            "pedestrians": [50.0],
        },
        {
            "sdt": "a3_sdt",
            "st": "a3_st",
            "scene_token": "sceneA",
            "scene_name": "scene-A",
            "timestamp": 1003,
            "is_night": False,
            "is_rain": False,
            "hard_braking": None,
            "accel": -0.2,
            "speed": 5.0,
            "pedestrians": [5.0],
        },
        {
            "sdt": "a4_sdt",
            "st": "a4_st",
            "scene_token": "sceneA",
            "scene_name": "scene-A",
            "timestamp": 1004,
            "is_night": False,
            "is_rain": False,
            "hard_braking": False,
            "accel": -0.3,
            "speed": 4.0,
        },
        {
            "sdt": "b3_sdt",
            "st": "b3_st",
            "scene_token": "sceneB",
            "scene_name": "scene-B",
            "timestamp": 2004,
            "is_night": True,
            "is_rain": False,
            "hard_braking": False,
            "accel": -0.4,
            "speed": 5.0,
            "pedestrians": [8.0],
        },
        {
            "sdt": "b2_sdt",
            "st": "b2_st",
            "scene_token": "sceneB",
            "scene_name": "scene-B",
            "timestamp": 2002,
            "is_night": False,
            "is_rain": True,
            "hard_braking": False,
            "accel": -0.4,
            "speed": 5.0,
            "pedestrians": [7.0],
        },
        {
            "sdt": "b_mid_sdt",
            "st": "b_mid_st",
            "scene_token": "sceneB",
            "scene_name": "scene-B",
            "timestamp": 2001,
            "is_night": False,
            "is_rain": False,
            "hard_braking": False,
            "accel": -0.3,
            "speed": 4.0,
        },
        {
            "sdt": "b0_sdt",
            "st": "b0_st",
            "scene_token": "sceneB",
            "scene_name": "scene-B",
            "timestamp": 2000,
            "is_night": False,
            "is_rain": False,
            "hard_braking": True,
            "accel": -6.0,
            "speed": 7.0,
            "pedestrians": [40.0],
        },
        {
            "sdt": "b1_sdt",
            "st": "b1_st",
            "scene_token": "sceneB",
            "scene_name": "scene-B",
            "timestamp": 1999,
            "is_night": False,
            "is_rain": False,
            "hard_braking": False,
            "accel": -0.5,
            "speed": 12.0,
            "cyclists": [6.0],
        },
    ]


def _build(
    tmp_path: Path,
    *,
    with_staging: bool = True,
    presets_cfg: dict[str, Any] | None = None,
    flagship_expected: int | None = 1,
    frames: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    processed_dir = tmp_path / "processed"
    _write_processed(processed_dir, frames if frames is not None else _base_frames())
    staging_dir = None
    if with_staging:
        staging_dir = tmp_path / "staging"
        _write_staging(
            staging_dir,
            val_tokens=["a2_sdt", "b0_sdt"],
            fn_pedestrian_tokens=["a2_sdt"],
            low_conf_tokens=["b0_sdt"],
        )
    return build_events(
        processed_dir=processed_dir,
        staging_dir=staging_dir,
        presets_cfg=presets_cfg or PRESETS_CFG,
        flagship_expected=flagship_expected,
    )


def _tags_of(events: pd.DataFrame, token: str) -> list[str]:
    row = events.loc[events["sample_data_token"] == token]
    assert len(row) == 1, f"expected exactly one row for {token!r}, found {len(row)}"
    return list(row.iloc[0]["preset_tags"])


def test_event_features_and_flagship_agrees_with_sql(tmp_path: Path) -> None:
    events = _build(tmp_path)
    flagship = events[events["preset_tags"].apply(lambda t: "hard_braking_near_pedestrians" in t)]
    assert len(flagship) == 1  # fixture has exactly one (a1)
    row = flagship.iloc[0]
    assert row["sample_data_token"] == "a1_sdt"
    assert row["min_dist_pedestrian_m"] < 10 and row["is_hard_braking"]
    # every documented per-event column is present
    for col in (
        "sample_data_token",
        "sample_token",
        "scene_name",
        "timestamp",
        "speed_mps",
        "accel_long_min_mps2",
        "is_hard_braking",
        "min_dist_pedestrian_m",
        "min_dist_vehicle_m",
        "min_dist_cyclist_m",
        "n_peds_within_10m",
        "is_night",
        "is_rain",
        "preset_tags",
        "in_curated_set",
    ):
        assert col in events.columns
    assert row["speed_mps"] == 8.0
    assert row["accel_long_min_mps2"] == -7.5
    assert row["min_dist_vehicle_m"] == 30.0
    assert bool(row["in_curated_set"]) is False  # a1 was never staged


def test_neighbors_ordered_and_na_at_scene_edges(tmp_path: Path) -> None:
    events = _build(tmp_path)

    # b1 is scene B's earliest-timestamp frame (the "edge start") and survives (it's
    # the fast-cyclist event) -> its t_minus neighbors must be NA...
    b1 = events.loc[events["sample_data_token"] == "b1_sdt"].iloc[0]
    assert pd.isna(b1["t_minus1"]) and pd.isna(b1["t_minus2"])
    assert pd.isna(b1["speed_t_minus1"]) and pd.isna(b1["accel_t_minus1"])
    # ...but its t_plus neighbors are real, timestamp-ordered tokens: b0 (next),
    # b_mid (the one after) — proving re-sort-by-timestamp overrode the shuffled
    # write order (b0 was written to parquet chronologically LAST among b0/b1/b_mid).
    assert b1["t_plus1"] == "b0_sdt"
    assert b1["t_plus2"] == "b_mid_sdt"
    assert b1["speed_t_plus1"] == 7.0  # b0's ego speed_mps
    assert b1["accel_t_plus1"] == -6.0  # b0's accel_long_min_mps2

    # b3 is scene B's latest-timestamp frame (the "edge end") and survives (the
    # night-pedestrians event) -> its t_plus neighbors must be NA...
    b3 = events.loc[events["sample_data_token"] == "b3_sdt"].iloc[0]
    assert pd.isna(b3["t_plus1"]) and pd.isna(b3["t_plus2"])
    # ...but its t_minus neighbors are real: b2 (previous), b_mid (before that).
    assert b3["t_minus1"] == "b2_sdt"
    assert b3["t_minus2"] == "b_mid_sdt"


def test_null_hard_braking_treated_false(tmp_path: Path) -> None:
    events = _build(tmp_path)
    # a3 has is_hard_braking=None AND a pedestrian at 5m (< near_dist_m) — if null
    # were ever coerced to True it would join the flagship set and inflate the
    # pre-cap count past 1, which _build's flagship_expected=1 would already catch.
    # Confirm directly too: a3 carries zero tags (null -> not hard braking, and it
    # has no other qualifying property), so it must be entirely absent.
    assert "a3_sdt" not in set(events["sample_data_token"])
    flagship_tokens = set(
        events.loc[
            events["preset_tags"].apply(lambda t: "hard_braking_near_pedestrians" in t),
            "sample_data_token",
        ]
    )
    assert flagship_tokens == {"a1_sdt"}


def test_model_presets_only_from_staging_val(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # With staging: fn_pedestrians_night tags exactly a2 (curated val, night,
    # baseline-unmatched pedestrian); low_conf_braking tags exactly b0 (curated val,
    # hard braking, a baseline low_conf prediction). The decoys in _write_staging
    # (graph_rate_night's low_conf pred on a2, the TP pedestrian on b0) must not
    # leak into either tag.
    events = _build(tmp_path, with_staging=True)
    assert _tags_of(events, "a2_sdt") == ["fn_pedestrians_night"]
    assert _tags_of(events, "b0_sdt") == ["low_conf_braking"]
    assert bool(events.loc[events["sample_data_token"] == "a2_sdt", "in_curated_set"].iloc[0])
    assert bool(events.loc[events["sample_data_token"] == "b0_sdt", "in_curated_set"].iloc[0])

    # Without staging: both model presets are skipped entirely (a2/b0 carry no
    # other tag, so they vanish from the output along with the warning).
    with caplog.at_level(logging.WARNING, logger="nuscenes_data_engine"):
        events_no_staging = _build(tmp_path, with_staging=False)
    assert "a2_sdt" not in set(events_no_staging["sample_data_token"])
    assert "b0_sdt" not in set(events_no_staging["sample_data_token"])
    assert any("model-result presets" in rec.message for rec in caplog.records)
    all_tags = {tag for tags in events_no_staging["preset_tags"] for tag in tags}
    assert "fn_pedestrians_night" not in all_tags
    assert "low_conf_braking" not in all_tags
    assert not events_no_staging["in_curated_set"].any()


def test_cap_and_rank_columns(tmp_path: Path) -> None:
    # 5 frames in one scene, all tagged fast_cyclists (speed>=10, cyclist at 5m).
    # c0 ALSO qualifies for rain_vru (is_rain + a pedestrian at 5m) — it is the
    # slowest fast_cyclists candidate, so a cap of 2 must drop its fast_cyclists tag
    # while it keeps rain_vru (the "capped out of one preset keeps other tags" case).
    frames = [
        {
            "sdt": f"c{i}_sdt",
            "st": f"c{i}_st",
            "scene_token": "sceneC",
            "scene_name": "scene-C",
            "timestamp": 3000 + i,
            "is_night": False,
            "is_rain": i == 0,
            "hard_braking": False,
            "accel": -0.5,
            "speed": 10.0 + i,
            "cyclists": [5.0],
            **({"pedestrians": [5.0]} if i == 0 else {}),
        }
        for i in range(5)
    ]
    events = _build(
        tmp_path,
        with_staging=False,
        presets_cfg={**PRESETS_CFG, "cap_per_preset": 2},
        flagship_expected=None,
        frames=frames,
    )

    # c1, c2 are capped out of fast_cyclists with no other tag -> absent entirely.
    remaining = set(events["sample_data_token"])
    assert "c1_sdt" not in remaining and "c2_sdt" not in remaining

    # c4 (speed 14, fastest) and c3 (speed 13) are the top-2 -> kept, ranked 1 and 2.
    c4 = events.loc[events["sample_data_token"] == "c4_sdt"].iloc[0]
    c3 = events.loc[events["sample_data_token"] == "c3_sdt"].iloc[0]
    assert c4["preset_tags"] == ["fast_cyclists"]
    assert c3["preset_tags"] == ["fast_cyclists"]
    assert int(c4["preset_rank_fast_cyclists"]) == 1
    assert int(c3["preset_rank_fast_cyclists"]) == 2

    # c0 (speed 10, slowest -> capped out of fast_cyclists) keeps rain_vru instead.
    c0 = events.loc[events["sample_data_token"] == "c0_sdt"].iloc[0]
    assert c0["preset_tags"] == ["rain_vru"]
    assert pd.isna(c0["preset_rank_fast_cyclists"])
    assert int(c0["preset_rank_rain_vru"]) == 1


def test_only_tagged_events_exported_and_deterministic(tmp_path: Path) -> None:
    events_a = _build(tmp_path)
    events_b = _build(tmp_path)

    # untagged fixture rows (a0, a3, a4, b_mid) are absent from the output.
    tokens = set(events_a["sample_data_token"])
    assert tokens == {"a1_sdt", "a2_sdt", "b0_sdt", "b1_sdt", "b2_sdt", "b3_sdt"}
    assert all(len(t) > 0 for t in events_a["preset_tags"])  # no zero-tag rows survive

    pd.testing.assert_frame_equal(events_a, events_b)
