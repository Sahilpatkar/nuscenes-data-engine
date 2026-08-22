"""Tests for the demo artifact builder (pure exporters; no torch, no network)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from nuscenes_data_engine.demo.exporters import export_overview


@pytest.fixture()
def tiny_inputs(tmp_path: Path) -> dict[str, Path]:
    """Miniature processed/ and active_learning/ trees with known numbers."""
    processed = tmp_path / "processed"
    al = tmp_path / "active_learning"
    processed.mkdir()
    al.mkdir()
    pd.DataFrame(
        {
            "sample_data_token": ["s1", "s2", "s3"],
            # Phase 5: demo/events.py::build_events reads samples.parquet too
            # (CAM_FRONT-filtered, joined to canbus/ego_pose by sample_token) --
            # s1/s2 double as their own sample_token here (this fixture's tiny
            # scale never needs the real per-channel token distinction) and are
            # CAM_FRONT so build_events sees exactly canbus/ego_pose's covered set
            # (2 rows, both already present below); s3 is a different channel so
            # it's invisible to build_events without needing a matching canbus/
            # ego_pose row of its own.
            "sample_token": ["s1", "s2", "s3"],
            "channel": ["CAM_FRONT", "CAM_FRONT", "CAM_BACK"],
            "scene_token": ["sceneX", "sceneX", "sceneX"],
            "scene_name": ["scene-X", "scene-X", "scene-X"],
            "timestamp": [1000, 1001, 1002],
            "is_night": [False, False, False],
            "is_rain": [False, False, False],
        }
    ).to_parquet(processed / "samples.parquet")
    # Phase 7 (Task 2): sample_data_token/category_group -- export_weaksup's
    # rejected-side recipe reads exactly these two columns. All 4 rows are s1's;
    # 3 carry a detector category_group (pedestrian/car/pedestrian) and the 4th is
    # None (a non-detector class, e.g. a traffic cone) -- ignored by the recipe, so
    # s1's detector count is 3, matching _write_demo_config's random_pseudo_
    # summary.json mean_gt_boxes_per_accepted_frame=3.0 below (accepted=["s1"]).
    pd.DataFrame(
        {
            "sample_token": ["s1"] * 4,
            "sample_data_token": ["s1"] * 4,
            "category_group": ["pedestrian", "car", "pedestrian", None],
        }
    ).to_parquet(processed / "annotations.parquet")
    pd.DataFrame(
        {
            # annotation_token: Task 1 (Phase 3) needs a joinable token on every
            # annotations_3d row for the gt_boxes distance/size enrichment (proven
            # 100% joinable on real data) -- f1..f3 are the flagship rows' tokens,
            # otherwise unused by the flagship SQL itself.
            "annotation_token": ["f1", "f2", "f3"],
            "sample_token": ["s1", "s1", "s2"],
            "category_group": ["pedestrian", "pedestrian", "car"],
            "distance_to_ego_m": [5.0, 20.0, 3.0],
        }
    ).to_parquet(processed / "annotations_3d.parquet")
    pd.DataFrame(
        {
            "sample_token": ["s1", "s2"],
            "has_canbus": [True, True],
            "can_vel_mps": [10.0, 5.0],
            "can_speed_kmh": [36.0, 18.0],  # /3.6 == can_vel_mps here, by construction
            "is_hard_braking": [True, False],
            # Phase 5: demo/events.py::build_events also reads accel_long_min_mps2
            # (the flagship preset's severity key) -- s1's magnitude is deliberately
            # a strong-braking value, consistent with is_hard_braking=True.
            "accel_long_min_mps2": [-7.5, -0.5],
        }
    ).to_parquet(processed / "canbus.parquet")
    pd.DataFrame(
        {"sample_token": ["s1", "s2"], "speed_mps": [10.1, 4.9]}
    ).to_parquet(processed / "ego_pose.parquet")
    # Phase 7 (Task 1): every arm carries night.per_class/precision/recall and
    # slices (day/rain/clear mAP50-95) -- the shape a well-formed results.json has
    # in production (see the plan's "Verified facts") -- plus n_boxes on the two
    # weak arms only, matching the real file. round_order is the insertion order
    # below (baseline=0, random=1, weak_random=2, graph_rate_night=3,
    # weak_graph_rate_night=4); the arm-family reshape is order-independent so the
    # arms don't need to be re-ordered to match results.json's real 13-arm order.
    (al / "results.json").write_text(
        json.dumps(
            {
                "baseline": {
                    "n_train_images": 100,
                    "overall": {"mAP50-95": 0.20, "mAP50": 0.35},
                    "night": {
                        "mAP50-95": 0.10, "mAP50": 0.18, "precision": 0.50, "recall": 0.40,
                        "per_class": {"pedestrian": 0.0826},
                    },
                    "slices": {
                        "time_of_day/day": {"mAP50-95": 0.22},
                        "weather/rain": {"mAP50-95": 0.15},
                        "weather/clear": {"mAP50-95": 0.21},
                    },
                },
                "random": {
                    "n_train_images": 115,
                    "overall": {"mAP50-95": 0.21, "mAP50": 0.36},
                    "night": {
                        "mAP50-95": 0.099, "mAP50": 0.17, "precision": 0.51, "recall": 0.41,
                        "per_class": {"pedestrian": 0.081},
                    },
                    "slices": {
                        "time_of_day/day": {"mAP50-95": 0.23},
                        "weather/rain": {"mAP50-95": 0.155},
                        "weather/clear": {"mAP50-95": 0.215},
                    },
                },
                "weak_random": {
                    "n_train_images": 110,
                    "overall": {"mAP50-95": 0.2018, "mAP50": 0.34},
                    "night": {
                        "mAP50-95": 0.095, "mAP50": 0.16, "precision": 0.49, "recall": 0.39,
                        "per_class": {"pedestrian": 0.06},
                    },
                    "slices": {
                        "time_of_day/day": {"mAP50-95": 0.21},
                        "weather/rain": {"mAP50-95": 0.14},
                        "weather/clear": {"mAP50-95": 0.20},
                    },
                    "n_boxes": 1942,
                },
                "graph_rate_night": {
                    "n_train_images": 130,
                    "overall": {"mAP50-95": 0.22, "mAP50": 0.38},
                    "night": {
                        "mAP50-95": 0.1101, "mAP50": 0.20, "precision": 0.55, "recall": 0.45,
                        "per_class": {"pedestrian": 0.1171},
                    },
                    "slices": {
                        "time_of_day/day": {"mAP50-95": 0.25},
                        "weather/rain": {"mAP50-95": 0.18},
                        "weather/clear": {"mAP50-95": 0.24},
                    },
                },
                "weak_graph_rate_night": {
                    "n_train_images": 122,
                    "overall": {"mAP50-95": 0.205, "mAP50": 0.35},
                    "night": {
                        "mAP50-95": 0.09, "mAP50": 0.16, "precision": 0.48, "recall": 0.38,
                        "per_class": {"pedestrian": 0.0203},
                    },
                    "slices": {
                        "time_of_day/day": {"mAP50-95": 0.20},
                        "weather/rain": {"mAP50-95": 0.13},
                        "weather/clear": {"mAP50-95": 0.19},
                    },
                    "n_boxes": 1500,
                },
            }
        )
    )
    # Phase 7 (Task 1): al_communities.parquet's inputs -- 2 communities whose
    # quotas each sum to n_mine=3 (the AL config below), identical size/mass/
    # night_members across both arms' files (only quota differs, matching the real
    # communities_graph_rate*.json pair).
    for arm, quotas in (("graph_rate", [2, 1]), ("graph_rate_night", [1, 2])):
        records = [
            {"community": 0, "size": 10, "mass": 5.0, "night_members": 2, "quota": quotas[0]},
            {"community": 1, "size": 8, "mass": 3.0, "night_members": 1, "quota": quotas[1]},
        ]
        (al / f"communities_{arm}.json").write_text(json.dumps(records))
    al_config = tmp_path / "active_learning.yaml"
    al_config.write_text(yaml.safe_dump({"mining": {"n_mine": 3}}))
    # Phase 7 (Task 1): weak_verdict/al_selected_by's arm parquets -- weak_arm ==
    # al_arm == "graph_rate_night" in every fixture here (matches configs/demo.yaml
    # today), so one candidate/accepted/selected trio covers both roles. "v0" is
    # the token every curation-staging helper below curates, so it comes out
    # weak_verdict="accepted" (in both the candidate pool and accepted) and
    # al_selected_by="graph_rate_night" (in the selected set).
    pd.DataFrame({"sample_data_token": ["v0"]}).to_parquet(al / "graph_rate_night.parquet")
    pd.DataFrame({"sample_data_token": ["v0"]}).to_parquet(
        al / "graph_rate_night_accepted.parquet"
    )
    # Phase 7 (Task 2): export_weaksup now reads <arm>.parquet/<arm>_accepted.
    # parquet for EVERY *_pseudo_summary.json arm -- _write_demo_config below
    # always writes a "random" summary, so every build_config-based test needs
    # these two files. candidates = s1 (accepted) + s2 (rejected, no annotations
    # rows at all -> 0 detector GT boxes by the reindex-fill-0 rule).
    pd.DataFrame({"sample_data_token": ["s1", "s2"]}).to_parquet(al / "random.parquet")
    pd.DataFrame({"sample_data_token": ["s1"]}).to_parquet(al / "random_accepted.parquet")
    # Phase 7 (Task 2): weak_labels/vlm_counts' own inputs -- present for every
    # curation-included build_config test (weak_arm == "graph_rate_night" in every
    # such fixture), one row each so the two exports have something to filter.
    pd.DataFrame(
        {
            "sample_data_token": ["v0"],
            "category_group": ["car"],
            "x_min": [100.0], "y_min": [100.0], "x_max": [200.0], "y_max": [200.0],
            "score": [0.75],
        }
    ).to_parquet(al / "graph_rate_night_pseudo_labels.parquet")
    (al / "autolabel_weak").mkdir()
    pd.DataFrame(
        {
            "sample_data_token": ["v0"],
            "model": ["qwen2.5-vl"],
            "parse_status": ["ok"],
            "time_of_day": ["night"],
            "weather": ["clear"],
            "hazards": ["[]"],
            "notable_conditions": ["[]"],
            "label_confidence": ["high"],
            "cars": [1.0], "trucks": [0.0], "buses": [0.0], "trailers": [0.0],
            "construction_vehicles": [0.0], "motorcycles": [0.0], "bicycles": [0.0],
            "pedestrians": [1.0], "traffic_cones": [0.0], "barriers": [0.0],
        }
    ).to_parquet(al / "autolabel_weak" / "labels.parquet")
    return {"processed": processed, "al": al, "al_config": al_config}


def test_export_overview_derives_every_number(tmp_path: Path, tiny_inputs: dict[str, Path]) -> None:
    out = tmp_path / "demo_data"
    metrics = export_overview(
        processed_dir=tiny_inputs["processed"],
        al_dir=tiny_inputs["al"],
        out_dir=out,
        flagship_cypher={"count": 30, "source": "docs/GRAPH.md"},
    )
    on_disk = json.loads((out / "overview_metrics.json").read_text())
    assert on_disk == metrics
    assert metrics["scale"]["images"] == 3
    assert metrics["scale"]["boxes_2d"] == 4
    assert metrics["scale"]["objects_3d"] == 3
    assert metrics["scale"]["canbus_rows"] == 2
    # flagship: s1 is hard-braking AND has a pedestrian within 10 m -> exactly 1
    assert metrics["flagship"]["sql"] == 1
    assert metrics["flagship"]["cypher"] == 30
    assert metrics["flagship"]["cypher_source"] == "docs/GRAPH.md"
    # CAN validation: corr of (can_speed_kmh/3.6 == 10.0, 5.0) vs (10.1, 4.9) is
    # ~1.0, derived not hardcoded
    assert 0.99 <= metrics["can_speed_r"] <= 1.0
    # headline results derived from results.json
    assert metrics["results"]["best_night_arm"] == "graph_rate_night"
    assert metrics["results"]["best_night_delta"] == pytest.approx(0.0101, abs=1e-6)
    # weak-supervision retention: per-pair, with an attributed headline (the
    # documented Overview headline is the weak_random/random pair, not whichever
    # weak_ arm happens to be present)
    weak_retention = metrics["results"]["weak_retention"]
    assert weak_retention["headline_arm"] == "random"
    assert weak_retention["headline"] == pytest.approx(0.18, abs=1e-6)
    assert weak_retention["by_base_arm"]["graph_rate_night"] == pytest.approx(0.25, abs=1e-6)


def test_export_overview_excludes_negative_gt_gain_pairs(
    tmp_path: Path, tiny_inputs: dict[str, Path]
) -> None:
    """A base arm that regressed vs baseline gets no weak-retention ratio.

    `if gt_gain:` (float-truthiness) only excluded an exactly-zero gain; a negative
    gt_gain would still produce a technically-computed but meaningless ratio (there is
    no such thing as "retention" of a regression). `if gt_gain > 0:` excludes it.
    """
    results_path = tiny_inputs["al"] / "results.json"
    results = json.loads(results_path.read_text())
    # graph_rate_night now UNDERPERFORMS baseline overall mAP (0.20) -> negative gt_gain.
    results["graph_rate_night"]["overall"]["mAP50-95"] = 0.15
    results_path.write_text(json.dumps(results))

    metrics = export_overview(
        processed_dir=tiny_inputs["processed"],
        al_dir=tiny_inputs["al"],
        out_dir=tmp_path / "demo_data",
        flagship_cypher={"count": 30, "source": "docs/GRAPH.md"},
    )
    weak_retention = metrics["results"]["weak_retention"]
    assert "graph_rate_night" not in weak_retention["by_base_arm"]
    # the headline (random) pair is a different base arm and is unaffected
    assert weak_retention["headline"] == pytest.approx(0.18, abs=1e-6)


def test_export_overview_missing_input_raises(tmp_path: Path, tiny_inputs: dict[str, Path]) -> None:
    (tiny_inputs["processed"] / "canbus.parquet").unlink()
    with pytest.raises(ValueError, match=r"canbus\.parquet"):
        export_overview(
            processed_dir=tiny_inputs["processed"],
            al_dir=tiny_inputs["al"],
            out_dir=tmp_path / "demo_data",
            flagship_cypher={"count": 30, "source": "docs/GRAPH.md"},
        )


_AL_RESULTS_COLUMNS = [
    "arm", "n_train_images", "overall_map5095", "night_map5095",
    "delta_overall", "delta_night",
    "round_order", "family", "overall_map50", "night_map50",
    "night_precision", "night_recall", "night_ped_map5095",
    "day_map5095", "rain_map5095", "clear_map5095",
    "n_boxes", "n_scenes", "night_share", "rain_share", "val_images",
]


def test_export_al_results_reshapes_and_sorts(tmp_path: Path, tiny_inputs: dict[str, Path]) -> None:
    from nuscenes_data_engine.demo.exporters import export_al_results

    out = tmp_path / "demo_data"
    df = export_al_results(
        al_dir=tiny_inputs["al"], out_dir=out, processed_dir=tiny_inputs["processed"]
    )
    on_disk = pd.read_parquet(out / "active_learning_results.parquet")
    pd.testing.assert_frame_equal(df, on_disk)
    assert list(df.columns) == _AL_RESULTS_COLUMNS
    assert list(df["arm"]) == sorted(df["arm"])          # deterministic order
    row = df.set_index("arm").loc["graph_rate_night"]
    assert row["delta_night"] == pytest.approx(0.0101, abs=1e-6)
    baseline = df.set_index("arm").loc["baseline"]
    assert baseline["delta_overall"] == 0.0


def test_export_al_results_missing_results_json_raises(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.exporters import export_al_results

    al = tmp_path / "active_learning"
    al.mkdir()
    with pytest.raises(ValueError, match=r"results\.json"):
        export_al_results(al_dir=al, out_dir=tmp_path / "demo_data", processed_dir=tmp_path)


def test_al_results_keeps_round_order_and_family(
    tmp_path: Path, tiny_inputs: dict[str, Path]
) -> None:
    """round_order is results.json's own insertion index (survives the arm-sorted
    output); family collapses baseline/weak_* and leaves every other arm as its own
    singleton family (spec §2)."""
    from nuscenes_data_engine.demo.exporters import export_al_results

    df = export_al_results(
        al_dir=tiny_inputs["al"], out_dir=tmp_path / "demo_data",
        processed_dir=tiny_inputs["processed"],
    ).set_index("arm")
    # insertion order in tiny_inputs' results.json: baseline, random, weak_random,
    # graph_rate_night, weak_graph_rate_night
    assert df.loc["baseline", "round_order"] == 0
    assert df.loc["random", "round_order"] == 1
    assert df.loc["weak_random", "round_order"] == 2
    assert df.loc["graph_rate_night", "round_order"] == 3
    assert df.loc["weak_graph_rate_night", "round_order"] == 4
    assert df.loc["baseline", "family"] == "baseline"
    assert df.loc["weak_random", "family"] == "weak"
    assert df.loc["weak_graph_rate_night", "family"] == "weak"
    assert df.loc["random", "family"] == "random"
    assert df.loc["graph_rate_night", "family"] == "graph_rate_night"


def test_al_results_night_pedestrian_and_slices_are_derived(
    tmp_path: Path, tiny_inputs: dict[str, Path]
) -> None:
    from nuscenes_data_engine.demo.exporters import export_al_results

    df = export_al_results(
        al_dir=tiny_inputs["al"], out_dir=tmp_path / "demo_data",
        processed_dir=tiny_inputs["processed"],
    ).set_index("arm")
    grn = df.loc["graph_rate_night"]
    assert grn["night_ped_map5095"] == pytest.approx(0.1171, abs=1e-6)
    assert grn["day_map5095"] == pytest.approx(0.25, abs=1e-6)
    assert grn["rain_map5095"] == pytest.approx(0.18, abs=1e-6)
    assert grn["clear_map5095"] == pytest.approx(0.24, abs=1e-6)
    # n_boxes: NA for a non-weak arm, present for the two weak arms.
    assert pd.isna(df.loc["random", "n_boxes"])
    assert df.loc["weak_random", "n_boxes"] == 1942
    assert df.loc["weak_graph_rate_night", "n_boxes"] == 1500


def test_al_results_carries_val_images_when_results_json_has_it(
    tmp_path: Path, tiny_inputs: dict[str, Path]
) -> None:
    """``val_images`` is results.json's own per-arm val-split size (6019 on every
    real arm) -- the Active Learning page's Evaluation beat names it. Nullable
    Int64: an arm (or a whole results.json) without it gets NA, not 0, and not a
    KeyError."""
    from nuscenes_data_engine.demo.exporters import export_al_results

    al = tiny_inputs["al"]
    results_path = al / "results.json"
    results = json.loads(results_path.read_text())
    results["baseline"]["val_images"] = 6019
    results_path.write_text(json.dumps(results))

    df = export_al_results(
        al_dir=al, out_dir=tmp_path / "demo_data", processed_dir=tiny_inputs["processed"]
    ).set_index("arm")
    assert df["val_images"].dtype == "Int64"
    assert df.loc["baseline", "val_images"] == 6019
    # every other arm in this fixture carries no val_images at all
    assert pd.isna(df.loc["random", "val_images"])


def test_al_results_composition_columns_come_from_arm_composition(
    tmp_path: Path, tiny_inputs: dict[str, Path]
) -> None:
    """n_scenes/night_share/rain_share come from active_learning.report.
    arm_composition for an arm with a token parquet whose tokens actually match
    samples.parquet (random.parquet here, "s1"/"s2" from tiny_inputs' processed/
    samples.parquet); an arm whose only matching tokens aren't real samples rows
    (graph_rate_night.parquet here holds only "v0", the weak_verdict/al_selected_by
    fixture token, which isn't in samples.parquet) ends up with zero matched rows,
    so composition has nothing to report and the column is NA -- same outward
    result as the file being absent entirely. baseline is always NA (arm_
    composition skips it by design)."""
    from nuscenes_data_engine.demo.exporters import export_al_results

    al_dir = tiny_inputs["al"]
    pd.DataFrame({"sample_data_token": ["s1", "s2"]}).to_parquet(al_dir / "random.parquet")

    df = export_al_results(
        al_dir=al_dir, out_dir=tmp_path / "demo_data", processed_dir=tiny_inputs["processed"]
    ).set_index("arm")
    assert df.loc["random", "n_scenes"] == 1        # s1/s2 both scene-X
    assert df.loc["random", "night_share"] == pytest.approx(0.0)
    assert df.loc["random", "rain_share"] == pytest.approx(0.0)
    assert pd.isna(df.loc["baseline", "n_scenes"])
    assert pd.isna(df.loc["graph_rate_night", "n_scenes"])


def _write_communities(
    al_dir: Path, *, arm: str, records: list[dict[str, Any]]
) -> None:
    (al_dir / f"communities_{arm}.json").write_text(json.dumps(records))


def test_al_communities_merges_both_quota_columns_and_checks_sums(
    tmp_path: Path, tiny_inputs: dict[str, Path]
) -> None:
    from nuscenes_data_engine.demo.exporters import export_al_communities

    out = tmp_path / "demo_data"
    df = export_al_communities(
        al_dir=tiny_inputs["al"], out_dir=out,
        arms=("graph_rate", "graph_rate_night"), n_mine=3,
    )
    on_disk = pd.read_parquet(out / "al_communities.parquet")
    pd.testing.assert_frame_equal(df, on_disk)
    assert set(df.columns) >= {
        "community", "size", "night_members", "mass", "quota_graph_rate",
        "quota_graph_rate_night",
    }
    assert list(df["community"]) == sorted(df["community"])
    assert df["quota_graph_rate"].sum() == 3
    assert df["quota_graph_rate_night"].sum() == 3


def test_al_communities_size_mismatch_raises(tmp_path: Path, tiny_inputs: dict[str, Path]) -> None:
    from nuscenes_data_engine.demo.exporters import export_al_communities

    al_dir = tiny_inputs["al"]
    records = json.loads((al_dir / "communities_graph_rate.json").read_text())
    records[0]["size"] = records[0]["size"] + 1     # disagree with the night file
    _write_communities(al_dir, arm="graph_rate", records=records)

    with pytest.raises(ValueError, match=r"community 0"):
        export_al_communities(
            al_dir=al_dir, out_dir=tmp_path / "demo_data",
            arms=("graph_rate", "graph_rate_night"), n_mine=3,
        )


def test_al_communities_quota_sum_mismatch_raises(
    tmp_path: Path, tiny_inputs: dict[str, Path]
) -> None:
    from nuscenes_data_engine.demo.exporters import export_al_communities

    with pytest.raises(ValueError, match=r"sums to 3.*n_mine=4"):
        export_al_communities(
            al_dir=tiny_inputs["al"], out_dir=tmp_path / "demo_data",
            arms=("graph_rate", "graph_rate_night"), n_mine=4,
        )


def _write_al_exemplar_inputs() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A minimal config + manifest + predictions + gt_boxes quad for
    export_al_exemplars: one val token ("good") that satisfies every validation
    clause, including the 2026-08-21 amendment (a visible GT box the arm claims at
    ``tp`` while the baseline only claims it at ``low_conf``).

    ``fixes_fn_vs_baseline_graph_rate_night`` is deliberately still present and
    True in the manifest -- the real curated manifest carries it, and it is what
    the validation USED to key off; the tests below prove the box-level rule, not
    the flag, is what decides now.
    """
    config = {
        "al": {"arm": "graph_rate_night", "baseline": "baseline", "exemplar_tokens": ["good"]},
        "models": {"baseline": {}, "graph_rate_night": {}},
    }
    manifest = pd.DataFrame({
        "sample_data_token": ["good"], "split": ["val"],
        "fixes_fn_vs_baseline_graph_rate_night": [True],
    })
    predictions = pd.DataFrame({
        "sample_data_token": ["good", "good"],
        "model": ["baseline", "graph_rate_night"],
        "status": ["low_conf", "tp"],
        "conf": [0.22, 0.61],
        "matched_annotation_token": ["g1", "g1"],
    })
    gt_boxes = pd.DataFrame({
        "annotation_token": ["g1"], "sample_data_token": ["good"],
        "category_group": ["pedestrian"], "below_visibility_min": [False],
    })
    return config, manifest, predictions, gt_boxes


def test_al_exemplars_validation(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.exporters import export_al_exemplars

    config, manifest, predictions, gt_boxes = _write_al_exemplar_inputs()
    out = tmp_path / "demo_data"

    # happy path first, to prove the fixture itself is valid before each failure
    # variant below mutates one thing at a time.
    tokens = export_al_exemplars(
        config=config, manifest=manifest, predictions=predictions, gt_boxes=gt_boxes,
        out_dir=out, curation_present=True,
    )
    assert tokens == ["good"]
    on_disk = json.loads((out / "al_exemplars.json").read_text())
    assert on_disk == {"arm": "graph_rate_night", "baseline": "baseline", "tokens": ["good"]}

    # non-val token
    bad_config = {**config, "al": {**config["al"], "exemplar_tokens": ["ghost"]}}
    with pytest.raises(ValueError, match="ghost"):
        export_al_exemplars(
            config=bad_config, manifest=manifest, predictions=predictions, gt_boxes=gt_boxes,
            out_dir=out, curation_present=True,
        )

    # missing predictions for one configured model
    partial_predictions = predictions.loc[predictions["model"] == "baseline"]
    with pytest.raises(ValueError, match="graph_rate_night"):
        export_al_exemplars(
            config=config, manifest=manifest, predictions=partial_predictions,
            gt_boxes=gt_boxes, out_dir=out, curation_present=True,
        )

    # empty list + curation absent -> ok, writes an empty tokens list
    empty_config = {**config, "al": {**config["al"], "exemplar_tokens": []}}
    tokens = export_al_exemplars(
        config=empty_config, manifest=manifest, predictions=predictions, gt_boxes=gt_boxes,
        out_dir=out, curation_present=False,
    )
    assert tokens == []
    on_disk = json.loads((out / "al_exemplars.json").read_text())
    assert on_disk == {"arm": "graph_rate_night", "baseline": "baseline", "tokens": []}

    # empty list + curation present -> error
    with pytest.raises(ValueError, match="empty"):
        export_al_exemplars(
            config=empty_config, manifest=manifest, predictions=predictions, gt_boxes=gt_boxes,
            out_dir=out, curation_present=True,
        )


def test_al_exemplars_require_a_confident_arm_box_the_baseline_missed(tmp_path: Path) -> None:
    """The 2026-08-21 amendment (spec §2): an exemplar is valid iff at least one
    VISIBLE GT box has the arm's prediction at ``tp`` while the baseline's matched
    prediction is absent or ``low_conf``.

    The old rule -- ``fixes_fn_vs_<baseline>_<arm> == True`` -- admitted frames whose
    only "fix" was a far-away car the arm itself only claims at conf 0.07-0.35
    (``matched_<model>`` counts a low-confidence claim as a match, so the flag fires
    on a low-conf-to-low-conf pair). Those are not demo-worthy before/afters, so the
    box-level rule replaces the flag entirely: the flag is neither necessary (last
    case) nor sufficient (first three).
    """
    from nuscenes_data_engine.demo.exporters import export_al_exemplars

    config, manifest, predictions, gt_boxes = _write_al_exemplar_inputs()
    out = tmp_path / "demo_data"

    # (1) the arm only claims the box at low confidence -> not an upgrade worth showing
    low_conf_arm = predictions.copy()
    low_conf_arm.loc[low_conf_arm["model"] == "graph_rate_night", "status"] = "low_conf"
    with pytest.raises(ValueError, match="good"):
        export_al_exemplars(
            config=config, manifest=manifest, predictions=low_conf_arm, gt_boxes=gt_boxes,
            out_dir=out, curation_present=True,
        )

    # (2) the baseline already detects the box confidently -> nothing was fixed
    baseline_tp = predictions.copy()
    baseline_tp.loc[baseline_tp["model"] == "baseline", "status"] = "tp"
    with pytest.raises(ValueError, match="good"):
        export_al_exemplars(
            config=config, manifest=manifest, predictions=baseline_tp, gt_boxes=gt_boxes,
            out_dir=out, curation_present=True,
        )

    # (3) the only qualifying box is below the visibility floor -> not shown on the
    # page at all (the pages drop those rows everywhere), so it cannot justify one
    invisible = gt_boxes.copy()
    invisible["below_visibility_min"] = [True]
    with pytest.raises(ValueError, match="good"):
        export_al_exemplars(
            config=config, manifest=manifest, predictions=predictions, gt_boxes=invisible,
            out_dir=out, curation_present=True,
        )

    # (4) the flag being False no longer matters -- the box-level upgrade is the rule
    flag_false = manifest.copy()
    flag_false["fixes_fn_vs_baseline_graph_rate_night"] = [False]
    tokens = export_al_exemplars(
        config=config, manifest=flag_false, predictions=predictions, gt_boxes=gt_boxes,
        out_dir=out, curation_present=True,
    )
    assert tokens == ["good"]


def test_frame_manifest_gains_weak_verdict_and_al_selected_by(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.build import _add_al_selection_columns

    al_dir = tmp_path / "active_learning"
    al_dir.mkdir()
    # "accepted": weak arm's candidate pool AND accepted set.
    # "candidate_only": weak arm's candidate pool but not accepted -> "rejected".
    # "other": in neither -> NA.
    pd.DataFrame({"sample_data_token": ["accepted", "candidate_only"]}).to_parquet(
        al_dir / "graph_rate_night.parquet"
    )
    pd.DataFrame({"sample_data_token": ["accepted"]}).to_parquet(
        al_dir / "graph_rate_night_accepted.parquet"
    )
    manifest = pd.DataFrame(
        {"sample_data_token": ["accepted", "candidate_only", "other"], "split": ["val"] * 3}
    )

    out = _add_al_selection_columns(
        manifest, al_dir=al_dir, weak_arm="graph_rate_night", al_arm="graph_rate_night"
    )
    result = out.set_index("sample_data_token")
    assert result.loc["accepted", "weak_verdict"] == "accepted"
    assert result.loc["candidate_only", "weak_verdict"] == "rejected"
    assert pd.isna(result.loc["other", "weak_verdict"])
    # al_selected_by: graph_rate_night.parquet is also the al_arm's selected set
    # here -- both accepted and candidate_only are in it, other is not.
    assert result.loc["accepted", "al_selected_by"] == "graph_rate_night"
    assert result.loc["candidate_only", "al_selected_by"] == "graph_rate_night"
    assert pd.isna(result.loc["other", "al_selected_by"])
    assert result["weak_verdict"].dtype == "string"
    assert result["al_selected_by"].dtype == "string"


def test_export_weaksup_empty_dir_raises(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.exporters import export_weaksup

    al = tmp_path / "active_learning"
    al.mkdir()
    with pytest.raises(ValueError, match=r"no \*_pseudo_summary\.json"):
        export_weaksup(
            al_dir=al, out_dir=tmp_path / "demo_data", processed_dir=tmp_path / "processed"
        )


def test_export_weaksup_reads_summaries(tmp_path: Path, tiny_inputs: dict[str, Path]) -> None:
    from nuscenes_data_engine.demo.exporters import export_weaksup

    al = tiny_inputs["al"]
    # mean_gt_boxes_per_accepted_frame=3.0 matches tiny_inputs' own fixture data
    # (random.parquet=["s1","s2"], random_accepted.parquet=["s1"], s1 has 3
    # detector-class annotations.parquet rows) -- export_weaksup pins the two
    # together, so this can't be an arbitrary number any more (see
    # test_weaksup_rejected_side_uses_detector_classes_only for the recipe itself).
    (al / "random_pseudo_summary.json").write_text(json.dumps({
        "arm": "random", "n_candidates": 1500, "n_accepted": 958, "retention": 0.639,
        "n_boxes": 1942, "mean_boxes_per_accepted_frame": 2.0,
        "mean_gt_boxes_per_accepted_frame": 3.0,
        "conf": 0.5, "tolerance": 1, "n_no_label": 10, "n_unparsed": 3,
        "rejected_by_class": {}, "accepted_mutual_zero_by_class": {},
    }))
    out = tmp_path / "demo_data"
    df = export_weaksup(al_dir=al, out_dir=out, processed_dir=tiny_inputs["processed"])
    assert (out / "weak_supervision_results.parquet").is_file()
    row = df.set_index("arm").loc["random"]
    assert row["n_accepted"] == 958
    assert row["verifier_retention"] == pytest.approx(0.639)
    assert row["n_rejected"] == 1500 - 958
    assert row["tolerance"] == 1
    assert row["conf"] == pytest.approx(0.5)
    assert row["n_unparsed"] == 3
    # arms without a summary file are simply absent, not an error
    assert "graph_rate_night" not in set(df["arm"])


def _write_weaksup_rejected_side_fixture(tmp_path: Path) -> dict[str, Path]:
    """5 candidate tokens (c1..c5) for arm "weakarm": c1/c2 accepted, c3/c4/c5
    rejected. Detector-class GT boxes per token: c1=2, c2=4 (accepted mean=3.0,
    matching the summary below), c3=1 (+1 non-detector row that must be ignored),
    c4=3, c5=0 (no annotations.parquet rows at all -> reindex fill 0)."""
    processed = tmp_path / "processed"
    al = tmp_path / "active_learning"
    processed.mkdir()
    al.mkdir()
    pd.DataFrame(
        {
            "sample_data_token": [
                "c1", "c1",
                "c2", "c2", "c2", "c2",
                "c3", "c3",
                "c4", "c4", "c4",
            ],
            "category_group": [
                "car", "pedestrian",
                "car", "car", "pedestrian", "bicycle",
                "car", None,
                "truck", "bus", "car",
            ],
        }
    ).to_parquet(processed / "annotations.parquet")
    pd.DataFrame({"sample_data_token": ["c1", "c2", "c3", "c4", "c5"]}).to_parquet(
        al / "weakarm.parquet"
    )
    pd.DataFrame({"sample_data_token": ["c1", "c2"]}).to_parquet(al / "weakarm_accepted.parquet")
    (al / "weakarm_pseudo_summary.json").write_text(json.dumps({
        "arm": "weakarm", "n_candidates": 5, "n_accepted": 2, "retention": 0.4,
        "n_boxes": 10, "mean_boxes_per_accepted_frame": 2.5,
        "mean_gt_boxes_per_accepted_frame": 3.0,   # (2 + 4) / 2
        "conf": 0.5, "tolerance": 1, "n_no_label": 0, "n_unparsed": 0,
        "rejected_by_class": {}, "accepted_mutual_zero_by_class": {},
    }))
    return {"processed": processed, "al": al}


def test_weaksup_rejected_side_uses_detector_classes_only(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.exporters import export_weaksup

    fixture = _write_weaksup_rejected_side_fixture(tmp_path)
    out = tmp_path / "demo_data"
    df = export_weaksup(al_dir=fixture["al"], out_dir=out, processed_dir=fixture["processed"])
    row = df.set_index("arm").loc["weakarm"]

    assert row["n_rejected"] == 3   # n_candidates(5) - n_accepted(2)
    # rejected = c3(1, non-detector ignored) + c4(3) + c5(0, missing token) -> 4/3
    assert row["gt_boxes_per_rejected_frame"] == pytest.approx(4 / 3, abs=1e-9)
    # candidate mean over all 5: c1=2, c2=4, c3=1, c4=3, c5=0 -> 10/5 = 2.0
    assert row["gt_boxes_per_candidate_frame"] == pytest.approx(2.0, abs=1e-9)
    # the accepted-side mean recomputed by the same recipe pins the summary's own
    # mean_gt_boxes_per_accepted_frame (both are 3.0 here by construction).
    assert row["gt_boxes_per_accepted_frame"] == pytest.approx(3.0, abs=1e-9)


def test_weaksup_accepted_mean_mismatch_raises(tmp_path: Path) -> None:
    """A summary whose mean_gt_boxes_per_accepted_frame disagrees with the
    recomputed recipe is a drifted pseudo-labelling run, not a silently-wrong
    published number."""
    from nuscenes_data_engine.demo.exporters import export_weaksup

    fixture = _write_weaksup_rejected_side_fixture(tmp_path)
    summary_path = fixture["al"] / "weakarm_pseudo_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["mean_gt_boxes_per_accepted_frame"] = 9.99
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="drifted"):
        export_weaksup(al_dir=fixture["al"], out_dir=tmp_path / "demo_data",
                        processed_dir=fixture["processed"])


def test_weak_loss_decomposition_reproduces_documented_shares(tmp_path: Path) -> None:
    """docs/ACTIVE_LEARNING.md's documented random (49.7%/32.1%/18.2%) and
    graph_rate_night (42.1%/18.5%/39.4%) loss splits, reproduced from a miniature
    results.json carrying only the mAP50-95 deltas the plan documents. A third base
    arm ("strat") has a weak_strat entry but no weak_strat_gt twin and must be
    skipped rather than error."""
    from nuscenes_data_engine.demo.exporters import export_weak_loss_decomposition

    al = tmp_path / "active_learning"
    al.mkdir()
    base = 0.10
    (al / "results.json").write_text(json.dumps({
        "baseline": {"overall": {"mAP50-95": base}},
        "random": {"overall": {"mAP50-95": base + 0.0340}},
        "weak_random_gt": {"overall": {"mAP50-95": base + 0.0171}},
        "weak_random": {"overall": {"mAP50-95": base + 0.0062}},
        "graph_rate_night": {"overall": {"mAP50-95": base + 0.0254}},
        "weak_graph_rate_night_gt": {"overall": {"mAP50-95": base + 0.0147}},
        "weak_graph_rate_night": {"overall": {"mAP50-95": base + 0.0100}},
        "strat": {"overall": {"mAP50-95": base + 0.02}},
        "weak_strat": {"overall": {"mAP50-95": base + 0.01}},
    }))

    df = export_weak_loss_decomposition(al_dir=al, out_dir=tmp_path / "demo_data").set_index(
        "base_arm"
    )
    assert list(df.index) == ["graph_rate_night", "random"]   # sorted; "strat" skipped

    random_row = df.loc["random"]
    assert random_row["gt_gain"] == pytest.approx(0.0340, abs=1e-9)
    assert random_row["weak_gt_gain"] == pytest.approx(0.0171, abs=1e-9)
    assert random_row["weak_gain"] == pytest.approx(0.0062, abs=1e-9)
    assert random_row["dropped_frame_cost"] == pytest.approx(0.0169, abs=1e-9)
    assert random_row["dropped_frame_share"] == pytest.approx(0.497, abs=1e-3)
    assert random_row["label_cost"] == pytest.approx(0.0109, abs=1e-9)
    assert random_row["label_share"] == pytest.approx(0.321, abs=1e-3)
    assert random_row["retention"] == pytest.approx(0.1824, abs=1e-6)
    assert bool(random_row["headline"]) is True
    # shares + the raw (unrounded) ratio sum to exactly 1 -- the stored "retention"
    # column is deliberately rounded to 4dp (to compare against overview's own
    # rounded number) and is NOT expected to satisfy this to 1e-9 itself.
    raw_retention = random_row["weak_gain"] / random_row["gt_gain"]
    assert (
        random_row["dropped_frame_share"] + random_row["label_share"] + raw_retention
    ) == pytest.approx(1.0, abs=1e-9)

    night_row = df.loc["graph_rate_night"]
    assert night_row["dropped_frame_share"] == pytest.approx(0.421, abs=1e-3)
    assert night_row["label_share"] == pytest.approx(0.185, abs=1e-3)
    assert night_row["retention"] == pytest.approx(0.3937, abs=1e-6)
    assert bool(night_row["headline"]) is False   # headline is random-only


def test_weak_verifier_by_class_long_table(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.exporters import export_weak_verifier_by_class

    al = tmp_path / "active_learning"
    al.mkdir()
    (al / "random_pseudo_summary.json").write_text(json.dumps({
        "arm": "random", "n_candidates": 1500, "n_accepted": 1000, "retention": 0.67,
        "n_boxes": 1, "mean_boxes_per_accepted_frame": 1.0,
        "mean_gt_boxes_per_accepted_frame": 1.0, "conf": 0.5, "tolerance": 1,
        "n_no_label": 0, "n_unparsed": 0,
        # "truck" only appears in rejected_by_class -> its accepted-mutual-zero
        # count must default to 0, not be dropped from the union.
        "rejected_by_class": {"pedestrian": 5, "car": 3, "truck": 2},
        "accepted_mutual_zero_by_class": {"pedestrian": 100, "car": 50},
    }))
    (al / "graph_rate_night_pseudo_summary.json").write_text(json.dumps({
        "arm": "graph_rate_night", "n_candidates": 1500, "n_accepted": 1101,
        "retention": 0.734, "n_boxes": 1, "mean_boxes_per_accepted_frame": 1.0,
        "mean_gt_boxes_per_accepted_frame": 1.0, "conf": 0.5, "tolerance": 1,
        "n_no_label": 0, "n_unparsed": 0,
        "rejected_by_class": {"pedestrian": 10},
        "accepted_mutual_zero_by_class": {"pedestrian": 870},
    }))

    df = export_weak_verifier_by_class(al_dir=al, out_dir=tmp_path / "demo_data")

    assert list(df.columns) == [
        "arm", "category_group", "n_rejected_disagreements", "n_accepted_mutual_zero",
        "mutual_zero_share",
    ]
    # sorted by (arm, category_group)
    assert list(zip(df["arm"], df["category_group"], strict=True)) == [
        ("graph_rate_night", "pedestrian"),
        ("random", "car"),
        ("random", "pedestrian"),
        ("random", "truck"),
    ]
    truck = df.set_index(["arm", "category_group"]).loc[("random", "truck")]
    assert truck["n_rejected_disagreements"] == 2
    assert truck["n_accepted_mutual_zero"] == 0
    assert truck["mutual_zero_share"] == pytest.approx(0.0)

    night_ped = df.set_index(["arm", "category_group"]).loc[("graph_rate_night", "pedestrian")]
    assert night_ped["mutual_zero_share"] == pytest.approx(870 / 1101, abs=1e-9)


def test_weak_labels_filtered_to_curated_tokens_native_coords(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.exporters import export_weak_labels

    al = tmp_path / "active_learning"
    al.mkdir()
    pd.DataFrame(
        {
            "sample_data_token": ["t2", "t1", "t1", "t3"],
            "category_group": ["car", "pedestrian", "car", "car"],
            "x_min": [500.0, 10.0, 300.0, 1.0],
            "y_min": [500.0, 20.0, 5.0, 1.0],
            "x_max": [600.0, 40.0, 320.0, 2.0],
            "y_max": [600.0, 60.0, 45.0, 2.0],
            "score": [0.5, 0.9, 0.6, 0.99],
        }
    ).to_parquet(al / "weakarm_pseudo_labels.parquet")

    out = tmp_path / "demo_data"
    df = export_weak_labels(al_dir=al, arm="weakarm", tokens=["t1", "t2"], out_dir=out)

    assert (out / "weak_labels.parquet").is_file()
    assert list(df.columns) == [
        "sample_data_token", "category_group", "x_min", "y_min", "x_max", "y_max", "score",
    ]
    assert set(df["sample_data_token"]) == {"t1", "t2"}   # t3 dropped (not curated)
    # sorted by (sample_data_token, category_group, x_min, y_min)
    assert list(zip(df["sample_data_token"], df["category_group"], df["x_min"], strict=True)) == [
        ("t1", "car", 300.0), ("t1", "pedestrian", 10.0), ("t2", "car", 500.0),
    ]
    # native (1600x900) coords untouched -- no scaling applied.
    row = df.set_index(["sample_data_token", "category_group"]).loc[("t2", "car")]
    assert row["x_min"] == 500.0 and row["y_max"] == 600.0


def test_vlm_counts_one_row_per_token_prefers_ok_parse_and_adds_gt_counts(
    tmp_path: Path,
) -> None:
    from nuscenes_data_engine.demo.exporters import export_vlm_counts

    al = tmp_path / "active_learning"
    processed = tmp_path / "processed"
    (al / "autolabel_weak").mkdir(parents=True)
    processed.mkdir()

    def _count_cols(n: int) -> dict[str, list[float | None]]:
        names = [
            "cars", "trucks", "buses", "trailers", "construction_vehicles",
            "motorcycles", "bicycles", "pedestrians", "traffic_cones", "barriers",
        ]
        return {name: [None] * n for name in names}

    rows: dict[str, Any] = {
        "sample_data_token": ["wA", "wA", "wR", "wBothBad", "wBothBad", "wBothOk"],
        "model": ["qwen2.5-vl"] * 6,
        # wA: first row truncated (must be skipped), second is "ok" -> kept.
        # wR: single ok row.
        # wBothBad: neither row is "ok" -> falls back to the LAST row in file order
        # (mirrors the run: keep="last" applies regardless of ok-ness too).
        # wBothOk: an "ok" row here too, in the SECOND (later-merged) table -- see
        # the other "ok" row for it in the autolabel table below.
        "parse_status": ["truncated", "ok", "ok", "truncated", "truncated", "ok"],
        "time_of_day": [None, "night", "day", "dusk", "dusk2", "day"],
        "weather": [None, "clear", "rain", "fog", "fog2", "rain"],
        "hazards": ["[]"] * 6,
        "notable_conditions": ["[]"] * 6,
        # A STRING enum in the real table ("high"/"low"/None), never a float
        # (consolidated review C1).
        "label_confidence": [None, "high", "low", None, None, "high"],
    }
    for name, values in _count_cols(6).items():
        rows[name] = values
    rows["cars"] = [None, 1.0, 0.0, 5.0, 6.0, 20.0]
    rows["pedestrians"] = [None, 2.0, 0.0, 0.0, 0.0, 9.0]
    pd.DataFrame(rows).to_parquet(al / "autolabel_weak" / "labels.parquet")

    pd.DataFrame({
        "sample_data_token": ["wA", "wA", "wR", "wBothBad"],
        "category_group": ["car", "pedestrian", "car", "bus"],
    }).to_parquet(processed / "annotations.parquet")

    # The frame the Phase-6b autolabel run labelled and the weak run never did:
    # its label lives ONLY in the OTHER table, which run_pseudo_label merges before
    # verifying (consolidated review C3). vlm_counts must find it there.
    autolabel = tmp_path / "autolabel"
    autolabel.mkdir()
    other: dict[str, Any] = {
        "sample_data_token": ["wOther", "wBothOk"],
        "model": ["qwen2.5-vl"] * 2,
        "parse_status": ["ok", "ok"],
        "time_of_day": ["night", "dawn"],
        "weather": ["clear", "snow"],
        "hazards": ["[]"] * 2,
        "notable_conditions": ["[]"] * 2,
        "label_confidence": ["high", "low"],
    }
    for name, values in _count_cols(2).items():
        other[name] = values
    other["cars"] = [3.0, 10.0]
    other["pedestrians"] = [1.0, 5.0]
    pd.DataFrame(other).to_parquet(autolabel / "labels.parquet")

    out = tmp_path / "demo_data"
    df = export_vlm_counts(
        label_paths=[autolabel / "labels.parquet", al / "autolabel_weak" / "labels.parquet"],
        processed_dir=processed,
        tokens=["wA", "wR", "wNoVlm", "wBothBad", "wOther", "wBothOk"],
        out_dir=out,
    ).set_index("sample_data_token")

    assert (out / "vlm_counts.parquet").is_file()
    assert list(df.reset_index().columns) == [
        "sample_data_token", "parse_status", "label_confidence", "vlm_time_of_day",
        "vlm_weather", "vlm_car", "vlm_truck", "vlm_bus", "vlm_pedestrian", "vlm_bicycle",
        "gt_car", "gt_truck", "gt_bus", "gt_pedestrian", "gt_bicycle",
    ]
    # wA: the "ok" row wins over the earlier "truncated" duplicate.
    assert df.loc["wA", "parse_status"] == "ok"
    assert df.loc["wA", "label_confidence"] == "high"
    assert df.loc["wA", "vlm_time_of_day"] == "night"
    assert df.loc["wA", "vlm_car"] == 1.0
    assert df.loc["wA", "vlm_pedestrian"] == 2.0
    assert df.loc["wA", "gt_car"] == 1
    assert df.loc["wA", "gt_pedestrian"] == 1
    assert df.loc["wA", "gt_truck"] == 0
    # wR: a single ok row, straightforward.
    assert df.loc["wR", "parse_status"] == "ok"
    assert df.loc["wR", "gt_car"] == 1
    # wBothBad: neither dup is "ok" -> falls back to the LAST row (cars=6.0,
    # dusk2), mirroring the run's keep="last" regardless of ok-ness.
    assert df.loc["wBothBad", "parse_status"] == "truncated"
    assert df.loc["wBothBad", "vlm_car"] == 6.0
    assert df.loc["wBothBad", "vlm_time_of_day"] == "dusk2"
    assert df.loc["wBothBad", "gt_bus"] == 1
    # wNoVlm: no row in labels.parquet at all -> vlm_* NA, parse_status NA, but GT
    # is a real fact (0), not "unknown".
    assert pd.isna(df.loc["wNoVlm", "parse_status"])
    assert pd.isna(df.loc["wNoVlm", "vlm_car"])
    assert df.loc["wNoVlm", "gt_car"] == 0
    # wOther: labelled only in the FIRST table -- the row the old single-table read
    # left entirely NA (8 of the 40 shipped rows).
    assert df.loc["wOther", "parse_status"] == "ok"
    assert df.loc["wOther", "label_confidence"] == "high"
    assert df.loc["wOther", "vlm_car"] == 3.0
    assert df.loc["wOther", "vlm_pedestrian"] == 1.0
    # wBothOk: an "ok" row in BOTH tables with DIFFERENT counts -- the run's rule
    # (LAST ok row in merged-table order wins) means the SECOND table's (weak)
    # row wins over the FIRST table's (autolabel) row, not the other way round.
    assert df.loc["wBothOk", "parse_status"] == "ok"
    assert df.loc["wBothOk", "label_confidence"] == "high"
    assert df.loc["wBothOk", "vlm_car"] == 20.0
    assert df.loc["wBothOk", "vlm_pedestrian"] == 9.0
    assert df.loc["wBothOk", "vlm_time_of_day"] == "day"


def test_vlm_counts_empty_tokens_writes_empty_table_with_schema(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.exporters import export_vlm_counts

    al = tmp_path / "active_learning"
    processed = tmp_path / "processed"
    al.mkdir()
    processed.mkdir()
    out = tmp_path / "demo_data"
    df = export_vlm_counts(
        label_paths=[al / "autolabel_weak" / "labels.parquet"],
        processed_dir=processed, tokens=[], out_dir=out,
    )
    assert df.empty
    assert (out / "vlm_counts.parquet").is_file()
    assert "gt_car" in df.columns


def test_vlm_counts_absent_label_table_is_a_logged_no_op(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A fresh clone has no ``data/autolabel/`` at all -- the missing table is
    logged and skipped, and the remaining one still produces the rows. Only when
    NO table exists does the export refuse (there is no VLM label anywhere)."""
    from nuscenes_data_engine.demo.exporters import export_vlm_counts

    al = tmp_path / "active_learning"
    processed = tmp_path / "processed"
    (al / "autolabel_weak").mkdir(parents=True)
    processed.mkdir()
    pd.DataFrame({
        "sample_data_token": ["wA"], "model": ["qwen2.5-vl"], "parse_status": ["ok"],
        "time_of_day": ["night"], "weather": ["clear"],
        "hazards": ["[]"], "notable_conditions": ["[]"], "label_confidence": ["high"],
        "cars": [1.0], "trucks": [0.0], "buses": [0.0], "trailers": [0.0],
        "construction_vehicles": [0.0], "motorcycles": [0.0], "bicycles": [0.0],
        "pedestrians": [0.0], "traffic_cones": [0.0], "barriers": [0.0],
    }).to_parquet(al / "autolabel_weak" / "labels.parquet")
    pd.DataFrame(
        {"sample_data_token": ["wA"], "category_group": ["car"]}
    ).to_parquet(processed / "annotations.parquet")

    missing = tmp_path / "autolabel" / "labels.parquet"
    with caplog.at_level(logging.WARNING, logger="nuscenes_data_engine"):
        df = export_vlm_counts(
            label_paths=[missing, al / "autolabel_weak" / "labels.parquet"],
            processed_dir=processed, tokens=["wA"], out_dir=tmp_path / "demo_data",
        )
    assert str(missing) in caplog.text
    assert df.set_index("sample_data_token").loc["wA", "parse_status"] == "ok"

    with pytest.raises(ValueError, match="no VLM label table found"):
        export_vlm_counts(
            label_paths=[missing], processed_dir=processed, tokens=["wA"],
            out_dir=tmp_path / "demo_data",
        )


def test_resolve_n_mine_prefers_the_graph_mining_override(tmp_path: Path) -> None:
    """The graph arms read ``graph_mining.n_mine`` first (graph_mining.py), so the
    demo's two readers -- build.py's quota-sum check and al_explain's
    re-derivation -- must too, or they measure different budgets (consolidated
    review M4)."""
    from nuscenes_data_engine.demo.exporters import resolve_n_mine

    assert resolve_n_mine({"mining": {"n_mine": 1500}, "graph_mining": {"n_mine": 900}}) == 900
    assert resolve_n_mine({"mining": {"n_mine": 1500}}) == 1500
    assert resolve_n_mine({"graph_mining": {"n_mine": 900}}) == 900
    assert resolve_n_mine({}) == 1500


def test_export_thumbs_writes_one_jpeg_per_token(tmp_path: Path) -> None:
    lancedb = pytest.importorskip("lancedb")
    from nuscenes_data_engine.demo.exporters import export_thumbs

    db = lancedb.connect(str(tmp_path / "lancedb"))
    db.create_table(
        "frames",
        data=[
            {"sample_data_token": "t1", "thumbnail": b"\xff\xd8\xff\xe0one"},
            {"sample_data_token": "t2", "thumbnail": b"\xff\xd8\xff\xe0two"},
            {"sample_data_token": "t3", "thumbnail": b"\xff\xd8\xff\xe0three"},
        ],
    )
    out = tmp_path / "demo_data"
    written = export_thumbs(
        lancedb_path=tmp_path / "lancedb", table="frames",
        tokens=["t1", "t3"], out_dir=out,
    )
    assert sorted(p.name for p in written) == ["t1.jpg", "t3.jpg"]
    assert (out / "sample_frames" / "thumbs" / "t1.jpg").read_bytes().endswith(b"one")


def test_export_thumbs_unknown_token_raises(tmp_path: Path) -> None:
    lancedb = pytest.importorskip("lancedb")
    from nuscenes_data_engine.demo.exporters import export_thumbs

    db = lancedb.connect(str(tmp_path / "lancedb"))
    db.create_table("frames", data=[{"sample_data_token": "t1", "thumbnail": b"x"}])
    with pytest.raises(ValueError, match="ghost"):
        export_thumbs(
            lancedb_path=tmp_path / "lancedb", table="frames",
            tokens=["t1", "ghost"], out_dir=tmp_path / "demo_data",
        )


def test_export_thumbs_empty_token_list_raises(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.exporters import export_thumbs

    with pytest.raises(ValueError, match="empty token list"):
        export_thumbs(
            lancedb_path=tmp_path / "lancedb", table="frames",
            tokens=[], out_dir=tmp_path / "demo_data",
        )


def test_export_thumbs_malformed_token_raises_value_error(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.exporters import export_thumbs

    bad_token = "abc'); DROP TABLE frames;--"
    with pytest.raises(ValueError, match="malformed tokens") as exc_info:
        export_thumbs(
            lancedb_path=tmp_path / "lancedb", table="frames",
            tokens=["t1", bad_token], out_dir=tmp_path / "demo_data",
        )
    assert bad_token in str(exc_info.value)


def _fake_search_fn(
    hits_by_query: dict[str, list[dict[str, Any]]],
) -> Any:
    """A ``search_fn(query, k) -> list[dict]`` stand-in: returns ``hits_by_query[query]``
    truncated to ``k`` (mimicking a real SearchEngine.search_text's own k-limit), and
    records every ``(query, k)`` call it received on ``.calls`` for assertions."""
    calls: list[tuple[str, int]] = []

    def _fn(query: str, k: int) -> list[dict[str, Any]]:
        calls.append((query, k))
        return hits_by_query.get(query, [])[:k]

    _fn.calls = calls  # type: ignore[attr-defined]
    return _fn


def test_export_semsearch_writes_ranked_rows_and_oversamples(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.exporters import export_semsearch

    hits = {
        "foggy road": [
            {"sample_data_token": "f1", "channel": "CAM_FRONT", "score": 0.9},
            {"sample_data_token": "b1", "channel": "CAM_BACK", "score": 0.85},
            {"sample_data_token": "f2", "channel": "CAM_FRONT", "score": 0.8},
            {"sample_data_token": "f3", "channel": "CAM_FRONT", "score": 0.7},
        ],
    }
    search_fn = _fake_search_fn(hits)
    out = tmp_path / "staging"
    df = export_semsearch(search_fn=search_fn, queries=["foggy road"], k=2, staging_dir=out)

    on_disk = pd.read_parquet(out / "semantic_search_results.parquet")
    pd.testing.assert_frame_equal(df, on_disk)
    assert list(df.columns) == ["query", "rank", "sample_data_token", "score", "k"]
    # CAM_FRONT filter applied, non-CAM_FRONT b1 dropped, rank order preserved
    # (nearest-first), truncated to k=2 -- f3 never makes it in.
    assert list(df["sample_data_token"]) == ["f1", "f2"]
    assert list(df["rank"]) == [1, 2]
    assert list(df["score"]) == [0.9, 0.8]
    assert (df["query"] == "foggy road").all()
    # k column carries the CONFIGURED target (2), not how many actually returned --
    # the demo page's gallery caption ("N of k front-camera hits") needs both.
    assert list(df["k"]) == [2, 2]
    # default oversample x8: search_fn called with k=2*8=16, not the raw k=2
    assert search_fn.calls == [("foggy road", 16)]  # type: ignore[attr-defined]


def test_export_semsearch_oversample_is_configurable(tmp_path: Path) -> None:
    """configs/demo.yaml's semsearch.oversample overrides the default 8x (item 7,
    consolidated review) -- passed straight through to the search_fn call size."""
    from nuscenes_data_engine.demo.exporters import export_semsearch

    hits = {"foggy road": [{"sample_data_token": "f1", "channel": "CAM_FRONT", "score": 0.9}]}
    search_fn = _fake_search_fn(hits)
    export_semsearch(
        search_fn=search_fn, queries=["foggy road"], k=2, staging_dir=tmp_path, oversample=16,
    )
    assert search_fn.calls == [("foggy road", 32)]  # type: ignore[attr-defined]  # 2*16, not 2*8


def test_export_semsearch_multiple_queries_in_order(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.exporters import export_semsearch

    hits = {
        "q1": [{"sample_data_token": "a1", "channel": "CAM_FRONT", "score": 0.5}],
        "q2": [{"sample_data_token": "b1", "channel": "CAM_FRONT", "score": 0.6}],
    }
    df = export_semsearch(
        search_fn=_fake_search_fn(hits), queries=["q1", "q2"], k=5, staging_dir=tmp_path
    )
    assert list(df["query"]) == ["q1", "q2"]
    assert list(df["sample_data_token"]) == ["a1", "b1"]
    assert list(df["rank"]) == [1, 1]  # rank resets per query


def test_export_semsearch_empty_queries_raises(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.exporters import export_semsearch

    with pytest.raises(ValueError, match="empty"):
        export_semsearch(search_fn=_fake_search_fn({}), queries=[], k=8, staging_dir=tmp_path)


def _write_demo_config(
    tmp_path: Path, tiny_inputs: dict[str, Path], *, models: dict[str, Any]
) -> Path:
    """Shared demo.yaml builder for ``build_config`` and the legacy-model-shape test
    below — factored out so the two differ only in ``models``, not in every other
    path/budget/flagship field.
    """
    al = tiny_inputs["al"]
    # Phase 7 (Task 2): mean_gt_boxes_per_accepted_frame=3.0 is pinned against
    # tiny_inputs' own annotations.parquet + random.parquet/random_accepted.parquet
    # (accepted=["s1"], 3 detector-class GT boxes) -- export_weaksup asserts the two
    # agree, so this can't drift from that fixture without both changing together.
    (al / "random_pseudo_summary.json").write_text(json.dumps({
        "arm": "random", "n_candidates": 10, "n_accepted": 6, "retention": 0.6,
        "n_boxes": 12, "mean_boxes_per_accepted_frame": 2.0,
        "mean_gt_boxes_per_accepted_frame": 3.0,
        "conf": 0.5, "tolerance": 1, "n_no_label": 1, "n_unparsed": 0,
        "rejected_by_class": {"pedestrian": 2}, "accepted_mutual_zero_by_class": {"car": 1},
    }))
    config = {
        "paths": {
            "processed_dir": str(tiny_inputs["processed"]),
            "active_learning_dir": str(al),
            "active_learning_config": str(tiny_inputs["al_config"]),
            # Phase 7 (consolidated review C3): vlm_counts reads BOTH label tables
            # run_pseudo_label merges. Pointed inside tmp_path so these fixtures
            # never reach for the machine's real data/autolabel/ -- absent by
            # default (a logged skip), staged by the test that exercises the merge.
            "autolabel_dir": str(tmp_path / "autolabel"),
            "mlruns_dir": str(tmp_path / "mlruns"),
            "lancedb_path": str(tmp_path / "lancedb"),
            "lancedb_table": "frames",
            "out_dir": str(tmp_path / "demo_data"),
        },
        "models": models,
        # Phase 3: hero is a hand-picked exemplar crop token from the curated-frames
        # group, not an mlruns mosaic (see the dated amendment in
        # docs/superpowers/specs/2026-08-12-demo-phase1-design.md) -- absent by
        # default here, same as `curation:` below; tests that need a successful
        # build opt in via `_stage_and_pick_hero`.
        "hero": {"token": None},
        "budgets": {"max_package_mb": 100},
        "flagship": {"expected_sql_count": 1, "cypher_count": 30,
                     "cypher_source": "docs/GRAPH.md"},
        # Phase 5: demo/events.py::build_events' preset thresholds -- mirrors
        # configs/demo.yaml's `presets:` section (see _include_events).
        "presets": {
            "cap_per_preset": 30,
            "near_dist_m": 10.0,
            "high_speed_mps": 10.0,
            "model_for_results": "baseline",
        },
        # Task 4: absent by default -- no `demo curate`/`demo infer` run has staged
        # anything under this dir in most fixtures here, so run_build must skip the
        # curation group loudly rather than erroring. The curation-specific tests
        # below stage real files under this same staging_dir. weak_arm/al_arm mirror
        # configs/demo.yaml's real values -- tiny_inputs' al dir already carries the
        # matching graph_rate_night(.parquet/_accepted.parquet) fixtures.
        "curation": {
            "staging_dir": str(tmp_path / "curation_staging"),
            "weak_arm": "graph_rate_night",
            "al_arm": "graph_rate_night",
        },
        # Phase 7 (Task 1): al_communities.parquet always runs (see run_build), so
        # every build_config-based test needs this section; exemplar_tokens is
        # empty by default (curation is absent by default too, so that's valid --
        # see export_al_exemplars) -- tests that stage curation opt into a real
        # exemplar token via _stage_and_pick_hero (or set it themselves).
        "al": {
            "arm": "graph_rate_night",
            "baseline": "baseline",
            "community_arms": ["graph_rate", "graph_rate_night"],
            "exemplar_tokens": [],
        },
    }
    path = tmp_path / "demo.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _stage_minimal_curation(staging_dir: Path) -> None:
    """Write a minimal, fully-staged curation dir: one val token "v0" with
    n_preds_baseline coverage, gt_boxes a1 (20x20 -> small) / a2 (200x200 -> large)
    for v0, one predictions row for v0, and a single crop file v0.jpg.

    Factored out of the per-test staging blocks below (test_build_includes_
    curation_group_when_staged and its siblings) for the Task 1 gt-box-enrichment
    and hero-token tests, which all need the same minimal shape -- the older
    curation tests keep their own inline staging rather than being churned to
    adopt this.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "crops").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sample_data_token": ["v0"], "split": ["val"], "filename": ["images/v0.jpg"],
        "curation_buckets": [["night_failure"]],
        "n_preds_baseline": pd.array([1], dtype="Int64"),
        # Phase 7 (Task 1): al.exemplar_tokens=["v0"] (set by _stage_and_pick_hero)
        # carries the real manifest's fix flag; since the 2026-08-21 amendment the
        # validation keys off the per-BOX upgrade staged below instead (see
        # test_al_exemplars_require_a_confident_arm_box_the_baseline_missed), but
        # the column stays here because the real curated manifest has it.
        "fixes_fn_vs_baseline_graph_rate_night": pd.array([True], dtype="boolean"),
    }).to_parquet(staging_dir / "frame_manifest.parquet")
    pd.DataFrame({
        # v0's exemplar shape: `baseline` claims a1 only; `graph_rate_night` also
        # claims a2 (tp) -- the confident detection the baseline never made, which
        # is what export_al_exemplars requires of a hand-approved exemplar token.
        "sample_data_token": ["v0", "v0"],
        "model": ["baseline", "graph_rate_night"],
        "category_group": ["car", "car"],
        "x_min": [1.0, 0.0], "y_min": [1.0, 0.0], "x_max": [2.0, 200.0],
        "y_max": [2.0, 200.0],
        "conf": [0.9, 0.61], "status": ["tp", "tp"],
        "matched_annotation_token": ["a1", "a2"],
    }).to_parquet(staging_dir / "predictions.parquet")
    pd.DataFrame({
        "annotation_token": ["a1", "a2"], "sample_data_token": ["v0", "v0"],
        "category_group": ["pedestrian", "car"],
        "x_min": [0.0, 0.0], "y_min": [0.0, 0.0],
        "x_max": [20.0, 200.0], "y_max": [20.0, 200.0],
        "matched_baseline": [True, True],
        # Real gt_boxes always carries the visibility floor flag (demo infer writes
        # it) -- export_al_exemplars reads it to keep an invisible box from
        # justifying an exemplar.
        "below_visibility_min": [False, False],
    }).to_parquet(staging_dir / "gt_boxes.parquet")
    (staging_dir / "crops" / "v0.jpg").write_bytes(b"\xff\xd8\xff\xe0crop")


def _stage_and_pick_hero(config_path: Path, *, token: str = "v0") -> dict[str, Any]:
    """Stage a minimal curated-frames group and point ``hero.token`` at it.

    Most ``build_config``-based tests below don't care about curation or the hero
    at all -- they need SOME successful build to exercise an unrelated code path
    (manifest determinism, budget/flagship validation, ...). Since Phase 3's hero
    flow always requires a real curated crop (see
    test_build_null_hero_token_fails_loudly /
    test_build_hero_token_without_curation_fails_loudly for the failure paths this
    guards), every such test opts in explicitly via this helper rather than
    ``build_config`` silently defaulting to "staged" -- which would make those two
    curation-absent-specific tests unable to express their own precondition.
    Returns the parsed config dict for further test-specific mutation.
    """
    config = yaml.safe_load(config_path.read_text())
    _stage_minimal_curation(Path(config["curation"]["staging_dir"]))
    config["hero"] = {"token": token}
    # Phase 7 (Task 1): export_al_exemplars requires a non-empty token list once
    # curation is included (see configs/demo.yaml's al.exemplar_tokens) -- reuse
    # the same token, which _stage_minimal_curation already gives the required
    # fixes_fn_vs_baseline_graph_rate_night=True flag and val split.
    config["al"]["exemplar_tokens"] = [token]
    config_path.write_text(yaml.safe_dump(config))
    return config


@pytest.fixture()
def build_config(tmp_path: Path, tiny_inputs: dict[str, Path]) -> Path:
    # Production shape: a {run, imgsz} mapping (configs/demo.yaml since the Task 3
    # review round) -- the shape every other fixture in this module now exercises.
    # (The flat run-id-string legacy shape's dedicated coverage,
    # test_build_accepts_legacy_flat_model_shape, was deleted in the Phase 3 quality
    # round: its subject -- run_build's hero accessor resolving config["models"] --
    # no longer exists, since the Phase 3 hero flow reads configs/demo.yaml
    # hero.token instead. _include_curation's val-coverage check only iterates
    # config["models"] keys, which is shape-agnostic already.)
    return _write_demo_config(
        tmp_path, tiny_inputs, models={"baseline": {"run": "runX", "imgsz": 640}}
    )


def test_build_writes_validated_manifest(build_config: Path, tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    manifest = run_build(build_config)
    out = Path(config["paths"]["out_dir"])
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk == manifest
    assert manifest["git_sha"]
    assert manifest["outputs"]["overview_metrics.json"]["sha256"]
    assert manifest["outputs"]["active_learning_results.parquet"]["rows"] == 5
    assert manifest["validation"]["flagship_sql_count"] == 1
    assert manifest["validation"]["package_mb"] < 1
    assert manifest["package_version"] == "0.7"


def test_build_is_deterministic(build_config: Path) -> None:
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    # Phase 6 (item M9, review): the graph-subgraph group is staged here too, so
    # byte-identical rebuilds are pinned WITH it present -- the copied
    # graph_subgraphs/*.json are package outputs like any other.
    _stage_subgraphs(Path(config["curation"]["staging_dir"]))
    out = Path(config["paths"]["out_dir"])
    run_build(build_config)
    first = {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
    run_build(build_config)
    second = {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
    assert first == second                    # byte-stable outputs (manifest has built_at)


def test_build_fails_on_wrong_flagship_count(build_config: Path) -> None:
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    config["flagship"]["expected_sql_count"] = 30      # tiny fixture yields 1, not 30
    build_config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="flagship"):
        run_build(build_config)


def test_build_fails_over_size_budget(build_config: Path) -> None:
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    config["budgets"]["max_package_mb"] = 0
    build_config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="budget"):
        run_build(build_config)


def test_git_sha_anchored_to_package_repo_not_cwd(tmp_path: Path) -> None:
    """_git_sha() must anchor to this package's repo, not an unrelated repo at cwd."""
    from nuscenes_data_engine.demo.build import _git_sha

    repo_root = Path(__file__).resolve().parents[1]
    expected = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        assert _git_sha() == expected
    finally:
        os.chdir(cwd)


def test_build_wipes_stale_residue(build_config: Path) -> None:
    """A leftover file from a removed exporter or failed prior build must not survive."""
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    out_dir = Path(config["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    # manifest.json makes this look like an existing demo package (not a typo'd
    # out_dir) so the wipe-guard (test_build_refuses_to_wipe_a_non_package_dir) lets
    # the rebuild proceed.
    (out_dir / "manifest.json").write_text("{}")
    (out_dir / "stale.json").write_text("{}")

    manifest = run_build(build_config)

    assert not (out_dir / "stale.json").exists()
    assert "stale.json" not in manifest["outputs"]


def test_build_refuses_to_wipe_a_non_package_dir(build_config: Path) -> None:
    """A typo'd out_dir pointing at a real, unrelated directory must not be wiped."""
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    out_dir = Path(config["paths"]["out_dir"])
    out_dir.mkdir(parents=True)
    (out_dir / "not_a_demo_package.txt").write_text("please don't delete me")

    with pytest.raises(ValueError, match="refusing to wipe"):
        run_build(build_config)

    assert (out_dir / "not_a_demo_package.txt").is_file()


def test_build_recovers_from_a_failed_build_without_manual_cleanup(build_config: Path) -> None:
    """A failed build must not self-lock the retry.

    manifest.json is written last, so a failed build (here: the documented flagship-
    mismatch failure) leaves overview_metrics.json on disk but no manifest.json. The
    wipe guard must still recognise that tree as a demo package (via
    _PACKAGE_MARKERS) and let the corrected retry through, not blame a config typo
    that doesn't exist and demand a manual rm -rf.
    """
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    config["flagship"]["expected_sql_count"] = 30  # tiny fixture yields 1, not 30 -> fails
    build_config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="flagship"):
        run_build(build_config)

    out_dir = Path(config["paths"]["out_dir"])
    assert (out_dir / "overview_metrics.json").is_file()
    assert not (out_dir / "manifest.json").is_file()

    config["flagship"]["expected_sql_count"] = 1  # fix the config back to reality
    build_config.write_text(yaml.safe_dump(config))
    manifest = run_build(build_config)  # must not raise "refusing to wipe"

    assert (out_dir / "manifest.json").is_file()
    assert manifest["validation"]["flagship_sql_count"] == 1


def test_build_rewipes_an_existing_demo_package(build_config: Path) -> None:
    """A dir that already looks like a demo package (has manifest.json) is fine to wipe."""
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    run_build(build_config)  # first build creates out_dir/manifest.json
    manifest = run_build(build_config)  # second build must not raise the new guard

    out_dir = Path(config["paths"]["out_dir"])
    assert (out_dir / "manifest.json").is_file()
    assert manifest["validation"]["n_arms"] == 5


def test_build_proceeds_when_out_dir_absent(build_config: Path) -> None:
    """An out_dir that simply doesn't exist yet (the common case) is fine, not an error."""
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    out_dir = Path(config["paths"]["out_dir"])
    assert not out_dir.exists()

    manifest = run_build(build_config)

    assert (out_dir / "manifest.json").is_file()
    assert manifest["validation"]["n_arms"] == 5


def test_build_fails_when_weak_retention_headline_missing(build_config: Path) -> None:
    """A results.json missing the documented headline (random) pair must fail loudly,
    not silently publish `"headline": null` to the committed package."""
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    al_dir = Path(config["paths"]["active_learning_dir"])
    results_path = al_dir / "results.json"
    results = json.loads(results_path.read_text())
    del results["random"]
    del results["weak_random"]
    results_path.write_text(json.dumps(results))

    with pytest.raises(ValueError, match="weak-retention headline"):
        run_build(build_config)


def test_build_hashes_all_real_inputs(build_config: Path) -> None:
    """manifest['inputs'] must cover everything the build actually reads, not just
    results.json. Exact composition for build_config + _stage_and_pick_hero: 1
    (active_learning_dir/results.json) + 5 (exporters.PROCESSED_INPUTS: samples/
    annotations/annotations_3d/canbus/ego_pose) + 1 (random_pseudo_summary.json) + 3
    (staged frame_manifest/gt_boxes/predictions.parquet, curation included) + 1
    (paths.active_learning_config) + 2 (communities_graph_rate[_night].json,
    always hashed) + 2 (weak_arm.parquet/weak_arm_accepted.parquet -- weak_arm ==
    al_arm == "graph_rate_night" here, so al_arm.parquet is the same file as
    weak_arm.parquet and dedupes under one dict key) = 15. Phase 7 (Task 2) adds 4
    more: random.parquet/random_accepted.parquet (export_weaksup's rejected-side
    recipe, for the "random" *_pseudo_summary.json arm -- a distinct pair from the
    weak_arm/al_arm ones above, since "random" != "graph_rate_night") + graph_
    rate_night_pseudo_labels.parquet (export_weak_labels' input, curation included)
    + autolabel_weak/labels.parquet (export_vlm_counts' input, curation included)
    = 19. Phase 8 (Task 2) adds 2 more when a chat-replay group is staged:
    chat_replays.json + chat_replay_summary.json = 21."""
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    _stage_chat_replays(Path(config["curation"]["staging_dir"]) / "chat_replays")
    manifest = run_build(build_config)
    assert len(manifest["inputs"]) == 21
    assert any(key.endswith("canbus.parquet") for key in manifest["inputs"])
    assert any(key.endswith("graph_rate_night_pseudo_labels.parquet") for key in manifest["inputs"])
    assert any(key.endswith("autolabel_weak/labels.parquet") for key in manifest["inputs"])
    assert any(
        key.endswith("random.parquet") and not key.endswith("random_accepted.parquet")
        for key in manifest["inputs"]
    )
    assert any(key.endswith("random_accepted.parquet") for key in manifest["inputs"])
    assert any(key.endswith("chat_replays.json") for key in manifest["inputs"])
    assert any(key.endswith("chat_replay_summary.json") for key in manifest["inputs"])


def test_build_vlm_counts_cover_every_weak_verdict_frame_from_both_label_tables(
    build_config: Path, tmp_path: Path
) -> None:
    """vlm_counts is scoped to ``frame_manifest.weak_verdict``, not the two
    curation buckets (consolidated review I1) -- the page's tabs render every
    verdict frame -- and it reads BOTH label tables (C3), hashing each one that
    exists.

    "v0" is the staged curated frame: it carries ``curation_buckets =
    ["night_failure"]`` (so the old bucket-scoped list skipped it entirely) but is
    in the weak arm's accepted set, so ``weak_verdict == "accepted"``. ``tiny_
    inputs`` already gives it an "ok" row in the weak table (cars=1.0); this test
    adds a SECOND "ok" row for it in the Phase-6b autolabel table (cars=3.0, a
    deliberately different count) so a populated, non-NA result proves both
    tables were read and hashed. The two tables disagreeing is also what pins the
    dedup rule end to end: the exporter mirrors ``run_pseudo_label``'s (the LAST
    ok row in merged-table order wins), and the weak table is listed SECOND
    (``vlm_label_tables``), so the weak row's cars=1.0 -- not the autolabel row's
    3.0 -- is what must come out here.
    """
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    autolabel_dir = Path(config["paths"]["autolabel_dir"])
    autolabel_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sample_data_token": ["v0"], "model": ["qwen2.5-vl"], "parse_status": ["ok"],
        "time_of_day": ["night"], "weather": ["clear"],
        "hazards": ["[]"], "notable_conditions": ["[]"], "label_confidence": ["high"],
        "cars": [3.0], "trucks": [0.0], "buses": [0.0], "trailers": [0.0],
        "construction_vehicles": [0.0], "motorcycles": [0.0], "bicycles": [0.0],
        "pedestrians": [1.0], "traffic_cones": [0.0], "barriers": [0.0],
    }).to_parquet(autolabel_dir / "labels.parquet")

    manifest = run_build(build_config)
    out = Path(config["paths"]["out_dir"])
    counts = pd.read_parquet(out / "vlm_counts.parquet").set_index("sample_data_token")
    assert list(counts.index) == ["v0"]
    assert manifest["validation"]["n_vlm_counts"] == 1
    assert counts.loc["v0", "parse_status"] == "ok"
    assert counts.loc["v0", "label_confidence"] == "high"
    # the weak table's row wins (listed second -> LAST in merged order), not the
    # autolabel table's -- see the docstring.
    assert counts.loc["v0", "vlm_car"] == 1.0
    # both tables hashed as inputs, the weak one and the Phase-6b one
    assert str(autolabel_dir / "labels.parquet") in manifest["inputs"]
    assert any(key.endswith("autolabel_weak/labels.parquet") for key in manifest["inputs"])


def test_vlm_label_paths_rejects_a_configured_but_missing_weak_config(
    tmp_path: Path,
) -> None:
    """`paths.autolabel_weak_config` pointing at a file that isn't there is a
    typo in demo.yaml, not "unconfigured" -- it must fail loudly rather than
    silently falling back to the shipped `<active_learning_dir>/autolabel_weak`
    default (consolidated review follow-up)."""
    from nuscenes_data_engine.demo.build import _vlm_label_paths

    missing = tmp_path / "configs" / "autolabel_weak.yaml"
    config: dict[str, Any] = {"paths": {"autolabel_weak_config": str(missing)}}

    with pytest.raises(ValueError, match="does not exist") as excinfo:
        _vlm_label_paths(config, al_dir=tmp_path / "active_learning")
    assert str(missing) in str(excinfo.value)
    assert "paths.autolabel_weak_config" in str(excinfo.value)


def test_build_resolves_n_mine_from_the_graph_mining_override(
    build_config: Path, tmp_path: Path
) -> None:
    """A config that overrides the budget for the GRAPH arms only must be read the
    same way graph_mining.py reads it, or export_al_communities' quota-sum check
    fires against a budget no arm ever used (consolidated review M4)."""
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    al_config_path = Path(config["paths"]["active_learning_config"])
    # tiny_inputs' communities_*.json quotas each sum to 3; mining.n_mine says 99
    # (a budget no graph arm used), graph_mining.n_mine says 3.
    al_config_path.write_text(
        yaml.safe_dump({"mining": {"n_mine": 99}, "graph_mining": {"n_mine": 3}})
    )
    run_build(build_config)
    assert pd.read_parquet(Path(config["paths"]["out_dir"]) / "al_communities.parquet").shape[0]


def test_build_without_an_al_section_fails_directively(build_config: Path) -> None:
    """A pre-Phase-7 configs/demo.yaml has no ``al:`` key at all; a bare
    ``KeyError: 'al'`` names neither the file nor the fix (consolidated review
    M9)."""
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    del config["al"]
    build_config.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match=r"needs an `al:` section"):
        run_build(build_config)


def test_build_wires_weak_tables_and_checks_retention_against_overview(
    build_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nuscenes_data_engine.demo import build as build_module
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    al_dir = Path(config["paths"]["active_learning_dir"])
    out = Path(config["paths"]["out_dir"])

    # weak_random_gt gives the "random" base arm both twins it needs for a
    # weak_loss_decomposition row (tiny_inputs only carries weak_random by
    # default) -- mutated here, not in the shared fixture, so every OTHER
    # build_config test keeps getting an (empty, still-valid) decomposition table.
    results_path = al_dir / "results.json"
    results = json.loads(results_path.read_text())
    results["weak_random_gt"] = {
        "n_train_images": 112,
        "overall": {"mAP50-95": 0.205, "mAP50": 0.35},
        "night": {"mAP50-95": 0.092, "mAP50": 0.16, "precision": 0.5, "recall": 0.4,
                   "per_class": {"pedestrian": 0.07}},
        "slices": {
            "time_of_day/day": {"mAP50-95": 0.21},
            "weather/rain": {"mAP50-95": 0.14},
            "weather/clear": {"mAP50-95": 0.20},
        },
    }
    results_path.write_text(json.dumps(results))

    manifest = run_build(build_config)
    for name in (
        "weak_loss_decomposition.parquet", "weak_verifier_by_class.parquet",
        "weak_labels.parquet", "vlm_counts.parquet",
    ):
        assert (out / name).is_file()
        assert name in manifest["outputs"]

    input_names = {Path(p).name for p in manifest["inputs"]}
    assert "annotations.parquet" in input_names
    assert "graph_rate_night_pseudo_labels.parquet" in input_names
    assert "labels.parquet" in input_names
    assert "random.parquet" in input_names
    assert "random_accepted.parquet" in input_names

    decomposition = pd.read_parquet(out / "weak_loss_decomposition.parquet")
    assert "random" in set(decomposition["base_arm"])
    assert manifest["validation"]["n_weak_labels"] >= 0
    assert manifest["validation"]["n_vlm_counts"] >= 0

    # Corrupt export_overview's weak_retention.by_base_arm so it disagrees with
    # weak_loss_decomposition's own (independently computed) retention for
    # "random" -- the build must fail loudly, not publish a silently-wrong pair.
    real_export_overview = build_module.exporters.export_overview

    def _corrupted_overview(**kwargs: Any) -> dict[str, Any]:
        metrics = real_export_overview(**kwargs)
        metrics["results"]["weak_retention"]["by_base_arm"]["random"] = 0.999
        build_module.write_json(kwargs["out_dir"] / "overview_metrics.json", metrics)
        return metrics

    monkeypatch.setattr(build_module.exporters, "export_overview", _corrupted_overview)
    with pytest.raises(ValueError, match="retention"):
        run_build(build_config)


def test_build_skips_absent_curation_with_manifest_note(build_config: Path) -> None:
    """No `demo curate`/`demo infer` staging (build_config's default) must not fail
    the build -- Task 2's TRINITY-unreachable fallback means Tasks 1-4 still merge
    on their own, and since data/demo_curation is gitignored, a fresh clone with a
    null hero token must still produce a working Phase-1-only package (spec-review
    fix: the null-token check must be gated on curation presence, not fire
    unconditionally). It must, however, say so loudly in the manifest, not silently
    proceed as if the curation group never existed.

    Whether hero.token is even required now depends on curation:
    test_build_hero_token_without_curation_fails_loudly covers "token configured
    but curation absent" (still an error -- the config references curated data that
    isn't there), and test_build_null_hero_token_fails_loudly covers "curation
    present but no token" (still an error -- pick one)."""
    from nuscenes_data_engine.demo.build import run_build

    manifest = run_build(build_config)
    assert manifest["validation"]["curation"] == "absent"
    out = Path(yaml.safe_load(build_config.read_text())["paths"]["out_dir"])
    assert not (out / "frame_manifest.parquet").exists()
    assert "sample_frames/hero.jpg" not in manifest["outputs"]
    assert not (out / "sample_frames" / "hero.jpg").exists()


def test_build_includes_curation_group_when_staged(build_config: Path, tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    # The staging this test used to write inline is exactly _stage_minimal_curation's
    # shape; it adopts the helper now that the exemplar rule (2026-08-21 amendment)
    # needs a per-box arm-beats-baseline upgrade staged as well -- one place to keep
    # that shape correct, rather than three copies of it in this module.
    _stage_minimal_curation(Path(config["curation"]["staging_dir"]))
    config["hero"] = {"token": "v0"}
    config["al"]["exemplar_tokens"] = ["v0"]
    build_config.write_text(yaml.safe_dump(config))
    manifest = run_build(build_config)
    assert manifest["validation"]["curation"] == "included"
    out = Path(config["paths"]["out_dir"])
    assert (out / "frame_manifest.parquet").is_file()
    assert (out / "sample_frames" / "crops" / "v0.jpg").is_file()
    assert "frame_manifest.parquet" in manifest["outputs"]
    # staging parquets are hashed as inputs (they are what the build actually read)
    assert any(
        Path(key).name == "frame_manifest.parquet" for key in manifest["inputs"]
    )


def test_build_raises_on_stale_crop_not_in_manifest(build_config: Path) -> None:
    """Licensing hazard: crops are nuScenes-derived imagery. A re-curation with a
    different token set must not silently ship a PRIOR run's crop alongside the
    current manifest -- stale imagery in staging means the operator's pipeline state
    is inconsistent (e.g. `demo curate` re-ran but `demo infer` hasn't cleaned up the
    old crops dir), not something to quietly carry forward into the package."""
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    staging = Path(config["curation"]["staging_dir"])
    (staging / "crops").mkdir(parents=True)
    pd.DataFrame({
        "sample_data_token": ["v0"], "split": ["val"], "filename": ["images/v0.jpg"],
        "curation_buckets": [["night_failure"]],
        "n_preds_baseline": pd.array([1], dtype="Int64"),
    }).to_parquet(staging / "frame_manifest.parquet")
    pd.DataFrame({
        "sample_data_token": ["v0"], "model": ["baseline"], "category_group": ["car"],
        "x_min": [1.0], "y_min": [1.0], "x_max": [2.0], "y_max": [2.0],
        "conf": [0.9], "status": ["tp"], "matched_annotation_token": ["a1"],
    }).to_parquet(staging / "predictions.parquet")
    pd.DataFrame({
        "annotation_token": ["a1"], "sample_data_token": ["v0"],
        "category_group": ["car"], "x_min": [1.0], "y_min": [1.0],
        "x_max": [2.0], "y_max": [2.0], "matched_baseline": [True],
    }).to_parquet(staging / "gt_boxes.parquet")
    (staging / "crops" / "v0.jpg").write_bytes(b"\xff\xd8\xff\xe0crop")
    # stale imagery: left over from a PRIOR curation run, not a token in this manifest.
    (staging / "crops" / "stale_token.jpg").write_bytes(b"\xff\xd8\xff\xe0stale")

    with pytest.raises(ValueError, match="stale_token"):
        run_build(build_config)


def test_build_fails_when_val_token_lacks_model_predictions(build_config: Path) -> None:
    """The never-ran case, column entirely absent: frame_manifest.parquet carries no
    n_preds_baseline column at all (demo infer never ran with this model) -- must
    fail loudly, not be conflated with a legitimate zero-prediction finding."""
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    staging = Path(config["curation"]["staging_dir"])
    (staging / "crops").mkdir(parents=True)
    pd.DataFrame({
        "sample_data_token": ["v0"], "split": ["val"], "filename": ["images/v0.jpg"],
        "curation_buckets": [["night_failure"]],
        # no n_preds_baseline column at all
    }).to_parquet(staging / "frame_manifest.parquet")
    pd.DataFrame(columns=["sample_data_token", "model", "category_group", "x_min",
                          "y_min", "x_max", "y_max", "conf", "status",
                          "matched_annotation_token"]).to_parquet(
        staging / "predictions.parquet")
    pd.DataFrame(columns=["annotation_token", "sample_data_token", "category_group",
                          "x_min", "y_min", "x_max", "y_max"]).to_parquet(
        staging / "gt_boxes.parquet")
    (staging / "crops" / "v0.jpg").write_bytes(b"\xff\xd8\xff\xe0crop")
    with pytest.raises(ValueError, match="predictions"):
        run_build(build_config)


def test_build_fails_when_val_token_has_na_n_preds(build_config: Path) -> None:
    """The never-ran case, column present but NA: distinct code path from the column
    being entirely absent (a model dropped from a later infer run, or leftover
    NA from a rerun with a different roster) -- must still fail loudly."""
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    staging = Path(config["curation"]["staging_dir"])
    (staging / "crops").mkdir(parents=True)
    pd.DataFrame({
        "sample_data_token": ["v0"], "split": ["val"], "filename": ["images/v0.jpg"],
        "curation_buckets": [["night_failure"]],
        "n_preds_baseline": pd.array([pd.NA], dtype="Int64"),
    }).to_parquet(staging / "frame_manifest.parquet")
    pd.DataFrame(columns=["sample_data_token", "model", "category_group", "x_min",
                          "y_min", "x_max", "y_max", "conf", "status",
                          "matched_annotation_token"]).to_parquet(
        staging / "predictions.parquet")
    pd.DataFrame(columns=["annotation_token", "sample_data_token", "category_group",
                          "x_min", "y_min", "x_max", "y_max"]).to_parquet(
        staging / "gt_boxes.parquet")
    (staging / "crops" / "v0.jpg").write_bytes(b"\xff\xd8\xff\xe0crop")
    with pytest.raises(ValueError, match="predictions"):
        run_build(build_config)


def test_build_passes_val_coverage_with_a_zero_prediction_count(build_config: Path) -> None:
    """Task-5 fix: 'ran and found nothing' (n_preds_<model> == 0) is legitimate
    coverage, not a validation failure -- real Task-5 case: baseline and
    graph_rate_night both found zero boxes on a genuine total-miss frame, and the
    old predictions.parquet-row-presence check couldn't tell that apart from
    'never ran', wrongly failing the build on a token the demo actually wants."""
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    staging = Path(config["curation"]["staging_dir"])
    (staging / "crops").mkdir(parents=True)
    # v1: a second, ordinarily-covered val token purely so al.exemplar_tokens (now
    # required once curation is included, Phase 7 Task 1) has a real exemplar to
    # validate -- v0 stays the dedicated "ran and found nothing" case below (zero
    # predictions.parquet rows is the whole point of this test), which
    # export_al_exemplars would otherwise (correctly) reject as "no predictions".
    pd.DataFrame({
        "sample_data_token": ["v0", "v1"], "split": ["val", "val"],
        "filename": ["images/v0.jpg", "images/v1.jpg"],
        "curation_buckets": [["night_failure"], ["night_failure"]],
        "n_preds_baseline": pd.array([0, 1], dtype="Int64"),
        "fixes_fn_vs_baseline_graph_rate_night": pd.array([False, True], dtype="boolean"),
    }).to_parquet(staging / "frame_manifest.parquet")
    # baseline ran and found nothing on v0 -- predictions.parquet has zero rows for
    # it, same as a model that never ran would; n_preds is what disambiguates. v1
    # gets two real rows: baseline's own claim, plus graph_rate_night's confident
    # claim on GT box b1 which baseline never makes -- the per-box upgrade
    # export_al_exemplars requires of an exemplar token since the 2026-08-21
    # amendment.
    pd.DataFrame({
        "sample_data_token": ["v1", "v1"],
        "model": ["baseline", "graph_rate_night"],
        "category_group": ["car", "car"],
        "x_min": [1.0, 1.0], "y_min": [1.0, 1.0], "x_max": [2.0, 2.0],
        "y_max": [2.0, 2.0],
        "conf": [0.9, 0.61], "status": ["tp", "tp"],
        "matched_annotation_token": [None, "b1"],
    }).to_parquet(staging / "predictions.parquet")
    # matched_baseline must exist here -- this staging is fully "included" (unlike
    # the two sibling tests above, which fail inside _include_curation before ever
    # reaching it), so build_events' _model_preset_tags reads it too
    # (presets.model_for_results in build_config is "baseline"). The single b1 row
    # is v1's exemplar box: unmatched by baseline, caught by graph_rate_night.
    pd.DataFrame({
        "annotation_token": ["b1"], "sample_data_token": ["v1"],
        "category_group": ["car"], "x_min": [1.0], "y_min": [1.0],
        "x_max": [2.0], "y_max": [2.0],
        "matched_baseline": pd.array([False], dtype="boolean"),
        "below_visibility_min": [False],
    }).to_parquet(staging / "gt_boxes.parquet")
    (staging / "crops" / "v0.jpg").write_bytes(b"\xff\xd8\xff\xe0crop")
    (staging / "crops" / "v1.jpg").write_bytes(b"\xff\xd8\xff\xe0crop1")
    config["hero"] = {"token": "v0"}
    config["al"]["exemplar_tokens"] = ["v1"]
    build_config.write_text(yaml.safe_dump(config))

    manifest = run_build(build_config)
    assert manifest["validation"]["curation"] == "included"
    assert manifest["validation"]["events"] == "included"


def test_build_raises_named_error_on_partial_curation_staging(build_config: Path) -> None:
    """Review round 2, item 1: a staging dir with ONLY frame_manifest.parquet (i.e.
    `demo curate` ran but `demo infer` has not yet) is an operator mid-flow, not a
    no-curation machine -- treating it as silently "absent" would hide their
    mistake. It must fail loudly and name `demo infer` as the next step, and it must
    not crash with a raw FileNotFoundError out of the copy2 loop first."""
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    staging = Path(config["curation"]["staging_dir"])
    staging.mkdir(parents=True)
    pd.DataFrame({
        "sample_data_token": ["v0"], "split": ["val"], "filename": ["images/v0.jpg"],
        "curation_buckets": [["night_failure"]],
    }).to_parquet(staging / "frame_manifest.parquet")
    # predictions.parquet / gt_boxes.parquet deliberately absent -- `demo infer` has
    # not run yet.
    with pytest.raises(ValueError, match="demo infer"):
        run_build(build_config)


def test_build_tolerates_a_corrupt_lancedb_store(
    build_config: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Review round 2, item 2: `lancedb_path.is_dir()` can be True while the store
    itself has no usable table inside (e.g. an existing-but-empty directory) --
    export_thumbs then raises from lancedb's own guts. Thumbnails are supplementary
    (crops already cover every token), so this must degrade to a warning and let the
    build finish, not crash it -- the crop-or-thumb validation still runs and would
    still fail loudly if coverage were actually lost."""
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    # Same shape as _stage_minimal_curation (see the sibling curation test above) --
    # this test is about the thumbnail export degrading, not about the staging.
    _stage_minimal_curation(Path(config["curation"]["staging_dir"]))
    config["hero"] = {"token": "v0"}
    config["al"]["exemplar_tokens"] = ["v0"]
    build_config.write_text(yaml.safe_dump(config))

    # Exists (is_dir() True) but has no valid lance table inside -- open_table raises.
    Path(config["paths"]["lancedb_path"]).mkdir(parents=True)

    with caplog.at_level(logging.WARNING, logger="nuscenes_data_engine"):
        manifest = run_build(build_config)
    assert manifest["validation"]["curation"] == "included"
    assert "thumbnail export skipped" in caplog.text


def test_build_enriches_gt_boxes_with_distance_and_size(build_config: Path) -> None:
    import yaml

    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    _stage_minimal_curation(Path(config["curation"]["staging_dir"]))
    config["hero"] = {"token": "v0"}
    config["al"]["exemplar_tokens"] = ["v0"]
    build_config.write_text(yaml.safe_dump(config))
    processed = Path(config["paths"]["processed_dir"])
    # a1/a2 join onto the annotation_token values _stage_minimal_curation already
    # staged for v0's gt_boxes. This APPENDS to (rather than overwrites) the fixture's
    # existing annotations_3d.parquet: the flagship rows (f1..f3, see tiny_inputs'
    # extended annotation_token column) must survive untouched so the flagship SQL
    # count doesn't move -- a1/a2 carry no sample_token, so the flagship join (which
    # requires a matching canbus.sample_token) simply never sees them.
    existing = pd.read_parquet(processed / "annotations_3d.parquet")
    extra = pd.DataFrame({
        "annotation_token": ["a1", "a2"],
        "distance_to_ego_m": [4.5, 31.0],
    })
    pd.concat([existing, extra], ignore_index=True).to_parquet(
        processed / "annotations_3d.parquet"
    )
    manifest = run_build(build_config)
    out = Path(config["paths"]["out_dir"])
    gt = pd.read_parquet(out / "gt_boxes.parquet").set_index("annotation_token")
    assert gt.loc["a1", "distance_to_ego_m"] == 4.5
    assert gt.loc["a1", "size_bucket"] == "small"     # 20x20 = 400 < 1024
    assert gt.loc["a2", "size_bucket"] == "large"     # 200x200 = 40000 >= 9216
    assert manifest["validation"]["curation"] == "included"


def test_size_bucket_boundaries_exact() -> None:
    from nuscenes_data_engine.demo.build import _size_bucket

    assert _size_bucket(32.0 * 32.0 - 1) == "small"
    assert _size_bucket(32.0 * 32.0) == "medium"
    assert _size_bucket(96.0 * 96.0 - 1) == "medium"
    assert _size_bucket(96.0 * 96.0) == "large"


def test_size_bucket_rejects_degenerate_area() -> None:
    """A NaN or negative box area is corrupt data (e.g. x_max < x_min upstream), not
    a legitimate "large" or silently-computed "small" bucket -- it must fail loudly
    naming the bad value, not fall through the < comparisons (NaN compares False to
    everything, which would otherwise silently fall through to "large")."""
    import math

    from nuscenes_data_engine.demo.build import _size_bucket

    with pytest.raises(ValueError, match="degenerate box area"):
        _size_bucket(math.nan)
    with pytest.raises(ValueError, match="degenerate box area"):
        _size_bucket(-1.0)


def test_build_hero_from_token_copies_crop_and_records_token(build_config: Path) -> None:
    import json as _json

    import yaml

    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    _stage_minimal_curation(Path(config["curation"]["staging_dir"]))
    config["hero"] = {"token": "v0"}          # v0 is a staged crop in the fixture
    config["al"]["exemplar_tokens"] = ["v0"]
    build_config.write_text(yaml.safe_dump(config))
    run_build(build_config)
    out = Path(config["paths"]["out_dir"])
    assert (out / "sample_frames" / "hero.jpg").read_bytes() == (
        out / "sample_frames" / "crops" / "v0.jpg"
    ).read_bytes()
    metrics = _json.loads((out / "overview_metrics.json").read_text())
    assert metrics["hero_token"] == "v0"


def test_build_null_hero_token_fails_loudly(build_config: Path) -> None:
    import yaml

    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    _stage_minimal_curation(Path(config["curation"]["staging_dir"]))
    config["hero"] = {"token": None}
    config["al"]["exemplar_tokens"] = ["v0"]
    build_config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="hero token"):
        run_build(build_config)


def test_build_hero_token_without_curation_fails_loudly(build_config: Path) -> None:
    import yaml

    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())   # fixture has no staging by default
    config["hero"] = {"token": "v0"}
    build_config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="hero"):
        run_build(build_config)


def test_build_raises_on_annotation_token_join_fanout(build_config: Path) -> None:
    """A duplicate annotation_token in annotations_3d would silently fan a single
    staged gt_boxes row out into two published rows via the left join -- caught as a
    row-count mismatch between the staged and enriched frames, not shipped quietly."""
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    processed = Path(config["paths"]["processed_dir"])
    existing = pd.read_parquet(processed / "annotations_3d.parquet")
    # Two rows both claiming annotation_token "a1" -- _stage_minimal_curation's
    # gt_boxes has exactly one "a1" row, so the join fans it out to two.
    dup = pd.DataFrame({
        "annotation_token": ["a1", "a1"],
        "distance_to_ego_m": [9.9, 10.1],
    })
    pd.concat([existing, dup], ignore_index=True).to_parquet(
        processed / "annotations_3d.parquet"
    )

    with pytest.raises(ValueError, match="fanout"):
        run_build(build_config)


def test_build_gt_boxes_transform_leaves_staging_untouched(build_config: Path) -> None:
    """Transform-on-copy (Task 1): the staged gt_boxes.parquet is read to derive the
    published copy but must not itself be modified on disk -- and since it (along
    with the other two staging parquets) is what the build actually read, all three
    must appear in manifest['inputs']."""
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    staging = Path(config["curation"]["staging_dir"])
    before = (staging / "gt_boxes.parquet").read_bytes()

    manifest = run_build(build_config)

    assert (staging / "gt_boxes.parquet").read_bytes() == before
    hashed_names = {Path(key).name for key in manifest["inputs"]}
    assert {"frame_manifest.parquet", "gt_boxes.parquet", "predictions.parquet"} <= hashed_names


# --- Phase 5: scenario events + recorded semsearch (Task 2) ---------------------


def test_build_includes_scenario_events_without_curation(build_config: Path) -> None:
    """Events are computed LIVE from processed_dir regardless of curation staging --
    build_config's default fixture has no curation staged at all, proving
    _include_events's "staging_dir from curation config when the three staging
    parquets exist, else None" branch resolves to None here (the two model-result
    presets are silently skipped via build_events' own warning, not a build
    failure) and events still ship. tiny_inputs' extended processed fixture (Task 2
    review round: samples.parquet/canbus.parquet now carry the columns build_events
    needs) yields exactly one flagship event (s1: hard braking + a pedestrian at
    5m), the same single row the pre-existing flagship SQL count already covers."""
    from nuscenes_data_engine.demo.build import run_build

    manifest = run_build(build_config)
    out = Path(yaml.safe_load(build_config.read_text())["paths"]["out_dir"])
    assert (out / "scenario_events.parquet").is_file()
    events = pd.read_parquet(out / "scenario_events.parquet")
    assert len(events) == 1
    assert events.iloc[0]["sample_data_token"] == "s1"
    assert manifest["validation"]["events"] == "included"
    assert manifest["validation"]["flagship_events"] == 1
    assert manifest["validation"]["n_events"] == 1
    assert "scenario_events.parquet" in manifest["outputs"]


def test_include_events_flagship_mismatch_raises(build_config: Path, tmp_path: Path) -> None:
    """_include_events threads config["flagship"]["expected_sql_count"] straight to
    build_events' own flagship_expected assertion -- a config claiming 2 flagship
    events when the processed data actually has 1 must fail loudly, from
    build_events' independent (pandas, not DuckDB) computation, not just the
    pre-existing SQL-level check in run_build (test_build_fails_on_wrong_flagship_
    count already covers that one)."""
    from nuscenes_data_engine.demo.build import _include_events

    config = yaml.safe_load(build_config.read_text())
    config["flagship"]["expected_sql_count"] = 2  # actual pre-cap count is 1
    out = tmp_path / "events_out"
    out.mkdir()
    with pytest.raises(ValueError, match="flagship"):
        _include_events(config, out)


def test_build_exports_thumbs_for_events_and_filmstrip_neighbors(
    build_config: Path,
) -> None:
    """Thumbs are exported for the event token AND its filmstrip neighbors, not just
    the event itself. s1/s2 are build_config's only two CAM_FRONT keyframes (same
    scene, s1 timestamp=1000, s2 timestamp=1001) -- s1 is the sole flagship event,
    so its filmstrip t_plus1 neighbor is s2 (t_minus1/t_minus2/t_plus2 are all NA,
    scene-edge). Both tokens' thumbnails must land, reusing the tmp-LanceDB-store
    pattern from test_export_thumbs_writes_one_jpeg_per_token."""
    lancedb = pytest.importorskip("lancedb")
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    db = lancedb.connect(config["paths"]["lancedb_path"])
    db.create_table(
        config["paths"]["lancedb_table"],
        data=[
            {"sample_data_token": "s1", "thumbnail": b"\xff\xd8\xff\xe0one"},
            {"sample_data_token": "s2", "thumbnail": b"\xff\xd8\xff\xe0two"},
        ],
    )

    manifest = run_build(build_config)

    out = Path(config["paths"]["out_dir"])
    assert (out / "sample_frames" / "thumbs" / "s1.jpg").is_file()
    assert (out / "sample_frames" / "thumbs" / "s2.jpg").is_file()
    assert manifest["validation"]["events"] == "included"


def test_build_events_thumbs_skip_warn_when_store_absent(
    build_config: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No LanceDB store at all (build_config's default) must degrade to a warning,
    same as _include_curation's own thumbnail handling -- not fail the build."""
    from nuscenes_data_engine.demo.build import run_build

    with caplog.at_level(logging.WARNING, logger="nuscenes_data_engine"):
        manifest = run_build(build_config)
    assert manifest["validation"]["events"] == "included"
    assert "event frames ship without thumbnails" in caplog.text


def test_include_events_dedups_thumbs_already_exported(
    build_config: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A token that already has a thumb on disk (e.g. exported by the curation step
    that runs earlier in run_build) must NOT be re-requested from the LanceDB store.
    Proven by pre-seeding out_dir with a thumb for s1 (the flagship event's own
    token) and populating the store with ONLY s2 (s1's t_plus1 filmstrip neighbor):
    if dedup didn't exclude s1 from the request, export_thumbs would raise "tokens
    missing from the LanceDB store" for s1 (genuinely absent from THIS store),
    which the broad except in build.py would swallow into a warning -- so the
    absence of that warning is the proof dedup worked, and s2's thumb still lands
    from the real request. s1's pre-seeded thumb is asserted untouched."""
    lancedb = pytest.importorskip("lancedb")
    from nuscenes_data_engine.demo.build import _include_events

    config = yaml.safe_load(build_config.read_text())
    db = lancedb.connect(config["paths"]["lancedb_path"])
    db.create_table(
        config["paths"]["lancedb_table"],
        data=[{"sample_data_token": "s2", "thumbnail": b"\xff\xd8\xff\xe0two"}],
    )
    out = tmp_path / "out"
    thumbs_dir = out / "sample_frames" / "thumbs"
    thumbs_dir.mkdir(parents=True)
    (thumbs_dir / "s1.jpg").write_bytes(b"\xff\xd8\xff\xe0already-there")

    with caplog.at_level(logging.WARNING, logger="nuscenes_data_engine"):
        result = _include_events(config, out)

    assert result["events"] == "included"
    assert "event thumbnail export skipped" not in caplog.text
    assert (thumbs_dir / "s2.jpg").is_file()
    assert (thumbs_dir / "s1.jpg").read_bytes() == b"\xff\xd8\xff\xe0already-there"


def test_build_notes_absent_semsearch(build_config: Path) -> None:
    """No `demo semsearch` staging (build_config's default) must not fail the build
    -- mirrors _include_curation's own absent-staging fallback."""
    from nuscenes_data_engine.demo.build import run_build

    manifest = run_build(build_config)
    assert manifest["validation"]["semsearch"] == "absent"
    out = Path(yaml.safe_load(build_config.read_text())["paths"]["out_dir"])
    assert not (out / "semantic_search_results.parquet").exists()
    assert "semantic_search_results.parquet" not in manifest["outputs"]


def test_build_includes_semsearch_when_staged(build_config: Path) -> None:
    """A staged semantic_search_results.parquet (from `demo semsearch`) is copied
    into the package, hashed as an input, and its tokens get thumbnails."""
    lancedb = pytest.importorskip("lancedb")
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    staging = Path(config["curation"]["staging_dir"])
    staging.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "query": ["foggy road", "foggy road"],
            "rank": [1, 2],
            "sample_data_token": ["semq1", "semq2"],
            "score": [0.9, 0.8],
            "k": [8, 8],
        }
    ).to_parquet(staging / "semantic_search_results.parquet")
    db = lancedb.connect(config["paths"]["lancedb_path"])
    db.create_table(
        config["paths"]["lancedb_table"],
        data=[
            {"sample_data_token": "semq1", "thumbnail": b"\xff\xd8\xff\xe0one"},
            {"sample_data_token": "semq2", "thumbnail": b"\xff\xd8\xff\xe0two"},
        ],
    )

    manifest = run_build(build_config)

    out = Path(config["paths"]["out_dir"])
    assert (out / "semantic_search_results.parquet").is_file()
    on_disk = pd.read_parquet(out / "semantic_search_results.parquet")
    assert list(on_disk["sample_data_token"]) == ["semq1", "semq2"]
    assert manifest["validation"]["semsearch"] == "included"
    assert "semantic_search_results.parquet" in manifest["outputs"]
    assert any(
        Path(key).name == "semantic_search_results.parquet" for key in manifest["inputs"]
    )
    assert (out / "sample_frames" / "thumbs" / "semq1.jpg").is_file()
    assert (out / "sample_frames" / "thumbs" / "semq2.jpg").is_file()


def test_build_rejects_malformed_staged_semsearch_before_copying(build_config: Path) -> None:
    """A staged semantic_search_results.parquet missing a required column (e.g. a
    stale pre-`k`-column file from before item 7's schema change) must fail the
    build BEFORE the file is copied into the package -- not ship a gallery the
    page can't fully render, and not leave a partially-included package behind."""
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    staging = Path(config["curation"]["staging_dir"])
    staging.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "query": ["foggy road"],
            "rank": [1],
            "sample_data_token": ["semq1"],
            "score": [0.9],
            # no "k" column -- the schema this build now requires.
        }
    ).to_parquet(staging / "semantic_search_results.parquet")

    with pytest.raises(ValueError, match="missing columns"):
        run_build(build_config)

    out = Path(config["paths"]["out_dir"])
    assert not (out / "semantic_search_results.parquet").exists()


# --- Phase 6: graph subgraph export (Task 2) -------------------------------------

_SUBGRAPH_PRESETS = (
    "hard_braking_near_pedestrians",
    "night_pedestrians",
    "fast_cyclists",
    "rain_vru",
    "fn_pedestrians_night",
    "low_conf_braking",
)


def _stage_subgraphs(
    staging_dir: Path,
    *,
    flagship_cypher_count: int = 1,
    omit: tuple[str, ...] = (),
    flagship_event_tokens: tuple[str, ...] = ("s1",),
) -> None:
    """Stage six tiny ``graph_subgraphs/<preset>.json`` files under ``staging_dir``.

    Mirrors ``demo subgraph_export.export_subgraphs``'s real output shape (top-level
    keys: preset, count_cypher, sql_count, cypher_count, parity, note?, events)
    closely enough for ``_include_subgraphs``'s copy/hash/flagship-read logic to
    exercise, without a live Neo4j connection -- the exporter itself (Cypher text,
    assembly, live parity) is Task 1's own coverage (tests/test_demo_subgraphs.py).
    ``flagship_cypher_count`` controls the flagship preset's ``cypher_count`` --
    the one value ``_include_subgraphs`` actually reads, for the overview patch and
    the sql==cypher assertion. ``omit`` skips writing named presets, to exercise
    the partial-staging ValueError path.

    The ``events`` KEYS matter too since the stale-staging guard (item M2, Phase 6
    review): they must be exactly the tokens this build's own events frame ranks
    for that preset. ``tiny_inputs`` tags exactly one event -- "s1", the flagship's
    (hard braking + a pedestrian at 5m) -- and nothing at all for the other five
    presets, so that is what ``flagship_event_tokens`` defaults to.
    ``flagship_event_tokens`` can be pointed at a token the events frame does NOT
    rank, to exercise that guard.
    """
    out = staging_dir / "graph_subgraphs"
    out.mkdir(parents=True, exist_ok=True)
    empty_subgraph = {"nodes": [], "edges": [], "path": []}
    for preset in _SUBGRAPH_PRESETS:
        if preset in omit:
            continue
        is_flagship = preset == "hard_braking_near_pedestrians"
        cypher_count = flagship_cypher_count if is_flagship else 1
        payload: dict[str, Any] = {
            "preset": preset,
            "count_cypher": "MATCH (n) RETURN count(n)",
            "sql_count": 1,
            "cypher_count": cypher_count,
            "parity": cypher_count == 1,
            "events": (
                {token: empty_subgraph for token in flagship_event_tokens}
                if is_flagship
                else {}
            ),
        }
        (out / f"{preset}.json").write_text(json.dumps(payload, sort_keys=True, indent=2))


def test_build_includes_subgraphs_when_staged(build_config: Path) -> None:
    """A fully-staged graph_subgraphs/ dir (all six presets) is copied into the
    package, hashed as an input, and the flagship's computed Cypher count both
    patches overview_metrics.json and passes the sql==cypher assertion (build_
    config's flagship expected_sql_count is 1, matching this fixture's default
    flagship_cypher_count=1)."""
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    staging = Path(config["curation"]["staging_dir"])
    _stage_subgraphs(staging, flagship_cypher_count=1)

    manifest = run_build(build_config)

    out = Path(config["paths"]["out_dir"])
    for preset in _SUBGRAPH_PRESETS:
        assert (out / "graph_subgraphs" / f"{preset}.json").is_file()
        assert (staging / "graph_subgraphs" / f"{preset}.json").is_file()  # staging untouched

    assert manifest["validation"]["subgraphs"] == "included"
    assert manifest["validation"]["flagship_cypher"] == 1

    hashed_names = {Path(key).name for key in manifest["inputs"]}
    for preset in _SUBGRAPH_PRESETS:
        assert f"{preset}.json" in hashed_names

    overview = json.loads((out / "overview_metrics.json").read_text())
    assert overview["flagship"]["cypher"] == 1
    assert overview["flagship"]["cypher_source"] == "computed (neo4j, demo subgraphs)"


def test_build_raises_on_subgraph_flagship_parity_mismatch(build_config: Path) -> None:
    """The flagship's computed Cypher count disagreeing with the SQL count (== the
    plan's headline "the SQL and Cypher counts agree" claim) must fail the build
    loudly, naming both counts -- never silently publish a mismatched claim."""
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    staging = Path(config["curation"]["staging_dir"])
    _stage_subgraphs(staging, flagship_cypher_count=2)  # sql count is 1 -- mismatch

    with pytest.raises(ValueError, match=r"(?s)1.*2|2.*1"):
        run_build(build_config)


def test_build_notes_absent_subgraphs(build_config: Path) -> None:
    """No `demo subgraphs` staging (build_config's default) must not fail the
    build -- mirrors _include_semsearch's own absent-staging fallback -- and the
    overview's flagship Cypher stays the sourced value/label exactly as before
    (the Phase-1 contract, unchanged by this phase when there's nothing to
    compute)."""
    from nuscenes_data_engine.demo.build import run_build

    manifest = run_build(build_config)

    assert manifest["validation"]["subgraphs"] == "absent"
    assert "flagship_cypher" not in manifest["validation"]
    out = Path(yaml.safe_load(build_config.read_text())["paths"]["out_dir"])
    assert not (out / "graph_subgraphs").exists()

    overview = json.loads((out / "overview_metrics.json").read_text())
    assert overview["flagship"]["cypher"] == 30           # build_config's sourced value
    assert overview["flagship"]["cypher_source"] == "docs/GRAPH.md"


def test_build_raises_named_error_on_partial_subgraph_staging(build_config: Path) -> None:
    """A graph_subgraphs/ dir with only SOME of the six presets (`demo subgraphs`
    interrupted mid-run, or a stale partial staging from before a preset existed)
    is an operator mid-flow state, same rationale as _include_curation's own
    partial-staging ValueError -- it must not be silently treated as "absent"
    (hiding the mistake) nor silently shipped partial (a page some presets can't
    render for). The error must name the missing preset."""
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    staging = Path(config["curation"]["staging_dir"])
    _stage_subgraphs(staging, omit=("rain_vru",))

    with pytest.raises(ValueError, match="rain_vru"):
        run_build(build_config)


def test_build_raises_on_stale_subgraph_staging(build_config: Path) -> None:
    """A staging dir holding all six presets but subgraphs for the WRONG events (a
    `demo subgraphs` run from before the processed tables/presets changed) must fail
    the build naming the preset, not ship a package whose graph panel silently shows
    the honest-but-avoidable "no subgraph staged for this event" note on every card.

    tiny_inputs ranks exactly one flagship event ("s1"); staging a subgraph for some
    other token instead is exactly the stale case.
    """
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    staging = Path(config["curation"]["staging_dir"])
    _stage_subgraphs(staging, flagship_event_tokens=("a_token_from_a_previous_run",))

    with pytest.raises(ValueError, match="hard_braking_near_pedestrians") as excinfo:
        run_build(build_config)
    assert "demo subgraphs" in str(excinfo.value)


def test_build_records_al_explain_group_absent_then_included(build_config: Path) -> None:
    """Phase 7 (Task 3): the `demo al-explain` staging is a declared-optional input
    group, exactly like the subgraphs one -- absent on a fresh clone (the ordinary
    state, a warning not a failure), and once staged copied flat next to the other
    tables AND hashed as a build INPUT (the outputs sweep hashes the copies)."""
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    assert run_build(build_config)["validation"]["al_explain"] == "absent"

    staging = Path(config["curation"]["staging_dir"]) / "al_explain"
    staging.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_data_token": ["t0"], "arm": ["graph_rate_night"]}).to_parquet(
        staging / "al_selection_explain.parquet", index=False
    )
    (staging / "al_explain_validation.json").write_text(json.dumps({
        "arm": "graph_rate_night", "selected_match": True, "communities_match": True,
        "n_selected": 1, "n_communities": 1,
    }))

    manifest = run_build(build_config)
    assert manifest["validation"]["al_explain"] == "included"
    out = Path(config["paths"]["out_dir"])
    assert (out / "al_selection_explain.parquet").is_file()
    assert (out / "al_explain_validation.json").is_file()
    assert str(staging / "al_selection_explain.parquet") in manifest["inputs"]
    assert str(staging / "al_explain_validation.json") in manifest["inputs"]
    assert "al_selection_explain.parquet" in manifest["outputs"]


def test_build_accepts_subgraph_staging_matching_the_events_frame(build_config: Path) -> None:
    """The other side of the stale guard: staging keyed by exactly the tokens this
    build's events frame ranks (the default helper shape) builds cleanly, and the
    package's flagship JSON really does carry that event's subgraph."""
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    _stage_subgraphs(Path(config["curation"]["staging_dir"]))

    manifest = run_build(build_config)
    assert manifest["validation"]["subgraphs"] == "included"

    out = Path(config["paths"]["out_dir"])
    shipped = json.loads((out / "graph_subgraphs" / "hard_braking_near_pedestrians.json").read_text())
    assert set(shipped["events"]) == {"s1"}


# --- Phase 8 (Task 2): recorded chat replays --------------------------------------


def _chat_frame(token: str, **overrides: Any) -> dict[str, Any]:
    """A frame dict shaped exactly like ``chat_record.FRAME_COLUMNS`` (the package's
    projected shape -- no thumbnail bytes, filename, or scene_description)."""
    frame: dict[str, Any] = {
        "sample_data_token": token,
        "scene_name": "scene-0916",
        "location": "singapore-onenorth",
        "is_night": True,
        "is_rain": True,
        "channel": "CAM_FRONT",
        "score": 0.87,
    }
    frame.update(overrides)
    return frame


def _chat_record_entry(
    *,
    id: str,
    kind: str = "showcase",
    question: str = "What happened?",
    answer: str | None = "An answer.",
    model: str | None = "claude-test",
    provider: str = "anthropic",
    steps: list[dict[str, Any]] | None = None,
    frames: list[dict[str, Any]] | None = None,
    charts: list[dict[str, Any]] | None = None,
    checks: dict[str, Any] | None = None,
    latency_s: float = 1.23,
    error: str | None = None,
) -> dict[str, Any]:
    """A replay dict shaped exactly like ``chat_record.RECORD_KEYS``."""
    return {
        "id": id,
        "kind": kind,
        "question": question,
        "answer": answer,
        "model": model,
        "provider": provider,
        "steps": steps or [],
        "frames": frames or [],
        "charts": charts or [],
        "checks": checks,
        "latency_s": latency_s,
        "error": error,
    }


def _default_chat_replays() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A fresh, valid (records, summary) pair each call: one showcase replay (1
    chart, 2 frames, 3 steps) plus two eval replays (one passed, one failed with a
    numeric-check miss) -- mirrors ``chat_record.record_session``'s real output
    shape closely enough for ``_validate_chat_replays``/``_include_chat_replay`` to
    exercise, without a live agent session."""
    records = [
        _chat_record_entry(
            id="show1",
            kind="showcase",
            steps=[
                {"tool": "search_frames", "input": {"query": "fog"}, "output": "2 rows"},
                {"tool": "search_frames", "input": {"query": "fog"}, "output": "2 rows"},
                {"tool": "make_chart", "input": {"kind": "bar"}, "output": "charted"},
            ],
            frames=[_chat_frame("f1"), _chat_frame("f2")],
            charts=[
                {
                    "kind": "bar",
                    "title": "Night frames per location",
                    "columns": ["location", "n"],
                    "rows": [["boston-seaport", 3], ["singapore-onenorth", 5]],
                }
            ],
        ),
        _chat_record_entry(
            id="eval_pass",
            kind="eval",
            question="How many samples are there?",
            checks={"passed": True, "numeric": True},
        ),
        _chat_record_entry(
            id="eval_fail",
            kind="eval",
            question="How many samples are in boston-seaport?",
            checks={"passed": False, "numeric": False},
        ),
    ]
    summary = {
        "model": "claude-test",
        "provider": "anthropic",
        "recorded_at": "2026-08-21T00:00:00+00:00",
        "git_sha": "deadbeef",
        "n_eval": 2,
        "n_passed": 1,
        "n_showcase": 1,
        "search_available": True,
        "graph_available": True,
        "max_turns": 8,
    }
    return records, summary


def _stage_chat_replays(
    staging_dir: Path,
    *,
    records: list[dict[str, Any]] | None = None,
    summary: dict[str, Any] | None = None,
) -> None:
    """Stage ``chat_replays.json`` + ``chat_replay_summary.json`` under
    ``staging_dir``, mirroring ``demo chat-record``'s ``write_staging`` output shape
    (Task 1) -- sorted keys, indent 2, newline-terminated."""
    default_records, default_summary = _default_chat_replays()
    if records is None:
        records = default_records
    if summary is None:
        summary = default_summary
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "chat_replays.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )
    (staging_dir / "chat_replay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def test_include_chat_replay_absent_returns_absent_and_copies_nothing(
    build_config: Path,
) -> None:
    """No `demo chat-record` staging (build_config's default) must not fail the
    build -- mirrors _include_al_explain's own absent-staging fallback."""
    from nuscenes_data_engine.demo.build import _include_chat_replay

    config = yaml.safe_load(build_config.read_text())
    out_dir = Path(config["paths"]["out_dir"])
    out_dir.mkdir(parents=True)

    assert _include_chat_replay(config, out_dir) == "absent"
    assert not (out_dir / "chat_replays.json").exists()
    assert not (out_dir / "chat_replay_summary.json").exists()

    # No `curation:` section at all is also "absent", same as every other optional
    # group.
    assert _include_chat_replay({}, out_dir) == "absent"


def test_include_chat_replay_partial_staging_raises_naming_the_missing_file(
    build_config: Path,
) -> None:
    """A staging dir holding only one of the two files is an operator mid-flow (an
    interrupted `demo chat-record`), not a no-replay machine."""
    from nuscenes_data_engine.demo.build import _include_chat_replay

    config = yaml.safe_load(build_config.read_text())
    staging_dir = Path(config["curation"]["staging_dir"]) / "chat_replays"
    staging_dir.mkdir(parents=True)
    (staging_dir / "chat_replays.json").write_text("[]")
    out_dir = Path(config["paths"]["out_dir"])
    out_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match=r"chat_replay_summary\.json"):
        _include_chat_replay(config, out_dir)
    assert not (out_dir / "chat_replays.json").exists()


def test_include_chat_replay_copies_files_and_records_validation(
    build_config: Path,
) -> None:
    """A fully-staged chat-replay group is copied flat into the package, hashed as
    an input, and its record-derived counts land in manifest['validation']."""
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    _stage_chat_replays(Path(config["curation"]["staging_dir"]) / "chat_replays")

    manifest = run_build(build_config)

    out = Path(config["paths"]["out_dir"])
    assert (out / "chat_replays.json").is_file()
    assert (out / "chat_replay_summary.json").is_file()
    assert "chat_replays.json" in manifest["outputs"]
    assert "chat_replay_summary.json" in manifest["outputs"]
    assert any(Path(key).name == "chat_replays.json" for key in manifest["inputs"])
    assert any(Path(key).name == "chat_replay_summary.json" for key in manifest["inputs"])
    assert manifest["validation"]["chat_replay"] == "included"
    assert manifest["validation"]["n_replays"] == 3
    assert manifest["validation"]["n_replay_passed"] == 1
    assert manifest["validation"]["n_replay_frames"] == 2


def test_include_chat_replay_exports_frame_thumbs_deduped(
    build_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The distinct frame tokens across every staged replay are exported as
    thumbnails, deduped through the same helper every other optional group uses."""
    from nuscenes_data_engine.demo import build as build_module

    config = yaml.safe_load(build_config.read_text())
    _stage_chat_replays(Path(config["curation"]["staging_dir"]) / "chat_replays")
    out_dir = Path(config["paths"]["out_dir"])
    out_dir.mkdir(parents=True)

    captured: dict[str, Any] = {}

    def fake_export_thumbs_deduped(*, config: Any, out_dir: Path, tokens: set, context: str) -> None:
        captured["tokens"] = tokens
        captured["context"] = context

    monkeypatch.setattr(build_module, "_export_thumbs_deduped", fake_export_thumbs_deduped)

    assert build_module._include_chat_replay(config, out_dir) == "included"
    assert captured == {"tokens": {"f1", "f2"}, "context": "chat_replay"}


def test_include_chat_replay_skips_thumb_export_when_no_frames(
    build_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A staged replay group with no frames at all (every showcase answer was
    text/chart/cypher-only) must not call the thumb exporter."""
    from nuscenes_data_engine.demo import build as build_module

    config = yaml.safe_load(build_config.read_text())
    records = [_chat_record_entry(id="eval_only", kind="eval", checks={"passed": True})]
    summary = {
        "model": "claude-test", "provider": "anthropic",
        "recorded_at": "2026-08-21T00:00:00+00:00", "git_sha": "deadbeef",
        "n_eval": 1, "n_passed": 1, "n_showcase": 0,
        "search_available": True, "graph_available": True, "max_turns": 8,
    }
    _stage_chat_replays(
        Path(config["curation"]["staging_dir"]) / "chat_replays", records=records, summary=summary
    )
    out_dir = Path(config["paths"]["out_dir"])
    out_dir.mkdir(parents=True)

    called = False

    def fake_export_thumbs_deduped(**kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(build_module, "_export_thumbs_deduped", fake_export_thumbs_deduped)

    assert build_module._include_chat_replay(config, out_dir) == "included"
    assert called is False


def test_validate_chat_replays_rejects_bad_staging(tmp_path: Path) -> None:
    """Every documented malformed-staging case must raise ValueError, naming the
    staged path -- a schema drift or hand-edited staging dir must never ship."""
    from nuscenes_data_engine.demo.build import _validate_chat_replays

    path = tmp_path / "chat_replays.json"

    records, summary = _default_chat_replays()
    records[1]["id"] = records[0]["id"]  # duplicate id
    with pytest.raises(ValueError, match="duplicate"):
        _validate_chat_replays(records, summary, path)

    records, summary = _default_chat_replays()
    records[0]["frames"][0]["sample_data_token"] = "../x"  # non-alphanumeric token
    with pytest.raises(ValueError):
        _validate_chat_replays(records, summary, path)

    records, summary = _default_chat_replays()
    records[0]["charts"][0]["rows"] = [["boston-seaport"]]  # ragged (columns has 2)
    with pytest.raises(ValueError):
        _validate_chat_replays(records, summary, path)

    records, summary = _default_chat_replays()
    records[1]["checks"] = None  # eval record with no checks
    with pytest.raises(ValueError):
        _validate_chat_replays(records, summary, path)

    records, summary = _default_chat_replays()
    del records[0]["latency_s"]  # missing a RECORD_KEYS key
    with pytest.raises(ValueError):
        _validate_chat_replays(records, summary, path)

    records, summary = _default_chat_replays()
    records[0]["kind"] = "bogus"  # unknown kind
    with pytest.raises(ValueError):
        _validate_chat_replays(records, summary, path)

    records, summary = _default_chat_replays()
    summary["n_eval"] = 99  # disagrees with the records
    with pytest.raises(ValueError):
        _validate_chat_replays(records, summary, path)

    records, summary = _default_chat_replays()
    summary["n_showcase"] = 99  # disagrees with the records
    with pytest.raises(ValueError):
        _validate_chat_replays(records, summary, path)

    records, summary = _default_chat_replays()
    del summary["max_turns"]  # missing a summary key
    with pytest.raises(ValueError):
        _validate_chat_replays(records, summary, path)

    # sanity: the unmodified fixture is valid and raises nothing.
    records, summary = _default_chat_replays()
    _validate_chat_replays(records, summary, path)


def test_build_is_deterministic_with_chat_replays_staged(build_config: Path) -> None:
    """Same guarantee as test_build_is_deterministic, extended to a staged chat-
    replay group (Phase 8, Task 2) -- the copied JSONs are byte-identical inputs, so
    the outputs stay byte-stable across rebuilds same as every other optional
    group's."""
    from nuscenes_data_engine.demo.build import run_build

    config = _stage_and_pick_hero(build_config)
    _stage_subgraphs(Path(config["curation"]["staging_dir"]))
    _stage_chat_replays(Path(config["curation"]["staging_dir"]) / "chat_replays")
    out = Path(config["paths"]["out_dir"])
    run_build(build_config)
    first = {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
    run_build(build_config)
    second = {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
    assert first == second
