"""Tests for the demo artifact builder (pure exporters; no torch, no network)."""

from __future__ import annotations

import json
from pathlib import Path

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
    pd.DataFrame({"sample_data_token": ["s1", "s2", "s3"]}).to_parquet(
        processed / "samples.parquet"
    )
    pd.DataFrame({"sample_token": ["s1"] * 4}).to_parquet(processed / "annotations.parquet")
    pd.DataFrame(
        {
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
        }
    ).to_parquet(processed / "canbus.parquet")
    pd.DataFrame(
        {"sample_token": ["s1", "s2"], "speed_mps": [10.1, 4.9]}
    ).to_parquet(processed / "ego_pose.parquet")
    (al / "results.json").write_text(
        json.dumps(
            {
                "baseline": {
                    "n_train_images": 100,
                    "overall": {"mAP50-95": 0.20},
                    "night": {"mAP50-95": 0.10},
                },
                "random": {
                    "n_train_images": 115,
                    "overall": {"mAP50-95": 0.21},
                    "night": {"mAP50-95": 0.099},
                },
                "weak_random": {
                    "n_train_images": 110,
                    "overall": {"mAP50-95": 0.2018},
                    "night": {"mAP50-95": 0.095},
                },
                "graph_rate_night": {
                    "n_train_images": 130,
                    "overall": {"mAP50-95": 0.22},
                    "night": {"mAP50-95": 0.1101},
                },
                "weak_graph_rate_night": {
                    "n_train_images": 122,
                    "overall": {"mAP50-95": 0.205},
                    "night": {"mAP50-95": 0.09},
                },
            }
        )
    )
    return {"processed": processed, "al": al}


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


def test_export_overview_missing_input_raises(tmp_path: Path, tiny_inputs: dict[str, Path]) -> None:
    (tiny_inputs["processed"] / "canbus.parquet").unlink()
    with pytest.raises(ValueError, match=r"canbus\.parquet"):
        export_overview(
            processed_dir=tiny_inputs["processed"],
            al_dir=tiny_inputs["al"],
            out_dir=tmp_path / "demo_data",
            flagship_cypher={"count": 30, "source": "docs/GRAPH.md"},
        )


def test_export_al_results_reshapes_and_sorts(tmp_path: Path, tiny_inputs: dict[str, Path]) -> None:
    from nuscenes_data_engine.demo.exporters import export_al_results

    out = tmp_path / "demo_data"
    df = export_al_results(al_dir=tiny_inputs["al"], out_dir=out)
    on_disk = pd.read_parquet(out / "active_learning_results.parquet")
    pd.testing.assert_frame_equal(df, on_disk)
    assert list(df.columns) == [
        "arm", "n_train_images", "overall_map5095", "night_map5095",
        "delta_overall", "delta_night",
    ]
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
        export_al_results(al_dir=al, out_dir=tmp_path / "demo_data")


def test_export_weaksup_empty_dir_raises(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.exporters import export_weaksup

    al = tmp_path / "active_learning"
    al.mkdir()
    with pytest.raises(ValueError, match=r"no \*_pseudo_summary\.json"):
        export_weaksup(al_dir=al, out_dir=tmp_path / "demo_data")


def test_export_weaksup_reads_summaries(tmp_path: Path, tiny_inputs: dict[str, Path]) -> None:
    from nuscenes_data_engine.demo.exporters import export_weaksup

    al = tiny_inputs["al"]
    (al / "random_pseudo_summary.json").write_text(json.dumps({
        "arm": "random", "n_candidates": 1500, "n_accepted": 958, "retention": 0.639,
        "n_boxes": 1942, "mean_boxes_per_accepted_frame": 2.0,
        "mean_gt_boxes_per_accepted_frame": 3.9,
    }))
    out = tmp_path / "demo_data"
    df = export_weaksup(al_dir=al, out_dir=out)
    assert (out / "weak_supervision_results.parquet").is_file()
    row = df.set_index("arm").loc["random"]
    assert row["n_accepted"] == 958
    assert row["verifier_retention"] == pytest.approx(0.639)
    # arms without a summary file are simply absent, not an error
    assert "graph_rate_night" not in set(df["arm"])


def test_export_hero_copies_the_configured_mosaic(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.exporters import export_hero

    mlruns = tmp_path / "mlruns"
    run_dir = mlruns / "artifacts" / "run123" / "artifacts" / "ultralytics_run"
    run_dir.mkdir(parents=True)
    (run_dir / "val_batch0_pred.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    out = tmp_path / "demo_data"
    dest = export_hero(mlruns_dir=mlruns, run_id="run123", mosaic="val_batch0_pred.jpg", out_dir=out)
    assert dest == out / "sample_frames" / "hero.jpg"
    assert dest.read_bytes().startswith(b"\xff\xd8")


def test_export_hero_missing_mosaic_raises(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.exporters import export_hero

    with pytest.raises(ValueError, match="hero mosaic"):
        export_hero(
            mlruns_dir=tmp_path, run_id="nope", mosaic="val_batch0_pred.jpg",
            out_dir=tmp_path / "demo_data",
        )


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


@pytest.fixture()
def build_config(tmp_path: Path, tiny_inputs: dict[str, Path]) -> Path:
    al = tiny_inputs["al"]
    (al / "random_pseudo_summary.json").write_text(json.dumps({
        "arm": "random", "n_candidates": 10, "n_accepted": 6, "retention": 0.6,
        "n_boxes": 12, "mean_boxes_per_accepted_frame": 2.0,
        "mean_gt_boxes_per_accepted_frame": 3.0,
    }))
    mlruns = tmp_path / "mlruns"
    hero = mlruns / "artifacts" / "runX" / "artifacts" / "ultralytics_run"
    hero.mkdir(parents=True)
    (hero / "val_batch0_pred.jpg").write_bytes(b"\xff\xd8\xff\xe0hero")
    config = {
        "paths": {
            "processed_dir": str(tiny_inputs["processed"]),
            "active_learning_dir": str(al),
            "mlruns_dir": str(mlruns),
            "lancedb_path": str(tmp_path / "lancedb"),
            "lancedb_table": "frames",
            "out_dir": str(tmp_path / "demo_data"),
        },
        "models": {"baseline": "runX"},
        "hero": {"run": "baseline", "mosaic": "val_batch0_pred.jpg"},
        "budgets": {"max_package_mb": 100},
        "flagship": {"expected_sql_count": 1, "cypher_count": 30,
                     "cypher_source": "docs/GRAPH.md"},
    }
    path = tmp_path / "demo.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_build_writes_validated_manifest(build_config: Path, tmp_path: Path) -> None:
    from nuscenes_data_engine.demo.build import run_build

    manifest = run_build(build_config)
    out = Path(yaml.safe_load(build_config.read_text())["paths"]["out_dir"])
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk == manifest
    assert manifest["git_sha"]
    assert manifest["outputs"]["overview_metrics.json"]["sha256"]
    assert manifest["outputs"]["active_learning_results.parquet"]["rows"] == 5
    assert manifest["validation"]["flagship_sql_count"] == 1
    assert manifest["validation"]["package_mb"] < 1


def test_build_is_deterministic(build_config: Path) -> None:
    from nuscenes_data_engine.demo.build import run_build

    out = Path(yaml.safe_load(build_config.read_text())["paths"]["out_dir"])
    run_build(build_config)
    first = {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
    run_build(build_config)
    second = {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
    assert first == second                    # byte-stable outputs (manifest has built_at)


def test_build_fails_on_wrong_flagship_count(build_config: Path) -> None:
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    config["flagship"]["expected_sql_count"] = 30      # tiny fixture yields 1, not 30
    build_config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="flagship"):
        run_build(build_config)


def test_build_fails_over_size_budget(build_config: Path) -> None:
    from nuscenes_data_engine.demo.build import run_build

    config = yaml.safe_load(build_config.read_text())
    config["budgets"]["max_package_mb"] = 0
    build_config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="budget"):
        run_build(build_config)
