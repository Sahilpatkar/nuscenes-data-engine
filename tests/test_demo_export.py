"""Tests for the demo artifact builder (pure exporters; no torch, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

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
    # CAN validation: corr of (10.0, 5.0) vs (10.1, 4.9) is ~1.0, derived not hardcoded
    assert 0.99 <= metrics["can_speed_r"] <= 1.0
    # headline results derived from results.json
    assert metrics["results"]["best_night_arm"] == "graph_rate_night"
    assert metrics["results"]["best_night_delta"] == pytest.approx(0.0101, abs=1e-6)
    assert metrics["results"]["weak_retention_of_gt_gain"] == pytest.approx(
        (0.205 - 0.20) / (0.22 - 0.20), abs=1e-6
    )


def test_export_overview_missing_input_raises(tmp_path: Path, tiny_inputs: dict[str, Path]) -> None:
    (tiny_inputs["processed"] / "canbus.parquet").unlink()
    with pytest.raises(ValueError, match=r"canbus\.parquet"):
        export_overview(
            processed_dir=tiny_inputs["processed"],
            al_dir=tiny_inputs["al"],
            out_dir=tmp_path / "demo_data",
            flagship_cypher={"count": 30, "source": "docs/GRAPH.md"},
        )
