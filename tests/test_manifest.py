"""Tests for the data-availability manifest (synthetic dataroot; offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nuscenes_data_engine.validation.manifest import (
    build_manifest,
    camera_keyframes_complete,
    summarize_manifest,
)

VERSION = "v1.0-test"


def _write_dataroot(root: Path, *, drop_camera_keyframe: bool = False) -> None:
    """Two scenes; camera keyframes + sweeps on disk, radar referenced but absent."""
    meta = root / VERSION
    meta.mkdir(parents=True)
    scenes = [{"token": f"scene{j}", "name": f"scene-000{j}"} for j in range(2)]
    samples = [{"token": f"s{j}", "scene_token": f"scene{j}"} for j in range(2)]
    sample_data = []
    for j in range(2):
        sample_data += [
            {
                "token": f"cam{j}",
                "sample_token": f"s{j}",
                "filename": f"samples/CAM_FRONT/frame{j}.jpg",
                "is_key_frame": True,
            },
            {
                "token": f"sweep{j}",
                "sample_token": f"s{j}",
                "filename": f"sweeps/CAM_FRONT/sweep{j}.jpg",
                "is_key_frame": False,
            },
            {
                "token": f"radar{j}",
                "sample_token": f"s{j}",
                "filename": f"samples/RADAR_FRONT/r{j}.pcd",
                "is_key_frame": True,
            },
        ]
    for name, table in (("scene", scenes), ("sample", samples), ("sample_data", sample_data)):
        (meta / f"{name}.json").write_text(json.dumps(table))

    for rec in sample_data:
        filename = str(rec["filename"])
        if filename.startswith("samples/RADAR"):
            continue  # radar referenced but never on disk
        if drop_camera_keyframe and filename == "samples/CAM_FRONT/frame1.jpg":
            continue
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


class TestBuildManifest:
    def test_presence_flags(self, tmp_path: Path) -> None:
        _write_dataroot(tmp_path)
        manifest = build_manifest(tmp_path, VERSION, tmp_path / "availability.parquet")
        assert len(manifest) == 6
        by_token = manifest.set_index("sample_data_token")
        assert by_token.loc["cam0", "present"]
        assert by_token.loc["sweep0", "present"]
        assert not by_token.loc["radar0", "present"]
        assert by_token.loc["cam0", "modality"] == "cam"
        assert by_token.loc["radar0", "modality"] == "radar"
        assert by_token.loc["cam1", "scene_name"] == "scene-0001"
        assert (tmp_path / "availability.parquet").is_file()

    def test_summary_counts_missing_channel(self, tmp_path: Path) -> None:
        _write_dataroot(tmp_path)
        manifest = build_manifest(tmp_path, VERSION, tmp_path / "a.parquet")
        summary = summarize_manifest(manifest).set_index(["channel", "is_key_frame"])
        assert summary.loc[("RADAR_FRONT", True), "n_referenced"] == 2
        assert summary.loc[("RADAR_FRONT", True), "n_present"] == 0
        assert summary.loc[("CAM_FRONT", True), "n_present"] == 2

    def test_camera_keyframes_gate(self, tmp_path: Path) -> None:
        _write_dataroot(tmp_path)
        complete = build_manifest(tmp_path, VERSION, tmp_path / "a.parquet")
        assert camera_keyframes_complete(complete) is True

        broken_root = tmp_path / "broken"
        _write_dataroot(broken_root, drop_camera_keyframe=True)
        broken = build_manifest(broken_root, VERSION, broken_root / "a.parquet")
        assert camera_keyframes_complete(broken) is False
        assert isinstance(broken, pd.DataFrame)
