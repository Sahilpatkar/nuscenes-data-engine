"""1-epoch CPU training smoke test (dedicated CI job; excluded from default runs).

Trains yolov8n for one epoch on a tiny synthetic dataset — no nuScenes access, no GPU.
Deliberately fails (rather than skips) when the weights download is unavailable: the
smoke-train CI job must never go green without actually training.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")  # train extra
pytest.importorskip("ultralytics")

import cv2
import numpy as np
import yaml

from nuscenes_data_engine.training.runtime import WEIGHTS_DIR, configure_ultralytics


def _make_yolo_dataset(root: Path, *, n_train: int = 8, n_val: int = 2, size: int = 64) -> Path:
    """Gray frames with one white rectangle each + matching YOLO labels + data.yaml."""
    rng = np.random.default_rng(0)
    for split, count in (("train", n_train), ("val", n_val)):
        images, labels = root / "images" / split, root / "labels" / split
        images.mkdir(parents=True), labels.mkdir(parents=True)
        for i in range(count):
            x, y = int(rng.integers(8, size - 24)), int(rng.integers(8, size - 24))
            w = h = 16
            img = np.full((size, size, 3), 96, np.uint8)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
            cv2.imwrite(str(images / f"{i}.jpg"), img)
            cx, cy = (x + w / 2) / size, (y + h / 2) / size
            (labels / f"{i}.txt").write_text(f"0 {cx} {cy} {w / size} {h / size}\n")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {"path": str(root), "train": "images/train", "val": "images/val", "names": {0: "car"}}
        )
    )
    return data_yaml


@pytest.mark.smoke_train
def test_one_epoch_cpu_train(tmp_path: Path) -> None:
    configure_ultralytics()  # before importing ultralytics: keep its writes in the repo
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset

    weights = WEIGHTS_DIR / "yolov8n.pt"
    if not weights.is_file():
        attempt_download_asset(str(weights))  # no suppress: a download failure must fail
    data_yaml = _make_yolo_dataset(tmp_path / "dataset")

    results = YOLO(str(weights)).train(
        data=str(data_yaml),
        epochs=1,
        imgsz=64,
        batch=4,
        device="cpu",
        workers=0,
        project=str(tmp_path / "runs"),
        name="smoke",
        exist_ok=True,
        plots=False,
        verbose=False,
        seed=0,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    assert best.is_file() and best.stat().st_size > 0
