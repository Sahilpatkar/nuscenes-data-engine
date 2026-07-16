"""Unit tests for the ingestion primitives (no dataset load required)."""

from __future__ import annotations

import numpy as np
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from nuscenes_data_engine.ingestion.categories import (
    CLASS_TO_INDEX,
    DETECTION_CLASSES,
    group_for,
)
from nuscenes_data_engine.ingestion.projection import project_box_to_2d

# A simple pinhole intrinsic: focal 1000 px, principal point at image centre.
INTRINSIC = np.array([[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]])
IMAGE_SIZE = (1600, 900)


def _box(center: tuple[float, float, float]) -> Box:
    """A 2x2x2 axis-aligned box (identity orientation) at ``center`` in the camera frame."""
    return Box(center=list(center), size=[2.0, 2.0, 2.0], orientation=Quaternion())


class TestCategories:
    def test_known_mappings(self) -> None:
        assert group_for("vehicle.car") == "car"
        assert group_for("vehicle.truck") == "truck"
        assert group_for("vehicle.bus.rigid") == "bus"
        assert group_for("vehicle.bus.bendy") == "bus"
        assert group_for("vehicle.bicycle") == "bicycle"
        assert group_for("human.pedestrian.adult") == "pedestrian"
        assert group_for("human.pedestrian.construction_worker") == "pedestrian"

    def test_unmapped_returns_none(self) -> None:
        assert group_for("movable_object.barrier") is None
        assert group_for("vehicle.motorcycle") is None
        assert group_for("animal") is None

    def test_class_index_ordering(self) -> None:
        assert DETECTION_CLASSES == ("car", "truck", "bus", "pedestrian", "bicycle")
        assert CLASS_TO_INDEX["car"] == 0
        assert CLASS_TO_INDEX["bicycle"] == len(DETECTION_CLASSES) - 1


class TestProjection:
    def test_box_in_front_projects_within_image(self) -> None:
        bbox = project_box_to_2d(_box((0.0, 0.0, 10.0)), INTRINSIC, IMAGE_SIZE)
        assert bbox is not None
        x_min, y_min, x_max, y_max = bbox
        # Box spans X,Y in [-1, 1] at Z in [9, 11]; nearest face dominates the extent.
        assert 0.0 <= x_min < 800.0 < x_max <= 1600.0
        assert 0.0 <= y_min < 450.0 < y_max <= 900.0

    def test_box_behind_camera_is_none(self) -> None:
        assert project_box_to_2d(_box((0.0, 0.0, -10.0)), INTRINSIC, IMAGE_SIZE) is None

    def test_clip_keeps_box_within_bounds(self) -> None:
        # Box pushed far to the right; clipped extent must stay inside the image.
        bbox = project_box_to_2d(_box((20.0, 0.0, 10.0)), INTRINSIC, IMAGE_SIZE, clip=True)
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            assert 0.0 <= x_min <= x_max <= 1600.0
            assert 0.0 <= y_min <= y_max <= 900.0
