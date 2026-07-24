"""Tests for Phase B geometry math (pure; no devkit / dataset needed)."""

from __future__ import annotations

import math

from nuscenes_data_engine.ingestion.geometry import (
    bev_distance,
    ego_relative,
    instance_records,
    quaternion_yaw,
    speed_between,
)


def test_quaternion_yaw_identity_and_ninety() -> None:
    assert quaternion_yaw([1.0, 0.0, 0.0, 0.0]) == 0.0  # identity -> heading 0
    # 90 deg about +z: q = [cos45, 0, 0, sin45] -> yaw = pi/2
    s = math.sqrt(0.5)
    assert math.isclose(quaternion_yaw([s, 0.0, 0.0, s]), math.pi / 2, abs_tol=1e-9)


def test_bev_distance_ignores_z() -> None:
    assert bev_distance(0.0, 0.0, 3.0, 4.0) == 5.0
    assert math.isclose(bev_distance(1.0, 1.0, 1.0, 1.0), 0.0)


def test_ego_relative_heading_zero_is_forward_left() -> None:
    # ego at origin facing +x; object at (3,4) -> 3 m ahead, 4 m to the left.
    fwd, left = ego_relative(0.0, 0.0, 0.0, 3.0, 4.0)
    assert math.isclose(fwd, 3.0) and math.isclose(left, 4.0)
    # distance is preserved: sqrt(fwd^2 + left^2) == BEV distance
    assert math.isclose(math.hypot(fwd, left), bev_distance(0.0, 0.0, 3.0, 4.0))


def test_ego_relative_heading_ninety_rotates_frame() -> None:
    # ego facing +y (heading 90 deg); object at (3,4) -> 4 m ahead, 3 m to the RIGHT.
    fwd, left = ego_relative(0.0, 0.0, math.pi / 2, 3.0, 4.0)
    assert math.isclose(fwd, 4.0, abs_tol=1e-9)
    assert math.isclose(left, -3.0, abs_tol=1e-9)


def test_speed_between_uses_microsecond_timestamps() -> None:
    # 10 m in 1 s (1e6 microseconds) -> 10 m/s
    assert math.isclose(speed_between(0.0, 0.0, 0, 10.0, 0.0, 1_000_000), 10.0)
    # zero displacement -> 0
    assert math.isclose(speed_between(5.0, 5.0, 0, 5.0, 5.0, 500_000), 0.0)


def test_instance_records_aggregate_observations_per_object() -> None:
    anns = [
        {"instance_token": "i1", "category_name": "vehicle.car", "category_group": "car",
         "scene_token": "s1"},
        {"instance_token": "i1", "category_name": "vehicle.car", "category_group": "car",
         "scene_token": "s1"},
        {"instance_token": "i2", "category_name": "human.pedestrian.adult",
         "category_group": "pedestrian", "scene_token": "s1"},
    ]
    rows = {r["instance_token"]: r for r in instance_records(anns)}
    assert rows["i1"] == {
        "instance_token": "i1", "category_name": "vehicle.car", "category_group": "car",
        "scene_token": "s1", "n_annotations": 2,
    }
    assert rows["i2"]["n_annotations"] == 1
