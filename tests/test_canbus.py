"""Tests for CAN-bus ingestion: pure alignment helpers + fake-devkit flatten."""

from __future__ import annotations

from typing import Any

import pytest

from nuscenes_data_engine.ingestion.canbus import (
    accel_window,
    nearest_message,
    scene_number,
)

# ---------------------------------------------------------------------------
# pure helpers — run in torch-free CI
# ---------------------------------------------------------------------------


def _msgs(*utimes: int) -> list[dict[str, Any]]:
    return [{"utime": u, "vehicle_speed": float(i)} for i, u in enumerate(utimes)]


def test_nearest_message_picks_closest_within_tolerance() -> None:
    messages = _msgs(1_000_000, 1_500_000, 2_100_000)
    msg = nearest_message(messages, t_us=1_600_000, tolerance_us=600_000)
    assert msg is not None and msg["utime"] == 1_500_000


def test_nearest_message_none_when_outside_tolerance() -> None:
    messages = _msgs(1_000_000)
    assert nearest_message(messages, t_us=2_000_000, tolerance_us=600_000) is None
    assert nearest_message([], t_us=0, tolerance_us=600_000) is None


def test_nearest_message_exact_tolerance_boundary_included() -> None:
    messages = _msgs(1_000_000)
    assert nearest_message(messages, t_us=1_600_000, tolerance_us=600_000) is not None


def test_accel_window_stats_and_nearest_speed() -> None:
    pose = [
        {"utime": 900_000, "accel": [-1.0, 0.0, 9.8], "vel": [10.0, 0.0, 0.0]},
        {"utime": 1_000_000, "accel": [-4.2, 0.1, 9.8], "vel": [8.0, 6.0, 0.0]},
        {"utime": 1_100_000, "accel": [0.5, 0.0, 9.8], "vel": [7.0, 0.0, 0.0]},
        {"utime": 9_000_000, "accel": [-9.0, 0.0, 9.8], "vel": [0.0, 0.0, 0.0]},  # outside
    ]
    accel_min, accel_max, speed = accel_window(pose, t_us=1_010_000, window_us=500_000)
    assert accel_min == pytest.approx(-4.2)
    assert accel_max == pytest.approx(0.5)
    assert speed == pytest.approx(10.0)  # nearest msg is utime=1_000_000 -> |[8,6,0]| = 10


def test_accel_window_empty_returns_nones() -> None:
    assert accel_window([], t_us=0, window_us=500_000) == (None, None, None)
    far = [{"utime": 9_000_000, "accel": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0]}]
    assert accel_window(far, t_us=0, window_us=500_000) == (None, None, None)


def test_scene_number() -> None:
    assert scene_number("scene-0161") == 161
    assert scene_number("scene-1094") == 1094
