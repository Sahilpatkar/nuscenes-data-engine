"""Tests for CAN-bus ingestion: pure alignment helpers + fake-devkit flatten."""

from __future__ import annotations

from typing import Any, ClassVar

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


def test_accel_window_exact_boundary_included() -> None:
    pose = [
        {"utime": 500_000, "accel": [-5.0, 0.0, 9.8], "vel": [1.0, 0.0, 0.0]},
        {"utime": 1_500_000, "accel": [-6.0, 0.0, 9.8], "vel": [2.0, 0.0, 0.0]},
    ]
    # Both sit exactly ±500_000 us from t; an exclusive window would drop both.
    accel_min, accel_max, speed = accel_window(pose, t_us=1_000_000, window_us=500_000)
    assert accel_min == pytest.approx(-6.0)
    assert accel_max == pytest.approx(-5.0)
    assert speed is not None


def test_scene_number() -> None:
    assert scene_number("scene-0161") == 161
    assert scene_number("scene-1094") == 1094


# ---------------------------------------------------------------------------
# flatten_canbus — fake nusc + fake can handles (no devkit needed)
# ---------------------------------------------------------------------------


class _FakeNusc:
    """Two-keyframe scenes with the minimal fields flatten_canbus touches."""

    def __init__(self, scenes: list[dict[str, Any]]) -> None:
        self.scene = scenes
        self._samples: dict[str, dict[str, Any]] = {}
        self._logs = {"log-1": {"location": "singapore-onenorth"}}
        for scene in scenes:
            base = scene["name"]
            for i in range(2):
                token = f"{base}-kf{i}"
                self._samples[token] = {
                    "token": token,
                    "timestamp": 1_000_000 + i * 2_000_000,  # kf1 sits at t=3.0s
                    "next": f"{base}-kf{i + 1}" if i == 0 else "",
                }

    def get(self, table: str, token: str) -> dict[str, Any]:
        if table == "sample":
            return self._samples[token]
        if table == "log":
            return self._logs[token]
        raise KeyError(table)


def _scene(name: str) -> dict[str, Any]:
    return {
        "token": f"tok-{name}",
        "name": name,
        "log_token": "log-1",
        "description": "Night, heavy rain",
        "first_sample_token": f"{name}-kf0",
    }


class _FakeCan:
    can_blacklist: ClassVar[list[int]] = [161]

    def __init__(self, monitor: list[dict[str, Any]], pose: list[dict[str, Any]],
                 missing: set[str] = frozenset()) -> None:
        self._monitor, self._pose, self._missing = monitor, pose, missing

    def get_messages(self, scene_name: str, message: str) -> list[dict[str, Any]]:
        if scene_name in self._missing:
            raise FileNotFoundError(f"no CAN data for {scene_name}")
        return self._monitor if message == "vehicle_monitor" else self._pose


_MONITOR = [{
    "utime": 1_050_000, "vehicle_speed": 14.7, "steering": 191.9, "steering_speed": 65.7,
    "brake": 20, "brake_switch": 1, "throttle": 55, "yaw_rate": 18.9,
    "left_signal": 0, "right_signal": 1,
}]
_POSE = [
    {"utime": 1_000_000, "accel": [-4.0, 0.0, 9.8], "vel": [3.0, 4.0, 0.0]},
    {"utime": 1_400_000, "accel": [1.0, 0.0, 9.8], "vel": [5.0, 0.0, 0.0]},
    {"utime": 2_900_000, "accel": [1.0, 0.0, 9.8], "vel": [6.0, 0.0, 0.0]},
]


def test_flatten_canbus_aligns_signals_and_flags() -> None:
    from nuscenes_data_engine.ingestion.canbus import flatten_canbus

    rows = flatten_canbus(
        _FakeNusc([_scene("scene-0001")]), _FakeCan(_MONITOR, _POSE),
        window_s=0.5, hard_braking_mps2=-3.0, monitor_tolerance_ms=600,
    )
    assert len(rows) == 2
    first, second = rows
    assert first["sample_token"] == "scene-0001-kf0" and first["has_canbus"] is True
    assert first["can_speed_kmh"] == pytest.approx(14.7)
    assert first["brake_pedal"] == 20 and first["right_signal"] == 1
    assert first["accel_long_min_mps2"] == pytest.approx(-4.0)
    assert first["accel_long_max_mps2"] == pytest.approx(1.0)
    assert first["is_hard_braking"] is True  # -4.0 <= -3.0
    assert first["can_vel_mps"] == pytest.approx(5.0)  # nearest pose |[3,4,0]| = 5
    assert first["is_night"] is True and first["is_rain"] is True
    assert first["scene_name"] == "scene-0001" and first["location"] == "singapore-onenorth"
    # Second keyframe (t=3.0s): the monitor msg is 1.95s stale (outside 600ms -> nulls)
    # but the utime=2_900_000 pose is in its ±0.5s window -> accel_min 1.0 -> no event.
    assert second["can_speed_kmh"] is None
    assert second["accel_long_min_mps2"] == pytest.approx(1.0)
    assert second["can_vel_mps"] == pytest.approx(6.0)
    assert second["is_hard_braking"] is False


def test_flatten_canbus_blacklisted_scene_yields_nulls() -> None:
    from nuscenes_data_engine.ingestion.canbus import flatten_canbus

    rows = flatten_canbus(_FakeNusc([_scene("scene-0161")]), _FakeCan(_MONITOR, _POSE))
    assert len(rows) == 2
    assert all(r["has_canbus"] is False for r in rows)
    assert all(r["can_speed_kmh"] is None for r in rows)
    assert all(r["is_hard_braking"] is None for r in rows)


def test_flatten_canbus_missing_files_treated_as_absent() -> None:
    from nuscenes_data_engine.ingestion.canbus import flatten_canbus

    rows = flatten_canbus(
        _FakeNusc([_scene("scene-0002")]),
        _FakeCan(_MONITOR, _POSE, missing={"scene-0002"}),
    )
    assert all(r["has_canbus"] is False and r["throttle"] is None for r in rows)


def test_flatten_canbus_monitor_outside_tolerance_is_null_but_pose_still_used() -> None:
    from nuscenes_data_engine.ingestion.canbus import flatten_canbus

    stale_monitor = [dict(_MONITOR[0], utime=99_000_000)]
    rows = flatten_canbus(_FakeNusc([_scene("scene-0003")]), _FakeCan(stale_monitor, _POSE))
    first = rows[0]
    assert first["has_canbus"] is True
    assert first["can_speed_kmh"] is None  # monitor too old
    assert first["accel_long_min_mps2"] == pytest.approx(-4.0)  # pose window still valid


def test_flatten_canbus_limit_scenes() -> None:
    from nuscenes_data_engine.ingestion.canbus import flatten_canbus

    rows = flatten_canbus(
        _FakeNusc([_scene("scene-0001"), _scene("scene-0004")]),
        _FakeCan(_MONITOR, _POSE),
        limit_scenes=1,
    )
    assert {r["scene_name"] for r in rows} == {"scene-0001"}


def test_flatten_canbus_propagates_unexpected_errors() -> None:
    """A devkit AssertionError (bad message name) must fail loudly, not degrade to nulls."""
    from nuscenes_data_engine.ingestion.canbus import flatten_canbus

    class _AssertingCan(_FakeCan):
        def get_messages(self, scene_name: str, message: str) -> list[dict[str, Any]]:
            raise AssertionError(f"Message {message} not found among all_messages")

    with pytest.raises(AssertionError, match="not found"):
        flatten_canbus(_FakeNusc([_scene("scene-0005")]), _AssertingCan(_MONITOR, _POSE))
