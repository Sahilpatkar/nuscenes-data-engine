"""CAN-bus ingestion: keyframe-aligned ego dynamics from the nuScenes can_bus expansion.

The pure helpers (nearest-message selection, windowed acceleration stats) carry the
alignment correctness and are unit-tested without the devkit. ``flatten_canbus`` walks
the loaded ``NuScenes`` handle plus a ``NuScenesCanBus`` handle to emit one record per
keyframe — JSON metadata only, so it runs where the dataset lives (TRINITY). Scenes in
the devkit's ``can_blacklist`` (or with missing CAN files) yield ``has_canbus=False``
rows with null signals; no keyframe is ever dropped.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger("nuscenes_data_engine")


def nearest_message(
    messages: list[dict[str, Any]], t_us: int, tolerance_us: int
) -> dict[str, Any] | None:
    """The message whose ``utime`` is closest to ``t_us``, or None if none within tolerance."""
    best: dict[str, Any] | None = None
    best_dt = tolerance_us + 1
    for msg in messages:
        dt = abs(int(msg["utime"]) - t_us)
        if dt < best_dt:
            best, best_dt = msg, dt
    return best


def accel_window(
    pose_msgs: list[dict[str, Any]], t_us: int, window_us: int
) -> tuple[float | None, float | None, float | None]:
    """(accel_long_min, accel_long_max, speed_mps at the nearest pose) over ``t±window``.

    Longitudinal acceleration is ``accel[0]`` (vehicle-frame x-forward); speed is the
    norm of the nearest in-window pose message's ``vel``. Empty window -> three Nones.
    """
    in_window = [m for m in pose_msgs if abs(int(m["utime"]) - t_us) <= window_us]
    if not in_window:
        return None, None, None
    longs = [float(m["accel"][0]) for m in in_window]
    nearest = min(in_window, key=lambda m: abs(int(m["utime"]) - t_us))
    speed = math.sqrt(sum(float(v) * float(v) for v in nearest["vel"]))
    return min(longs), max(longs), speed


def scene_number(scene_name: str) -> int:
    """``'scene-0161'`` -> ``161`` (the devkit blacklist holds scene numbers as ints)."""
    return int(scene_name.rsplit("-", 1)[1])


def flatten_canbus(
    nusc: Any,
    can: Any,
    *,
    window_s: float = 0.5,
    hard_braking_mps2: float = -3.0,
    monitor_tolerance_ms: int = 600,
    limit_scenes: int | None = None,
) -> list[dict[str, Any]]:
    """One record per keyframe: nearest vehicle_monitor signals + windowed pose accel.

    Blacklisted scenes (``can.can_blacklist`` holds scene numbers) and scenes whose CAN
    files are missing produce ``has_canbus=False`` rows with null signals.
    """
    from nuscenes_data_engine.ingestion.parse import _scene_conditions

    window_us = int(window_s * 1_000_000)
    tolerance_us = monitor_tolerance_ms * 1000
    blacklist = set(getattr(can, "can_blacklist", ()))
    rows: list[dict[str, Any]] = []
    scenes = nusc.scene[:limit_scenes] if limit_scenes else nusc.scene

    for si, scene in enumerate(scenes):
        log = nusc.get("log", scene["log_token"])
        is_night, is_rain = _scene_conditions(scene["description"])
        ctx = {
            "scene_token": scene["token"],
            "scene_name": scene["name"],
            "location": log["location"],
            "is_night": is_night,
            "is_rain": is_rain,
        }
        monitor: list[dict[str, Any]] = []
        pose: list[dict[str, Any]] = []
        has_canbus = scene_number(scene["name"]) not in blacklist
        if has_canbus:
            try:
                monitor = can.get_messages(scene["name"], "vehicle_monitor")
                pose = can.get_messages(scene["name"], "pose")
            except Exception:  # devkit raises plain Exception for absent scene files
                logger.warning("CAN data missing for %s; writing nulls", scene["name"])
                has_canbus = False

        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            t_us = int(sample["timestamp"])
            msg = nearest_message(monitor, t_us, tolerance_us) if has_canbus else None
            accel_min, accel_max, vel_mps = (
                accel_window(pose, t_us, window_us) if has_canbus else (None, None, None)
            )
            rows.append(
                {
                    "sample_token": sample_token,
                    "timestamp": t_us,
                    "has_canbus": has_canbus,
                    "can_speed_kmh": None if msg is None else float(msg["vehicle_speed"]),
                    "steering_deg": None if msg is None else float(msg["steering"]),
                    "steering_speed": None if msg is None else float(msg["steering_speed"]),
                    "brake_pedal": None if msg is None else int(msg["brake"]),
                    "brake_switch": None if msg is None else int(msg["brake_switch"]),
                    "throttle": None if msg is None else int(msg["throttle"]),
                    "yaw_rate": None if msg is None else float(msg["yaw_rate"]),
                    "left_signal": None if msg is None else int(msg["left_signal"]),
                    "right_signal": None if msg is None else int(msg["right_signal"]),
                    "accel_long_min_mps2": accel_min,
                    "accel_long_max_mps2": accel_max,
                    "can_vel_mps": vel_mps,
                    "is_hard_braking": None if accel_min is None else accel_min <= hard_braking_mps2,
                    **ctx,
                }
            )
            sample_token = sample["next"]
        if (si + 1) % 100 == 0 or si + 1 == len(scenes):
            logger.info("canbus scene %d/%d: %d keyframes so far", si + 1, len(scenes), len(rows))
    return rows
