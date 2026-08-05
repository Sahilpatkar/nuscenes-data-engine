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
