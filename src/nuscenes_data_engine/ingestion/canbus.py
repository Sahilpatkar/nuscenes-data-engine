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
from pathlib import Path
from typing import Any

logger = logging.getLogger("nuscenes_data_engine")


def nearest_message(
    messages: list[dict[str, Any]], t_us: int, tolerance_us: int
) -> dict[str, Any] | None:
    """The message whose ``utime`` is closest to ``t_us``, or None if none within tolerance.

    Ties (two messages equidistant) resolve to the earlier one in ``messages``.
    """
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
            except (FileNotFoundError, OSError):  # absent CAN files for this scene
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


def speed_correlation(canbus: Any, ego: Any) -> float | None:
    """Pearson corr of CAN speed (km/h -> m/s) vs GT-derived ego speed; None if <2 rows.

    A free wiring sanity-check: the two speeds come from independent sources
    (vehicle CAN vs consecutive GT poses), so high correlation validates alignment.
    """
    merged = canbus.merge(ego[["sample_token", "speed_mps"]], on="sample_token").dropna(
        subset=["can_speed_kmh", "speed_mps"]
    )
    if len(merged) < 2:
        return None
    corr = float(merged["can_speed_kmh"].div(3.6).corr(merged["speed_mps"]))
    return None if math.isnan(corr) else corr


def run_canbus_ingestion(
    config_path: Path, *, limit_scenes: int | None = None
) -> dict[str, Any]:
    """Parse the CAN-bus expansion into the keyframe-aligned canbus Parquet table."""
    import pandas as pd

    from nuscenes_data_engine.config import get_settings, load_yaml
    from nuscenes_data_engine.ingestion.parquet import write_parquet
    from nuscenes_data_engine.ingestion.parse import load_nusc

    settings = get_settings()
    cfg = load_yaml(config_path)
    source = cfg.get("source", {})
    dataroot = Path(settings.nuscenes_dataroot or source.get("dataroot"))
    version = settings.nuscenes_version or source.get("version")
    can_cfg = cfg.get("canbus", {})
    out = cfg.get("output", {})
    processed_dir = Path(out.get("processed_dir", settings.processed_dir))
    canbus_path = processed_dir / out.get("parquet", {}).get("canbus", "canbus.parquet")

    from nuscenes.can_bus.can_bus_api import NuScenesCanBus

    can = NuScenesCanBus(dataroot=str(dataroot))
    nusc = load_nusc(dataroot, version)
    rows = flatten_canbus(
        nusc,
        can,
        window_s=float(can_cfg.get("window_s", 0.5)),
        hard_braking_mps2=float(can_cfg.get("hard_braking_mps2", -3.0)),
        monitor_tolerance_ms=int(can_cfg.get("monitor_tolerance_ms", 600)),
        limit_scenes=limit_scenes,
    )
    n_rows = write_parquet(rows, canbus_path)

    corr = None
    ego_path = processed_dir / "ego_pose.parquet"
    if ego_path.is_file():
        try:
            corr = speed_correlation(
                pd.DataFrame(rows),
                pd.read_parquet(ego_path, columns=["sample_token", "speed_mps"]),
            )
        except Exception as exc:  # sanity cross-check only — never fail a written table
            logger.warning("Speed cross-check skipped (%s); canbus.parquet is unaffected", exc)
    summary = {
        "version": version,
        "scenes_processed": limit_scenes if limit_scenes else len(nusc.scene),
        "canbus_rows": n_rows,
        "n_missing_canbus": sum(1 for r in rows if not r["has_canbus"]),
        "n_hard_braking": sum(1 for r in rows if r["is_hard_braking"]),
        "speed_correlation_vs_gt": corr,
        "canbus_parquet": str(canbus_path),
    }
    logger.info(
        "CAN ingestion wrote %d rows (%d without CAN, %d hard-braking; speed corr vs GT: %s)",
        n_rows, summary["n_missing_canbus"], summary["n_hard_braking"],
        "n/a" if corr is None else f"{corr:.3f}",
    )
    return summary
