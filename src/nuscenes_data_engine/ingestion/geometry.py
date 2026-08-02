"""Phase B geometry: ego-pose + 3D annotation extraction and the math around it.

The pure helpers (quaternion->yaw, BEV distance, ego-relative coords, speed) carry the
geometric correctness and are unit-tested without the devkit. ``flatten_geometry`` (added
alongside) walks the loaded ``NuScenes`` handle to emit the ego-pose / 3D-annotation /
instance records — GT metadata only, no LiDAR blobs, so it runs where the dataset lives.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger("nuscenes_data_engine")


def quaternion_yaw(rotation: list[float]) -> float:
    """Heading (yaw about +z), in radians, from a nuScenes ``[w, x, y, z]`` quaternion."""
    w, x, y, z = rotation
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def bev_distance(ex: float, ey: float, ox: float, oy: float) -> float:
    """Bird's-eye (ground-plane) distance between ego ``(ex,ey)`` and object ``(ox,oy)``."""
    return math.hypot(ox - ex, oy - ey)


def ego_relative(
    ex: float, ey: float, heading: float, ox: float, oy: float
) -> tuple[float, float]:
    """Object position in the ego frame as ``(forward, left)``.

    Rotates the global delta into the ego frame by ``-heading``. ``forward`` is along the
    ego's facing direction, ``left`` is 90 deg counter-clockwise from it;
    ``hypot(forward, left)`` equals the BEV distance.
    """
    dx, dy = ox - ex, oy - ey
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    forward = dx * cos_h + dy * sin_h
    left = -dx * sin_h + dy * cos_h
    return forward, left


def speed_between(
    x0: float, y0: float, t0_us: int, x1: float, y1: float, t1_us: int
) -> float:
    """BEV speed (m/s) between two poses; timestamps are microseconds."""
    dt_s = (t1_us - t0_us) / 1_000_000.0
    if dt_s <= 0:
        return 0.0
    return math.hypot(x1 - x0, y1 - y0) / dt_s


def instance_records(annotations_3d: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One ``ObjectInstance`` row per physical object: category + observation count.

    Pure: derived from the collected 3D observations so it honours ``--limit-scenes``.
    """
    seen: dict[str, dict[str, Any]] = {}
    for ann in annotations_3d:
        token = ann["instance_token"]
        record = seen.get(token)
        if record is None:
            seen[token] = {
                "instance_token": token,
                "category_name": ann["category_name"],
                "category_group": ann["category_group"],
                "scene_token": ann["scene_token"],
                "n_annotations": 1,
            }
        else:
            record["n_annotations"] += 1
    return list(seen.values())


def flatten_geometry(
    nusc: Any, *, limit_scenes: int | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Walk keyframes to emit (ego poses, 3D observations, instances) as flat records.

    GT metadata only: the reference ego pose per keyframe is the ``LIDAR_TOP`` sample_data's
    ``ego_pose`` (no point cloud is read). Distances/ego-relative coords are computed in
    the global/map frame; object velocity comes from ``nusc.box_velocity``.
    """
    from nuscenes_data_engine.ingestion.categories import group_for
    from nuscenes_data_engine.ingestion.parse import _scene_conditions

    ego_poses: list[dict[str, Any]] = []
    annotations_3d: list[dict[str, Any]] = []
    scenes = nusc.scene[:limit_scenes] if limit_scenes else nusc.scene

    for si, scene in enumerate(scenes):
        log = nusc.get("log", scene["log_token"])
        is_night, is_rain = _scene_conditions(scene["description"])
        ctx = {
            "scene_token": scene["token"],
            "location": log["location"],
            "is_night": is_night,
            "is_rain": is_rain,
        }
        prev: tuple[float, float, int] | None = None  # (x, y, timestamp_us)
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego = nusc.get("ego_pose", lidar["ego_pose_token"])
            ex, ey, ez = (float(v) for v in ego["translation"])
            heading = quaternion_yaw(ego["rotation"])
            et = int(ego["timestamp"])
            speed = speed_between(prev[0], prev[1], prev[2], ex, ey, et) if prev else None
            ego_poses.append(
                {
                    "ego_pose_token": lidar["ego_pose_token"],
                    "sample_token": sample_token,
                    "timestamp": et,
                    "x": ex, "y": ey, "z": ez,
                    "heading": heading,
                    "speed_mps": speed,
                    **ctx,
                }
            )
            for ann_token in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                ox, oy, oz = (float(v) for v in ann["translation"])
                width, length, height = (float(v) for v in ann["size"])
                vel = nusc.box_velocity(ann_token)
                vx, vy = float(vel[0]), float(vel[1])
                nan_v = math.isnan(vx) or math.isnan(vy)
                forward, left = ego_relative(ex, ey, heading, ox, oy)
                category = ann["category_name"]
                annotations_3d.append(
                    {
                        "annotation_token": ann_token,
                        "sample_token": sample_token,
                        "instance_token": ann["instance_token"],
                        "category_name": category,
                        "category_group": group_for(category),
                        "x": ox, "y": oy, "z": oz,
                        "width": width, "length": length, "height": height,
                        "yaw": quaternion_yaw(ann["rotation"]),
                        "vx": None if nan_v else vx,
                        "vy": None if nan_v else vy,
                        "speed_mps": None if nan_v else math.hypot(vx, vy),
                        "num_lidar_pts": int(ann["num_lidar_pts"]),
                        "num_radar_pts": int(ann["num_radar_pts"]),
                        "visibility_token": ann["visibility_token"],
                        "distance_to_ego_m": bev_distance(ex, ey, ox, oy),
                        "ego_rel_x": forward,
                        "ego_rel_y": left,
                        **ctx,
                    }
                )
            prev = (ex, ey, et)
            sample_token = sample["next"]
        logger.info(
            "geometry scene %d/%d (%s): %d poses, %d 3D boxes so far",
            si + 1, len(scenes), scene["name"], len(ego_poses), len(annotations_3d),
        )

    instances = instance_records(annotations_3d)
    logger.info(
        "Geometry complete: %d poses, %d 3D boxes, %d instances",
        len(ego_poses), len(annotations_3d), len(instances),
    )
    return ego_poses, annotations_3d, instances


def run_geometry_ingestion(
    config_path: Path, *, limit_scenes: int | None = None
) -> dict[str, Any]:
    """Phase B: parse nuScenes ego-pose + 3D geometry into Parquet tables."""
    from nuscenes_data_engine.config import get_settings, load_yaml
    from nuscenes_data_engine.ingestion.parquet import write_parquet
    from nuscenes_data_engine.ingestion.parse import load_nusc

    settings = get_settings()
    cfg = load_yaml(config_path)
    source = cfg.get("source", {})
    dataroot = Path(settings.nuscenes_dataroot or source.get("dataroot"))
    version = settings.nuscenes_version or source.get("version")
    out = cfg.get("output", {})
    processed_dir = Path(out.get("processed_dir", settings.processed_dir))
    names = out.get("parquet", {})
    ego_path = processed_dir / names.get("ego_pose", "ego_pose.parquet")
    ann3d_path = processed_dir / names.get("annotations_3d", "annotations_3d.parquet")
    inst_path = processed_dir / names.get("instances", "instances.parquet")

    nusc = load_nusc(dataroot, version)
    ego_poses, annotations_3d, instances = flatten_geometry(nusc, limit_scenes=limit_scenes)

    n_ego = write_parquet(ego_poses, ego_path)
    n_ann = write_parquet(annotations_3d, ann3d_path)
    n_inst = write_parquet(instances, inst_path)
    summary = {
        "version": version,
        "scenes_processed": limit_scenes if limit_scenes else len(nusc.scene),
        "ego_poses": n_ego,
        "annotations_3d": n_ann,
        "instances": n_inst,
        "ego_pose_parquet": str(ego_path),
        "annotations_3d_parquet": str(ann3d_path),
        "instances_parquet": str(inst_path),
    }
    logger.info(
        "Geometry ingestion wrote %d poses, %d 3D boxes, %d instances -> %s",
        n_ego, n_ann, n_inst, processed_dir,
    )
    return summary
