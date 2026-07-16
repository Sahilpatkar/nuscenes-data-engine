"""Flatten the nuScenes relational tables into denormalized records.

Walks ``scene -> sample -> sample_data`` for each camera and joins each keyframe image
with its projected 2D annotations plus scene/log/weather context. Produces two record
lists:

* **images** — one row per (keyframe sample, camera) image.
* **annotations** — one row per projected 2D bounding box.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility

from nuscenes_data_engine.ingestion.categories import group_for
from nuscenes_data_engine.ingestion.projection import project_box_to_2d

logger = logging.getLogger("nuscenes_data_engine")

# Default surround-view cameras.
DEFAULT_CAMERAS: tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def load_nusc(dataroot: Path, version: str) -> NuScenes:
    """Load the nuScenes metadata tables via the devkit (loads all tables into memory)."""
    logger.info("Loading nuScenes %s from %s ...", version, dataroot)
    return NuScenes(version=version, dataroot=str(dataroot), verbose=False)


def _scene_conditions(description: str) -> tuple[bool, bool]:
    """Return ``(is_night, is_rain)`` parsed from a scene description string."""
    lowered = description.lower()
    return "night" in lowered, "rain" in lowered


def flatten(
    nusc: NuScenes,
    *,
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
    visibility_min: int = 1,
    min_box_area_px: float = 0.0,
    clip_to_image: bool = True,
    limit_scenes: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Flatten nuScenes into (image records, annotation records).

    Args:
        nusc: A loaded :class:`NuScenes` instance.
        cameras: Camera channels to process.
        visibility_min: Drop annotations whose visibility token is below this (1..4).
        min_box_area_px: Drop projected boxes smaller than this area, in pixels².
        clip_to_image: Clamp projected boxes to image bounds.
        limit_scenes: If set, only process the first N scenes (fast dev runs).

    Returns:
        ``(images, annotations)`` as two lists of flat dicts.
    """
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    scenes = nusc.scene[:limit_scenes] if limit_scenes else nusc.scene
    n_dropped = 0

    for si, scene in enumerate(scenes):
        log = nusc.get("log", scene["log_token"])
        is_night, is_rain = _scene_conditions(scene["description"])
        scene_ctx = {
            "scene_token": scene["token"],
            "scene_name": scene["name"],
            "scene_description": scene["description"],
            "log_token": scene["log_token"],
            "location": log["location"],
            "is_night": is_night,
            "is_rain": is_rain,
        }

        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            for cam in cameras:
                sd_token = sample["data"][cam]
                sd = nusc.get("sample_data", sd_token)
                image_size = (sd["width"], sd["height"])
                _data_path, boxes, cam_intrinsic = nusc.get_sample_data(
                    sd_token, box_vis_level=BoxVisibility.ANY
                )

                n_boxes = 0
                for box in boxes:
                    ann = nusc.get("sample_annotation", box.token)
                    if int(ann["visibility_token"]) < visibility_min:
                        n_dropped += 1
                        continue
                    bbox = project_box_to_2d(box, cam_intrinsic, image_size, clip=clip_to_image)
                    if bbox is None:
                        n_dropped += 1
                        continue
                    x_min, y_min, x_max, y_max = bbox
                    area = (x_max - x_min) * (y_max - y_min)
                    if area < min_box_area_px:
                        n_dropped += 1
                        continue

                    annotations.append(
                        {
                            "annotation_token": box.token,
                            "sample_data_token": sd_token,
                            "sample_token": sample_token,
                            "channel": cam,
                            "category_name": box.name,
                            "category_group": group_for(box.name),
                            "visibility_token": ann["visibility_token"],
                            "num_lidar_pts": ann["num_lidar_pts"],
                            "num_radar_pts": ann["num_radar_pts"],
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max,
                            "bbox_area": area,
                            **scene_ctx,
                        }
                    )
                    n_boxes += 1

                images.append(
                    {
                        "sample_data_token": sd_token,
                        "sample_token": sample_token,
                        "channel": cam,
                        "filename": sd["filename"],
                        "width": sd["width"],
                        "height": sd["height"],
                        "timestamp": sd["timestamp"],
                        "n_boxes": n_boxes,
                        **scene_ctx,
                    }
                )
            sample_token = sample["next"]

        logger.info(
            "scene %d/%d (%s): %d images, %d boxes so far",
            si + 1,
            len(scenes),
            scene["name"],
            len(images),
            len(annotations),
        )

    logger.info(
        "Flatten complete: %d images, %d annotations, %d boxes dropped",
        len(images),
        len(annotations),
        n_dropped,
    )
    return images, annotations
