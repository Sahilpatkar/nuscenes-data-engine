"""Project nuScenes 3D annotation boxes into 2D camera-frame bounding boxes.

Uses `calibrated_sensor` (intrinsics + extrinsics) and `ego_pose` to transform a
3D box from the global frame into the camera frame, then to pixel coordinates,
returning an axis-aligned 2D box clipped to the image.
"""

from __future__ import annotations

from typing import Any


def box_3d_to_2d(
    annotation: dict[str, Any],
    calibrated_sensor: dict[str, Any],
    ego_pose: dict[str, Any],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    """Return an ``(x_min, y_min, x_max, y_max)`` 2D box, or ``None`` if not visible.

    Args:
        annotation: A ``sample_annotation`` record (3D box in global frame).
        calibrated_sensor: The camera's ``calibrated_sensor`` record.
        ego_pose: The ``ego_pose`` for the sample_data timestamp.
        image_size: ``(width, height)`` of the camera image in pixels.
    """
    # TODO(Phase 1): reuse nuscenes.utils.geometry_utils (Box, view_points) to
    # transform corners global -> ego -> camera -> pixels, then take the AABB.
    raise NotImplementedError
