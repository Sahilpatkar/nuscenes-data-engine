"""Project nuScenes 3D annotation boxes into 2D camera-frame bounding boxes.

The devkit's :meth:`NuScenes.get_sample_data` already returns each 3D box transformed
into the **camera coordinate frame** (global -> ego -> sensor). This module takes such a
box plus the camera intrinsic and produces an axis-aligned 2D pixel box, clipped to the
image. Only corners in front of the image plane (z > ``min_z``) contribute to the extent.
"""

from __future__ import annotations

from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points

# (x_min, y_min, x_max, y_max) in pixels.
BBox2D = tuple[float, float, float, float]


def project_box_to_2d(
    box: Box,
    camera_intrinsic: object,
    image_size: tuple[int, int],
    *,
    clip: bool = True,
    min_z: float = 0.1,
) -> BBox2D | None:
    """Return the 2D pixel box for a camera-frame 3D ``box``, or ``None`` if not visible.

    Args:
        box: A :class:`nuscenes.utils.data_classes.Box` already in the camera frame
            (as returned by :meth:`NuScenes.get_sample_data`).
        camera_intrinsic: 3x3 camera intrinsic matrix.
        image_size: ``(width, height)`` of the image in pixels.
        clip: Clamp the box to the image bounds.
        min_z: Minimum camera-frame depth (metres) for a corner to count; drops boxes
            entirely behind the image plane.

    Returns:
        ``(x_min, y_min, x_max, y_max)`` in pixels, or ``None`` if the box has no
        positive-depth corners or collapses to zero area after clipping.
    """
    corners_3d = box.corners()  # (3, 8)
    in_front = corners_3d[2, :] > min_z
    if not in_front.any():
        return None

    corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]  # (2, 8)
    xs = corners_2d[0, in_front]
    ys = corners_2d[1, in_front]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    width, height = image_size
    if clip:
        x_min = min(max(x_min, 0.0), float(width))
        x_max = min(max(x_max, 0.0), float(width))
        y_min = min(max(y_min, 0.0), float(height))
        y_max = min(max(y_max, 0.0), float(height))

    if x_max <= x_min or y_max <= y_min:
        return None
    return (x_min, y_min, x_max, y_max)
