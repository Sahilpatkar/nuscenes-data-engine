"""Project nuScenes 3D annotation boxes into 2D camera-frame bounding boxes.

The devkit's :meth:`NuScenes.get_sample_data` already returns each 3D box transformed into
the **camera coordinate frame** (global -> ego -> sensor). This module projects the box
corners that are in front of the image plane, then takes the **intersection of their convex
hull with the image canvas** (the devkit's ``post_process_coords`` approach) and returns its
axis-aligned bounding box. This is tighter and more correct for boxes that straddle the
image edge than a plain corner min/max + clamp.
"""

from __future__ import annotations

from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from shapely.geometry import MultiPoint
from shapely.geometry import box as shapely_box

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
        clip: Intersect the projected box with the image canvas (drop out-of-frame area).
        min_z: Minimum camera-frame depth (metres) for a corner to count; drops boxes
            entirely behind the image plane.

    Returns:
        ``(x_min, y_min, x_max, y_max)`` in pixels, or ``None`` if the box has no
        positive-depth corners, does not intersect the canvas, or is degenerate.
    """
    corners_3d = box.corners()  # (3, 8)
    in_front = corners_3d[2, :] > min_z
    if in_front.sum() < 3:  # need >=3 points for a 2D polygon
        return None

    corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]  # (2, 8)
    points = [(float(x), float(y)) for x, y in corners_2d[:, in_front].T]
    hull = MultiPoint(points).convex_hull

    if clip:
        width, height = image_size
        canvas = shapely_box(0, 0, float(width), float(height))
        if not hull.intersects(canvas):
            return None
        hull = hull.intersection(canvas)

    if hull.is_empty:
        return None
    x_min, y_min, x_max, y_max = hull.bounds
    if x_max <= x_min or y_max <= y_min:
        return None
    return (float(x_min), float(y_min), float(x_max), float(y_max))
