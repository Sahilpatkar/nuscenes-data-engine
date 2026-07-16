"""Map the 23 fine-grained nuScenes categories onto the detector's coarse classes.

Training/eval use a small class set (see ``configs/train.yaml``): car, truck, bus,
pedestrian, bicycle. Every nuScenes category name is mapped to one of those groups, or
to ``None`` (kept in the metadata but not a training target).
"""

from __future__ import annotations

# The 23 canonical nuScenes category names (from category.json).
ALL_NUSCENES_CATEGORIES: tuple[str, ...] = (
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.wheelchair",
    "human.pedestrian.stroller",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer",
    "human.pedestrian.construction_worker",
    "animal",
    "vehicle.car",
    "vehicle.motorcycle",
    "vehicle.bicycle",
    "vehicle.bus.bendy",
    "vehicle.bus.rigid",
    "vehicle.truck",
    "vehicle.construction",
    "vehicle.emergency.ambulance",
    "vehicle.emergency.police",
    "vehicle.trailer",
    "movable_object.barrier",
    "movable_object.trafficcone",
    "movable_object.pushable_pullable",
    "movable_object.debris",
    "static_object.bicycle_rack",
)

# Exact-name overrides first; prefix rules below handle the rest.
_EXACT: dict[str, str] = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bicycle": "bicycle",
}

# (prefix, group) — first match wins.
_PREFIX: tuple[tuple[str, str], ...] = (
    ("vehicle.bus", "bus"),
    ("human.pedestrian", "pedestrian"),
)

# The coarse classes, in the canonical order used for YOLO class indices.
DETECTION_CLASSES: tuple[str, ...] = ("car", "truck", "bus", "pedestrian", "bicycle")
CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(DETECTION_CLASSES)}


def group_for(category_name: str) -> str | None:
    """Return the coarse detector class for a nuScenes category, or ``None``.

    ``None`` means the category is retained in the metadata but is not one of the
    detector's training classes.
    """
    if category_name in _EXACT:
        return _EXACT[category_name]
    for prefix, group in _PREFIX:
        if category_name.startswith(prefix):
            return group
    return None
