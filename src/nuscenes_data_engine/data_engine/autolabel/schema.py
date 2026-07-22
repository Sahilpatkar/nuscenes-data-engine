"""The structured scene-label schema the VLM must produce.

Design notes:
- Enums only where ground truth exists to evaluate against (`is_night`, `is_rain`);
  free-text lists where the signal is exploratory (hazards, notable conditions).
- ``dusk_dawn`` exists because GT is binary and description-derived — forcing twilight
  frames into day/night would manufacture eval noise. Same for ``overcast``/``fog``.
- Object counts are explicit int fields (a dict would violate the structured-output
  requirement of ``additionalProperties: false``), aligned to a 10-class view over the
  fine-grained nuScenes ``category_name``. This mapping is EVAL-ONLY — the trainer's
  5-class taxonomy in ingestion/categories.py is untouched.
- The Claude structured-outputs endpoint rejects numeric constraints (minimum/maximum),
  so non-negativity is a client-side validator and the emitted JSON schema is sanitized.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Fine-grained nuScenes category_name -> count field. Categories not listed (animal,
# debris, pushable/pullable, bicycle racks) are excluded from the count evaluation.
GT_COUNT_GROUPS: dict[str, str] = {
    "vehicle.car": "cars",
    "vehicle.emergency.police": "cars",
    "vehicle.truck": "trucks",
    "vehicle.emergency.ambulance": "trucks",
    "vehicle.bus.rigid": "buses",
    "vehicle.bus.bendy": "buses",
    "vehicle.trailer": "trailers",
    "vehicle.construction": "construction_vehicles",
    "vehicle.motorcycle": "motorcycles",
    "vehicle.bicycle": "bicycles",
    "human.pedestrian.adult": "pedestrians",
    "human.pedestrian.child": "pedestrians",
    "human.pedestrian.construction_worker": "pedestrians",
    "human.pedestrian.police_officer": "pedestrians",
    "human.pedestrian.personal_mobility": "pedestrians",
    "human.pedestrian.stroller": "pedestrians",
    "human.pedestrian.wheelchair": "pedestrians",
    "movable_object.trafficcone": "traffic_cones",
    "movable_object.barrier": "barriers",
}

COUNT_FIELDS: tuple[str, ...] = (
    "cars",
    "trucks",
    "buses",
    "trailers",
    "construction_vehicles",
    "motorcycles",
    "bicycles",
    "pedestrians",
    "traffic_cones",
    "barriers",
)


class ObjectCounts(BaseModel):
    """Instances of each class clearly visible in the frame."""

    model_config = ConfigDict(extra="forbid")

    cars: int = Field(description="Cars, including parked ones.")
    trucks: int = Field(description="Trucks and lorries.")
    buses: int = Field(description="Buses.")
    trailers: int = Field(description="Truck trailers.")
    construction_vehicles: int = Field(description="Cranes, excavators, and similar.")
    motorcycles: int = Field(description="Motorcycles and scooters.")
    bicycles: int = Field(description="Bicycles, ridden or parked.")
    pedestrians: int = Field(description="People on foot.")
    traffic_cones: int = Field(description="Orange traffic cones.")
    barriers: int = Field(description="Temporary road barriers.")

    @field_validator("*")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("counts must be non-negative")
        return value


class SceneLabel(BaseModel):
    """One VLM-produced label for a single front-camera frame."""

    model_config = ConfigDict(extra="forbid")

    time_of_day: Literal["day", "dusk_dawn", "night"] = Field(
        description="Lighting conditions; dusk_dawn for twilight."
    )
    weather: Literal["clear", "overcast", "rain", "fog"] = Field(
        description="Dominant weather; rain includes wet roads with active rainfall."
    )
    object_counts: ObjectCounts
    hazards: list[str] = Field(
        description="Short phrases for anything requiring driver caution, "
        "e.g. 'pedestrian crossing ahead'. Empty if none."
    )
    notable_conditions: list[str] = Field(
        description="Short phrases for unusual scene properties, e.g. 'construction zone', "
        "'glare'. Empty if none."
    )
    label_confidence: Literal["low", "medium", "high"] = Field(
        description="Your confidence in this label overall."
    )


_UNSUPPORTED_KEYS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "default",
        "title",
    }
)


def _sanitize(node: Any) -> Any:
    """Strip schema keywords the structured-outputs endpoint rejects; lock objects down."""
    if isinstance(node, dict):
        cleaned = {k: _sanitize(v) for k, v in node.items() if k not in _UNSUPPORTED_KEYS}
        if cleaned.get("type") == "object" and "properties" in cleaned:
            cleaned["additionalProperties"] = False
            cleaned["required"] = list(cleaned["properties"].keys())
        return cleaned
    if isinstance(node, list):
        return [_sanitize(item) for item in node]
    return node


def structured_output_schema() -> dict[str, Any]:
    """JSON schema for ``output_config.format``, sanitized for the Claude API."""
    schema = _sanitize(SceneLabel.model_json_schema())
    assert isinstance(schema, dict)
    return schema
