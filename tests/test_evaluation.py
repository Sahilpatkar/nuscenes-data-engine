"""Tests for condition-slice assignment (pure logic; no model/dataset)."""

from __future__ import annotations

from nuscenes_data_engine.evaluation.slices import _matches, assign_slices

SLICE_CONFIG = {
    "time_of_day": {
        "day": {"exclude": ["night"]},
        "night": {"include": ["night"]},
    },
    "weather": {
        "clear": {"exclude": ["rain"]},
        "rain": {"include": ["rain"]},
    },
}


class TestMatches:
    def test_include_rule(self) -> None:
        assert _matches("Night, parked cars", {"include": ["night"]}) is True
        assert _matches("Sunny afternoon", {"include": ["night"]}) is False

    def test_exclude_rule(self) -> None:
        assert _matches("Sunny afternoon", {"exclude": ["night"]}) is True
        assert _matches("Driving at night", {"exclude": ["night"]}) is False

    def test_case_insensitive(self) -> None:
        assert _matches("Heavy RAIN on highway", {"include": ["rain"]}) is True


class TestAssignSlices:
    def test_night_rain(self) -> None:
        labels = assign_slices("Night drive in heavy rain", SLICE_CONFIG)
        assert labels == {"time_of_day": "night", "weather": "rain"}

    def test_day_clear(self) -> None:
        labels = assign_slices("Sunny day, light traffic", SLICE_CONFIG)
        assert labels == {"time_of_day": "day", "weather": "clear"}

    def test_night_clear(self) -> None:
        labels = assign_slices("Nighttime, clear skies", SLICE_CONFIG)
        assert labels["time_of_day"] == "night"
        assert labels["weather"] == "clear"
