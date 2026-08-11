"""Tests for the chat-agent eval harness (pure graders; no LLM, no network)."""

from __future__ import annotations

import pytest

from nuscenes_data_engine.data_engine.chat.evaluate import (
    extract_numbers,
    numeric_matches,
)


def test_extract_numbers_handles_plain_decimal_percent_and_thousands() -> None:
    text = "There are 1,234 night scenes, 12.5 boxes per frame, and 45% are in Boston."
    assert extract_numbers(text) == [1234.0, 12.5, 45.0]


def test_extract_numbers_ignores_words_and_returns_empty() -> None:
    assert extract_numbers("No numeric content here at all.") == []


def test_extract_numbers_keeps_negatives_and_ignores_trailing_period() -> None:
    assert extract_numbers("The delta was -0.0262.") == [-0.0262]


def test_numeric_matches_absolute_tolerance_boundary() -> None:
    # tolerance is absolute and inclusive
    assert numeric_matches([101.0], expected=100.0, tolerance=1.0) is True
    assert numeric_matches([102.0], expected=100.0, tolerance=1.0) is False
    assert numeric_matches([100.0], expected=100.0, tolerance=0.0) is True


def test_numeric_matches_relative_tolerance_boundary() -> None:
    assert numeric_matches([105.0], expected=100.0, tolerance_pct=5.0) is True
    assert numeric_matches([106.0], expected=100.0, tolerance_pct=5.0) is False


def test_numeric_matches_any_number_in_the_answer_counts() -> None:
    # The agent may narrate several figures; one correct match is a pass.
    assert numeric_matches([7.0, 100.0, 3.5], expected=100.0, tolerance=0.0) is True


def test_numeric_matches_empty_answer_fails() -> None:
    assert numeric_matches([], expected=100.0, tolerance=0.0) is False


def test_numeric_matches_rejects_both_tolerance_kinds() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        numeric_matches([1.0], expected=1.0, tolerance=1.0, tolerance_pct=5.0)
