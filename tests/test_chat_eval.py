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


def test_extract_numbers_ignores_digits_inside_identifiers() -> None:
    # Real token from data/chat/log.jsonl — its digit runs must not become values.
    assert extract_numbers("frame 8afcbca8a9c9462eb61bb71caa644127 matched") == []
    assert extract_numbers("CAM_FRONT_2 and LIDAR_TOP") == []


def test_extract_numbers_hyphenated_names_do_not_become_negatives() -> None:
    assert extract_numbers("Scene-0992 has 41 keyframes.") == [41.0]
    assert extract_numbers("on 2026-08-11 we ran it") == [2026.0]


def test_extract_numbers_keeps_genuine_negatives_and_ranges() -> None:
    assert extract_numbers("delta was -0.0262 overall") == [-0.0262]
    assert extract_numbers("between 12-15 objects") == [12.0]  # range start only


def test_extract_numbers_still_handles_punctuation_neighbours() -> None:
    assert extract_numbers("(67.8%) and $1,234.56 and 45%.") == [67.8, 1234.56, 45.0]


def test_extract_numbers_on_a_real_logged_answer() -> None:
    """Guards the token-shredding regression using the repo's own agent output."""
    import json
    from pathlib import Path

    log = Path("data/chat/log.jsonl")
    if not log.is_file():
        pytest.skip("no chat log in this checkout")
    answers = [json.loads(line)["answer"] for line in log.read_text().splitlines() if line.strip()]
    hex_like = [a for a in answers if "sample_data_token" in a or "8afcbca8" in a]
    if not hex_like:
        pytest.skip("no token-citing answer in the log")
    for answer in hex_like:
        for value in extract_numbers(answer):
            assert abs(value) < 1e6, f"phantom value {value} extracted from an identifier"


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


def test_numeric_matches_defaults_to_exact_when_no_tolerance_given() -> None:
    assert numeric_matches([100.0], expected=100.0) is True
    assert numeric_matches([100.1], expected=100.0) is False


def test_is_english_accepts_normal_answer_and_rejects_thai() -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import is_english

    assert is_english("There are 12 night scenes in singapore-onenorth.") is True
    # The exact drift observed in data/chat/log.jsonl's first record.
    assert is_english("ในเซ็นซิเนกา (Singapore), มีสถานการณ์แสงสว่างน้อยที่บันทึกไว้") is False


def test_is_english_accepts_latin_accents() -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import is_english

    assert is_english("Résumé café naïve — a Boston scene.") is True


def test_is_english_ignores_digits_and_punctuation() -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import is_english

    # Mostly numbers/symbols with a little Latin text still counts as English.
    assert is_english("12,345 (67.8%) — ok") is True


def test_is_english_empty_answer_is_not_english() -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import is_english

    assert is_english("") is False
    assert is_english("   ") is False


def test_used_tools_detects_any_step() -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import used_tools

    assert used_tools([{"tool": "run_sql", "input": {"sql": "SELECT 1"}, "output": "1 rows"}])
    assert not used_tools([])


def test_returned_frames() -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import returned_frames

    assert returned_frames([{"sample_data_token": "t1"}]) is True
    assert returned_frames([]) is False
