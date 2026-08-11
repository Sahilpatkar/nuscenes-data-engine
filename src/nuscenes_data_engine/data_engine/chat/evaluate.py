"""Answer-correctness eval for the dataset-chat agent — deterministic, no LLM judge.

Ground truth comes from a ``reference_sql`` per case, executed through the SAME guarded
catalog the agent uses, so a case can never ask for something the agent is structurally
unable to answer. Every check is a pure function over the agent's own output, so the
grading logic is unit-testable without a live model (design:
docs/superpowers/specs/2026-08-11-chat-eval-harness-design.md).
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("nuscenes_data_engine")

# 1,234 | 12.5 | -0.0262 | 45 (a trailing % is stripped by the caller's float()).
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_numbers(text: str) -> list[float]:
    """Numeric literals in the order they appear; handles thousands separators."""
    values: list[float] = []
    for match in _NUMBER.finditer(text):
        try:
            values.append(float(match.group().replace(",", "")))
        except ValueError:  # pragma: no cover - the regex only yields parseable text
            continue
    return values


def numeric_matches(
    values: list[float],
    expected: float,
    tolerance: float | None = None,
    tolerance_pct: float | None = None,
) -> bool:
    """True when any value is within tolerance of ``expected``.

    Exactly one of ``tolerance`` (absolute) / ``tolerance_pct`` (relative) may be given;
    both bounds are inclusive. An answer with no numbers never matches.
    """
    if tolerance is not None and tolerance_pct is not None:
        raise ValueError("Pass exactly one of tolerance / tolerance_pct, not both")
    if tolerance_pct is not None:
        allowed = abs(expected) * tolerance_pct / 100.0
    else:
        allowed = tolerance if tolerance is not None else 0.0
    return any(abs(value - expected) <= allowed for value in values)


# Share of *alphabetic* characters that must be Latin for an answer to count as English.
# A heuristic, deliberately not a language model: it exists to catch the wholesale
# script drift seen in data/chat/log.jsonl (an answer returned in Thai), not to judge
# fluency. Digits, punctuation and whitespace are ignored entirely.
_LATIN_SHARE_MIN = 0.9


def is_english(text: str) -> bool:
    """True when the answer's alphabetic characters are overwhelmingly Latin script.

    Heuristic by design — see ``_LATIN_SHARE_MIN``. An empty or non-alphabetic answer
    is not English (there is nothing to answer with).
    """
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    latin = sum(1 for char in letters if char.isascii())
    return latin / len(letters) >= _LATIN_SHARE_MIN


def used_tools(steps: list[dict[str, Any]]) -> bool:
    """True when the agent called at least one tool (queried rather than recalled)."""
    return bool(steps)


def returned_frames(frames: list[dict[str, Any]]) -> bool:
    """True when the agent attached at least one example frame."""
    return bool(frames)
