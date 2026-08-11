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
import unicodedata
from typing import Any

from nuscenes_data_engine.data_engine.chat import catalog

logger = logging.getLogger("nuscenes_data_engine")

# A standalone number: 1,234 | 12.5 | -0.0262 | 45%. The boundaries matter — without
# them the digit runs inside identifiers become phantom values (the hex token
# "8afcbca8a9c9462eb..." yielded 9462.0, and "Scene-0992" yielded -992.0), and since
# numeric_matches passes on ANY match, a phantom can score a wrong answer as correct.
# A leading "-" counts as a sign only when it does not follow a word character, so
# "Scene-0992" and "2026-08-11" no longer produce negatives.
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.-])-?\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9_])")


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
    both bounds are inclusive. An answer with no numbers never matches. With neither
    given, the match is exact (tolerance 0).
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


def _is_latin(char: str) -> bool:
    """True for Latin-script letters, including accented ones (é, ñ, ø)."""
    try:
        return unicodedata.name(char).startswith("LATIN")
    except ValueError:  # unnamed character
        return False


def is_english(text: str) -> bool:
    """True when the answer's alphabetic characters are overwhelmingly Latin script.

    Heuristic by design — see ``_LATIN_SHARE_MIN``. An empty or non-alphabetic answer
    is not English (there is nothing to answer with).
    """
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    latin = sum(1 for char in letters if _is_latin(char))
    return latin / len(letters) >= _LATIN_SHARE_MIN


def used_tools(steps: list[dict[str, Any]]) -> bool:
    """True when the agent called at least one tool (queried rather than recalled)."""
    return bool(steps)


def returned_frames(frames: list[dict[str, Any]]) -> bool:
    """True when the agent attached at least one example frame."""
    return bool(frames)


def observed_numbers(con: Any, steps: list[dict[str, Any]]) -> set[float]:
    """Every numeric value the agent's own SQL actually returns, plus row counts.

    ``steps`` records tool output as a summary string, not rows, so the SQL is
    re-executed through the same guarded catalog to recover the values. Steps that are
    not ``run_sql``, or whose SQL errors, contribute nothing.
    """
    values: set[float] = set()
    for step in steps:
        if step.get("tool") != "run_sql":
            continue
        sql = (step.get("input") or {}).get("sql")
        if not sql:
            continue
        result = catalog.run_sql(con, sql)
        if "error" in result:
            logger.debug("grounding: step SQL failed (%s)", result["error"])
            continue
        values.add(float(result["row_count"]))
        for row in result["rows"]:
            for cell in row:
                if isinstance(cell, bool):
                    continue  # bools are ints in Python; never a cited figure
                if isinstance(cell, int | float):
                    values.add(float(cell))
    return values


def is_grounded(answer: str, con: Any, steps: list[dict[str, Any]]) -> bool:
    """True when every number in the answer appears in the agent's own tool output.

    A number that happens to equal the reference but was never retrieved still fails —
    being right by luck is not being grounded. An answer citing no numbers is grounded
    vacuously.
    """
    cited = extract_numbers(answer)
    if not cited:
        return True
    observed = observed_numbers(con, steps)
    return all(any(abs(value - seen) <= 1e-6 for seen in observed) for value in cited)
