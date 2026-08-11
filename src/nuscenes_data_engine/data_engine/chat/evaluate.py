"""Answer-correctness eval for the dataset-chat agent — deterministic, no LLM judge.

Ground truth comes from a ``reference_sql`` per case, executed through the SAME guarded
catalog the agent uses, so a case can never ask for something the agent is structurally
unable to answer. Every check is a pure function over the agent's own output, so the
grading logic is unit-testable without a live model (design:
docs/superpowers/specs/2026-08-11-chat-eval-harness-design.md).
"""

from __future__ import annotations

import contextlib
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
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
    not ``run_sql``, or whose SQL errors, contribute nothing. DuckDB types a plain
    decimal literal (e.g. from a ``run_sql`` that echoes a value back) as DECIMAL, not
    DOUBLE; ``catalog._clip`` stringifies that (it isn't ``bool``/``int``/``float``), so
    a numeric-looking string cell is parsed too rather than silently dropped.
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
                elif isinstance(cell, str):
                    # Not a numeric-looking string (category name, token, ...) is fine.
                    with contextlib.suppress(ValueError):
                        values.add(float(cell))
    return values


def _matches_at_cited_precision(cited: float, observed: float) -> bool:
    """True when ``cited`` is ``observed`` rounded to the precision the answer used.

    An agent writing "about 16.3 km/h" for a retrieved 16.25118931104633 is grounded —
    it is reporting the value it saw at a sensible precision, not inventing one. An
    exact comparison here would contradict the numeric check, which tolerates the same
    rounding, and would penalise whichever model rounds more sensibly.
    """
    if abs(cited - observed) <= 1e-6:
        return True
    text = f"{cited}"
    decimals = len(text.split(".")[1]) if "." in text else 0
    return abs(cited - round(observed, decimals)) <= 1e-6


def is_grounded(
    answer: str, con: Any, steps: list[dict[str, Any]], question: str = ""
) -> bool:
    """True when every number in the answer traces to the agent's own tool output.

    Grounded means "derived from what was retrieved", so it accepts a value rounded to
    the precision the answer states, and accepts constants restated from the question
    itself (a question asking about 5 metres invites an answer that says 5). A number
    that is neither retrieved, nor a rounding of a retrieved value, nor present in the
    question, is unsupported — being plausible, or even correct, is not being grounded.
    """
    cited = extract_numbers(answer)
    if not cited:
        return True
    observed = observed_numbers(con, steps) | set(extract_numbers(question))
    return all(
        any(_matches_at_cited_precision(value, seen) for seen in observed)
        for value in cited
    )


@dataclass
class EvalCase:
    """One eval question plus how to check its answer."""

    id: str
    question: str
    reference_sql: str | None = None
    tolerance: float | None = None
    tolerance_pct: float | None = None
    expect_frames: bool = False


def load_cases(path: Path) -> list[EvalCase]:
    """Parse the eval-case YAML, validating ids and tolerance usage."""
    from nuscenes_data_engine.config import load_yaml

    raw = load_yaml(path).get("cases") or []
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for entry in raw:
        if "id" not in entry:
            raise ValueError(f"case missing required field 'id' in {path}: {entry!r}")
        case_id = str(entry["id"])
        if case_id in seen:
            raise ValueError(f"duplicate case id {case_id!r} in {path}")
        seen.add(case_id)
        if "question" not in entry:
            raise ValueError(f"case {case_id!r} missing required field 'question' in {path}")
        tolerance = entry.get("tolerance")
        tolerance_pct = entry.get("tolerance_pct")
        if tolerance is not None and tolerance_pct is not None:
            raise ValueError(
                f"case {case_id!r}: pass exactly one of tolerance / tolerance_pct"
            )
        cases.append(
            EvalCase(
                id=case_id,
                question=str(entry["question"]),
                reference_sql=entry.get("reference_sql"),
                tolerance=None if tolerance is None else float(tolerance),
                tolerance_pct=None if tolerance_pct is None else float(tolerance_pct),
                expect_frames=bool(entry.get("expect_frames", False)),
            )
        )
    return cases


def reference_value(con: Any, case: EvalCase) -> float:
    """The case's ground-truth number, from its reference SQL via the guarded catalog.

    A reference that errors or returns nothing is a HARNESS bug, not an agent failure,
    so it raises rather than scoring the case.
    """
    if case.reference_sql is None:
        raise ValueError(f"case {case.id!r} has no reference_sql")
    result = catalog.run_sql(con, case.reference_sql)
    if "error" in result:
        raise ValueError(f"case {case.id!r}: reference SQL failed — {result['error']}")
    if len(result["rows"]) != 1 or len(result["rows"][0]) != 1:
        raise ValueError(
            f"case {case.id!r}: reference SQL must return exactly one cell, got "
            f"{len(result['rows'])} rows x {len(result['rows'][0]) if result['rows'] else 0} cols"
        )
    cell = result["rows"][0][0]
    if isinstance(cell, bool) or not isinstance(cell, int | float):
        raise ValueError(
            f"case {case.id!r}: reference SQL must return a number, got {cell!r}"
        )
    return float(cell)


def validate_cases(con: Any, cases: list[EvalCase]) -> None:
    """Resolve every reference up front so a harness bug never scores as agent failure."""
    for case in cases:
        if case.reference_sql is not None:
            reference_value(con, case)  # raises with the case id on any problem
        elif not case.expect_frames:
            raise ValueError(
                f"case {case.id!r} has neither reference_sql nor expect_frames — "
                "it would pass on english/tool_use alone"
            )
