"""Answer-correctness eval for the dataset-chat agent — deterministic, no LLM judge.

Ground truth comes from a ``reference_sql`` per case, executed through the SAME guarded
catalog the agent uses, so a case can never ask for something the agent is structurally
unable to answer. Every check is a pure function over the agent's own output, so the
grading logic is unit-testable without a live model (design:
docs/superpowers/specs/2026-08-11-chat-eval-harness-design.md).
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import statistics
import time
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


def _observed_split(con: Any, steps: list[dict[str, Any]]) -> tuple[set[float], set[float]]:
    """(measurements, row_counts): cells and column sums vs result-shape counts.

    Row counts stay directly citable ("the query returned 8 rows") but must not feed
    derivations — a LIMIT-8 query's 8 is a property of the SQL text, not of the world,
    and reviews measured it manufacturing collisions (8/40x100 admitted a memorised 20).

    ``steps`` records tool output as a summary string, not rows, so the SQL is
    re-executed through the same guarded catalog to recover the values. Steps that are
    not ``run_sql``, or whose SQL errors, contribute nothing. DuckDB types a plain
    decimal literal (e.g. from a ``run_sql`` that echoes a value back) as DECIMAL, not
    DOUBLE; ``catalog._clip`` stringifies that (it isn't ``bool``/``int``/``float``), so
    a numeric-looking string cell is parsed too rather than silently dropped. Each
    numeric result column also contributes its sum — an answer totalling the
    per-channel counts it retrieved is reporting tool-derived data — unless the result
    was truncated (catalog.MAX_ROWS): a 50-row slice of a bigger result sums to a
    semi-arbitrary partial total, not a real measurement.
    """
    measurements: set[float] = set()
    row_counts: set[float] = set()
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
        row_counts.add(float(result["row_count"]))
        columns: dict[int, list[float]] = {}
        for row in result["rows"]:
            for index, cell in enumerate(row):
                if isinstance(cell, bool):
                    continue  # bools are ints in Python; never a cited figure
                if isinstance(cell, int | float):
                    columns.setdefault(index, []).append(float(cell))
                elif isinstance(cell, str):
                    # Not a numeric-looking string (category name, token, ...) is fine.
                    with contextlib.suppress(ValueError):
                        columns.setdefault(index, []).append(float(cell))
        for cells in columns.values():
            measurements.update(cells)
            if len(cells) >= 2 and not result["truncated"]:
                measurements.add(sum(cells))  # answers often total a result column
    return measurements, row_counts


def observed_numbers(con: Any, steps: list[dict[str, Any]]) -> set[float]:
    """Every numeric value the agent's own SQL actually returns, plus row counts.

    A thin union over ``_observed_split`` — see its docstring for what counts as a
    measurement vs a row count and why the split matters for derivations.
    """
    measurements, row_counts = _observed_split(con, steps)
    return measurements | row_counts


def _matches_at_cited_precision(cited: float, observed: float) -> bool:
    """True when ``cited`` is ``observed`` rounded to the precision the answer used.

    An agent writing "about 16.3 km/h" for a retrieved 16.25118931104633 is grounded —
    it is reporting the value it saw at a sensible precision, not inventing one. An
    exact comparison here would contradict the numeric check, which tolerates the same
    rounding, and would penalise whichever model rounds more sensibly.

    A citation written as an integer is matched at one decimal (float formatting yields
    "14.0"), i.e. a 0.1 window — deliberately tighter than the naive 0-decimal reading;
    widening it doubles the measured false-admission rate.
    """
    if abs(cited - observed) <= 1e-6:
        return True
    text = f"{cited}"
    decimals = len(text.split(".")[1]) if "." in text else 0
    return abs(cited - round(observed, decimals)) <= 1e-6


# One-step derivations the grounding check admits, each motivated by a real failure
# from the v1 review taxonomy (docs/superpowers/specs/2026-08-11-grounding-v2-design.md):
# a difference ("4,986 of 5,000, so 14 failed"), a percentage (99.66 = 4969/4986*100),
# and the m/s<->km/h conversion (18.4 = 5.1049*3.6). No products, no other quotients,
# no chaining: derived values are never re-derived, which is what keeps a WRONG
# computation (the historical 98.5% for a true 99.66%) failing.
_MS_TO_KMH = 3.6


def _derived_candidates(observed: set[float]) -> set[float]:
    """The full one-step admitted-derivation set: unit conversions + guarded pairs.

    A percentage candidate is admitted only as a share of a plausible (>=10) total —
    without that guard, any two small observed numbers manufacture a percentage for
    almost any cited value (1/2*100 = 50, 2/1*100 = 200), which review measured
    defeating a negative control. Materialized once per ``is_grounded`` call instead of
    recomputed per cited number.
    """
    obs = sorted(observed)
    candidates: set[float] = set()
    for a in obs:
        candidates.add(a * _MS_TO_KMH)
        candidates.add(a / _MS_TO_KMH)
    for i, a in enumerate(obs):
        for b in obs[i + 1 :]:
            candidates.add(a + b)
            candidates.add(a - b)
            candidates.add(b - a)
            if b >= 10 and a <= b:
                candidates.add(a / b * 100.0)  # a as a share of total b
            if a >= 10 and b <= a:
                candidates.add(b / a * 100.0)
    return candidates


def _derived_matches(cited: float, observed: set[float]) -> bool:
    """True when ``cited`` is one admitted arithmetic step from observed values."""
    return any(
        _matches_at_cited_precision(cited, value) for value in _derived_candidates(observed)
    )


# A permissive extractor used ONLY for the question side of grounding (the allowlist),
# never for the answer's own citations. `_NUMBER` (above) suppresses digits adjacent to
# a hyphen so identifiers like "Scene-0992" don't yield phantom values — but that same
# suppression means a question phrased "the 1-4 scale" (plain ASCII hyphen) never yields
# "4" as an echoable constant, even though the same constant cited with a non-ASCII dash
# (e.g. an en dash, as answers often render it) extracts fine from the answer side: the
# identical constant, checked but never admitted. Widening only the allowlist can remove
# false grounding failures, never create them — a citation still has to match some
# admitted value, this only grows the set it may match against.
_ANY_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _allowlist_numbers(text: str) -> set[float]:
    """Numbers a citation may legitimately echo (question constants).

    Deliberately more permissive than ``extract_numbers``: it also matches digits
    adjacent to punctuation such as the hyphen in "the 1-4 scale", which the citation
    extractor suppresses to avoid phantom values. Widening only the allowlist can
    remove false grounding failures, never create them.
    """
    return {float(m.group().replace(",", "")) for m in _ANY_NUMBER.finditer(text)}


def is_grounded(
    answer: str, con: Any, steps: list[dict[str, Any]], question: str = ""
) -> bool:
    """True when every number in the answer traces to the agent's own tool output.

    Grounded means "derived from what was retrieved", so it accepts a value rounded to
    the precision the answer states, and accepts constants restated from the question
    itself (a question asking about 5 metres invites an answer that says 5). A number
    that is neither retrieved, nor a rounding of a retrieved value, nor present in the
    question, is unsupported — being plausible, or even correct, is not being grounded.
    …or one admitted arithmetic step (sum, difference, percentage, m/s↔km/h) from
    observed values — showing your arithmetic is grounded; getting it wrong is not.
    """
    cited = extract_numbers(answer)
    if not cited:
        return True
    measurements, row_counts = _observed_split(con, steps)
    allowed = measurements | row_counts | _allowlist_numbers(question)
    derived = _derived_candidates(measurements)  # derivations draw on measurements ONLY
    return all(
        any(_matches_at_cited_precision(value, seen) for seen in allowed | derived)
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


def grade_case(
    con: Any,
    case: EvalCase,
    *,
    answer: str,
    steps: list[dict[str, Any]],
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-check verdicts for one answered case, plus the overall ``passed``.

    Checks that do not apply to a case are ABSENT from the dict rather than True, so a
    case is never credited with a check it never faced.
    """
    checks: dict[str, Any] = {
        "english": is_english(answer),
        "tool_use": used_tools(steps),
        "grounded": is_grounded(answer, con, steps, case.question),
    }
    if case.reference_sql is not None:
        expected = reference_value(con, case)
        checks["numeric"] = numeric_matches(
            extract_numbers(answer), expected, case.tolerance, case.tolerance_pct
        )
        checks["expected"] = expected
    if case.expect_frames:
        checks["frames"] = returned_frames(frames)
    checks["passed"] = all(
        value for key, value in checks.items() if isinstance(value, bool)
    )
    return checks


CHECK_NAMES = ("numeric", "english", "tool_use", "grounded", "frames")


def render_report(records: list[dict[str, Any]], *, model: str, provider: str) -> str:
    """Markdown summary: overall + per-check pass rates, then a per-case table."""
    total = len(records)
    passed = sum(1 for record in records if record["checks"].get("passed"))
    lines = [
        "# Chat-agent eval report\n",
        f"- provider: `{provider}`  model: `{model}`",
        f"- cases: **{passed}/{total} passed**"
        + (f" ({100.0 * passed / total:.0f}%)" if total else ""),
    ]
    latencies = [record["latency_s"] for record in records if record.get("latency_s")]
    if latencies:
        lines.append(f"- median latency: {statistics.median(latencies):.1f}s")
    lines.append("\n## Per-check pass rates\n")
    lines.append("| check | passed | applicable |")
    lines.append("|---|---:|---:|")
    for name in CHECK_NAMES:
        applicable = [r for r in records if name in r["checks"]]
        if not applicable:
            continue
        ok = sum(1 for r in applicable if r["checks"][name])
        lines.append(f"| {name} | {ok} | {len(applicable)} |")
    lines.append("\n## Cases\n")
    lines.append("| id | passed | failed checks | question |")
    lines.append("|---|---|---|---|")
    for record in records:
        checks = record["checks"]
        failed = ", ".join(
            name for name in CHECK_NAMES if name in checks and not checks[name]
        ) or "—"
        mark = "✅" if checks.get("passed") else "❌"
        lines.append(f"| {record['id']} | {mark} | {failed} | {record['question']} |")
    return "\n".join(lines) + "\n"


def run_eval(
    cases: list[EvalCase],
    *,
    con: Any,
    transport: Any,
    search_engine: Any | None,
    out_dir: Path,
    provider: str,
    limit: int | None = None,
) -> dict[str, Any]:
    """Answer every case, grade it, and write results.jsonl + report.md."""
    from nuscenes_data_engine.data_engine.chat import agent

    selected = cases[:limit] if limit else cases
    validate_cases(con, selected)  # a harness bug must never score as an agent failure
    records: list[dict[str, Any]] = []
    for index, case in enumerate(selected, start=1):
        started = time.time()
        try:
            result = agent.answer(
                case.question, transport=transport, con=con, search_engine=search_engine
            )
            checks = grade_case(
                con, case, answer=result.answer, steps=result.steps, frames=result.frames
            )
            record: dict[str, Any] = {
                "id": case.id, "question": case.question, "answer": result.answer,
                "model": result.model, "steps": result.steps,
                "n_frames": len(result.frames), "checks": checks,
                "latency_s": round(time.time() - started, 2),
            }
        except Exception as exc:  # a broken case must not abort the suite
            logger.warning("case %s raised: %s", case.id, exc)
            record = {
                "id": case.id, "question": case.question, "answer": "",
                "model": getattr(transport, "model", "unknown"), "steps": [],
                "n_frames": 0, "error": str(exc), "checks": {"passed": False},
                "latency_s": round(time.time() - started, 2),
            }
        records.append(record)
        logger.info(
            "[%d/%d] %s: %s", index, len(selected), case.id,
            "pass" if record["checks"].get("passed") else "FAIL",
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    model = records[0]["model"] if records else getattr(transport, "model", "unknown")
    results_path = out_dir / f"results_{provider}.jsonl"
    with open(results_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")
    report_path = out_dir / f"report_{provider}.md"
    report_path.write_text(render_report(records, model=model, provider=provider))

    passed = sum(1 for record in records if record["checks"].get("passed"))
    summary = {
        "provider": provider, "model": model, "n_cases": len(records),
        "n_passed": passed,
        "pass_rate": round(passed / len(records), 3) if records else 0.0,
        "results": str(results_path), "report": str(report_path),
    }
    logger.info("Eval: %d/%d passed -> %s", passed, len(records), report_path)
    return summary
