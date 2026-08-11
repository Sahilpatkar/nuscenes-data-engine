# Chat-Agent Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the dataset-chat agent's answer correctness with deterministic, LLM-free grading — per the approved spec `docs/superpowers/specs/2026-08-11-chat-eval-harness-design.md`.

**Architecture:** A new `data_engine/chat/evaluate.py` holds pure grader functions (numeric match, English-script ratio, tool use, grounding, frames) plus a runner that calls the existing `agent.answer()` unchanged and writes `results.jsonl` + `report.md`. Cases live in a version-controlled `configs/chat_eval.yaml` whose `reference_sql` runs through the same guarded catalog the agent uses.

**Tech Stack:** Python 3.11, DuckDB (via the existing guarded catalog), pandas, Typer CLI, pytest. Strict mypy + ruff; guards raise `ValueError`, never `assert`.

**Working branch:** `chat-eval` (exists; spec committed as 32cec83).

**Conventions:** Repo root `/Users/sahilpatkar/Curosr_repos/nuscenes-data-engine`. Tests: `uv run pytest tests/test_chat_eval.py -v`. Lint/type: `uv run ruff check . && uv run mypy`. Do NOT bulk-`ruff format` (CI gates `ruff check` only).

**Verified facts (read these before starting; do not re-derive):**
- `agent.answer(question, *, transport, con, search_engine, history=None, max_turns=MAX_TURNS, log_path=None, graph_driver=None, graph_database="neo4j") -> ChatResult`.
- `ChatResult` is a dataclass: `answer: str`, `model: str`, `steps: list[dict]`, `frames: list[dict]`.
- **Each step is `{"tool": name, "input": args, "output": _summarize(output)}` — `output` is a SUMMARY STRING** (e.g. `"12 rows"`, `"3 frames found"`, `"error: ..."`), **not the rows**. The raw numbers the agent saw are therefore NOT in `steps`. For `run_sql` steps, `input` is `{"sql": "..."}`, so the SQL is recoverable and can be re-executed to recover the values — that is how the grounding check works (Task 3). This keeps `agent.py` untouched, as the spec requires.
- `catalog.run_sql(con, sql, max_rows=MAX_ROWS)` returns `{"columns": [...], "rows": [[...]], "row_count": int, "truncated": bool}` or `{"error": "..."}`.
- `catalog.open_catalog(processed_dir, labels_path=None)` builds the read-only DuckDB connection.
- The `chat` CLI command is a top-level `@app.command()` in `src/nuscenes_data_engine/cli.py` (~line 378) and shows how to build transport/catalog/search/graph — mirror it.
- `data/chat/log.jsonl` has 28 rows / 20 unique questions; keys `ts, question, model, steps, n_frames, answer, latency_s`.

---

### Task 1: Numeric extraction and tolerance matching

**Files:**
- Create: `src/nuscenes_data_engine/data_engine/chat/evaluate.py`
- Create: `tests/test_chat_eval.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chat_eval.py`:

```python
"""Tests for the chat-agent eval harness (pure graders; no LLM, no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_chat_eval.py -v`
Expected: FAIL at import — `ModuleNotFoundError` for `...chat.evaluate`.

- [ ] **Step 3: Implement**

Create `src/nuscenes_data_engine/data_engine/chat/evaluate.py`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_chat_eval.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuscenes_data_engine/data_engine/chat/evaluate.py tests/test_chat_eval.py
git commit -m "chat eval: numeric extraction + tolerance matching"
```

---

### Task 2: The English-script, tool-use and frames checks

**Files:**
- Modify: `src/nuscenes_data_engine/data_engine/chat/evaluate.py`
- Test: `tests/test_chat_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chat_eval.py`:

```python
def test_is_english_accepts_normal_answer_and_rejects_thai() -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import is_english

    assert is_english("There are 12 night scenes in singapore-onenorth.") is True
    # The exact drift observed in data/chat/log.jsonl's first record.
    assert is_english("ในเซ็นซิเนกา (Singapore), มีสถานการณ์แสงสว่างน้อยที่บันทึกไว้") is False


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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_chat_eval.py -k "english or used_tools or returned_frames" -v`
Expected: FAIL — `ImportError` for `is_english`.

- [ ] **Step 3: Implement**

Append to `evaluate.py`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_chat_eval.py -v`
Expected: 12 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuscenes_data_engine/data_engine/chat/evaluate.py tests/test_chat_eval.py
git commit -m "chat eval: english-script, tool-use and frames checks"
```

---

### Task 3: The grounding check (re-executes the agent's own SQL)

`steps[i]["output"]` is a summary string, so the numbers the agent saw are not in the
step record. Recover them by re-running each `run_sql` step's SQL through the same
guarded catalog. This keeps `agent.py` untouched and additionally verifies the SQL
really produces what the answer claims.

**Files:**
- Modify: `src/nuscenes_data_engine/data_engine/chat/evaluate.py`
- Test: `tests/test_chat_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chat_eval.py`:

```python
@pytest.fixture()
def tiny_con() -> Any:
    """A real DuckDB catalog over a two-row samples table."""
    pytest.importorskip("duckdb")
    import duckdb

    con = duckdb.connect()
    con.execute(
        "CREATE VIEW samples AS SELECT * FROM (VALUES "
        "('t1','boston-seaport',TRUE), ('t2','singapore-onenorth',FALSE)) "
        "AS s(sample_data_token, location, is_night)"
    )
    return con


def test_observed_numbers_reexecutes_sql_steps(tiny_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import observed_numbers

    steps = [{"tool": "run_sql", "input": {"sql": "SELECT count(*) FROM samples"},
              "output": "1 rows"}]
    assert 2.0 in observed_numbers(tiny_con, steps)


def test_observed_numbers_skips_non_sql_and_bad_sql(tiny_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import observed_numbers

    steps = [
        {"tool": "search_frames", "input": {"query": "foggy"}, "output": "3 frames found"},
        {"tool": "run_sql", "input": {"sql": "SELECT * FROM nope"}, "output": "error: x"},
    ]
    assert observed_numbers(tiny_con, steps) == set()  # neither contributes values


def test_is_grounded_passes_when_every_number_was_observed(tiny_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import is_grounded

    steps = [{"tool": "run_sql", "input": {"sql": "SELECT count(*) FROM samples"},
              "output": "1 rows"}]
    assert is_grounded("There are 2 frames.", tiny_con, steps) is True


def test_is_grounded_fails_on_a_number_never_observed(tiny_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import is_grounded

    steps = [{"tool": "run_sql", "input": {"sql": "SELECT count(*) FROM samples"},
              "output": "1 rows"}]
    # 99 appears nowhere in the tool output: being plausible is not being grounded.
    assert is_grounded("There are 2 frames, 99 of them at night.", tiny_con, steps) is False


def test_is_grounded_answer_without_numbers_is_grounded(tiny_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import is_grounded

    assert is_grounded("I could not find any matching frames.", tiny_con, []) is True
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_chat_eval.py -k "observed_numbers or grounded" -v`
Expected: FAIL — `ImportError` for `observed_numbers`.

- [ ] **Step 3: Implement**

Append to `evaluate.py` (add `from nuscenes_data_engine.data_engine.chat import catalog` to the module imports):

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_chat_eval.py -v`
Expected: 17 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuscenes_data_engine/data_engine/chat/evaluate.py tests/test_chat_eval.py
git commit -m "chat eval: grounding check via re-executed agent SQL"
```

---

### Task 4: Case loading and validation

**Files:**
- Modify: `src/nuscenes_data_engine/data_engine/chat/evaluate.py`
- Create: `configs/chat_eval.yaml`
- Test: `tests/test_chat_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chat_eval.py`:

```python
def test_load_cases_parses_and_defaults(tmp_path: Path) -> None:
    import yaml

    from nuscenes_data_engine.data_engine.chat.evaluate import load_cases

    path = tmp_path / "cases.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "cases": [
                    {"id": "a", "question": "How many?", "reference_sql": "SELECT 1",
                     "tolerance": 2},
                    {"id": "b", "question": "Show frames.", "expect_frames": True},
                ]
            }
        )
    )
    cases = load_cases(path)
    assert [c.id for c in cases] == ["a", "b"]
    assert cases[0].tolerance == 2.0 and cases[0].tolerance_pct is None
    assert cases[1].reference_sql is None and cases[1].expect_frames is True


def test_load_cases_rejects_both_tolerances(tmp_path: Path) -> None:
    import yaml

    from nuscenes_data_engine.data_engine.chat.evaluate import load_cases

    path = tmp_path / "cases.yaml"
    path.write_text(
        yaml.safe_dump(
            {"cases": [{"id": "a", "question": "q", "reference_sql": "SELECT 1",
                        "tolerance": 1, "tolerance_pct": 5}]}
        )
    )
    with pytest.raises(ValueError, match="exactly one"):
        load_cases(path)


def test_load_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    import yaml

    from nuscenes_data_engine.data_engine.chat.evaluate import load_cases

    path = tmp_path / "cases.yaml"
    path.write_text(
        yaml.safe_dump({"cases": [{"id": "a", "question": "q1"},
                                  {"id": "a", "question": "q2"}]})
    )
    with pytest.raises(ValueError, match="duplicate case id"):
        load_cases(path)


def test_reference_value_runs_through_the_guarded_catalog(tiny_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import EvalCase, reference_value

    case = EvalCase(id="a", question="q", reference_sql="SELECT count(*) FROM samples")
    assert reference_value(tiny_con, case) == 2.0


def test_reference_value_raises_on_broken_reference_sql(tiny_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import EvalCase, reference_value

    case = EvalCase(id="bad", question="q", reference_sql="SELECT * FROM nope")
    with pytest.raises(ValueError, match="reference SQL"):
        reference_value(tiny_con, case)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_chat_eval.py -k "load_cases or reference_value" -v`
Expected: FAIL — `ImportError` for `load_cases`.

- [ ] **Step 3: Implement**

Append to `evaluate.py` (add `from dataclasses import dataclass` and `from pathlib import Path` to the imports):

```python
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
        case_id = str(entry["id"])
        if case_id in seen:
            raise ValueError(f"duplicate case id {case_id!r} in {path}")
        seen.add(case_id)
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
    if not result["rows"] or not result["rows"][0]:
        raise ValueError(f"case {case.id!r}: reference SQL returned no rows")
    cell = result["rows"][0][0]
    if isinstance(cell, bool) or not isinstance(cell, int | float):
        raise ValueError(
            f"case {case.id!r}: reference SQL must return a number, got {cell!r}"
        )
    return float(cell)
```

- [ ] **Step 4: Create `configs/chat_eval.yaml`**

Seed it from the logged questions plus schema coverage. Before writing each case, **run its `reference_sql` yourself** against the real catalog to confirm it returns exactly one numeric cell:

```bash
uv run python -c "
from pathlib import Path
from nuscenes_data_engine.data_engine.chat.catalog import open_catalog, run_sql
con = open_catalog(Path('data/processed'), labels_path=Path('data/autolabel/labels.parquet'))
print(run_sql(con, 'SELECT count(DISTINCT scene_token) FROM samples WHERE is_night'))
"
```

Write ~20 cases covering `samples`, `annotations`, `labels`, `canbus`, `ego_pose`, each with a verified `reference_sql`. Start from this shape and extend — every case's SQL must be one you actually ran:

```yaml
# Chat-agent answer-correctness cases (docs/superpowers/specs/2026-08-11-chat-eval-harness-design.md).
# Each reference_sql runs through the same guarded catalog the agent uses and must
# return exactly one numeric cell. Seeded from data/chat/log.jsonl's real questions.
cases:
  - id: night_scenes_total
    question: How many distinct scenes are recorded at night?
    reference_sql: SELECT count(DISTINCT scene_token) FROM samples WHERE is_night
    tolerance: 0

  - id: hard_braking_keyframes
    question: How many keyframes have hard braking?
    reference_sql: SELECT count(*) FROM canbus WHERE is_hard_braking
    tolerance: 0

  - id: foggy_frame_examples
    question: Find frames that look foggy or have heavy lens glare, and show a few examples.
    expect_frames: true
```

Record in your report how many cases you wrote and which tables they cover.

- [ ] **Step 5: Verify**

Run: `uv run pytest tests/test_chat_eval.py -v` (22 PASS) and confirm the real config loads and every reference resolves:

```bash
uv run python -c "
from pathlib import Path
from nuscenes_data_engine.data_engine.chat.catalog import open_catalog
from nuscenes_data_engine.data_engine.chat.evaluate import load_cases, reference_value
con = open_catalog(Path('data/processed'), labels_path=Path('data/autolabel/labels.parquet'))
cases = load_cases(Path('configs/chat_eval.yaml'))
print(len(cases), 'cases')
for c in cases:
    if c.reference_sql:
        print(f'  {c.id}: {reference_value(con, c)}')
"
```

Every case must print a number. If one raises, fix its SQL — a broken reference is a harness bug.

- [ ] **Step 6: Commit**

```bash
git add src/nuscenes_data_engine/data_engine/chat/evaluate.py configs/chat_eval.yaml tests/test_chat_eval.py
git commit -m "chat eval: case schema, validation, and the seeded eval set"
```

---

### Task 5: The runner and report

**Files:**
- Modify: `src/nuscenes_data_engine/data_engine/chat/evaluate.py`
- Test: `tests/test_chat_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chat_eval.py`:

```python
def test_grade_case_reports_each_check(tiny_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import EvalCase, grade_case

    case = EvalCase(
        id="a", question="How many?",
        reference_sql="SELECT count(*) FROM samples", tolerance=0,
    )
    steps = [{"tool": "run_sql", "input": {"sql": "SELECT count(*) FROM samples"},
              "output": "1 rows"}]
    checks = grade_case(tiny_con, case, answer="There are 2 samples.", steps=steps, frames=[])
    assert checks["numeric"] is True
    assert checks["english"] is True
    assert checks["tool_use"] is True
    assert checks["grounded"] is True
    assert "frames" not in checks  # only for expect_frames cases
    assert checks["passed"] is True


def test_grade_case_marks_the_failing_check(tiny_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import EvalCase, grade_case

    case = EvalCase(
        id="a", question="How many?",
        reference_sql="SELECT count(*) FROM samples", tolerance=0,
    )
    checks = grade_case(
        tiny_con, case, answer="There are 99 samples.",
        steps=[{"tool": "run_sql", "input": {"sql": "SELECT count(*) FROM samples"},
                "output": "1 rows"}],
        frames=[],
    )
    assert checks["numeric"] is False and checks["grounded"] is False
    assert checks["passed"] is False


def test_grade_case_without_reference_skips_numeric(tiny_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import EvalCase, grade_case

    case = EvalCase(id="f", question="Show frames.", expect_frames=True)
    checks = grade_case(
        tiny_con, case, answer="Here are some frames.",
        steps=[{"tool": "search_frames", "input": {"query": "fog"}, "output": "2 frames found"}],
        frames=[{"sample_data_token": "t1"}],
    )
    assert "numeric" not in checks  # skipped, not passed
    assert checks["frames"] is True and checks["passed"] is True


def test_render_report_summarises_pass_rates() -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import render_report

    records = [
        {"id": "a", "question": "q1", "latency_s": 1.0,
         "checks": {"numeric": True, "english": True, "tool_use": True,
                    "grounded": True, "passed": True}},
        {"id": "b", "question": "q2", "latency_s": 3.0,
         "checks": {"numeric": False, "english": True, "tool_use": True,
                    "grounded": True, "passed": False}},
    ]
    markdown = render_report(records, model="qwen2.5:14b", provider="local")
    assert "qwen2.5:14b" in markdown
    assert "1/2" in markdown or "50" in markdown  # overall pass rate
    assert "numeric" in markdown and "| a " in markdown and "| b " in markdown
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_chat_eval.py -k "grade_case or render_report" -v`
Expected: FAIL — `ImportError` for `grade_case`.

- [ ] **Step 3: Implement**

Append to `evaluate.py` (add `import json`, `import statistics`, `import time` to the imports):

```python
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
        "grounded": is_grounded(answer, con, steps),
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
            name for name in CHECK_NAMES
            if name in checks and not checks[name]
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
            record = {
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
                "n_frames": 0, "error": str(exc),
                "checks": {"passed": False},
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_chat_eval.py -v`
Expected: 26 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuscenes_data_engine/data_engine/chat/evaluate.py tests/test_chat_eval.py
git commit -m "chat eval: case grading, runner and markdown report"
```

---

### Task 6: CLI + an end-to-end run with a stub transport

**Files:**
- Modify: `src/nuscenes_data_engine/cli.py` (after the `chat` command, ~line 440)
- Test: `tests/test_chat_eval.py`

- [ ] **Step 1: Write the failing test** (end-to-end through `run_eval` with a stub transport — no LLM, no network)

Append to `tests/test_chat_eval.py`:

```python
def test_run_eval_end_to_end_with_a_stub_transport(tmp_path: Path, tiny_con: Any) -> None:
    """The whole loop: answer -> grade -> artifacts, with a scripted transport."""
    from nuscenes_data_engine.data_engine.chat.evaluate import EvalCase, run_eval

    class _StubTransport:
        model = "stub-model"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
            self.calls += 1
            if self.calls == 1:  # first turn: call the SQL tool
                return {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "run_sql",
                                "arguments": '{"sql": "SELECT count(*) FROM samples"}',
                            },
                        }
                    ],
                }
            return {"content": "There are 2 samples.", "tool_calls": []}

    cases = [
        EvalCase(id="a", question="How many samples?",
                 reference_sql="SELECT count(*) FROM samples", tolerance=0)
    ]
    summary = run_eval(
        cases, con=tiny_con, transport=_StubTransport(), search_engine=None,
        out_dir=tmp_path / "eval", provider="stub",
    )
    assert summary["n_cases"] == 1 and summary["n_passed"] == 1
    assert summary["pass_rate"] == 1.0
    assert (tmp_path / "eval" / "results_stub.jsonl").is_file()
    assert "stub-model" in (tmp_path / "eval" / "report_stub.md").read_text()


def test_run_eval_records_a_failing_case_without_aborting(tmp_path: Path, tiny_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import EvalCase, run_eval

    class _BoomTransport:
        model = "stub-model"

        def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
            raise RuntimeError("transport exploded")

    cases = [
        EvalCase(id="a", question="q1"),
        EvalCase(id="b", question="q2"),
    ]
    summary = run_eval(
        cases, con=tiny_con, transport=_BoomTransport(), search_engine=None,
        out_dir=tmp_path / "eval", provider="stub",
    )
    assert summary["n_cases"] == 2 and summary["n_passed"] == 0  # both recorded, none lost
```

The stub's reply shape must match what `agent.answer` expects — **read `agent.answer`'s
loop first** (it reads `reply.get("tool_calls")` and `reply["content"]`, and parses
`function.arguments` as JSON). If the real shape differs from the stub above, adapt the
stub (not the agent) and say so in your report.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_chat_eval.py -k run_eval_end_to_end -v`
Expected: FAIL (the stub's shape or the artifacts don't line up yet) — iterate on the stub until `agent.answer` drives it, but do NOT change `agent.py`.

- [ ] **Step 3: Add the CLI command**

In `src/nuscenes_data_engine/cli.py`, after the `chat` command:

```python
@app.command("chat-eval")
def chat_eval(
    config: Path = typer.Option(Path("configs/chat_eval.yaml"), "--config", "-c"),
    provider: str | None = typer.Option(None, "--provider", help="local | anthropic."),
    model: str | None = typer.Option(None, "--model", help="Override the chat model."),
    limit: int | None = typer.Option(None, "--limit", help="First N cases (smoke runs)."),
    processed_dir: Path = typer.Option(Path("data/processed"), "--processed-dir"),
) -> None:
    """Score the chat agent's answers against reference SQL (deterministic, no judge)."""
    from nuscenes_data_engine.config import get_settings
    from nuscenes_data_engine.data_engine.chat.catalog import open_catalog
    from nuscenes_data_engine.data_engine.chat.evaluate import load_cases, run_eval
    from nuscenes_data_engine.data_engine.chat.transports import make_transport

    settings = get_settings()
    transport = make_transport(settings, provider=provider, model=model)
    con = open_catalog(
        processed_dir, labels_path=Path(settings.data_dir) / "autolabel" / "labels.parquet"
    )
    try:
        from nuscenes_data_engine.data_engine.search import SearchEngine

        engine: Any | None = SearchEngine(
            Path(settings.search_lancedb_path), settings.search_table,
            settings.search_model_name, device=settings.search_device,
        )
    except (ImportError, FileNotFoundError) as exc:
        logger.warning("Vector search unavailable (%s) — SQL-only eval.", exc)
        engine = None

    summary = run_eval(
        load_cases(config), con=con, transport=transport, search_engine=engine,
        out_dir=Path(settings.data_dir) / "chat" / "eval",
        provider=provider or "default", limit=limit,
    )
    logger.info("Eval summary: %s", summary)
```

(Check `make_transport`'s real signature before writing this — the `chat` command calls
it as `make_transport(settings, provider=..., model=..., base_url=...)`. Match it.)

- [ ] **Step 4: Verify**

Run: `uv run pytest tests/test_chat_eval.py -v` (28 PASS), `uv run nuscenes-data-engine chat-eval --help` (shows the five options). Do NOT run a live eval yet — that is Task 7.

- [ ] **Step 5: Commit**

```bash
git add src/nuscenes_data_engine/data_engine/chat/evaluate.py src/nuscenes_data_engine/cli.py tests/test_chat_eval.py
git commit -m "chat eval: chat-eval CLI + end-to-end stub-transport test"
```

---

### Task 7: CI-parity sweep and the two real runs (operational)

- [ ] **Step 1:** `uv run pytest -q` — report exact counts. `uv run ruff check .` clean. Bare `uv run mypy` clean. Commit any fixes as `chat eval: lint/type fixes`.
- [ ] **Step 2: Local provider run.** Ollama must be serving `qwen2.5:14b` on the infra Mac (see docs/DATASET_CHAT.md for the exact command). Then:
  ```bash
  uv run nuscenes-data-engine chat-eval --provider local
  ```
  Record: pass rate, per-check pass rates, median latency, and which case ids failed which checks. **A low pass rate is the finding, not a failure** — do not tune the cases to make it look better.
- [ ] **Step 3: Anthropic provider run.** Needs `ANTHROPIC_API_KEY` in `.env`. If the key is absent, STOP and report — do not skip silently, since the provider comparison is the deliverable.
  ```bash
  uv run nuscenes-data-engine chat-eval --provider anthropic
  ```
  Record the same numbers.
- [ ] **Step 4:** Confirm both artifact pairs exist and did not overwrite each other:
  `ls data/chat/eval/` → `results_local.jsonl`, `report_local.md`, `results_anthropic.jsonl`, `report_anthropic.md`.
- [ ] **Step 5:** Check specifically whether the local run reproduced the logged Thai drift (any case failing `english`) and report it either way — that observation motivated the whole harness.

---

### Task 8: Docs

**Files:** `docs/DATASET_CHAT.md`, `docs/PROJECT.md`

- [ ] **Step 1:** `docs/DATASET_CHAT.md` — add an "Answer-correctness evaluation" section: what the harness measures and why there is no LLM judge (reference SQL through the same guarded catalog); the five checks with one line each, noting `english` is a script-ratio heuristic and `grounded` re-executes the agent's own SQL because steps store summaries; the case-set location and size; the **measured** per-provider table from Task 7; and the runbook. Replace the existing qualitative claim about local-model limits with the measured numbers, keeping any limitation the data still supports.
- [ ] **Step 2:** `docs/PROJECT.md` — §9's chat-agent-upgrades item: mark the eval harness DONE with the headline pass rates; add a line to §5 results if that section carries chat entries.
- [ ] **Step 3:** Verify: `grep -n "chat-eval\|Answer-correctness" docs/*.md | head`; `uv run pytest -q` still green.
- [ ] **Step 4:** Commit: `git add docs && git commit -m "chat eval: docs (method + measured per-provider pass rates)"`

---

### After the plan (not plan tasks)

1. Final whole-branch review, then superpowers:finishing-a-development-branch (push, PR, CI).
2. The remaining chat upgrades — streaming, chart generation, the saved-questions gallery — are separate features; the eval harness now gives them a regression net.
