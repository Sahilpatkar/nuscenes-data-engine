# Grounding v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the grounding check's bias against models that show their arithmetic (per spec `docs/superpowers/specs/2026-08-11-grounding-v2-design.md`), add an offline replay re-grade, then measure: replay both stored runs and live-run `qwen2.5:32b`.

**Architecture:** All grader changes live in `data_engine/chat/evaluate.py` (pure functions; `agent.py`/`catalog.py` untouched). The allowed set for a citation grows by: per-column sums, four one-step derivations over observed values, schema-prompt constants, and the attached-frame count. A `regrade()` function replays stored `results_*.jsonl` through the current graders with zero LLM calls.

**Tech Stack:** Python 3.11, DuckDB via the existing guarded catalog, Typer CLI, pytest. Strict mypy + ruff (100-char lines). Guards raise `ValueError`, never `assert`.

**Working branch:** `grounding-v2` (exists, stacked on `chat-eval`; spec committed as bc88bb1). One commit per task.

**Verified facts (read before starting; do not re-derive):**
- `evaluate.py` (~430 lines) currently has: `_NUMBER` (citation extractor, hyphen-suppressing) at :33, `extract_numbers`, `numeric_matches`, `is_english`, `used_tools`, `returned_frames`, `observed_numbers(con, steps)` at :106 (flat `set[float]`: row_count + cells, numeric-looking string cells parsed), `_matches_at_cited_precision(cited, observed)` at :141, `_ANY_NUMBER`/`_allowlist_numbers` at :165 (allowlist extractor, **no `-?` sign** — "≤ -3.0" yields `3.0`), `is_grounded(answer, con, steps, question="")` at :179, `EvalCase`, `load_cases`, `reference_value`, `validate_cases`, `grade_case(con, case, *, answer, steps, frames)` at :283, `CHECK_NAMES`, `render_report(records, *, model, provider)`, `run_eval(...)`.
- The CLI command `chat-eval` is at `src/nuscenes_data_engine/cli.py:450-497`: builds transport via `make_transport(settings, provider=provider, model=model)`, opens the catalog, probes `SearchEngine` (construction + `search_text("probe", k=1)` inside one try/except), calls `run_eval(..., provider=provider or settings.chat_provider, ...)` with `out_dir=Path(settings.data_dir) / "chat" / "eval"`.
- Stored records in `data/chat/eval/results_{local,anthropic}.jsonl` carry: `id, question, answer, model, steps, n_frames, checks, latency_s` (and `error` for transport failures). They do NOT carry a frames list — only the count.
- The agent's system prompt embeds `catalog.schema_prompt(catalog.catalog_tables(con))` (see `agent.answer`). The eval runs without the graph, so that text is exactly what the model saw.
- Tests live in `tests/test_chat_eval.py` (46 tests; `tiny_con` fixture = bare DuckDB with a 3-column `samples` view). CI has no `torch` and no network; it must stay that way.
- Backups `results_local_qwen2.5-14b.jsonl` / `report_local_qwen2.5-14b.md` already exist in `data/chat/eval/` — do not touch them.
- `qwen2.5:32b` is being pulled by a background task; Task 7 checks `ollama list` before running.

**Real numbers used by the tests below (from the final-review failure categorisation — treat as given):**
- `labels_parse_ok_count`: answer cites 5,000 (schema constant) and 14 = 5000 − 4986 (derived difference).
- `labels_night_agreement_pct` (Claude run): observed includes 4,969 and 4,986; the answer's 99.66 = 4969/4986×100.
- Historical log record 5 (the miscalculation): observed = {4213, 756, 4986}; the answer said 98.5%, which is not reachable by any single admitted operation on those values (the correct path is two steps: 4213+756=4969, then 4969/4986×100=99.66 — v2 is one-step only, so even the correct value would fail *from those inputs*; from {4969, 4986} it passes).
- `ego_pose_avg_speed_onenorth`: 18.4 = 5.104916902905238 × 3.6 at 1-decimal cited precision.
- `missing_cam_files`: 1,183,790 = sum of six per-channel counts.

---

### Task 1: Per-column sums in `observed_numbers`

**Files:**
- Modify: `src/nuscenes_data_engine/data_engine/chat/evaluate.py:106-138`
- Test: `tests/test_chat_eval.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_chat_eval.py`):

```python
def test_observed_numbers_includes_per_column_sums(tiny_con: Any) -> None:
    """An answer totalling a result column cites tool-derived data (missing_cam_files)."""
    from nuscenes_data_engine.data_engine.chat.evaluate import observed_numbers

    steps = [{"tool": "run_sql",
              "input": {"sql": "SELECT * FROM (VALUES (10, 'a'), (20, 'b'), (30, 'c')) t(n, s)"},
              "output": "3 rows"}]
    observed = observed_numbers(tiny_con, steps)
    assert {10.0, 20.0, 30.0} <= observed   # cells, as before
    assert 60.0 in observed                  # the column sum
    assert 3.0 in observed                   # row_count, as before
```

- [ ] **Step 2:** `uv run pytest tests/test_chat_eval.py::test_observed_numbers_includes_per_column_sums -v` → FAIL (60.0 not in the set).

- [ ] **Step 3: Implement.** Replace the row loop in `observed_numbers` (keep the docstring's first two paragraphs; extend it with the column-sum sentence shown):

```python
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
            values.update(cells)
            if len(cells) >= 2:
                values.add(sum(cells))  # answers often total a result column
    return values
```

Docstring addition: "Each numeric result column also contributes its sum — an answer totalling the per-channel counts it retrieved is reporting tool-derived data."

- [ ] **Step 4:** `uv run pytest tests/test_chat_eval.py -q` → all pass (47).

- [ ] **Step 5: Commit:** `git add -u && git add tests/test_chat_eval.py && git commit -m "grounding v2: per-column sums join the observed set"`

---

### Task 2: Targeted derivations (`_derived_matches`) wired into `is_grounded`

**Files:**
- Modify: `src/nuscenes_data_engine/data_engine/chat/evaluate.py` (new helper after `_matches_at_cited_precision`; rewire `is_grounded`)
- Test: `tests/test_chat_eval.py`

- [ ] **Step 1: Write the failing tests:**

```python
def test_derived_matches_admits_the_four_taxonomy_rules() -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import _derived_matches

    # difference: "4,986 of 5,000, so 14 failed"
    assert _derived_matches(14.0, {5000.0, 4986.0}) is True
    # sum
    assert _derived_matches(9986.0, {5000.0, 4986.0}) is True
    # percentage at cited precision: 99.66 = 4969/4986*100
    assert _derived_matches(99.66, {4969.0, 4986.0}) is True
    # m/s -> km/h at cited precision: 18.4 = 5.1049... * 3.6
    assert _derived_matches(18.4, {5.104916902905238}) is True
    # and km/h -> m/s
    assert _derived_matches(5.1, {18.377700850458857}) is True


def test_derived_matches_rejects_wrong_arithmetic_record5() -> None:
    """The historical miscalculation must STILL fail: correct inputs, wrong result."""
    from nuscenes_data_engine.data_engine.chat.evaluate import _derived_matches

    observed = {4213.0, 756.0, 4986.0}
    assert _derived_matches(98.5, observed) is False
    # Even the CORRECT value is not one-step-derivable from these inputs (it needs
    # 4213+756=4969 first). v2 is one step by design; this documents the bound.
    assert _derived_matches(99.66, observed) is False


def test_derived_matches_collision_guard() -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import _derived_matches

    observed = {40.0, 2.0}
    assert _derived_matches(43.0, observed) is False   # near-miss must not pass
    assert _derived_matches(42.0, observed) is True    # 40+2: the documented trade-off
    assert _derived_matches(38.0, observed) is True    # 40-2


def test_is_grounded_accepts_a_derived_difference(tiny_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import is_grounded

    steps = [{"tool": "run_sql", "input": {"sql": "SELECT 5000, 4986"}, "output": "1 rows"}]
    assert is_grounded("4986 parsed, so 14 failed.", tiny_con, steps) is True
    # An underived number is still unsupported.
    assert is_grounded("4986 parsed, so 15 failed and 77 crashed.", tiny_con, steps) is False
```

- [ ] **Step 2:** Run them → FAIL (ImportError for `_derived_matches`).

- [ ] **Step 3: Implement.** Insert after `_matches_at_cited_precision`:

```python
# One-step derivations the grounding check admits, each motivated by a real failure
# from the v1 review taxonomy (docs/superpowers/specs/2026-08-11-grounding-v2-design.md):
# a difference ("4,986 of 5,000, so 14 failed"), a percentage (99.66 = 4969/4986*100),
# and the m/s<->km/h conversion (18.4 = 5.1049*3.6). No products, no other quotients,
# no chaining: derived values are never re-derived, which is what keeps a WRONG
# computation (the historical 98.5% for a true 99.66%) failing.
_MS_TO_KMH = 3.6


def _derived_matches(cited: float, observed: set[float]) -> bool:
    """True when ``cited`` is one admitted arithmetic step from observed values."""
    obs = sorted(observed)
    for a in obs:
        if _matches_at_cited_precision(cited, a * _MS_TO_KMH):
            return True
        if _matches_at_cited_precision(cited, a / _MS_TO_KMH):
            return True
    for i, a in enumerate(obs):
        for b in obs[i + 1 :]:
            candidates = [a + b, a - b, b - a]
            if b != 0:
                candidates.append(a / b * 100.0)
            if a != 0:
                candidates.append(b / a * 100.0)
            if any(_matches_at_cited_precision(cited, value) for value in candidates):
                return True
    return False
```

Rewire `is_grounded`'s return (keep the signature for now; Task 3 extends it):

```python
    observed = observed_numbers(con, steps)
    allowed = observed | _allowlist_numbers(question)
    return all(
        any(_matches_at_cited_precision(value, seen) for seen in allowed)
        or _derived_matches(value, observed)   # derivations draw on observed ONLY
        for value in cited
    )
```

Extend `is_grounded`'s docstring: "…or one admitted arithmetic step (sum, difference, percentage, m/s↔km/h) from observed values — showing your arithmetic is grounded; getting it wrong is not."

- [ ] **Step 4:** `uv run pytest tests/test_chat_eval.py -q` → all pass (51). All pre-existing `is_grounded` tests must pass unchanged — if one now passes/fails differently, STOP and report (it would mean a derivation collides with an existing fixture).

- [ ] **Step 5: Commit:** `git commit -am "grounding v2: one-step targeted derivations (sum/diff/pct/unit)"`

---

### Task 3: Schema-prompt constants + frame count; `grade_case` takes `n_frames`

**Files:**
- Modify: `src/nuscenes_data_engine/data_engine/chat/evaluate.py` (`_ANY_NUMBER`, new `_schema_constants`, `is_grounded` signature, `grade_case`, `run_eval` call site)
- Test: `tests/test_chat_eval.py` (new tests + mechanical updates to the four existing `grade_case` tests)

- [ ] **Step 1: Write the failing tests:**

```python
def test_allowlist_numbers_captures_signed_and_hyphenated() -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import _allowlist_numbers

    allowed = _allowlist_numbers("deceleration <= -3.0 m/s2 within a +-0.5 s window, 1-4 scale")
    # Sign symmetry: schema text says -3.0, an answer may cite 3.0 (or vice versa).
    assert {3.0, -3.0, 0.5, -0.5, 1.0, 4.0} <= allowed


def test_is_grounded_admits_schema_prompt_constants(tiny_con: Any, monkeypatch: Any) -> None:
    from nuscenes_data_engine.data_engine.chat import evaluate

    monkeypatch.setattr(
        evaluate, "_schema_constants",
        lambda con: {3.0, -3.0, 0.5, -0.5},
    )
    steps = [{"tool": "run_sql", "input": {"sql": "SELECT 94"}, "output": "1 rows"}]
    assert evaluate.is_grounded(
        "94 keyframes brake harder than 3.0 m/s2 in the 0.5 s window.", tiny_con, steps
    ) is True


def test_is_grounded_admits_the_attached_frame_count(tiny_con: Any, monkeypatch: Any) -> None:
    from nuscenes_data_engine.data_engine.chat import evaluate

    # Pin schema constants to empty so this test isolates the n_frames mechanism.
    monkeypatch.setattr(evaluate, "_schema_constants", lambda con: set())
    steps = [{"tool": "search_frames", "input": {"query": "fog"}, "output": "6 frames found"}]
    assert evaluate.is_grounded("I attached 6 example frames.", tiny_con, steps, n_frames=6) is True
    assert evaluate.is_grounded("I attached 6 example frames.", tiny_con, steps, n_frames=0) is False


def test_schema_constants_never_raises_on_a_bare_connection(tiny_con: Any) -> None:
    """The tiny fixture lacks most catalog views; extraction degrades to a set, not a crash."""
    from nuscenes_data_engine.data_engine.chat.evaluate import _schema_constants

    assert isinstance(_schema_constants(tiny_con), set)
```

- [ ] **Step 2:** Run them → FAIL (`_schema_constants` missing; `n_frames` unexpected keyword).

- [ ] **Step 3: Implement.**

(a) Give `_ANY_NUMBER` an optional sign and symmetrize the allowlist:

```python
_ANY_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _allowlist_numbers(text: str) -> set[float]:
    """Numbers a citation may legitimately echo (question / schema constants).

    Deliberately more permissive than ``extract_numbers``: it matches digits adjacent
    to punctuation ("the 1-4 scale") and keeps sign symmetry — schema text saying
    ``-3.0`` licenses an answer citing ``3.0`` and vice versa (a deceleration cited as
    a positive magnitude is the same constant). Widening only the allowlist can remove
    false grounding failures, never create them.
    """
    values = {float(m.group().replace(",", "")) for m in _ANY_NUMBER.finditer(text)}
    return values | {-v for v in values}
```

NOTE: with the sign added, "1-4 scale" now yields `{1.0, -4.0}` and symmetry restores `4.0` — verify `test_is_grounded_accepts_a_question_constant_next_to_an_ascii_hyphen` (existing) still passes.

(b) New `_schema_constants`, placed after `_allowlist_numbers`:

```python
def _schema_constants(con: Any) -> set[float]:
    """Numbers in the schema-prompt text the agent's system prompt embeds.

    A model quoting a documented threshold ("hard braking means <= -3.0 m/s2") is
    reporting what it was shown, not inventing. Extracted from the SAME
    ``catalog.schema_prompt`` text ``agent.answer`` uses — never a hardcoded list, so
    it cannot drift. On a connection lacking the catalog views (tests), degrade to
    whatever extracts rather than raising: grading must never crash on the fixture.
    """
    try:
        return _allowlist_numbers(catalog.schema_prompt(catalog.catalog_tables(con)))
    except Exception as exc:
        logger.debug("schema constants unavailable (%s)", exc)
        return set()
```

READ `catalog.schema_prompt` / `catalog.catalog_tables` first and check what they do on the bare fixture connection; if they raise, the except path covers it — but confirm the real catalog path returns text containing `-3.0` and `0.5` (canbus column docs) with a quick `uv run python -c` probe against `data/processed`, and paste the extracted set in your report.

(c) `is_grounded` gains `n_frames`:

```python
def is_grounded(
    answer: str,
    con: Any,
    steps: list[dict[str, Any]],
    question: str = "",
    *,
    n_frames: int = 0,
) -> bool:
```

body:

```python
    cited = extract_numbers(answer)
    if not cited:
        return True
    observed = observed_numbers(con, steps)
    allowed = observed | _allowlist_numbers(question) | _schema_constants(con)
    if n_frames > 0:
        allowed.add(float(n_frames))  # "6 examples attached" is tool-visible data
    return all(
        any(_matches_at_cited_precision(value, seen) for seen in allowed)
        or _derived_matches(value, observed)
        for value in cited
    )
```

(d) `grade_case` takes the count, not the list (stored records only carry `n_frames`, which Task 4's replay needs):

```python
def grade_case(
    con: Any,
    case: EvalCase,
    *,
    answer: str,
    steps: list[dict[str, Any]],
    n_frames: int,
) -> dict[str, Any]:
```

with `"grounded": is_grounded(answer, con, steps, case.question, n_frames=n_frames)` and `checks["frames"] = n_frames > 0` (the `returned_frames` helper and its test remain — it is still the public predicate for a frames list). In `run_eval`, the call site becomes `n_frames=len(result.frames)`.

(e) Mechanically update the four existing `grade_case` tests: `frames=[]` → `n_frames=0`; `frames=[{"sample_data_token": "t1"}]` → `n_frames=1`.

- [ ] **Step 4:** `uv run pytest tests/test_chat_eval.py -q` → all pass (55). Then `uv run pytest -q` (suite) and `uv run mypy` — the signature change must not break other callers (grep `grade_case(` to confirm only `run_eval` + tests call it).

- [ ] **Step 5: Commit:** `git commit -am "grounding v2: schema-prompt constants, frame count, signed allowlist"`

---

### Task 4: `regrade()` — offline replay through the current graders

**Files:**
- Modify: `src/nuscenes_data_engine/data_engine/chat/evaluate.py` (new function after `run_eval`; add `replace` to the dataclasses import)
- Test: `tests/test_chat_eval.py`

- [ ] **Step 1: Write the failing tests:**

```python
def test_regrade_replays_stored_records_without_llm(tmp_path: Path, tiny_con: Any) -> None:
    import json as _json

    from nuscenes_data_engine.data_engine.chat.evaluate import EvalCase, regrade

    stored = [
        {"id": "a", "question": "How many samples?", "answer": "There are 2 samples.",
         "model": "stub-model", "steps": [{"tool": "run_sql",
          "input": {"sql": "SELECT count(*) FROM samples"}, "output": "1 rows"}],
         "n_frames": 0, "checks": {"passed": False}, "latency_s": 1.0},
        {"id": "ghost", "question": "?", "answer": "", "model": "stub-model",
         "steps": [], "n_frames": 0, "checks": {"passed": False}, "latency_s": 1.0},
    ]
    src = tmp_path / "results_stub.jsonl"
    src.write_text("\n".join(_json.dumps(r) for r in stored) + "\n")

    cases = [EvalCase(id="a", question="How many samples?",
                      reference_sql="SELECT count(*) FROM samples", tolerance=0)]
    summary = regrade(src, con=tiny_con, cases=cases)

    assert summary["n_cases"] == 1 and summary["n_passed"] == 1   # re-graded to a pass
    assert summary["n_skipped"] == 1                              # unknown id skipped
    out = tmp_path / "results_stub_v2.jsonl"
    assert out.is_file()
    regraded = _json.loads(out.read_text().splitlines()[0])
    assert regraded["checks"]["passed"] is True
    assert (tmp_path / "results_stub_v2.md").is_file()
    # The input is untouched.
    assert _json.loads(src.read_text().splitlines()[0])["checks"]["passed"] is False


def test_regrade_uses_the_stored_question_not_the_configs(tmp_path: Path, tiny_con: Any) -> None:
    """Echo-grounding must reflect what the model was actually asked at the time."""
    import json as _json

    from nuscenes_data_engine.data_engine.chat.evaluate import EvalCase, regrade

    stored = [{"id": "a", "question": "How many within 5 metres?",
               "answer": "2 samples within 5 metres.", "model": "m",
               "steps": [{"tool": "run_sql",
                          "input": {"sql": "SELECT count(*) FROM samples"},
                          "output": "1 rows"}],
               "n_frames": 0, "checks": {}, "latency_s": 1.0}]
    src = tmp_path / "r.jsonl"
    src.write_text(_json.dumps(stored[0]) + "\n")
    # The current config has since reworded the question (no "5" in it).
    cases = [EvalCase(id="a", question="How many samples are nearby?",
                      reference_sql="SELECT count(*) FROM samples", tolerance=0)]
    summary = regrade(src, con=tiny_con, cases=cases)
    assert summary["n_passed"] == 1   # the echoed 5 is grounded via the STORED question
```

- [ ] **Step 2:** Run them → FAIL (ImportError for `regrade`).

- [ ] **Step 3: Implement** (import: `from dataclasses import dataclass, replace`):

```python
def regrade(
    results_path: Path, *, con: Any, cases: list[EvalCase]
) -> dict[str, Any]:
    """Re-grade a stored run's answers with the CURRENT graders — no LLM calls.

    Instrument isolation: same answers, new grader, so any pass-rate delta is
    attributable to the grader alone. Writes ``<stem>_v2.jsonl`` / ``<stem>_v2.md``
    next to the input and never modifies it. Grounding's question-echo allowlist uses
    the STORED question (what the model actually saw), while the reference value and
    tolerance come from the current config by case id; a record whose id is missing
    from the config is skipped with a warning and counted in neither numerator nor
    denominator.
    """
    by_id = {case.id: case for case in cases}
    records: list[dict[str, Any]] = []
    skipped = 0
    with open(results_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            stored = json.loads(line)
            case = by_id.get(stored.get("id"))
            if case is None:
                logger.warning(
                    "regrade: case %r not in the current config — skipped", stored.get("id")
                )
                skipped += 1
                continue
            asked = replace(case, question=stored.get("question", case.question))
            checks = grade_case(
                con, asked,
                answer=stored.get("answer", ""),
                steps=stored.get("steps") or [],
                n_frames=int(stored.get("n_frames") or 0),
            )
            records.append({**stored, "checks": checks})

    label = f"{results_path.stem}_v2"
    out_jsonl = results_path.with_name(f"{label}.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")
    model = records[0]["model"] if records else "unknown"
    report_path = results_path.with_name(f"{label}.md")
    report_path.write_text(render_report(records, model=model, provider=label))

    passed = sum(1 for record in records if record["checks"].get("passed"))
    summary = {
        "provider": label, "model": model, "n_cases": len(records),
        "n_passed": passed, "n_skipped": skipped,
        "pass_rate": round(passed / len(records), 3) if records else 0.0,
        "results": str(out_jsonl), "report": str(report_path),
    }
    logger.info("Regrade: %d/%d passed (%d skipped) -> %s",
                passed, len(records), skipped, report_path)
    return summary
```

- [ ] **Step 4:** `uv run pytest tests/test_chat_eval.py -q` → all pass (57).

- [ ] **Step 5: Commit:** `git commit -am "grounding v2: offline replay re-grade (regrade())"`

---

### Task 5: CLI — `--regrade` and model-suffixed artifacts; full CI sweep

**Files:**
- Modify: `src/nuscenes_data_engine/cli.py:450-497` (the `chat_eval` command)

- [ ] **Step 1: Modify the command.** Add two behaviors, reading the current body first (it changed during review fixes — trust the file, not this plan, for surrounding lines):

```python
@app.command("chat-eval")
def chat_eval(
    config: Path = typer.Option(Path("configs/chat_eval.yaml"), "--config", "-c"),
    provider: str | None = typer.Option(None, "--provider", help="local | anthropic."),
    model: str | None = typer.Option(None, "--model", help="Override the chat model."),
    limit: int | None = typer.Option(None, "--limit", help="First N cases (smoke runs)."),
    regrade_path: Path | None = typer.Option(
        None, "--regrade",
        help="Replay a stored results_*.jsonl through the current graders (no LLM calls).",
    ),
    processed_dir: Path = typer.Option(Path("data/processed"), "--processed-dir"),
) -> None:
    """Score the chat agent's answers against reference SQL (deterministic, no judge)."""
    from nuscenes_data_engine.config import get_settings
    from nuscenes_data_engine.data_engine.chat.catalog import open_catalog
    from nuscenes_data_engine.data_engine.chat.evaluate import load_cases, regrade, run_eval

    settings = get_settings()
    con = open_catalog(
        processed_dir, labels_path=Path(settings.data_dir) / "autolabel" / "labels.parquet"
    )
    if regrade_path is not None:
        summary = regrade(regrade_path, con=con, cases=load_cases(config))
        logger.info("Regrade summary: %s", summary)
        return
    ...
```

The transport and SearchEngine construction move BELOW the `--regrade` branch (a replay must not require an API key or Ollama). Keep the existing transport/search-probe/run_eval code otherwise unchanged, except the label:

```python
    label = provider or settings.chat_provider
    if model is not None:
        label = f"{label}_{model.replace(':', '-').replace('/', '-')}"
    summary = run_eval(
        load_cases(config), con=con, transport=transport, search_engine=engine,
        out_dir=Path(settings.data_dir) / "chat" / "eval",
        provider=label, limit=limit,
    )
```

(`run_eval` uses `provider` only for filenames and the report header, so `results_local_qwen2.5-32b.jsonl` comes out of the existing plumbing; without `--model`, filenames are unchanged.)

- [ ] **Step 2: Verify:** `uv run nuscenes-data-engine chat-eval --help` shows `--regrade`; then a smoke replay against the fixture-free real artifact:

```bash
uv run nuscenes-data-engine chat-eval --regrade data/chat/eval/results_local.jsonl
ls data/chat/eval/ | grep _v2
```

Both `results_local_v2.jsonl` and `results_local_v2.md` must appear; the command must run with NO Ollama and NO `ANTHROPIC_API_KEY` needed. (This also produces the first half of Task 6's measurement — fine; Task 6 interprets it.)

- [ ] **Step 3: CI parity:** `uv run pytest -q` (report the exact count), `uv run ruff check .`, `uv run mypy` — all clean.

- [ ] **Step 4: Commit:** `git commit -am "grounding v2: --regrade CLI + model-suffixed artifacts"`

---

### Task 6: Replay both stored runs; score the pre-registered predictions (operational)

- [ ] **Step 1:** Replay both:

```bash
uv run nuscenes-data-engine chat-eval --regrade data/chat/eval/results_local.jsonl
uv run nuscenes-data-engine chat-eval --regrade data/chat/eval/results_anthropic.jsonl
```

- [ ] **Step 2:** Score against the spec §2 predictions — do this mechanically, per case:

```bash
uv run python -c "
import json
for p in ('local', 'anthropic'):
    old = {r['id']: r for r in map(json.loads, open(f'data/chat/eval/results_{p}.jsonl'))}
    new = {r['id']: r for r in map(json.loads, open(f'data/chat/eval/results_{p}_v2.jsonl'))}
    flips = [(i, old[i]['checks'].get('grounded'), new[i]['checks'].get('grounded'))
             for i in old if old[i]['checks'].get('grounded') != new[i]['checks'].get('grounded')]
    print(p, 'passed:', sum(1 for r in old.values() if r['checks'].get('passed')), '->',
          sum(1 for r in new.values() if r['checks'].get('passed')), '| grounded flips:', flips)
"
```

Predictions to check (from the spec — report each as HELD or MISSED with the actual value):
1. Claude `grounded` 11/20 → 18/20; the two still-failing are exactly `max_instance_keyframes` and `foggy_misty_glare_frames`.
2. Claude composite → 17/20 (not 18 — `avg_visibility_pedestrian_night` still fails `numeric` on replay).
3. Local composite barely moves; its two v1 `grounded` failures flip.
4. No case's `grounded` goes from True to False (v2 only widens).

- [ ] **Step 3:** Also run the record-5 sanity check against the historical log (the graded suite doesn't include it): confirm via a small script that the 98.5% answer still fails `is_grounded` under v2, and paste the result.

- [ ] **Step 4:** Report all numbers verbatim. A MISSED prediction is a finding to report, not a bug to fix — do NOT touch the graders in this task. If a prediction missed because of an implementation defect (not a design miss), STOP and report BLOCKED with the evidence instead.

---

### Task 7: Live `qwen2.5:32b` run (operational)

- [ ] **Step 1:** `ollama list` must show `qwen2.5:32b` (a background pull was started; if absent, `ollama pull qwen2.5:32b` first). Confirm Ollama is serving (`curl -s localhost:11434/api/tags`).
- [ ] **Step 2:** Run the suite — 32b on CPU/Metal is slow (expect 30-90 s/case, i.e. 15-30 min total), so run it in the background and poll, do not block a 10-minute Bash timeout on it:

```bash
uv run nuscenes-data-engine chat-eval --provider local --model qwen2.5:32b
```

Artifacts must land as `results_local_qwen2.5-32b.jsonl` / `report_local_qwen2.5-32b.md` (the Task 5 suffix at work — verify, and verify `results_local.jsonl` was NOT overwritten).
- [ ] **Step 3:** Report: composite, per-check rates, median latency, and the language-drift count (english failures) side by side with the 14b v2 replay. The comparison question is pre-registered in the spec: does 32b clear the bar or not — either answer is the finding.

---

### Task 8: Docs

**Files:** `docs/DATASET_CHAT.md`, `docs/PROJECT.md`

- [ ] **Step 1:** In `docs/DATASET_CHAT.md`'s "Answer-correctness evaluation" section: replace the v1 results table with the v2 table (columns: check × {14b v2 replay, 32b live, claude v2 replay}); update the `grounded` check's one-line description (one-step derivations, schema constants, frame count — and what still correctly fails: memory facts, hedged estimates, wrong arithmetic); replace the "pre-registered future fix" paragraph with the outcome — each §2 prediction marked held/missed with actuals; keep the v1 numbers in a brief "v1 (superseded)" note so the correction is visible rather than overwritten; add the `--regrade` runbook line and the model-sweep decision rule with the 32b verdict.
- [ ] **Step 2:** `docs/PROJECT.md`: update the chat-eval lines in §5/§9 with the v2 headline numbers and the 32b verdict.
- [ ] **Step 3:** Verify every number you write against the artifacts (`results_*_v2.jsonl`, `report_local_qwen2.5-32b.md`) — no number in the docs may lack an artifact. `uv run pytest -q` still green.
- [ ] **Step 4:** `git add docs && git commit -m "grounding v2: docs (replayed + 32b results, prediction outcomes)"`

---

### After the plan (not plan tasks)

Final whole-branch review (against both the spec and the chat-eval PR base), then superpowers:finishing-a-development-branch. The PR stacks on `chat-eval`; if that PR has merged by then, rebase onto main first.
