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
    assert observed_numbers(tiny_con, steps) == set()


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


def test_is_grounded_accepts_a_correctly_rounded_citation(tiny_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import is_grounded

    steps = [{"tool": "run_sql", "input": {"sql": "SELECT 16.25118931104633"},
              "output": "1 rows"}]
    assert is_grounded("The average speed is about 16.3 km/h.", tiny_con, steps) is True
    assert is_grounded("The average speed is about 16.25 km/h.", tiny_con, steps) is True
    # A different value at the same precision is still ungrounded.
    assert is_grounded("The average speed is about 19.4 km/h.", tiny_con, steps) is False


def test_is_grounded_accepts_a_constant_echoed_from_the_question(tiny_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import is_grounded

    steps = [{"tool": "run_sql", "input": {"sql": "SELECT count(*) FROM samples"},
              "output": "1 rows"}]
    assert is_grounded(
        "2 pedestrians come within 5 metres of the ego vehicle.",
        tiny_con, steps, question="How many pedestrians come within 5 metres at night?",
    ) is True


def test_reference_value_rejects_multiple_rows(tiny_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import EvalCase, reference_value

    case = EvalCase(
        id="multi-row", question="q", reference_sql="SELECT is_night FROM samples"
    )
    with pytest.raises(ValueError, match="exactly one cell"):
        reference_value(tiny_con, case)


def test_reference_value_rejects_multiple_columns(tiny_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import EvalCase, reference_value

    case = EvalCase(
        id="multi-col", question="q",
        reference_sql="SELECT count(*), count(DISTINCT location) FROM samples",
    )
    with pytest.raises(ValueError, match="exactly one cell"):
        reference_value(tiny_con, case)


def test_validate_cases_passes_for_a_good_case_list(tiny_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import EvalCase, validate_cases

    cases = [
        EvalCase(id="a", question="q", reference_sql="SELECT count(*) FROM samples"),
        EvalCase(id="b", question="q", expect_frames=True),
    ]
    validate_cases(tiny_con, cases)  # must not raise


def test_validate_cases_raises_naming_the_case_id_on_broken_sql(tiny_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import EvalCase, validate_cases

    cases = [EvalCase(id="broken", question="q", reference_sql="SELECT * FROM nope")]
    with pytest.raises(ValueError, match="broken"):
        validate_cases(tiny_con, cases)


def test_validate_cases_raises_when_a_case_has_neither_reference_nor_frames(
    tiny_con: Any,
) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import EvalCase, validate_cases

    cases = [EvalCase(id="vacuous", question="q")]
    with pytest.raises(ValueError, match="neither reference_sql nor expect_frames"):
        validate_cases(tiny_con, cases)


def test_load_cases_rejects_a_case_missing_id(tmp_path: Path) -> None:
    import yaml

    from nuscenes_data_engine.data_engine.chat.evaluate import load_cases

    path = tmp_path / "cases.yaml"
    path.write_text(yaml.safe_dump({"cases": [{"question": "q"}]}))
    with pytest.raises(ValueError, match="id"):
        load_cases(path)


def test_load_cases_rejects_a_case_missing_question(tmp_path: Path) -> None:
    import yaml

    from nuscenes_data_engine.data_engine.chat.evaluate import load_cases

    path = tmp_path / "cases.yaml"
    path.write_text(yaml.safe_dump({"cases": [{"id": "a"}]}))
    with pytest.raises(ValueError, match="question"):
        load_cases(path)


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


def test_grade_case_passes_the_question_to_grounding(tiny_con: Any) -> None:
    """A constant echoed from the question must not count as ungrounded."""
    from nuscenes_data_engine.data_engine.chat.evaluate import EvalCase, grade_case

    case = EvalCase(
        id="a", question="How many samples are within 5 metres?",
        reference_sql="SELECT count(*) FROM samples", tolerance=0,
    )
    checks = grade_case(
        tiny_con, case, answer="2 samples are within 5 metres.",
        steps=[{"tool": "run_sql", "input": {"sql": "SELECT count(*) FROM samples"},
                "output": "1 rows"}],
        frames=[],
    )
    assert checks["grounded"] is True


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
    assert "1/2" in markdown or "50" in markdown
    assert "numeric" in markdown and "| a " in markdown and "| b " in markdown


def test_run_eval_end_to_end_with_a_stub_transport(tmp_path: Path, tiny_con: Any) -> None:
    """The whole loop: answer -> grade -> artifacts, with a scripted transport."""
    from nuscenes_data_engine.data_engine.chat.evaluate import EvalCase, run_eval

    class _StubTransport:
        model = "stub-model"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
            self.calls += 1
            if self.calls == 1:
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

    # Both cases must satisfy validate_cases (neither ref_sql nor expect_frames is
    # rejected up front), so this exercises a mid-suite transport failure rather than
    # a harness/config error — expect_frames=True is enough to pass validation without
    # pretending either case has a numeric reference.
    cases = [
        EvalCase(id="a", question="q1", expect_frames=True),
        EvalCase(id="b", question="q2", expect_frames=True),
    ]
    summary = run_eval(
        cases, con=tiny_con, transport=_BoomTransport(), search_engine=None,
        out_dir=tmp_path / "eval", provider="stub",
    )
    assert summary["n_cases"] == 2 and summary["n_passed"] == 0
