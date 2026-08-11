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


def test_is_grounded_accepts_a_question_constant_next_to_an_ascii_hyphen(
    tiny_con: Any,
) -> None:
    """A question's "1-4" (plain ASCII hyphen) must echo "4" as reliably as an answer's
    "1-4" written with a non-ASCII dash already does.

    ``extract_numbers`` deliberately suppresses hyphen-adjacent digits everywhere (so
    "Scene-0992" doesn't yield -992), but that suppression must not carry over to the
    allowlist: a question phrased with an ASCII hyphen ("the 1-4 scale") has to admit
    "4" as an echoable constant just like an answer citing the same range with a
    non-ASCII dash does, or the same constant is checked on one side and silently
    dropped on the other.
    """
    from nuscenes_data_engine.data_engine.chat.evaluate import is_grounded

    steps = [{"tool": "run_sql", "input": {"sql": "SELECT 2"}, "output": "1 rows"}]
    assert is_grounded(
        "the scale runs 1–4 and the mean is 2",  # noqa: RUF001 - en dash, as models render it
        tiny_con, steps, question="What is the mean on the 1-4 scale?",
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
    checks = grade_case(tiny_con, case, answer="There are 2 samples.", steps=steps, n_frames=0)
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
        n_frames=0,
    )
    assert checks["numeric"] is False and checks["grounded"] is False
    assert checks["passed"] is False


def test_grade_case_without_reference_skips_numeric(tiny_con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import EvalCase, grade_case

    case = EvalCase(id="f", question="Show frames.", expect_frames=True)
    checks = grade_case(
        tiny_con, case, answer="Here are some frames.",
        steps=[{"tool": "search_frames", "input": {"query": "fog"}, "output": "2 frames found"}],
        n_frames=1,
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
        n_frames=0,
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


def test_derived_matches_admits_the_four_taxonomy_rules() -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import _derived_matches

    # difference: "4,986 of 5,000, so 14 failed"
    assert _derived_matches(14.0, {5000.0, 4986.0}) is True
    # difference, the other direction: "4986 - 4969 = 17"
    assert _derived_matches(17.0, {4986.0, 4969.0}) is True
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


def test_row_count_excluded_from_derivation_pairs(tiny_con: Any) -> None:
    """A LIMIT-8 row count must not pair with a real value to manufacture a percentage.

    Mirrors the real max_instance_keyframes collision measured in review: pre-fix,
    row_count=8 paired with a genuine 40 to fabricate "20" (=8/40*100). Row counts are
    a property of the SQL text (the LIMIT clause), not a measurement of the world, so
    they must never feed a derivation — only direct/rounded citation of the count itself.
    """
    from nuscenes_data_engine.data_engine.chat.evaluate import is_grounded

    steps = [{"tool": "run_sql",
              "input": {"sql": "SELECT n FROM (VALUES (34),(35),(36),(37),(38),(39),(40),(41)) "
                                "t(n) LIMIT 8"},
              "output": "8 rows"}]
    assert is_grounded("About 20% stood out.", tiny_con, steps) is False
    # The row count itself is still directly citable — only feeding derivations is cut.
    assert is_grounded("The query returned 8 rows.", tiny_con, steps) is True


def test_percentage_guard_requires_a_plausible_total() -> None:
    """A percentage is a share of a plausible total, not any quotient of two numbers."""
    from nuscenes_data_engine.data_engine.chat.evaluate import _derived_matches

    # 1/2*100 = 50: without the guard this "explains" almost any round answer.
    assert _derived_matches(50.0, {1.0, 2.0}) is False
    # 2/1*100 = 200: same problem in the other direction.
    assert _derived_matches(200.0, {1.0, 2.0}) is False
    # The guard doesn't touch a real percentage over a plausible (>=10) total.
    assert _derived_matches(99.66, {4969.0, 4986.0}) is True


def test_column_sum_skipped_when_the_result_is_truncated(tiny_con: Any) -> None:
    """A 50-row slice of a bigger result must not manufacture a partial-total citation."""
    from nuscenes_data_engine.data_engine.chat.evaluate import observed_numbers

    steps = [{"tool": "run_sql", "input": {"sql": "SELECT i FROM range(60) t(i)"},
              "output": "50 rows (truncated)"}]
    observed = observed_numbers(tiny_con, steps)
    # Individual cells (0..49, the retained slice) are still observed...
    assert 0.0 in observed and 49.0 in observed
    # ...but their sum (a semi-arbitrary partial total: 0+1+...+49 = 1225) must not be.
    assert 1225.0 not in observed


def test_integer_citations_match_at_one_decimal_not_zero() -> None:
    from nuscenes_data_engine.data_engine.chat.evaluate import _matches_at_cited_precision

    assert _matches_at_cited_precision(14.0, 14.04) is True    # within the 0.1 window
    assert _matches_at_cited_precision(14.0, 13.6) is False    # 0-decimals would admit this


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
    assert evaluate.is_grounded(
        "I attached 6 example frames.", tiny_con, steps, n_frames=6
    ) is True
    assert evaluate.is_grounded(
        "I attached 6 example frames.", tiny_con, steps, n_frames=0
    ) is False


def test_schema_constants_never_raises_on_a_bare_connection(tiny_con: Any) -> None:
    """The tiny fixture lacks most catalog views; extraction degrades to a set, not a crash."""
    from nuscenes_data_engine.data_engine.chat.evaluate import _schema_constants

    assert isinstance(_schema_constants(tiny_con), set)


def test_max_instance_keyframes_stays_ungrounded() -> None:
    """Hard negative control: must never pass, whatever the allowlist grows.

    Local, data-dependent; skips in CI (data/ is gitignored). Its residual citation
    (20, from "~20 s at 2 Hz") is nuScenes domain knowledge recalled from model memory
    — absent from both the observed values and the schema-prompt constants.
    """
    import json
    from pathlib import Path

    path = Path("data/chat/eval/results_anthropic.jsonl")
    if not path.is_file():
        pytest.skip("no stored anthropic run in this checkout")
    if not Path("data/processed").is_dir():
        pytest.skip("no data/processed in this checkout")
    pytest.importorskip("duckdb")
    from nuscenes_data_engine.data_engine.chat.catalog import open_catalog
    from nuscenes_data_engine.data_engine.chat.evaluate import is_grounded

    con = open_catalog(Path("data/processed"), labels_path=Path("data/autolabel/labels.parquet"))
    records = {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}
    rec = records["max_instance_keyframes"]
    assert is_grounded(rec["answer"], con, rec["steps"], rec["question"]) is False


def test_foggy_flips_via_schema_constants_documented_limitation() -> None:
    """Local, data-dependent; skips in CI (data/ is gitignored).

    A units-blind allowlist admits '50 Hz' (canbus schema text) against a hallucinated
    "~50+" frame count, and '5,000 CAM_FRONT frames' (labels schema text) against the
    answer's own "5,000 CAM_FRONT frames" restatement — pre-registered as a MISSED
    prediction, see spec §2 amendment (2026-08-11): admitting schema-prompt constants
    was projected to flip this record from ungrounded to grounded not because the
    figures are genuinely supported, but because the check cannot distinguish a
    hallucinated frame count from a schema constant that happens to share its value.
    Documented as a limitation of v2's allowlist, not adjusted away.
    """
    import json
    from pathlib import Path

    path = Path("data/chat/eval/results_anthropic.jsonl")
    if not path.is_file():
        pytest.skip("no stored anthropic run in this checkout")
    if not Path("data/processed").is_dir():
        pytest.skip("no data/processed in this checkout")
    pytest.importorskip("duckdb")
    from nuscenes_data_engine.data_engine.chat.catalog import open_catalog
    from nuscenes_data_engine.data_engine.chat.evaluate import is_grounded

    con = open_catalog(Path("data/processed"), labels_path=Path("data/autolabel/labels.parquet"))
    records = {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}
    rec = records["foggy_misty_glare_frames"]
    assert is_grounded(rec["answer"], con, rec["steps"], rec["question"]) is True


def test_max_instance_keyframes_collision_ceiling_is_bounded() -> None:
    """Local, data-dependent ceiling on the pre-registered collision probe (integers
    1..200); skips in CI (data/ is gitignored), so it is NOT CI-enforced — see
    ``test_max_instance_keyframes_collision_ceiling_is_ci_enforced`` below for the
    committed-literal variant that actually runs in CI. Pre-fix this record admitted
    68/200 (34%); post-fix (row-count exclusion, percentage-share guard, truncation
    skip) it measures 48/200 (24%) — the residual is genuine sum/difference collisions
    among the record's own close-together keyframe counts (34..41), the documented
    one-step-derivation trade-off, not a bug. Ceiling is set with modest headroom
    (+6pp) over that measured rate so the guards are enforced without the test being a
    tautology.
    """
    import json
    from pathlib import Path

    path = Path("data/chat/eval/results_anthropic.jsonl")
    if not path.is_file():
        pytest.skip("no stored anthropic run in this checkout")
    if not Path("data/processed").is_dir():
        pytest.skip("no data/processed in this checkout")
    pytest.importorskip("duckdb")
    from nuscenes_data_engine.data_engine.chat.catalog import open_catalog
    from nuscenes_data_engine.data_engine.chat.evaluate import (
        _allowlist_numbers,
        _derived_candidates,
        _matches_at_cited_precision,
        _observed_split,
    )

    con = open_catalog(Path("data/processed"), labels_path=Path("data/autolabel/labels.parquet"))
    records = {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}
    rec = records["max_instance_keyframes"]
    measurements, row_counts = _observed_split(con, rec["steps"])
    allowed = measurements | row_counts | _allowlist_numbers(rec["question"])
    derived = _derived_candidates(measurements)
    admitted = sum(
        1
        for n in range(1, 201)
        if any(_matches_at_cited_precision(float(n), seen) for seen in allowed | derived)
    )
    rate = admitted / 200
    assert rate < 0.30, f"collision rate {rate:.1%} exceeds the enforced ceiling"


def test_max_instance_keyframes_collision_ceiling_is_ci_enforced(
    tiny_con: Any, monkeypatch: Any
) -> None:
    """CI-enforced version of the ceiling above: a committed literal of the real
    ``max_instance_keyframes`` observed sets (extracted once from
    ``data/chat/eval/results_anthropic.jsonl`` via the real catalog — see the sibling
    test for the live extraction), exercised through the PUBLIC ``is_grounded`` with
    ``_observed_split`` monkeypatched so it needs no ``data/`` checkout and runs in CI.

    Because this still goes through the real wiring (``_allowlist_numbers``,
    ``_derived_candidates``, ``is_grounded`` itself), a wiring regression — e.g. row
    counts leaking back into derivations — shows up as a shift in the measured
    admitted fraction, not just as a change to a mocked return value.
    """
    from nuscenes_data_engine.data_engine.chat import evaluate

    measurements = {
        34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 300.0, 410.0, 631.0, 636.0,
        658.0, 680.0, 720.0, 1344.0, 1894.0, 4051.0, 10614.0,
    }
    row_counts = {8.0, 10.0}
    monkeypatch.setattr(
        evaluate, "_observed_split", lambda con, steps: (measurements, row_counts)
    )
    dummy_steps = [{"tool": "run_sql", "input": {"sql": "SELECT 1"}, "output": "1 rows"}]
    admitted = sum(
        1
        for n in range(1, 201)
        if evaluate.is_grounded(f"The value is {n}.", tiny_con, dummy_steps)
    )
    rate = admitted / 200
    assert rate < 0.30, f"collision rate {rate:.1%} exceeds the enforced ceiling"


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
