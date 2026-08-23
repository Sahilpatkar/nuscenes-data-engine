"""Tests for ``demo chat-record`` (Phase 8, Task 1) — fakes only, no network.

The recorder answers the graded eval cases plus a hand-written showcase set through
the real agent, so every test here drives ``agent.answer`` for real against a tiny
DuckDB catalog and a scripted transport (``tests/test_chat.py``'s fakes): the only
thing faked is the model turn itself.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("duckdb")

from nuscenes_data_engine.data_engine.chat.agent import ChatResult
from nuscenes_data_engine.data_engine.chat.evaluate import EvalCase
from nuscenes_data_engine.demo.chat_record import (
    EXPECTATIONS,
    FRAME_COLUMNS,
    RECORD_KEYS,
    ShowcaseQuestion,
    build_record,
    check_expectation,
    load_showcase,
    project_frames,
    record_session,
    write_staging,
)
from tests.test_chat import FakeSearchEngine, ScriptedTransport

EXPECTED_RECORD_KEYS = {
    "id", "kind", "question", "answer", "model", "provider", "steps", "frames",
    "charts", "checks", "latency_s", "error",
}


# ---------------------------------------------------------------------------
# fixtures + local fakes
# ---------------------------------------------------------------------------


@pytest.fixture()
def con() -> Any:
    """A real DuckDB catalog over a two-row samples table (as tests/test_chat_eval)."""
    import duckdb

    connection = duckdb.connect()
    connection.execute(
        "CREATE VIEW samples AS SELECT * FROM (VALUES "
        "('t1','boston-seaport',TRUE), ('t2','singapore-onenorth',FALSE)) "
        "AS s(sample_data_token, location, is_night)"
    )
    return connection


CASE_TOTAL = EvalCase(
    id="samples_total",
    question="How many samples are there?",
    reference_sql="SELECT count(*) FROM samples",
    tolerance=0,
)
CASE_BOSTON = EvalCase(
    id="samples_boston",
    question="How many samples are in boston-seaport?",
    reference_sql="SELECT count(*) FROM samples WHERE location = 'boston-seaport'",
    tolerance=0,
)


class QuestionTransport:
    """Answers each question with a canned final reply keyed by the question text.

    A reply that is an ``Exception`` is raised instead — the transport-failure path
    (``run_eval``'s ``except`` branch) without an order-sensitive script.
    """

    model = "fake-model"

    def __init__(self, replies: dict[str, str | Exception]) -> None:
        self.replies = replies
        self.asked: list[str] = []

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        question = [message for message in messages if message["role"] == "user"][-1]["content"]
        self.asked.append(str(question))
        reply = self.replies[str(question)]
        if isinstance(reply, Exception):
            raise reply
        return {"role": "assistant", "content": reply}


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _frame(token: str, **overrides: Any) -> dict[str, Any]:
    """A search-result row, thumbnail bytes and all (data_engine/search.py's columns)."""
    frame: dict[str, Any] = {
        "sample_data_token": token,
        "scene_name": "scene-0916",
        "scene_description": "night rain, pedestrians",
        "channel": "CAM_FRONT",
        "filename": "samples/CAM_FRONT/n015.jpg",
        "timestamp": 1533151603512404,
        "location": "singapore-onenorth",
        "is_night": True,
        "is_rain": True,
        "score": 0.87,
        "thumbnail": b"\xff\xd8jpegbytes",
    }
    frame.update(overrides)
    return frame


def _result(
    *,
    answer: str = "Recorded answer.",
    steps: list[dict[str, Any]] | None = None,
    frames: list[dict[str, Any]] | None = None,
    charts: list[dict[str, Any]] | None = None,
) -> ChatResult:
    return ChatResult(
        answer=answer,
        model="claude-test",
        steps=steps or [],
        frames=frames or [],
        charts=charts or [],
    )


CHART = {
    "kind": "bar",
    "title": "Night keyframes per location",
    "columns": ["location", "n"],
    "rows": [["boston-seaport", 12], ["singapore-onenorth", 7]],
}


# ---------------------------------------------------------------------------
# build_record / project_frames
# ---------------------------------------------------------------------------


def test_replay_record_projects_frames_and_keeps_charts_and_steps() -> None:
    steps = [
        {"tool": "run_sql", "input": {"sql": "SELECT count(*) FROM samples"}, "output": "1 rows"},
        {"tool": "make_chart", "input": {"kind": "bar"}, "output": "charted: Night keyframes"},
    ]
    result = _result(
        answer="Two frames attached.",
        steps=steps,
        frames=[_frame("tok0"), _frame("tok1", score=0.61)],
        charts=[CHART],
    )

    record = build_record(
        id="showcase_chart_night_by_location",
        kind="showcase",
        question="Chart the number of night keyframes per location.",
        result=result,
        provider="anthropic",
        checks=None,
        latency_s=12.3456,
    )

    assert set(record) == EXPECTED_RECORD_KEYS
    assert record["id"] == "showcase_chart_night_by_location"
    assert record["kind"] == "showcase"
    assert record["question"] == "Chart the number of night keyframes per location."
    assert record["answer"] == "Two frames attached."
    assert record["model"] == "claude-test"
    assert record["provider"] == "anthropic"
    assert record["steps"] == steps
    assert record["charts"] == [CHART]
    assert record["checks"] is None
    assert record["error"] is None
    assert record["latency_s"] == 12.35
    # Frames are projected to exactly the seven package columns — no thumbnail bytes
    # (0.4 MB budget) and no filename (the package's own thumbs are keyed by token).
    assert record["frames"] == [
        {
            "sample_data_token": "tok0",
            "scene_name": "scene-0916",
            "location": "singapore-onenorth",
            "is_night": True,
            "is_rain": True,
            "channel": "CAM_FRONT",
            "score": 0.87,
        },
        {
            "sample_data_token": "tok1",
            "scene_name": "scene-0916",
            "location": "singapore-onenorth",
            "is_night": True,
            "is_rain": True,
            "channel": "CAM_FRONT",
            "score": 0.61,
        },
    ]
    assert set(FRAME_COLUMNS) == set(record["frames"][0])
    # The published constant (what `demo build` validates against) must not drift
    # from the record this function actually writes.
    assert set(RECORD_KEYS) == EXPECTED_RECORD_KEYS
    json.dumps(record)  # the record must be JSON-serialisable as staged


def test_build_record_of_a_failed_question_has_no_answer_or_model() -> None:
    record = build_record(
        id="samples_total",
        kind="eval",
        question=CASE_TOTAL.question,
        result=None,
        provider="anthropic",
        checks={"passed": False},
        latency_s=0.5,
        error="overloaded_error",
    )
    assert set(record) == EXPECTED_RECORD_KEYS
    assert record["answer"] is None and record["model"] is None
    assert record["steps"] == [] and record["frames"] == [] and record["charts"] == []
    assert record["error"] == "overloaded_error"
    assert record["checks"] == {"passed": False}


def test_build_record_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="kind"):
        build_record(
            id="x", kind="graded", question="q", result=_result(), provider="local",
            checks=None, latency_s=0.1,
        )


def test_project_frames_coerces_numpy_scalars_to_json_safe_values() -> None:
    """LanceDB rows arrive through pandas, so bools/floats are numpy scalars — which
    ``json.dumps`` refuses. Coercing here is what keeps the paid run from dying at the
    write step after every answer has been paid for."""
    numpy = pytest.importorskip("numpy")

    projected = project_frames(
        [_frame("tok0", is_night=numpy.bool_(True), score=numpy.float32(0.5))]
    )[0]
    assert projected["is_night"] is True
    assert isinstance(projected["score"], float)
    json.dumps(projected)


# ---------------------------------------------------------------------------
# check_expectation
# ---------------------------------------------------------------------------


def test_showcase_expectations() -> None:
    chart_step = {"tool": "make_chart", "input": {}, "output": "charted: Night keyframes"}
    cypher_step = {"tool": "run_cypher", "input": {"cypher": "MATCH (n) RETURN 1"},
                   "output": "1 rows"}
    search_step = {"tool": "search_frames", "input": {"query": "fog"}, "output": "6 frames found"}
    failed_search = {"tool": "search_frames", "input": {"query": "fog"},
                     "output": "error: search failed: No module named 'torch'"}

    # met
    assert check_expectation(_result(steps=[chart_step], charts=[CHART]), "chart") is None
    assert check_expectation(_result(steps=[cypher_step]), "cypher") is None
    assert check_expectation(_result(frames=[_frame("tok0")]), "frames") is None
    assert check_expectation(_result(steps=[search_step]), "search") is None
    assert check_expectation(_result(), "none") is None

    # missed — each reason names what was missing
    assert "make_chart" in str(check_expectation(_result(), "chart"))
    # the call was made but produced no chart (a rejected make_chart argument)
    assert "chart" in str(check_expectation(_result(steps=[chart_step]), "chart"))
    assert "run_cypher" in str(check_expectation(_result(steps=[chart_step]), "cypher"))
    assert "frame" in str(check_expectation(_result(steps=[chart_step]), "frames"))
    assert "search_frames" in str(check_expectation(_result(), "search"))
    assert "search_frames" in str(check_expectation(_result(steps=[failed_search]), "search"))


def test_check_expectation_rejects_an_unknown_expectation() -> None:
    with pytest.raises(ValueError, match="expect"):
        check_expectation(_result(), "sql")
    assert set(EXPECTATIONS) == {"chart", "cypher", "frames", "search", "none"}


# ---------------------------------------------------------------------------
# load_showcase
# ---------------------------------------------------------------------------


def test_load_showcase_validates() -> None:
    config = {
        "chat_replay": {
            "showcase": [
                {"id": "a", "question": "Chart it.", "expect": "chart"},
                {"id": "b", "question": "Just answer.", "expect": "none"},
            ]
        }
    }
    assert load_showcase(config) == [
        ShowcaseQuestion(id="a", question="Chart it.", expect="chart"),
        ShowcaseQuestion(id="b", question="Just answer.", expect="none"),
    ]

    # An absent section / empty list is allowed (a recording may be graded-only).
    assert load_showcase({}) == []
    assert load_showcase({"chat_replay": {}}) == []
    assert load_showcase({"chat_replay": {"showcase": []}}) == []

    with pytest.raises(ValueError, match="duplicate"):
        load_showcase({"chat_replay": {"showcase": [
            {"id": "a", "question": "q1", "expect": "none"},
            {"id": "a", "question": "q2", "expect": "none"},
        ]}})
    with pytest.raises(ValueError, match="expect"):
        load_showcase({"chat_replay": {"showcase": [
            {"id": "a", "question": "q1", "expect": "sql"},
        ]}})
    with pytest.raises(ValueError, match="question"):
        load_showcase({"chat_replay": {"showcase": [{"id": "a", "expect": "none"}]}})
    with pytest.raises(ValueError, match="id"):
        load_showcase({"chat_replay": {"showcase": [{"question": "q", "expect": "none"}]}})


def test_the_shipped_demo_config_showcase_is_loadable(configs_dir: Path) -> None:
    """The config edit is part of Task 1 — a typo in an `expect` must fail here, not
    after the paid run has answered five questions."""
    from nuscenes_data_engine.config import load_yaml

    config = load_yaml(configs_dir / "demo.yaml")
    showcase = load_showcase(config)
    assert len(showcase) == 5
    assert {question.expect for question in showcase} <= set(EXPECTATIONS)
    assert Path(config["chat_replay"]["cases"]).is_file()
    assert int(config["chat_replay"]["max_turns"]) >= 1


# ---------------------------------------------------------------------------
# record_session
# ---------------------------------------------------------------------------


def test_eval_cases_are_graded_with_the_harness(con: Any) -> None:
    transport = ScriptedTransport(
        [
            # case 1: query, then answer with the retrieved number -> passes
            {"role": "assistant", "content": None,
             "tool_calls": [_tool_call("c1", "run_sql", {"sql": "SELECT count(*) FROM samples"})]},
            {"role": "assistant", "content": "There are 2 samples."},
            # case 2: answers from memory, and wrongly -> fails
            {"role": "assistant", "content": "About 9 samples are in boston-seaport."},
        ]
    )

    records, summary = record_session(
        con=con, transport=transport, search_engine=None, graph_driver=None,
        graph_database="neo4j", cases=[CASE_TOTAL, CASE_BOSTON], showcase=[],
        provider="anthropic", max_turns=8,
    )

    assert [record["id"] for record in records] == ["samples_total", "samples_boston"]
    assert [record["kind"] for record in records] == ["eval", "eval"]
    # The STORED question is the one the model saw.
    assert [record["question"] for record in records] == [CASE_TOTAL.question, CASE_BOSTON.question]
    passed, failed = records
    assert passed["checks"]["passed"] is True
    assert passed["checks"]["numeric"] is True and passed["checks"]["expected"] == 2.0
    assert failed["checks"]["passed"] is False
    assert failed["checks"]["numeric"] is False and failed["checks"]["tool_use"] is False
    assert summary["n_eval"] == 2 and summary["n_passed"] == 1 and summary["n_showcase"] == 0


def test_error_is_recorded_and_run_continues(con: Any) -> None:
    showcase = [
        ShowcaseQuestion(id="show_ok", question="Anything interesting?", expect="none"),
        ShowcaseQuestion(id="show_boom", question="And the graph?", expect="none"),
    ]
    transport = QuestionTransport(
        {
            "Anything interesting?": "Yes, plenty.",
            "And the graph?": RuntimeError("overloaded_error: retry later"),
            CASE_TOTAL.question: "There are 2 samples.",
        }
    )

    records, summary = record_session(
        con=con, transport=transport, search_engine=None, graph_driver=None,
        graph_database="neo4j", cases=[CASE_TOTAL], showcase=showcase,
        provider="anthropic", max_turns=8,
    )

    assert [record["id"] for record in records] == ["show_ok", "show_boom", "samples_total"]
    broken = records[1]
    assert "overloaded_error" in str(broken["error"])
    assert broken["answer"] is None and broken["checks"] is None  # showcase -> not graded
    assert records[0]["error"] is None and records[0]["checks"] is None
    # The question after the failure was still asked, answered and graded.
    assert records[2]["error"] is None
    assert records[2]["checks"]["numeric"] is True and records[2]["checks"]["expected"] == 2.0
    assert transport.asked == ["Anything interesting?", "And the graph?", CASE_TOTAL.question]
    assert summary["n_showcase"] == 2 and summary["n_eval"] == 1


def test_run_fails_when_nothing_succeeded(con: Any) -> None:
    boom = RuntimeError("connection reset")
    transport = QuestionTransport({"Anything interesting?": boom, CASE_TOTAL.question: boom})

    with pytest.raises(ValueError, match="no question was answered"):
        record_session(
            con=con, transport=transport, search_engine=None, graph_driver=None,
            graph_database="neo4j", cases=[CASE_TOTAL],
            showcase=[ShowcaseQuestion(id="show_ok", question="Anything interesting?",
                                       expect="none")],
            provider="anthropic", max_turns=8,
        )


def test_run_fails_when_a_showcase_expectation_is_missed(con: Any) -> None:
    showcase = [
        ShowcaseQuestion(id="show_chart", question="Chart it.", expect="chart"),
        ShowcaseQuestion(id="show_cypher", question="Use the graph.", expect="cypher"),
        ShowcaseQuestion(id="show_fine", question="Anything interesting?", expect="none"),
    ]
    transport = QuestionTransport(
        {
            "Chart it.": "Here is the breakdown, in words.",
            "Use the graph.": "I answered with SQL instead.",
            "Anything interesting?": "Yes, plenty.",
        }
    )

    with pytest.raises(ValueError) as excinfo:
        record_session(
            con=con, transport=transport, search_engine=None, graph_driver=None,
            graph_database="neo4j", cases=[], showcase=showcase,
            provider="anthropic", max_turns=8,
        )
    message = str(excinfo.value)
    assert "show_chart" in message and "make_chart" in message
    assert "show_cypher" in message and "run_cypher" in message
    assert "show_fine" not in message


def test_limit_caps_the_total_number_of_questions_showcase_first(con: Any) -> None:
    transport = QuestionTransport(
        {
            "Anything interesting?": "Yes, plenty.",
            CASE_TOTAL.question: "There are 2 samples.",
            CASE_BOSTON.question: "There is 1 sample.",
        }
    )
    records, summary = record_session(
        con=con, transport=transport, search_engine=None, graph_driver=None,
        graph_database="neo4j", cases=[CASE_TOTAL, CASE_BOSTON],
        showcase=[ShowcaseQuestion(id="show_ok", question="Anything interesting?",
                                   expect="none")],
        provider="anthropic", max_turns=8, limit=2,
    )
    assert [record["id"] for record in records] == ["show_ok", "samples_total"]
    assert transport.asked == ["Anything interesting?", CASE_TOTAL.question]
    assert summary["n_eval"] == 1 and summary["n_showcase"] == 1


def test_record_session_validates_the_cases_before_paying_for_answers(con: Any) -> None:
    transport = QuestionTransport({})
    broken = EvalCase(id="broken", question="q", reference_sql="SELECT * FROM nope")
    with pytest.raises(ValueError, match="broken"):
        record_session(
            con=con, transport=transport, search_engine=None, graph_driver=None,
            graph_database="neo4j", cases=[broken], showcase=[],
            provider="anthropic", max_turns=8,
        )
    assert transport.asked == []


def test_summary_fields_and_staging_files(con: Any, tmp_path: Path) -> None:
    engine = FakeSearchEngine()
    transport = ScriptedTransport(
        [
            # showcase: search + attach frames
            {"role": "assistant", "content": None,
             "tool_calls": [_tool_call("c1", "search_frames", {"query": "fog", "k": 2})]},
            {"role": "assistant", "content": "Two foggy frames are attached."},
            # eval case
            {"role": "assistant", "content": None,
             "tool_calls": [_tool_call("c2", "run_sql", {"sql": "SELECT count(*) FROM samples"})]},
            {"role": "assistant", "content": "There are 2 samples."},
        ]
    )
    showcase = [ShowcaseQuestion(id="show_search", question="Find foggy frames.",
                                 expect="search")]

    records, summary = record_session(
        con=con, transport=transport, search_engine=engine, graph_driver=object(),
        graph_database="neo4j", cases=[CASE_TOTAL], showcase=showcase,
        provider="anthropic", max_turns=5,
    )

    assert set(summary) == {
        "model", "provider", "recorded_at", "git_sha", "n_eval", "n_passed",
        "n_showcase", "search_available", "graph_available", "max_turns",
    }
    assert summary["model"] == "fake-model" and summary["provider"] == "anthropic"
    assert summary["n_eval"] == 1 and summary["n_passed"] == 1 and summary["n_showcase"] == 1
    assert summary["search_available"] is True and summary["graph_available"] is True
    assert summary["max_turns"] == 5
    assert isinstance(summary["git_sha"], str) and summary["git_sha"]
    assert datetime.fromisoformat(str(summary["recorded_at"])).tzinfo is not None

    out_dir = tmp_path / "chat_replays"
    write_staging(records, summary, out_dir)

    replays_text = (out_dir / "chat_replays.json").read_text()
    summary_text = (out_dir / "chat_replay_summary.json").read_text()
    staged = json.loads(replays_text)
    assert isinstance(staged, list)
    # showcase first (config order), then eval (config order)
    assert [replay["id"] for replay in staged] == ["show_search", "samples_total"]
    assert staged[0]["frames"] and set(staged[0]["frames"][0]) == set(FRAME_COLUMNS)
    assert json.loads(summary_text) == summary
    # sorted keys, indent 2, newline-terminated -- byte-identical to write_json's shape
    assert replays_text == json.dumps(staged, indent=2, sort_keys=True) + "\n"
    assert summary_text == json.dumps(summary, indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("force_terminal", [None, True], ids=["plain", "colour"])
def test_chat_record_help_carries_the_cost_note(
    monkeypatch: pytest.MonkeyPatch, force_terminal: bool | None
) -> None:
    """The help text names the cost and the dry-run flag -- with colour on as well.

    GitHub Actions sets GITHUB_ACTIONS, which makes Typer 0.27 force a colour
    terminal for its Rich help (typer.rich_utils.FORCE_TERMINAL, read when the
    console is built); the option highlighter then emits ``--limit`` as two
    styled segments (``-`` and ``-limit``) with escape codes between them, so
    the plain substring never appears. The ``colour`` case reproduces CI's
    condition here without the env var.
    """
    import typer.rich_utils
    from typer.testing import CliRunner

    from nuscenes_data_engine.cli import app

    monkeypatch.setattr(typer.rich_utils, "FORCE_TERMINAL", force_terminal)
    result = CliRunner().invoke(app, ["demo", "chat-record", "--help"])
    assert result.exit_code == 0
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)  # drop SGR colour codes
    text = " ".join(plain.split())
    assert "PAID" in text
    assert "$4" in text and "--limit" in text


def test_chat_record_without_a_catalog_raises_a_directive_error(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from nuscenes_data_engine.cli import app

    config_path = tmp_path / "demo.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {"processed_dir": str(tmp_path / "nowhere")},
                "curation": {"staging_dir": str(tmp_path / "staging")},
                "chat_replay": {"cases": "configs/chat_eval.yaml", "max_turns": 8,
                                "showcase": []},
            }
        )
    )
    result = CliRunner().invoke(app, ["demo", "chat-record", "--config", str(config_path)])
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "processed" in str(result.exception)
    assert not (tmp_path / "staging").exists()
