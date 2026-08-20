"""Streaming-transport tests: pure SSE assembly, no network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from nuscenes_data_engine.data_engine.chat.transports import assemble_openai_stream


@pytest.fixture()
def con() -> Any:
    """A real DuckDB catalog over a two-row samples table (mirrors test_chat_eval.py's
    ``tiny_con``, named ``con`` here to match agent.answer's parameter name)."""
    pytest.importorskip("duckdb")
    import duckdb

    con = duckdb.connect()
    con.execute(
        "CREATE VIEW samples AS SELECT * FROM (VALUES "
        "('t1','boston-seaport',TRUE), ('t2','singapore-onenorth',FALSE)) "
        "AS s(sample_data_token, location, is_night)"
    )
    return con


def test_assemble_content_deltas_and_forwards_tokens() -> None:
    lines = [
        'data: {"choices":[{"delta":{"role":"assistant","content":"Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        "data: [DONE]",
    ]
    seen: list[str] = []
    message = assemble_openai_stream(lines, on_token=seen.append)
    assert message["content"] == "Hello"
    assert seen == ["Hel", "lo"]
    assert not message.get("tool_calls")


def test_assemble_tool_call_fragments_across_chunks() -> None:
    lines = [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"run_sql","arguments":"{\\"sq"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"l\\": \\"SELECT 1\\"}"}}]}}]}',
        "data: [DONE]",
    ]
    message = assemble_openai_stream(lines, on_token=lambda _t: None)
    calls = message["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["id"] == "c1"
    assert calls[0]["function"]["name"] == "run_sql"
    assert calls[0]["function"]["arguments"] == '{"sql": "SELECT 1"}'


def test_assemble_interleaved_text_and_two_tool_calls() -> None:
    lines = [
        'data: {"choices":[{"delta":{"content":"Let me check."}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"run_sql","arguments":"{}"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"b","function":{"name":"search_frames","arguments":"{\\"query\\""}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":": \\"fog\\"}"}}]}}]}',
        "data: [DONE]",
    ]
    message = assemble_openai_stream(lines, on_token=lambda _t: None)
    assert message["content"] == "Let me check."
    assert [c["function"]["name"] for c in message["tool_calls"]] == ["run_sql", "search_frames"]
    assert message["tool_calls"][1]["function"]["arguments"] == '{"query": "fog"}'


def test_assemble_ignores_blank_and_non_data_lines() -> None:
    lines = ["", ": keepalive", 'data: {"choices":[{"delta":{"content":"x"}}]}', "data: [DONE]"]
    assert assemble_openai_stream(lines, on_token=lambda _t: None)["content"] == "x"


def test_openai_complete_stream_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirrors tests/test_chat.py's test_openai_transport_roundtrip (MockTransport
    pattern) but for the streaming path: an SSE body with content deltas and a
    split tool-call fragment, through the real client.stream(...) code path."""
    httpx = pytest.importorskip("httpx")
    from nuscenes_data_engine.data_engine.chat.transports import OpenAICompatTransport

    sse_body = (
        b'data: {"choices":[{"delta":{"role":"assistant","content":"Hel"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1",'
        b'"function":{"name":"run_sql","arguments":"{\\"sql\\": \\"SELECT 1\\"}"}}]}}]}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(request: Any) -> Any:
        payload = json.loads(request.content)
        assert payload["model"] == "test-model"
        assert payload["stream"] is True
        return httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"})

    real_client = httpx.Client

    def patched(**kwargs: Any) -> Any:
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", patched)
    transport = OpenAICompatTransport("http://fake:1/v1", "test-model")
    seen: list[str] = []
    message = transport.complete_stream(
        [{"role": "user", "content": "q"}], [], on_token=seen.append
    )
    assert message["content"] == "Hello"
    assert seen == ["Hel", "lo"]
    assert message["tool_calls"][0]["function"]["name"] == "run_sql"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"sql": "SELECT 1"}'


def test_openai_complete_stream_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirrors test_openai_transport_unreachable: a connect error surfaces as the
    same TransportError complete() raises, not a raw httpx exception."""
    httpx = pytest.importorskip("httpx")
    from nuscenes_data_engine.data_engine.chat.transports import (
        OpenAICompatTransport,
        TransportError,
    )

    def handler(request: Any) -> Any:
        raise httpx.ConnectError("refused")

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)
    )
    transport = OpenAICompatTransport("http://fake:1/v1", "m")
    with pytest.raises(TransportError, match="ollama serve"):
        transport.complete_stream([{"role": "user", "content": "q"}], [], on_token=lambda _t: None)


def test_anthropic_stream_forwards_text_and_converts_final() -> None:
    """The SDK stream object is stubbed; conversion must match the non-stream path."""
    from nuscenes_data_engine.data_engine.chat.transports import AnthropicTransport

    class _Block:
        def __init__(self, type_: str, **kw: Any) -> None:
            self.type = type_
            for k, v in kw.items():
                setattr(self, k, v)

    class _FinalMessage:
        # ADAPT (ruff RUF012): class-level mutable default needs ClassVar.
        content: ClassVar[list[_Block]] = [
            _Block("thinking", thinking="hmm"),
            _Block("text", text="Two samples."),
            _Block("tool_use", id="c9", name="run_sql", input={"sql": "SELECT 1"}),
        ]

    class _Stream:
        text_stream = iter(["Two ", "samples."])

        def __enter__(self) -> _Stream:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def get_final_message(self) -> _FinalMessage:
            return _FinalMessage()

    transport = AnthropicTransport.__new__(AnthropicTransport)  # skip __init__ (no SDK client)
    # ADAPT: `model` is a read-only @property backed by `_model` (see transports.py);
    # `transport.model = ...` raises AttributeError ("has no setter"). Set the
    # backing attribute instead — `complete_stream` reads `self._model`, not the
    # property, so this is behaviorally equivalent for the transport under test.
    transport._model = "stub-claude"  # type: ignore[attr-defined]

    class _Messages:
        def stream(self, **kwargs: Any) -> _Stream:
            return _Stream()

    class _Client:
        messages = _Messages()

    transport._client = _Client()  # type: ignore[attr-defined]
    seen: list[str] = []
    reply = transport.complete_stream([], [], on_token=seen.append)
    assert seen == ["Two ", "samples."]
    assert reply["content"] == "Two samples."
    assert reply["tool_calls"][0]["function"]["name"] == "run_sql"
    # ADAPT attribute names (_client, kwargs) to the real class — behavior is the contract.


def test_fallback_transport_without_complete_stream_still_streams_one_token() -> None:
    from nuscenes_data_engine.data_engine.chat.transports import stream_with_fallback

    class _PlainFake:
        model = "fake"

        def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            return {"content": "whole answer", "tool_calls": []}

    seen: list[str] = []
    reply = stream_with_fallback(_PlainFake(), [], [], on_token=seen.append)
    assert reply["content"] == "whole answer"
    assert seen == ["whole answer"]


def test_answer_fires_callbacks_in_order(con: Any) -> None:
    """turn -> tokens -> step -> turn -> tokens, with the final answer intact."""
    from nuscenes_data_engine.data_engine.chat import agent

    class _StreamFake:
        model = "fake"

        def __init__(self) -> None:
            self.turn = 0

        def complete_stream(self, messages, tools, on_token):
            self.turn += 1
            if self.turn == 1:
                on_token("checking…")
                return {"content": "checking…", "tool_calls": [
                    {"id": "c1", "function": {"name": "run_sql",
                                              "arguments": '{"sql": "SELECT count(*) FROM samples"}'}}]}
            for piece in ("There are ", "2 samples."):
                on_token(piece)
            return {"content": "There are 2 samples.", "tool_calls": []}

        def complete(self, messages, tools):  # unused when streaming
            raise AssertionError("complete must not be called when on_token is set")

    events: list[tuple[str, str]] = []
    result = agent.answer(
        "How many?", transport=_StreamFake(), con=con, search_engine=None,
        on_turn=lambda: events.append(("turn", "")),
        on_token=lambda t: events.append(("token", t)),
        on_step=lambda s: events.append(("step", s["tool"])),
    )
    assert result.answer == "There are 2 samples."
    kinds = [k for k, _v in events]
    assert kinds == ["turn", "token", "step", "turn", "token", "token"]


def test_answer_without_callbacks_unchanged(con: Any) -> None:
    """No callbacks -> the plain complete path; ScriptedTransport-style fakes fine."""
    from nuscenes_data_engine.data_engine.chat import agent

    class _Plain:
        model = "fake"

        def complete(self, messages, tools):
            return {"content": "plain answer", "tool_calls": []}

    result = agent.answer("q", transport=_Plain(), con=con, search_engine=None)
    assert result.answer == "plain answer"


def _chart_args(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "kind": "bar", "title": "Night scenes per location",
        "columns": ["location", "scenes"],
        "rows": [["boston", 3], ["singapore", 5]],
    }
    args.update(overrides)
    return args


def test_make_chart_appends_to_result(con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat import agent

    result = agent.ChatResult(answer="", model="stub")
    output = agent._run_tool(
        "make_chart", _chart_args(), con=con, search_engine=None,
        result=result, graph_driver=None, graph_database="neo4j",
    )
    assert output == {
        "charted": True,
        "title": "Night scenes per location",
        "note": "Displayed to the user automatically — do not embed an image or link "
        "in your answer.",
    }
    assert len(result.charts) == 1
    assert result.charts[0]["kind"] == "bar"


def test_make_chart_guards_are_model_visible(con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat import agent

    result = agent.ChatResult(answer="", model="stub")
    bad_kind = agent._run_tool("make_chart", _chart_args(kind="pie"), con=con,
                               search_engine=None, result=result,
                               graph_driver=None, graph_database="neo4j")
    assert "error" in bad_kind and "pie" in bad_kind["error"]
    ragged = agent._run_tool("make_chart", _chart_args(rows=[["a", 1], ["b"]]), con=con,
                             search_engine=None, result=result,
                             graph_driver=None, graph_database="neo4j")
    assert "error" in ragged
    huge = agent._run_tool("make_chart",
                           _chart_args(rows=[["r", 1]] * 1001), con=con,
                           search_engine=None, result=result,
                           graph_driver=None, graph_database="neo4j")
    assert "error" in huge and "2000" in huge["error"]
    assert result.charts == []          # none of the bad calls appended


def test_chart_reaches_the_jsonl_log(con: Any, tmp_path: Path) -> None:
    """The logged record carries n_charts (spec: chart present in the log)."""
    import json as _json

    from nuscenes_data_engine.data_engine.chat import agent

    class _Charting:
        model = "fake"

        def __init__(self) -> None:
            self.turn = 0

        def complete(self, messages, tools):
            self.turn += 1
            if self.turn == 1:
                return {"content": "", "tool_calls": [
                    {"id": "c1", "function": {"name": "make_chart",
                                              "arguments": _json.dumps(_chart_args())}}]}
            return {"content": "Charted.", "tool_calls": []}

    log = tmp_path / "log.jsonl"
    result = agent.answer("chart it", transport=_Charting(), con=con,
                          search_engine=None, log_path=log)
    assert len(result.charts) == 1
    record = _json.loads(log.read_text().splitlines()[-1])
    assert record["n_charts"] == 1


# ---------------------------------------------------------------------------
# Quality fast-follow: callback containment (a raising callback must not lose
# the answer or the JSONL log record — mirrors _log's own OSError containment).
# ---------------------------------------------------------------------------


def test_assemble_openai_stream_survives_a_raising_on_token() -> None:
    """The forwarding call site inside the SSE assembler must not let a broken
    on_token abort assembly — the content is still accumulated."""
    lines = [
        'data: {"choices":[{"delta":{"role":"assistant","content":"Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        "data: [DONE]",
    ]

    def _boom(_token: str) -> None:
        raise RuntimeError("callback exploded")

    message = assemble_openai_stream(lines, on_token=_boom)
    assert message["content"] == "Hello"


def test_anthropic_complete_stream_survives_a_raising_on_token() -> None:
    """The text_stream forwarding loop must not let a broken on_token abort the
    turn — the final message conversion still completes."""
    from nuscenes_data_engine.data_engine.chat.transports import AnthropicTransport

    class _Block:
        def __init__(self, type_: str, **kw: Any) -> None:
            self.type = type_
            for k, v in kw.items():
                setattr(self, k, v)

    class _FinalMessage:
        content: ClassVar[list[_Block]] = [_Block("text", text="Two samples.")]

    class _Stream:
        text_stream = iter(["Two ", "samples."])

        def __enter__(self) -> _Stream:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def get_final_message(self) -> _FinalMessage:
            return _FinalMessage()

    transport = AnthropicTransport.__new__(AnthropicTransport)  # skip __init__
    transport._model = "stub-claude"  # type: ignore[attr-defined]

    class _Messages:
        def stream(self, **kwargs: Any) -> _Stream:
            return _Stream()

    class _Client:
        messages = _Messages()

    transport._client = _Client()  # type: ignore[attr-defined]

    def _boom(_token: str) -> None:
        raise RuntimeError("callback exploded")

    reply = transport.complete_stream([], [], on_token=_boom)
    assert reply["content"] == "Two samples."


def test_stream_with_fallback_survives_a_raising_on_token() -> None:
    """The one-shot forwarding site (no complete_stream on the transport) must not
    let a broken on_token lose the whole-answer reply."""
    from nuscenes_data_engine.data_engine.chat.transports import stream_with_fallback

    class _PlainFake:
        model = "fake"

        def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            return {"content": "whole answer", "tool_calls": []}

    def _boom(_token: str) -> None:
        raise RuntimeError("callback exploded")

    reply = stream_with_fallback(_PlainFake(), [], [], on_token=_boom)
    assert reply["content"] == "whole answer"


def test_answer_survives_a_raising_on_token_and_still_logs(con: Any, tmp_path: Path) -> None:
    from nuscenes_data_engine.data_engine.chat import agent

    class _Plain:
        model = "fake"

        def complete(self, messages, tools):
            return {"content": "plain answer", "tool_calls": []}

    def _boom(_token: str) -> None:
        raise RuntimeError("on_token blew up")

    log = tmp_path / "log.jsonl"
    result = agent.answer(
        "q", transport=_Plain(), con=con, search_engine=None,
        on_token=_boom, log_path=log,
    )
    assert result.answer == "plain answer"
    record = json.loads(log.read_text().splitlines()[-1])
    assert record["answer"] == "plain answer"


def test_answer_survives_a_raising_on_step_and_still_logs(con: Any, tmp_path: Path) -> None:
    from nuscenes_data_engine.data_engine.chat import agent

    class _Transport:
        model = "fake"

        def __init__(self) -> None:
            self.turn = 0

        def complete(self, messages, tools):
            self.turn += 1
            if self.turn == 1:
                return {"content": "", "tool_calls": [
                    {"id": "c1", "function": {"name": "run_sql",
                                              "arguments": '{"sql": "SELECT count(*) FROM samples"}'}}]}
            return {"content": "There are 2 samples.", "tool_calls": []}

    def _boom(_step: dict[str, Any]) -> None:
        raise RuntimeError("on_step blew up")

    log = tmp_path / "log.jsonl"
    result = agent.answer(
        "How many?", transport=_Transport(), con=con, search_engine=None,
        on_step=_boom, log_path=log,
    )
    assert result.answer == "There are 2 samples."
    record = json.loads(log.read_text().splitlines()[-1])
    assert record["answer"] == "There are 2 samples."


# ---------------------------------------------------------------------------
# Quality fast-follow: make_chart shape guards (Task 6's renderer would
# IndexError/choke on these shapes) + _summarize's "charted" branch.
# ---------------------------------------------------------------------------


def test_make_chart_shape_guards_reject_bad_data(con: Any) -> None:
    from nuscenes_data_engine.data_engine.chat import agent

    result = agent.ChatResult(answer="", model="stub")

    zero_columns = agent._run_tool(
        "make_chart", _chart_args(columns=[], rows=[]), con=con,
        search_engine=None, result=result, graph_driver=None, graph_database="neo4j",
    )
    assert "error" in zero_columns

    one_column = agent._run_tool(
        "make_chart", _chart_args(columns=["location"], rows=[["boston"]]), con=con,
        search_engine=None, result=result, graph_driver=None, graph_database="neo4j",
    )
    assert "error" in one_column

    empty_rows = agent._run_tool(
        "make_chart", _chart_args(rows=[]), con=con,
        search_engine=None, result=result, graph_driver=None, graph_database="neo4j",
    )
    assert "error" in empty_rows

    string_cell = agent._run_tool(
        "make_chart", _chart_args(rows=[["boston", "not-a-number"]]), con=con,
        search_engine=None, result=result, graph_driver=None, graph_database="neo4j",
    )
    assert "error" in string_cell and "not-a-number" in string_cell["error"]

    none_cell = agent._run_tool(
        "make_chart", _chart_args(rows=[["boston", None]]), con=con,
        search_engine=None, result=result, graph_driver=None, graph_database="neo4j",
    )
    assert "error" in none_cell and "None" in none_cell["error"]

    duplicate_columns = agent._run_tool(
        "make_chart",
        _chart_args(columns=["location", "location"], rows=[["boston", 1]]),
        con=con, search_engine=None, result=result, graph_driver=None, graph_database="neo4j",
    )
    assert "error" in duplicate_columns and "unique" in duplicate_columns["error"]

    non_scalar_x = agent._run_tool(
        "make_chart",
        _chart_args(rows=[[["nested"], 1], ["singapore", 5]]),
        con=con, search_engine=None, result=result, graph_driver=None, graph_database="neo4j",
    )
    assert "error" in non_scalar_x and "x-values" in non_scalar_x["error"]

    assert result.charts == []  # none of the bad calls appended

    valid = agent._run_tool(
        "make_chart", _chart_args(), con=con, search_engine=None,
        result=result, graph_driver=None, graph_database="neo4j",
    )
    assert valid == {
        "charted": True,
        "title": "Night scenes per location",
        "note": "Displayed to the user automatically — do not embed an image or link "
        "in your answer.",
    }
    assert len(result.charts) == 1


def test_summarize_reports_charted_output() -> None:
    from nuscenes_data_engine.data_engine.chat.agent import _summarize

    assert _summarize({"charted": True, "title": "Night scenes per location"}) == (
        "charted: Night scenes per location"
    )
