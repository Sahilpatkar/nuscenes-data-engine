"""Streaming-transport tests: pure SSE assembly, no network."""

from __future__ import annotations

from typing import Any, ClassVar

from nuscenes_data_engine.data_engine.chat.transports import assemble_openai_stream


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
