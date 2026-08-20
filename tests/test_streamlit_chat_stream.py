"""Pure-function tests for the Streamlit chat SSE client parsing — no Streamlit
runtime, no network. Mirrors tests/test_serving.py's ``_parse_sse`` server-side
counterpart, but exercises the client's own ``iter_stream_events`` so a drift
between the two implementations would show up as a client-side test failure, not
just at demo time.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from streamlit_app import iter_stream_events  # noqa: E402


def test_iter_stream_events_pairs_event_and_data_lines() -> None:
    """requests.iter_lines shape: decoded strings, blank lines between events."""
    lines = [
        "event: turn",
        "data: ",
        "",
        "event: token",
        'data: "Hel"',
        "",
        "event: token",
        'data: "lo"',
        "",
        "event: step",
        'data: {"tool": "run_sql", "input": {}, "output": "2 rows"}',
        "",
        "event: final",
        'data: {"answer": "Hello", "charts": [], "frames": [], "steps": [], "model": "m"}',
        "",
    ]
    events = list(iter_stream_events(lines))
    assert events == [
        ("turn", ""),
        ("token", '"Hel"'),
        ("token", '"lo"'),
        ("step", '{"tool": "run_sql", "input": {}, "output": "2 rows"}'),
        ("final", '{"answer": "Hello", "charts": [], "frames": [], "steps": [], "model": "m"}'),
    ]


def test_iter_stream_events_token_payload_stays_a_verbatim_json_string() -> None:
    """The function itself does no decoding: a token event's data is left exactly as
    received (the JSON-encoded string serving/app.py's chat_stream docstring
    describes) — the caller is responsible for a separate json.loads. This is what
    makes embedded newlines survive: an un-escaped newline in a bare token delta
    would otherwise split into a stray, un-prefixed second line."""
    lines = ["event: token", 'data: "line1\\nline2"', ""]
    events = list(iter_stream_events(lines))
    assert events == [("token", '"line1\\nline2"')]
    # The payload round-trips to the real delta only once the caller decodes it.
    assert json.loads(events[0][1]) == "line1\nline2"


def test_iter_stream_events_skips_blank_and_comment_lines() -> None:
    lines = ["", ": keepalive", "event: turn", "data: ", "", ": another comment", ""]
    events = list(iter_stream_events(lines))
    assert events == [("turn", "")]


def test_iter_stream_events_error_payload_is_plain_text_not_json() -> None:
    """error payloads are NOT JSON-encoded on the server side (see chat_stream's
    docstring) — the function still yields it verbatim; the client passes it
    straight to st.error without a json.loads."""
    lines = ["event: error", "data: internal error — see server logs", ""]
    events = list(iter_stream_events(lines))
    assert events == [("error", "internal error — see server logs")]


def test_iter_stream_events_ignores_a_data_line_with_no_preceding_event() -> None:
    """A stray data: line before any event: line has no (event, data) pair to join —
    it is dropped rather than yielded with a placeholder kind."""
    lines = ["data: orphan", "event: turn", "data: "]
    events = list(iter_stream_events(lines))
    assert events == [("turn", "")]


def test_iter_stream_events_is_a_generator() -> None:
    """Lazy like requests.iter_lines() — nothing is consumed until iterated, so
    wiring it directly into a live streaming response doesn't buffer the whole
    answer before the UI can render the first token."""
    assert inspect.isgeneratorfunction(iter_stream_events)
