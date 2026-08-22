"""Tests for Phase 6c dataset chat (offline: fakes + tiny fixtures, no model calls)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from nuscenes_data_engine.data_engine.chat import agent, catalog
from nuscenes_data_engine.data_engine.chat.transports import (
    OpenAICompatTransport,
    TransportError,
    convert_messages,
    convert_tool,
    make_transport,
)

# ---------------------------------------------------------------------------
# catalog.py — views + the SQL guard
# ---------------------------------------------------------------------------


@pytest.fixture()
def con(tmp_path: Path) -> Any:
    pd.DataFrame(
        {
            "sample_data_token": ["t1", "t2", "t3"],
            "scene_name": ["scene-a", "scene-a", "scene-b"],
            "channel": ["CAM_FRONT"] * 3,
            "n_boxes": [5, 2, 9],
            "is_night": [False, False, True],
        }
    ).to_parquet(tmp_path / "samples.parquet")
    pd.DataFrame(
        {
            "sample_data_token": ["t1", "t1", "t3"],
            "category_group": ["car", "pedestrian", None],
            "bbox_area": [100.0, 50.0, 20.0],
        }
    ).to_parquet(tmp_path / "annotations.parquet")
    pd.DataFrame({"sample_data_token": ["t1"], "parse_status": ["ok"]}).to_parquet(
        tmp_path / "labels.parquet"
    )
    return catalog.open_catalog(tmp_path, labels_path=tmp_path / "labels.parquet")


def test_catalog_views_and_missing_files(con: Any, tmp_path: Path) -> None:
    assert catalog.catalog_tables(con) == ["annotations", "labels", "samples"]
    # availability.parquet absent -> view simply not registered.
    empty = catalog.open_catalog(tmp_path / "nowhere")
    assert catalog.catalog_tables(empty) == []


def test_run_sql_select_and_cte(con: Any) -> None:
    out = catalog.run_sql(con, "SELECT scene_name, count(*) AS n FROM samples GROUP BY 1 ORDER BY 1")
    assert out["columns"] == ["scene_name", "n"]
    assert out["rows"] == [["scene-a", 2], ["scene-b", 1]]
    assert out["row_count"] == 2 and not out["truncated"]

    cte = catalog.run_sql(con, "WITH c AS (SELECT n_boxes FROM samples) SELECT max(n_boxes) FROM c")
    assert cte["rows"] == [[9]]


def test_run_sql_join_across_views(con: Any) -> None:
    out = catalog.run_sql(
        con,
        "SELECT s.scene_name, count(a.category_group) AS boxes FROM samples s "
        "LEFT JOIN annotations a USING (sample_data_token) GROUP BY 1 ORDER BY 1",
    )
    assert out["rows"] == [["scene-a", 2], ["scene-b", 0]]


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO samples VALUES (1)",
        "DROP VIEW samples",
        "COPY samples TO 'out.csv'",
        "CREATE TABLE x AS SELECT 1",
        "SELECT 1; SELECT 2",
        "PRAGMA database_list",
        "SET memory_limit='1GB'",
        "ATTACH 'other.db'",
        "SELECT * FROM read_parquet('/etc/passwd')",
        "SELECT * FROM 'data/processed/samples.parquet'",
        "SELECT * FROM read_csv_auto('x.csv')",
        "INSTALL httpfs",
        "SELECT getenv('HOME')",
    ],
)
def test_run_sql_guard_rejects(con: Any, sql: str) -> None:
    out = catalog.run_sql(con, sql)
    assert set(out) == {"error"}, sql


def test_run_sql_allows_division_between_two_string_literals(con: Any) -> None:
    """A filtered-percentage query is legitimate SQL, not a file-path escape."""
    sql = (
        "SELECT 100.0 * count(*) FILTER (WHERE scene_name = 'scene-a') / count(*) "
        "FROM samples WHERE channel = 'CAM_FRONT'"
    )
    result = catalog.run_sql(con, sql)
    assert "error" not in result, result.get("error")


def test_run_sql_still_blocks_a_real_file_path_literal(con: Any) -> None:
    """The guard must still reject genuine file-reading attempts, in every quoting form
    DuckDB accepts in a FROM clause: single-quoted string, double-quoted identifier, and
    dollar-quoted string (untagged and tagged)."""
    assert "error" in catalog.run_sql(con, "SELECT * FROM '/etc/passwd'")
    assert "error" in catalog.run_sql(con, "SELECT * FROM 'data/processed/samples.parquet'")
    assert "error" in catalog.run_sql(con, "SELECT * FROM read_parquet('/any/path.parquet')")
    assert "error" in catalog.run_sql(con, 'SELECT * FROM "data/processed/samples.parquet"')
    assert "error" in catalog.run_sql(con, "SELECT * FROM $$/etc/passwd$$")
    assert "error" in catalog.run_sql(con, "SELECT * FROM $tag$data/processed/samples.parquet$tag$")


def test_run_sql_row_cap_and_errors(con: Any) -> None:
    out = catalog.run_sql(con, "SELECT * FROM range(100)", max_rows=10)
    assert out["row_count"] == 10 and out["truncated"]

    bad = catalog.run_sql(con, "SELECT nope FROM samples")
    assert "error" in bad and "nope" in bad["error"]

    unparsable = catalog.run_sql(con, "SELEKT 1")
    assert "error" in unparsable


def test_run_sql_cursor_still_sees_views_and_the_guard(con: Any) -> None:
    """Regression pin for the per-call-cursor fix (concurrency race): ``con.cursor()``
    is an independent connection to the same in-memory database, so it must still see
    the parent connection's registered views and still enforce the SQL guard."""
    out = catalog.run_sql(con, "SELECT count(*) FROM samples")
    assert out["rows"] == [[3]]
    assert "error" in catalog.run_sql(con, "SELECT * FROM read_parquet('/etc/passwd')")


def test_run_sql_concurrent_calls_do_not_cross_contaminate(con: Any) -> None:
    """Regression test for the per-call-cursor fix: DuckDB's ``execute()`` mutates
    the connection's own live-result state, so two threads sharing one ``con`` and
    calling ``execute``/``fetchmany`` concurrently could interleave and each get back
    the OTHER thread's rows (proven: 105 cross-contaminated rows in a two-thread
    probe against the pre-fix implementation). Each thread runs 50 queries whose
    result is a function of (thread id, iteration) so any contamination is
    immediately visible as a value mismatch."""
    import threading

    mismatches: list[str] = []

    def worker(thread_id: int) -> None:
        for i in range(50):
            n = thread_id * 1000 + i  # unique per (thread, iteration)
            out = catalog.run_sql(con, f"SELECT {n} AS n")
            if out.get("rows") != [[n]]:
                mismatches.append(f"thread {thread_id} call {i}: expected [[{n}]], got {out!r}")

    threads = [threading.Thread(target=worker, args=(thread_id,)) for thread_id in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert mismatches == []


def test_schema_prompt_matches_available_tables(con: Any) -> None:
    prompt = catalog.schema_prompt(catalog.catalog_tables(con))
    assert "samples —" in prompt and "labels —" in prompt
    assert "availability —" not in prompt  # not registered on this catalog
    assert "no ego-pose" in prompt


def test_catalog_exposes_canbus_view(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    duckdb = pytest.importorskip("duckdb")
    del duckdb
    from nuscenes_data_engine.data_engine.chat.catalog import catalog_tables, open_catalog

    pd.DataFrame(
        {"sample_token": ["s1"], "has_canbus": [True], "can_speed_kmh": [14.7],
         "is_hard_braking": [True]}
    ).to_parquet(tmp_path / "canbus.parquet")
    con = open_catalog(tmp_path)
    assert "canbus" in catalog_tables(con)
    assert con.execute("SELECT count(*) FROM canbus WHERE is_hard_braking").fetchone()[0] == 1


def test_schema_prompt_notes_canbus_when_present(tmp_path: Path, con: Any) -> None:
    # `con` was opened before canbus.parquet existed -> its prompt has no canbus text.
    assert "canbus" not in catalog.schema_prompt(catalog.catalog_tables(con))

    pd.DataFrame(
        {"sample_token": ["s1"], "has_canbus": [True], "can_speed_kmh": [14.7],
         "is_hard_braking": [True]}
    ).to_parquet(tmp_path / "canbus.parquet")
    con_with_canbus = catalog.open_catalog(tmp_path)
    prompt = catalog.schema_prompt(catalog.catalog_tables(con_with_canbus))
    assert "is_hard_braking" in prompt and "canbus" in prompt


# ---------------------------------------------------------------------------
# transports.py
# ---------------------------------------------------------------------------


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def test_openai_transport_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    httpx = pytest.importorskip("httpx")

    def handler(request: Any) -> Any:
        payload = json.loads(request.content)
        assert payload["model"] == "test-model"
        assert payload["tools"][0]["function"]["name"] == "run_sql"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]},
        )

    real_client = httpx.Client

    def patched(**kwargs: Any) -> Any:
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", patched)
    transport = OpenAICompatTransport("http://fake:1/v1", "test-model")
    reply = transport.complete(
        [{"role": "user", "content": "q"}], agent.TOOL_SPECS[:1]
    )
    assert reply == {"role": "assistant", "content": "hi"}


def test_openai_transport_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    httpx = pytest.importorskip("httpx")

    def handler(request: Any) -> Any:
        raise httpx.ConnectError("refused")

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)
    )
    transport = OpenAICompatTransport("http://fake:1/v1", "m")
    with pytest.raises(TransportError, match="ollama serve"):
        transport.complete([{"role": "user", "content": "q"}], [])


def test_convert_tool_and_messages() -> None:
    anthropic_tool = convert_tool(agent.TOOL_SPECS[0])
    assert anthropic_tool["name"] == "run_sql"
    assert anthropic_tool["input_schema"]["required"] == ["sql"]

    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "how many?"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                _tool_call("c1", "run_sql", {"sql": "SELECT 1"}),
                _tool_call("c2", "run_sql", {"sql": "SELECT 2"}),
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"rows": []}'},
        {"role": "tool", "tool_call_id": "c2", "content": '{"rows": []}'},
        {"role": "assistant", "content": "42"},
    ]
    system, converted = convert_messages(messages)
    assert system == "be terse"
    assert [message["role"] for message in converted] == ["user", "assistant", "user", "assistant"]
    blocks = converted[1]["content"]
    assert blocks[0] == {"type": "text", "text": "checking"}
    assert blocks[1]["type"] == "tool_use" and blocks[1]["input"] == {"sql": "SELECT 1"}
    # Both tool results merged into ONE user turn (Claude requirement).
    results = converted[2]["content"]
    assert [block["tool_use_id"] for block in results] == ["c1", "c2"]


def test_make_transport_selects_provider() -> None:
    class FakeSettings:
        chat_provider = "local"
        chat_base_url = "http://x:1/v1"
        chat_model = "m-local"
        chat_anthropic_model = "m-claude"
        anthropic_api_key = ""

    transport = make_transport(FakeSettings())
    assert isinstance(transport, OpenAICompatTransport) and transport.model == "m-local"

    with pytest.raises(ValueError, match="Unknown chat provider"):
        make_transport(FakeSettings(), provider="wat")

    pytest.importorskip("anthropic")
    with pytest.raises(TransportError, match="ANTHROPIC_API_KEY"):
        make_transport(FakeSettings(), provider="anthropic")


# ---------------------------------------------------------------------------
# agent.py — the tool loop with a scripted transport
# ---------------------------------------------------------------------------


class ScriptedTransport:
    """Replays a fixed list of assistant replies; records what it was sent."""

    model = "fake-model"

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self._replies = replies
        self.seen: list[list[dict[str, Any]]] = []

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self.seen.append([dict(message) for message in messages])
        return self._replies[len(self.seen) - 1]


class FakeSearchEngine:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    @staticmethod
    def _frame(token: str, score: float = 0.9) -> dict[str, Any]:
        return {
            "sample_data_token": token,
            "scene_name": "scene-x",
            "scene_description": "desc",
            "channel": "CAM_FRONT",
            "filename": "f.jpg",
            "timestamp": 0,
            "location": "boston-seaport",
            "is_night": False,
            "is_rain": False,
            "score": score,
            "thumbnail": b"jpegbytes",
        }

    def search_text(self, query: str, k: int) -> list[dict[str, Any]]:
        self.calls.append(("text", query, k))
        return [self._frame(f"tok{i}") for i in range(min(k, 2))]

    def frames_by_tokens(self, tokens: list[str]) -> list[dict[str, Any]]:
        self.calls.append(("tokens", tuple(tokens)))
        return [self._frame(token, score=1.0) for token in tokens]


def test_agent_sql_then_answer(con: Any, tmp_path: Path) -> None:
    transport = ScriptedTransport(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _tool_call("c1", "run_sql", {"sql": "SELECT count(*) AS n FROM samples"})
                ],
            },
            {"role": "assistant", "content": "There are 3 frames."},
        ]
    )
    log = tmp_path / "log.jsonl"
    result = agent.answer(
        "how many frames?", transport=transport, con=con, search_engine=None, log_path=log
    )
    assert result.answer == "There are 3 frames."
    assert result.steps == [
        {"tool": "run_sql", "input": {"sql": "SELECT count(*) AS n FROM samples"}, "output": "1 rows"}
    ]
    # The tool result actually reached the model on turn 2.
    tool_message = transport.seen[1][-1]
    assert tool_message["role"] == "tool" and '"rows": [[3]]' in tool_message["content"]
    # Interaction logged.
    record = json.loads(log.read_text().splitlines()[0])
    assert record["question"] == "how many frames?" and record["model"] == "fake-model"
    assert record["steps"][0]["tool"] == "run_sql"


def test_agent_collects_frames_and_dedupes(con: Any) -> None:
    engine = FakeSearchEngine()
    transport = ScriptedTransport(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c1", "search_frames", {"query": "fog", "k": 2})],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c2", "show_frames", {"sample_data_tokens": ["tok0", "tokZ"]})],
            },
            {"role": "assistant", "content": "Here are foggy frames."},
        ]
    )
    result = agent.answer("foggy?", transport=transport, con=con, search_engine=engine)
    assert [frame["sample_data_token"] for frame in result.frames] == ["tok0", "tok1", "tokZ"]
    assert engine.calls[0] == ("text", "fog", 2)
    # Model-visible payload carries metadata but never thumbnail bytes.
    assert "thumbnail" not in json.loads(transport.seen[1][-1]["content"])["results"][0]


def test_agent_search_unavailable_and_bad_args(con: Any) -> None:
    transport = ScriptedTransport(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _tool_call("c1", "search_frames", {"query": "x"}),
                    {"id": "c2", "type": "function",
                     "function": {"name": "run_sql", "arguments": "{not json"}},
                    _tool_call("c3", "nope", {}),
                ],
            },
            {"role": "assistant", "content": "done"},
        ]
    )
    result = agent.answer("q", transport=transport, con=con, search_engine=None)
    assert result.answer == "done"
    outputs = [step["output"] for step in result.steps]
    assert "not available" in outputs[0]
    assert "Bad tool arguments" in outputs[1]
    assert "Unknown tool" in outputs[2]


def test_agent_normalizes_double_escaped_sql(con: Any) -> None:
    transport = ScriptedTransport(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _tool_call("c1", "run_sql", {"sql": "SELECT\\n count(*)\\nFROM samples"})
                ],
            },
            {"role": "assistant", "content": "3"},
        ]
    )
    result = agent.answer("q", transport=transport, con=con, search_engine=None)
    assert result.steps[0]["output"] == "1 rows"


def test_agent_breaks_repeated_tool_call_loop(con: Any) -> None:
    engine = FakeSearchEngine()
    repeat = {
        "role": "assistant",
        "content": None,
        "tool_calls": [_tool_call("c", "show_frames", {"sample_data_tokens": ["t3"]})],
    }
    transport = ScriptedTransport([repeat, repeat, {"role": "assistant", "content": "here they are"}])
    result = agent.answer("q", transport=transport, con=con, search_engine=engine, max_turns=6)
    assert result.answer == "here they are"
    # First call attaches; the identical second call is short-circuited, not re-run.
    assert result.steps[0]["output"] == "1 frames attached"
    assert result.steps[1]["output"] == "repeat (skipped)"
    # Exactly one underlying tool execution -> no duplicated frames.
    assert [frame["sample_data_token"] for frame in result.frames] == ["t3"]
    assert len(engine.calls) == 1


def test_summarize_reports_a_successful_chart_not_a_repeat() -> None:
    """``_make_chart``'s success payload carries both ``charted`` and ``note`` (the
    note tells the model not to embed the image) -- ``_summarize`` must check
    ``charted`` before ``note``, or every successful chart step reads as
    "repeat (skipped)" in the UI/log/Phase-8 recording (regression since Phase 4)."""
    result = agent.ChatResult(answer="", model="stub")
    output = agent._make_chart(
        {
            "kind": "bar",
            "title": "Night frames per location",
            "columns": ["location", "n"],
            "rows": [["boston-seaport", 3]],
        },
        result,
    )
    assert "charted" in output and "note" in output  # both keys really are present
    assert agent._summarize(output) == "charted: Night frames per location"


def test_agent_max_turns(con: Any) -> None:
    looping = {
        "role": "assistant",
        "content": None,
        "tool_calls": [_tool_call("c", "run_sql", {"sql": "SELECT 1"})],
    }
    transport = ScriptedTransport([looping] * 3)
    result = agent.answer("q", transport=transport, con=con, search_engine=None, max_turns=3)
    assert "tool-call budget" in result.answer
    assert len(result.steps) == 3


def test_agent_history_precedes_question(con: Any) -> None:
    transport = ScriptedTransport([{"role": "assistant", "content": "as before"}])
    history = [
        {"role": "user", "content": "earlier q"},
        {"role": "assistant", "content": "earlier a"},
    ]
    agent.answer("follow-up", transport=transport, con=con, search_engine=None, history=history)
    roles = [message["role"] for message in transport.seen[0]]
    assert roles == ["system", "user", "assistant", "user"]


def test_show_frames_tool_survives_a_broken_store(con: Any) -> None:
    """frames_by_tokens raising becomes a model-visible tool error, not a crash."""
    from nuscenes_data_engine.data_engine.chat import agent

    class _BrokenStore:
        def frames_by_tokens(self, tokens: list[str]) -> list[dict[str, Any]]:
            raise RuntimeError("store exploded")

    result = agent.ChatResult(answer="", model="stub")
    output = agent._run_tool(
        "show_frames", {"tokens": ["t1"]}, con=con, search_engine=_BrokenStore(),
        result=result, graph_driver=None, graph_database="neo4j",
    )
    assert "error" in output and "store exploded" in output["error"]
    assert result.frames == []


# ---------------------------------------------------------------------------
# POST /chat — endpoint wiring (degraded detection model; fakes on app.state)
# ---------------------------------------------------------------------------


@pytest.fixture()
def chat_client(
    con: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Any:
    fastapi = pytest.importorskip("fastapi")
    del fastapi
    from fastapi.testclient import TestClient

    import nuscenes_data_engine.serving.app as serving_app

    monkeypatch.setattr(
        serving_app, "load_production_model", lambda settings: (_ for _ in ()).throw(RuntimeError)
    )
    monkeypatch.setenv("CHAT_LOG_PATH", str(tmp_path / "chatlog.jsonl"))
    monkeypatch.setenv("SEARCH_LANCEDB_PATH", str(tmp_path / "no-lancedb"))
    with TestClient(serving_app.app) as client:
        client.app.state.chat_catalog = con
        client.app.state.search_engine = FakeSearchEngine()
        yield client


def _patch_transport(monkeypatch: pytest.MonkeyPatch, transport: Any) -> None:
    from nuscenes_data_engine.data_engine.chat import transports

    monkeypatch.setattr(transports, "make_transport", lambda settings, **kw: transport)


def test_chat_endpoint_happy_path(
    chat_client: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_transport(
        monkeypatch,
        ScriptedTransport(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        _tool_call("c1", "run_sql", {"sql": "SELECT count(*) FROM samples"}),
                        _tool_call("c2", "show_frames", {"sample_data_tokens": ["t3"]}),
                    ],
                },
                {"role": "assistant", "content": "3 frames; example attached."},
            ]
        ),
    )
    response = chat_client.post(
        "/chat", json={"message": "how many frames?", "history": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "3 frames; example attached."
    assert body["model"] == "fake-model"
    assert [step["tool"] for step in body["steps"]] == ["run_sql", "show_frames"]
    assert body["frames"][0]["sample_data_token"] == "t3"
    assert base64.b64decode(body["frames"][0]["thumbnail_b64"]) == b"jpegbytes"
    assert (tmp_path / "chatlog.jsonl").is_file()

    health = chat_client.get("/health").json()
    assert health["chat_provider"] == "local" and health["chat_model"]


def test_chat_endpoint_transport_down(
    chat_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DownTransport:
        model = "m"

        def complete(self, messages: Any, tools: Any) -> dict[str, Any]:
            raise TransportError("connection refused — is `ollama serve` running?")

    _patch_transport(monkeypatch, DownTransport())
    response = chat_client.post("/chat", json={"message": "q"})
    assert response.status_code == 503
    assert "ollama serve" in response.json()["detail"]


def test_chat_endpoint_no_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fastapi = pytest.importorskip("fastapi")
    del fastapi
    from fastapi.testclient import TestClient

    import nuscenes_data_engine.serving.app as serving_app

    monkeypatch.setattr(
        serving_app, "load_production_model", lambda settings: (_ for _ in ()).throw(RuntimeError)
    )
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "empty"))
    with TestClient(serving_app.app) as client:
        response = client.post("/chat", json={"message": "q"})
    assert response.status_code == 503
    assert "no Parquet tables" in response.json()["detail"]


# ---------------------------------------------------------------------------
# chat-eval CLI — probe seam (Task 1). Only a probe.search_text() failure may
# keep the engine (token-based frame attachment still works without torch);
# a construction failure still disables the engine outright. Exercised through
# the real `chat-eval` Typer command, following the `--regrade` test's pattern
# of stubbing `make_transport`/`run_eval` so no LLM or real data is needed.
# ---------------------------------------------------------------------------


class _FakeCliTransport:
    model = "fake"


class _ProbeConstructionFails:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError("no lancedb store")


class _ProbeSearchTextFails:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def search_text(self, *args: Any, **kwargs: Any) -> list[Any]:
        raise RuntimeError("No module named 'torch'")


class _ProbeAllOk:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def search_text(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


def _run_chat_eval_capturing_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, probe_cls: type
) -> Any:
    """Invoke the real `chat-eval` command with the transport/run_eval/SearchEngine
    seams stubbed; returns whatever `search_engine` reached `run_eval`."""
    from typer.testing import CliRunner

    import nuscenes_data_engine.data_engine.search as search_module
    from nuscenes_data_engine.cli import app
    from nuscenes_data_engine.data_engine.chat import evaluate, transports

    captured: dict[str, Any] = {}

    def fake_run_eval(
        cases: Any, *, con: Any, transport: Any, search_engine: Any,
        out_dir: Any, provider: Any, limit: Any = None,
    ) -> dict[str, Any]:
        captured["search_engine"] = search_engine
        return {"n_cases": 0}

    monkeypatch.setattr(evaluate, "run_eval", fake_run_eval)
    monkeypatch.setattr(transports, "make_transport", lambda settings, **kw: _FakeCliTransport())
    monkeypatch.setattr(search_module, "SearchEngine", probe_cls)
    # The Typer callback reconfigures root logging with `force=True` on every
    # invocation, which would tear down caplog's handler; skip it here so the
    # probe's warning (or lack of one) is observable.
    monkeypatch.setattr("nuscenes_data_engine.cli._configure_logging", lambda verbose: None)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    config_path = tmp_path / "chat_eval.yaml"
    config_path.write_text('cases:\n  - id: a\n    question: "How many samples?"\n')

    runner = CliRunner()
    result = runner.invoke(app, [
        "chat-eval", "--config", str(config_path),
        "--processed-dir", str(tmp_path / "processed"),
    ])
    assert result.exit_code == 0, result.output + repr(result.exception)
    return captured["search_engine"]


def test_chat_eval_probe_construction_failure_disables_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("WARNING", logger="nuscenes_data_engine")
    engine = _run_chat_eval_capturing_engine(monkeypatch, tmp_path, _ProbeConstructionFails)
    assert engine is None
    assert "vector search unavailable" in caplog.text.lower()


def test_chat_eval_probe_search_failure_keeps_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("WARNING", logger="nuscenes_data_engine")
    engine = _run_chat_eval_capturing_engine(monkeypatch, tmp_path, _ProbeSearchTextFails)
    assert isinstance(engine, _ProbeSearchTextFails)
    assert "semantic search unavailable" in caplog.text.lower()
    assert "token-based frame attachment still works" in caplog.text.lower()


def test_chat_eval_probe_all_ok_keeps_engine_and_warns_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("WARNING", logger="nuscenes_data_engine")
    engine = _run_chat_eval_capturing_engine(monkeypatch, tmp_path, _ProbeAllOk)
    assert isinstance(engine, _ProbeAllOk)
    assert caplog.text == ""
