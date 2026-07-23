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


def test_run_sql_row_cap_and_errors(con: Any) -> None:
    out = catalog.run_sql(con, "SELECT * FROM range(100)", max_rows=10)
    assert out["row_count"] == 10 and out["truncated"]

    bad = catalog.run_sql(con, "SELECT nope FROM samples")
    assert "error" in bad and "nope" in bad["error"]

    unparsable = catalog.run_sql(con, "SELEKT 1")
    assert "error" in unparsable


def test_schema_prompt_matches_available_tables(con: Any) -> None:
    prompt = catalog.schema_prompt(catalog.catalog_tables(con))
    assert "samples —" in prompt and "labels —" in prompt
    assert "availability —" not in prompt  # not registered on this catalog
    assert "no ego-pose" in prompt


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
