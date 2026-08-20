"""Tests for the Phase 4 serving API.

Runs fully offline against a local yolov8n.pt loaded through the SERVING_WEIGHTS
fallback (downloaded once into weights/ when reachable, skipped otherwise). CI installs
only the dev extra, so the whole module skips there via importorskip.
"""

from __future__ import annotations

import base64
import contextlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")  # serve extra
pytest.importorskip("ultralytics")  # train extra

import cv2
import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient

from nuscenes_data_engine.data_engine.chat.transports import TransportError
from nuscenes_data_engine.serving import model as serving_model
from nuscenes_data_engine.serving.app import app
from nuscenes_data_engine.serving.model import _localize_source_uri
from nuscenes_data_engine.serving.schemas import PredictResponse
from nuscenes_data_engine.training.runtime import WEIGHTS_DIR, configure_ultralytics


@pytest.fixture(scope="session")
def yolov8n_weights() -> Path:
    """A tiny stock checkpoint in weights/; downloaded once, else the tests skip."""
    weights = WEIGHTS_DIR / "yolov8n.pt"
    if not weights.is_file():
        configure_ultralytics()
        from ultralytics.utils.downloads import attempt_download_asset

        with contextlib.suppress(Exception):
            attempt_download_asset(str(weights))
    if not weights.is_file():
        pytest.skip("yolov8n.pt unavailable (offline)")
    return weights


@pytest.fixture()
def capture_path(tmp_path: Path) -> Path:
    """Where the client fixture directs the per-request monitoring capture."""
    return tmp_path / "requests.jsonl"


@pytest.fixture()
def client(
    yolov8n_weights: Path, capture_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """API client backed by yolov8n via SERVING_WEIGHTS (env beats .env)."""
    monkeypatch.setenv("SERVING_WEIGHTS", str(yolov8n_weights))
    monkeypatch.setenv("SERVING_CAPTURE_PATH", str(capture_path))
    serving_model.reset_model_cache()
    with TestClient(app) as test_client:  # context manager: runs the lifespan
        yield test_client
    serving_model.reset_model_cache()


@pytest.fixture()
def degraded_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """API client whose model load failed at startup."""

    def boom(settings: object = None) -> tuple[object, str]:
        raise RuntimeError("model load failed")

    monkeypatch.setattr("nuscenes_data_engine.serving.app.load_production_model", boom)
    with TestClient(app) as test_client:
        yield test_client


def _jpeg_bytes(width: int = 320, height: int = 240) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((height, width, 3), np.uint8))
    assert ok
    return buf.tobytes()


class TestHealth:
    def test_reports_loaded_model(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_version"].startswith("local:")


class TestPredict:
    def test_synthetic_image(self, client: TestClient) -> None:
        resp = client.post("/predict", files={"file": ("black.jpg", _jpeg_bytes(), "image/jpeg")})
        assert resp.status_code == 200
        body = PredictResponse.model_validate(resp.json())
        assert body.image_width == 320
        assert body.image_height == 240
        assert body.n_detections == len(body.detections)
        assert body.model_version.startswith("local:")

    def test_undecodable_upload(self, client: TestClient) -> None:
        resp = client.post("/predict", files={"file": ("junk.jpg", b"not an image", "image/jpeg")})
        assert resp.status_code == 400


class TestPredictAnnotated:
    def test_returns_png(self, client: TestClient) -> None:
        resp = client.post(
            "/predict/annotated", files={"file": ("black.jpg", _jpeg_bytes(), "image/jpeg")}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.content.startswith(b"\x89PNG")


class TestDegraded:
    def test_health_and_predict_without_model(self, degraded_client: TestClient) -> None:
        body = degraded_client.get("/health").json()
        assert body["model_loaded"] is False
        assert body["model_version"] is None
        resp = degraded_client.post(
            "/predict", files={"file": ("black.jpg", _jpeg_bytes(), "image/jpeg")}
        )
        assert resp.status_code == 503


class TestCapture:
    def test_appends_one_row_per_request(self, client: TestClient, capture_path: Path) -> None:
        files = {"file": ("black.jpg", _jpeg_bytes(), "image/jpeg")}
        assert client.post("/predict", files=files).status_code == 200
        assert client.post("/predict/annotated", files=files).status_code == 200
        rows = [json.loads(line) for line in capture_path.read_text().splitlines()]
        assert len(rows) == 2
        for row in rows:
            assert row["image_width"] == 320
            assert row["image_height"] == 240
            assert row["brightness"] == 0.0  # black frame
            assert row["model_version"].startswith("local:")
            assert row["latency_ms"] >= 0
            assert "ts" in row and "n_detections" in row

    def test_empty_path_disables_capture(
        self, yolov8n_weights: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SERVING_WEIGHTS", str(yolov8n_weights))
        monkeypatch.setenv("SERVING_CAPTURE_PATH", "")
        monkeypatch.chdir(tmp_path)  # any accidental default-path write would land here
        serving_model.reset_model_cache()
        with TestClient(app) as test_client:
            files = {"file": ("black.jpg", _jpeg_bytes(), "image/jpeg")}
            assert test_client.post("/predict", files=files).status_code == 200
        serving_model.reset_model_cache()
        assert not (tmp_path / "data").exists()


def _search_row(score: float) -> dict[str, object]:
    return {
        "sample_data_token": "tok1",
        "scene_name": "scene-0001",
        "scene_description": "Sunny day",
        "channel": "CAM_FRONT",
        "filename": "samples/CAM_FRONT/x.jpg",
        "timestamp": 1_000_000,
        "location": "boston-seaport",
        "is_night": False,
        "is_rain": False,
        "score": score,
        "thumbnail": b"\xff\xd8fakejpeg",
    }


class _FakeSearchEngine:
    model_name = "fake-model"

    def search_text(self, query: str, k: int) -> list[dict[str, object]]:
        return [_search_row(0.9)] * min(k, 2)

    def search_image(self, data: bytes, k: int) -> list[dict[str, object]]:
        if not data.startswith(b"\xff\xd8"):
            raise ValueError("Not a decodable image")
        return [_search_row(0.8)]

    def search_similar(self, token: str, k: int) -> list[dict[str, object]]:
        if token == "unknown":
            raise KeyError(token)
        return [_search_row(0.7)]


class TestSearchApi:
    def test_text_search(self, client: TestClient) -> None:
        client.app.state.search_engine = _FakeSearchEngine()
        body = client.get("/search", params={"q": "night construction", "k": 2}).json()
        assert len(body["results"]) == 2
        assert body["embedding_model"] == "fake-model"
        assert base64.b64decode(body["results"][0]["thumbnail_b64"]).startswith(b"\xff\xd8")
        assert body["results"][0]["score"] == 0.9

    def test_image_search_and_bad_upload(self, client: TestClient) -> None:
        client.app.state.search_engine = _FakeSearchEngine()
        ok = client.post("/search/image", files={"file": ("q.jpg", b"\xff\xd8data", "image/jpeg")})
        assert ok.status_code == 200
        bad = client.post("/search/image", files={"file": ("q.jpg", b"junk", "image/jpeg")})
        assert bad.status_code == 400

    def test_similar_and_unknown_token(self, client: TestClient) -> None:
        client.app.state.search_engine = _FakeSearchEngine()
        assert client.get("/search/similar/tok1").status_code == 200
        assert client.get("/search/similar/unknown").status_code == 404

    def test_search_unavailable_without_store(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client.app.state.search_engine = None
        client.app.state.settings.search_lancedb_path = str(tmp_path / "absent")
        resp = client.get("/search", params={"q": "anything"})
        assert resp.status_code == 503
        health = client.get("/health").json()
        assert health["search_ready"] is False


class TestLocalizeSourceUri:
    def test_foreign_mlruns_uri_rebased(self, tmp_path: Path) -> None:
        source = "file:///home/mgaur/sahil/nuscenes-data-engine/mlruns/artifacts/1/abc/weights"
        localized = _localize_source_uri(source, tmp_path / "mlruns")
        assert localized == (tmp_path / "mlruns" / "artifacts" / "1" / "abc" / "weights").as_uri()

    def test_existing_local_path_untouched(self, tmp_path: Path) -> None:
        source = tmp_path.as_uri()
        assert _localize_source_uri(source, tmp_path / "mlruns") == source

    def test_non_file_uri_untouched(self, tmp_path: Path) -> None:
        source = "s3://bucket/mlflow/artifacts/weights"
        assert _localize_source_uri(source, tmp_path / "mlruns") == source

    def test_foreign_uri_without_mlruns_untouched(self, tmp_path: Path) -> None:
        source = "file:///nonexistent/elsewhere/weights"
        assert _localize_source_uri(source, tmp_path / "mlruns") == source


# ---------------------------------------------------------------------------
# POST /chat, POST /chat/stream — Task 5 (demo-phase-4): the additive `charts`
# field on /chat's response, and the new SSE streaming endpoint. This file had no
# prior /chat coverage of its own — transport injection mirrors the ESTABLISHED
# pattern from tests/test_chat.py's `chat_client` fixture + `_patch_transport`
# helper (which already exercise POST /chat end-to-end against this same app),
# adapted to this file's TestClient conventions (string-path monkeypatch.setattr,
# like the existing `degraded_client` fixture above).
# ---------------------------------------------------------------------------


@pytest.fixture()
def con() -> Any:
    """A tiny DuckDB catalog with one `samples` view (mirrors
    tests/test_chat_stream.py's `con` fixture — duckdb lives in the 'engine' extra,
    so this skips gracefully when it's absent rather than failing at import time)."""
    pytest.importorskip("duckdb")
    import duckdb

    connection = duckdb.connect()
    connection.execute(
        "CREATE VIEW samples AS SELECT * FROM (VALUES "
        "('t1','boston-seaport',TRUE), ('t2','singapore-onenorth',FALSE)) "
        "AS s(sample_data_token, location, is_night)"
    )
    return connection


@pytest.fixture()
def chat_client(con: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Client for the chat endpoints: detection model degraded (irrelevant here), a
    DuckDB catalog and no vector store stubbed onto app.state — mirrors
    tests/test_chat.py's chat_client fixture."""

    def boom(settings: object = None) -> tuple[object, str]:
        raise RuntimeError("model load failed")

    monkeypatch.setattr("nuscenes_data_engine.serving.app.load_production_model", boom)
    monkeypatch.setenv("CHAT_LOG_PATH", str(tmp_path / "chatlog.jsonl"))
    monkeypatch.setenv("SEARCH_LANCEDB_PATH", str(tmp_path / "no-lancedb"))
    with TestClient(app) as test_client:
        test_client.app.state.chat_catalog = con
        test_client.app.state.search_engine = None
        yield test_client


def _patch_transport(monkeypatch: pytest.MonkeyPatch, transport: Any) -> None:
    from nuscenes_data_engine.data_engine.chat import transports

    monkeypatch.setattr(transports, "make_transport", lambda settings, **kw: transport)


def _parse_sse(lines: Iterable[str]) -> list[tuple[str, str]]:
    """Pair up `event:`/`data:` lines from an SSE stream into (kind, payload) tuples."""
    events: list[tuple[str, str]] = []
    kind: str | None = None
    for line in lines:
        if line.startswith("event: "):
            kind = line[len("event: ") :]
        elif line.startswith("data: ") and kind is not None:
            events.append((kind, line[len("data: ") :]))
            kind = None
    return events


class _StreamScriptedTransport:
    """Stream-capable transport: turn 1 calls run_sql, turn 2 answers with tokens
    (mirrors tests/test_chat_stream.py's _StreamFake)."""

    model = "fake-model"

    def __init__(self) -> None:
        self.turn = 0

    def complete_stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], on_token: Any
    ) -> dict[str, Any]:
        self.turn += 1
        if self.turn == 1:
            on_token("checking…")
            return {
                "content": "checking…",
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
        for piece in ("There are ", "line1\nline2", "2 samples."):
            on_token(piece)
        return {"content": "There are 2 samples.", "tool_calls": []}

    def complete(self, messages: Any, tools: Any) -> dict[str, Any]:
        raise AssertionError("complete must not be called when streaming")


class _RaisingStreamTransport:
    """A transport whose complete_stream always fails, for the error-event test."""

    model = "fake-model"

    def complete_stream(self, messages: Any, tools: Any, on_token: Any) -> dict[str, Any]:
        raise TransportError("connection refused — is `ollama serve` running?")


class _MultilineTransportErrorTransport:
    """A transport whose TransportError message spans two lines, the way
    httpx.HTTPStatusError's own message does BY CONSTRUCTION ("...for url
    '...'\\nFor more information check: ..."), which OpenAICompatTransport wraps
    verbatim. Regression fixture for the error payload's JSON-encoding: a
    line-based SSE parser must not truncate the message — and drop the trailing
    "is `ollama serve` running?" hint — at the embedded newline."""

    model = "fake-model"

    MESSAGE = (
        "Chat model at http://localhost:11434/v1 unavailable (Client error '404 Not "
        "Found' for url 'http://localhost:11434/v1/chat/completions'\n"
        "For more information check: https://developer.mozilla.org/en-US/docs/Web/"
        "HTTP/Status/404). Is the local model server (e.g. `ollama serve`) running?"
    )

    def complete_stream(self, messages: Any, tools: Any, on_token: Any) -> dict[str, Any]:
        raise TransportError(self.MESSAGE)


class _CrashingStreamTransport:
    """A transport whose complete_stream fails with a non-TransportError, for the
    generic-error-message + server-side-logging test (worker exceptions that are
    NOT TransportError must not leak their text to the client)."""

    model = "fake-model"

    def complete_stream(self, messages: Any, tools: Any, on_token: Any) -> dict[str, Any]:
        raise RuntimeError("boom: unexpected crash")


class _ChartingTransport:
    """Non-streaming transport scripted to call make_chart on its first turn."""

    model = "fake-model"

    def __init__(self) -> None:
        self.turn = 0

    def complete(self, messages: Any, tools: Any) -> dict[str, Any]:
        self.turn += 1
        if self.turn == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "make_chart",
                            "arguments": json.dumps(
                                {
                                    "kind": "bar",
                                    "title": "Night scenes per location",
                                    "columns": ["location", "scenes"],
                                    "rows": [["boston", 3], ["singapore", 5]],
                                }
                            ),
                        },
                    }
                ],
            }
        return {"content": "Charted.", "tool_calls": []}


class TestChatCharts:
    def test_chat_response_carries_charts(
        self, chat_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_transport(monkeypatch, _ChartingTransport())
        response = chat_client.post("/chat", json={"message": "chart it", "history": []})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["charts"][0]["kind"] == "bar"
        assert body["charts"][0]["title"] == "Night scenes per location"


class TestChatStream:
    def test_emits_turn_token_step_final(
        self, chat_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_transport(monkeypatch, _StreamScriptedTransport())
        with chat_client.stream(
            "POST", "/chat/stream", json={"message": "How many?", "history": []}
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            events = _parse_sse(response.iter_lines())
        kinds = [kind for kind, _payload in events]
        assert kinds[0] == "turn"
        assert "token" in kinds
        assert "step" in kinds
        assert kinds[-1] == "final"
        final_payload = next(payload for kind, payload in reversed(events) if kind == "final")
        final = json.loads(final_payload)
        assert final["answer"] == "There are 2 samples."
        assert "charts" in final and "frames" in final

        # Newline safety: one on_token delta is "line1\nline2". token payloads are
        # JSON-encoded (json.dumps(delta)) specifically so an embedded raw newline is
        # escaped onto a single `data:` line instead of splitting into a stray,
        # un-prefixed line that a line-based SSE parser (this test's _parse_sse, or a
        # browser/EventSource client) would silently drop. Both effects are checked:
        # framing survives (all 4 on_token calls are still 4 distinct token events)
        # and the payload round-trips with the real newline intact.
        token_payloads = [payload for kind, payload in events if kind == "token"]
        assert len(token_payloads) == 4
        decoded_tokens = [json.loads(payload) for payload in token_payloads]
        assert "line1\nline2" in decoded_tokens

    def test_error_event_on_transport_failure(
        self, chat_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_transport(monkeypatch, _RaisingStreamTransport())
        with chat_client.stream(
            "POST", "/chat/stream", json={"message": "q", "history": []}
        ) as response:
            assert response.status_code == 200
            events = _parse_sse(response.iter_lines())
        kinds = [kind for kind, _payload in events]
        assert kinds[-1] == "error"
        error_payload = next(payload for kind, payload in reversed(events) if kind == "error")
        assert "ollama serve" in json.loads(error_payload)

    def test_error_event_payload_is_json_encoded_for_multiline_messages(
        self, chat_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The error payload is JSON-encoded (json.dumps(message)) for the same
        newline-safety reason token payloads are: a TransportError's message can
        itself contain an embedded newline (httpx.HTTPStatusError's message is
        two-line by construction), and an un-escaped newline in a bare `data:` line
        would split into a second, un-prefixed line that a line-based SSE parser
        silently drops — truncating exactly the "is `ollama serve` running?" hint
        at the end of this fixture's message."""
        transport = _MultilineTransportErrorTransport()
        _patch_transport(monkeypatch, transport)
        with chat_client.stream(
            "POST", "/chat/stream", json={"message": "q", "history": []}
        ) as response:
            assert response.status_code == 200
            events = _parse_sse(response.iter_lines())
        kinds = [kind for kind, _payload in events]
        assert kinds[-1] == "error"
        # Framing survives: exactly one error event, not split into stray lines.
        assert kinds.count("error") == 1
        error_payload = next(payload for kind, payload in reversed(events) if kind == "error")
        decoded = json.loads(error_payload)
        assert decoded == transport.MESSAGE
        assert "\n" in decoded
        assert decoded.endswith("Is the local model server (e.g. `ollama serve`) running?")

    def test_error_event_generic_message_on_non_transport_exception(
        self,
        chat_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Parity with /chat: only TransportError's message reaches the client. Any
        other worker exception gets a generic client-visible message, but the full
        traceback must still be logged server-side (previously: not logged at all)."""
        caplog.set_level("ERROR", logger="nuscenes_data_engine")
        _patch_transport(monkeypatch, _CrashingStreamTransport())
        with chat_client.stream(
            "POST", "/chat/stream", json={"message": "q", "history": []}
        ) as response:
            assert response.status_code == 200
            events = _parse_sse(response.iter_lines())
        kinds = [kind for kind, _payload in events]
        assert kinds[-1] == "error"
        error_payload = next(payload for kind, payload in reversed(events) if kind == "error")
        assert json.loads(error_payload) == "internal error — see server logs"
        assert "boom: unexpected crash" not in error_payload
        assert "chat_stream worker failed" in caplog.text
        assert "boom: unexpected crash" in caplog.text  # traceback captured server-side

    def test_chat_stream_returns_503_before_streaming(
        self, chat_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resources are resolved before the generator is defined — a getter
        failure must surface as a normal 503 JSON response, never a 200 SSE stream
        that then fails mid-body."""
        import nuscenes_data_engine.serving.app as serving_app

        def boom(request: Any) -> Any:
            raise HTTPException(status_code=503, detail="Chat unavailable: no catalog")

        monkeypatch.setattr(serving_app, "_get_chat_catalog", boom)
        response = chat_client.post("/chat/stream", json={"message": "q", "history": []})
        assert response.status_code == 503
        assert response.headers["content-type"].startswith("application/json")
        assert "no catalog" in response.json()["detail"]
