"""Tests for the Phase 4 serving API.

Runs fully offline against a local yolov8n.pt loaded through the SERVING_WEIGHTS
fallback (downloaded once into weights/ when reachable, skipped otherwise). CI installs
only the dev extra, so the whole module skips there via importorskip.
"""

from __future__ import annotations

import base64
import contextlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")  # serve extra
pytest.importorskip("ultralytics")  # train extra

import cv2
import numpy as np
from fastapi.testclient import TestClient

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
