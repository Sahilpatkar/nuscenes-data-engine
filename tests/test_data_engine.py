"""Tests for the Phase 6a data engine (fake embedder; offline, CI-safe)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("lancedb")  # engine extra

import cv2
import numpy as np
import pandas as pd
import yaml

from nuscenes_data_engine.data_engine import store
from nuscenes_data_engine.data_engine.embeddings import run_embedding
from nuscenes_data_engine.data_engine.search import SearchEngine

DIM = 8


class FakeEmbedder:
    """Deterministic embedder: vector depends only on the input bytes/text."""

    dim = DIM
    name = "fake"

    @staticmethod
    def _vec(seed_bytes: bytes) -> np.ndarray[Any, Any]:
        seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:4], "big")
        vec = np.random.default_rng(seed).normal(size=DIM)
        return (vec / np.linalg.norm(vec)).astype(np.float32)

    def embed_images(self, images: Sequence[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        return np.stack([self._vec(img.tobytes()) for img in images])

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray[Any, Any]:
        return np.stack([self._vec(t.encode()) for t in texts])


def _write_dataset(root: Path, n_scenes: int = 3, frames_per_scene: int = 4) -> Path:
    """Synthetic samples.parquet + distinct tiny images under a fake dataroot."""
    rows = []
    for s in range(n_scenes):
        for f in range(frames_per_scene):
            filename = f"samples/CAM_FRONT/s{s}f{f}.jpg"
            img = np.full((90, 160, 3), 40 * s + 10 * f + 10, np.uint8)
            img[:20, :20] = (s * 80, f * 60, 200)  # make every frame unique
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), img)
            rows.append(
                {
                    "sample_data_token": f"tok_s{s}f{f}",
                    "sample_token": f"s{s}",
                    "scene_token": f"scene{s}",
                    "scene_name": f"scene-000{s}",
                    "scene_description": "Night drive" if s == 2 else "Sunny day",
                    "channel": "CAM_FRONT",
                    "filename": filename,
                    "width": 160,
                    "height": 90,
                    "timestamp": 1_000_000 + s * 100 + f,
                    "n_boxes": 3,
                    "location": "boston-seaport",
                    "is_night": s == 2,
                    "is_rain": False,
                }
            )
    processed = root / "processed"
    processed.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(processed / "samples.parquet", index=False)
    return processed


@pytest.fixture()
def engine_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Synthetic dataroot + config; env points Settings at them."""
    processed = _write_dataset(tmp_path)
    monkeypatch.setenv("NUSCENES_DATAROOT", str(tmp_path))
    monkeypatch.setenv("PROCESSED_DIR", str(processed))
    config = tmp_path / "engine.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "embedding": {"batch_size": 3, "thumbnail_max_px": 64},
                "lancedb": {"path": str(tmp_path / "lancedb"), "table": "frames"},
            }
        )
    )
    return config


class TestEmbeddingJob:
    def test_embeds_all_frames_with_thumbnails(self, engine_env: Path, tmp_path: Path) -> None:
        summary = run_embedding(engine_env, embedder=FakeEmbedder())
        assert summary["frames_added"] == 12
        assert summary["total_frames"] == 12
        tbl = store.open_frames_table(tmp_path / "lancedb", "frames", DIM)
        frames = tbl.to_pandas()
        thumb = cv2.imdecode(np.frombuffer(frames.iloc[0]["thumbnail"], np.uint8), cv2.IMREAD_COLOR)
        assert thumb is not None and max(thumb.shape[:2]) <= 64

    def test_resume_skips_completed_scenes(self, engine_env: Path) -> None:
        first = run_embedding(engine_env, embedder=FakeEmbedder())
        again = run_embedding(engine_env, embedder=FakeEmbedder())
        assert first["frames_added"] == 12
        assert again["frames_added"] == 0
        assert again["scenes_skipped"] == 3
        assert again["total_frames"] == 12

    def test_limit_scenes_and_rebuild(self, engine_env: Path) -> None:
        partial = run_embedding(engine_env, limit_scenes=1, embedder=FakeEmbedder())
        assert partial["total_frames"] == 4
        full = run_embedding(engine_env, rebuild=True, embedder=FakeEmbedder())
        assert full["total_frames"] == 12

    def test_manifest_filters_missing(self, engine_env: Path, tmp_path: Path) -> None:
        samples = pd.read_parquet(tmp_path / "processed" / "samples.parquet")
        manifest = samples[["sample_data_token"]].copy()
        manifest["present"] = manifest["sample_data_token"] != "tok_s0f0"
        manifest.to_parquet(tmp_path / "processed" / "availability.parquet", index=False)
        summary = run_embedding(engine_env, embedder=FakeEmbedder())
        assert summary["total_frames"] == 11


class TestSearch:
    def test_image_query_ranks_matching_frame_first(self, engine_env: Path, tmp_path: Path) -> None:
        run_embedding(engine_env, embedder=FakeEmbedder())
        engine = SearchEngine(tmp_path / "lancedb", "frames", "fake-model")
        engine._embedder = FakeEmbedder()  # bypass SigLIP construction
        # The original file bytes decode to exactly the pixels the job embedded, so the
        # fake (hash-of-pixels) embedder reproduces the stored vector -> score ~= 1.
        data = (tmp_path / "samples/CAM_FRONT/s1f2.jpg").read_bytes()
        results = engine.search_image(data, k=3)
        assert results[0]["sample_data_token"] == "tok_s1f2"
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)
        assert {"scene_name", "thumbnail", "is_night"} <= set(results[0])

    def test_similar_excludes_self(self, engine_env: Path, tmp_path: Path) -> None:
        run_embedding(engine_env, embedder=FakeEmbedder())
        engine = SearchEngine(tmp_path / "lancedb", "frames", "fake-model")
        results = engine.search_similar("tok_s0f0", k=5)
        assert len(results) == 5
        assert all(r["sample_data_token"] != "tok_s0f0" for r in results)

    def test_similar_unknown_token(self, engine_env: Path, tmp_path: Path) -> None:
        run_embedding(engine_env, embedder=FakeEmbedder())
        engine = SearchEngine(tmp_path / "lancedb", "frames", "fake-model")
        with pytest.raises(KeyError):
            engine.search_similar("nope", k=3)

    def test_missing_store_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            SearchEngine(tmp_path / "empty", "frames", "fake-model")


@pytest.mark.engine_smoke
def test_real_siglip_smoke() -> None:
    """Load the configured SigLIP model and sanity-check embeddings (manual; downloads)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from nuscenes_data_engine.config import load_yaml
    from nuscenes_data_engine.data_engine.embedder import SiglipEmbedder

    model_name = load_yaml(Path("configs/engine.yaml"))["embedding"]["model_name"]
    embedder = SiglipEmbedder(model_name, device="cpu")
    bright_a = np.full((256, 256, 3), 220, np.uint8)
    bright_b = np.full((256, 256, 3), 200, np.uint8)
    dark = np.full((256, 256, 3), 10, np.uint8)
    vectors = embedder.embed_images([bright_a, bright_b, dark])
    assert vectors.shape == (3, embedder.dim)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]  # bright pair closer
    texts = embedder.embed_texts(["a photo of a road at night", "a sunny highway"])
    assert texts.shape == (2, embedder.dim)
