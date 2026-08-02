"""Semantic search over the embedded frame store (text, image, and frame-to-frame)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from nuscenes_data_engine.data_engine import store
from nuscenes_data_engine.data_engine.embedder import Embedder

logger = logging.getLogger("nuscenes_data_engine")

_RESULT_COLUMNS = [
    "sample_data_token",
    "scene_name",
    "scene_description",
    "channel",
    "filename",
    "timestamp",
    "location",
    "is_night",
    "is_rain",
    "score",
    "thumbnail",
]


class SearchEngine:
    """Query the LanceDB frame store; builds the text/image encoder only when needed.

    ``search_similar`` reuses stored vectors, so it works without the encoder (and
    without torch) — only text and image queries construct the SigLIP embedder.
    """

    def __init__(self, db_path: Path, table: str, model_name: str, device: str = "cpu") -> None:
        self._table = store.open_frames_table(db_path, table, dim=0, create=False)
        self._model_name = model_name
        self._device = device
        self._embedder: Embedder | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _encoder(self) -> Embedder:
        if self._embedder is None:
            from nuscenes_data_engine.data_engine.embedder import SiglipEmbedder

            logger.info("Loading %s for query encoding (%s)", self._model_name, self._device)
            self._embedder = SiglipEmbedder(self._model_name, device=self._device)
        return self._embedder

    def search_text(self, query: str, k: int) -> list[dict[str, Any]]:
        vec = self._encoder().embed_texts([query])[0]
        return self._results(store.search_frames(self._table, vec, k))

    def search_image(self, data: bytes, k: int) -> list[dict[str, Any]]:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Not a decodable image")
        vec = self._encoder().embed_images([img])[0]
        return self._results(store.search_frames(self._table, vec, k))

    def search_similar(self, token: str, k: int) -> list[dict[str, Any]]:
        vec = store.vector_for(self._table, token)
        if vec is None:
            raise KeyError(f"Unknown frame token: {token}")
        return self._results(store.search_frames(self._table, vec, k, exclude_token=token))

    def frames_by_tokens(self, tokens: list[str]) -> list[dict[str, Any]]:
        """Metadata + thumbnails for specific frames (no similarity involved; score=1).

        Unknown tokens are silently dropped; tokens containing quotes are ignored
        rather than escaped (they cannot occur in real nuScenes tokens).
        """
        safe = [token for token in tokens if token and "'" not in token]
        if not safe:
            return []
        quoted = ", ".join(f"'{token}'" for token in safe)
        frames = (
            self._table.search()
            .where(f"sample_data_token IN ({quoted})")
            .limit(len(safe))
            .to_pandas()
        )
        frames = frames.assign(score=1.0)
        return self._results(frames)

    @staticmethod
    def _results(frames: pd.DataFrame) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = frames[_RESULT_COLUMNS].to_dict("records")
        return records
