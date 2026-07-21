"""Image/text embedders for semantic frame search.

Module top stays torch/transformers-free (they live in the train/engine extras and are
absent in the CI quality env); :class:`SiglipEmbedder` imports them lazily. The
:class:`Embedder` protocol lets tests substitute a deterministic fake.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from nuscenes_data_engine.training.runtime import REPO_ROOT


class Embedder(Protocol):
    """Anything that maps images/texts into a shared vector space."""

    dim: int
    name: str

    def embed_images(self, images: Sequence[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        """BGR uint8 images -> L2-normalized float32 vectors, shape (n, dim)."""
        ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray[Any, Any]:
        """Texts -> L2-normalized float32 vectors, shape (n, dim)."""
        ...


def configure_huggingface() -> None:
    """Keep Hugging Face downloads/caches inside the repo (workspace boundary).

    Call before importing transformers. Mirrors ``configure_ultralytics``, including
    the certifi CA bundle for SSL-strict hosts.
    """
    cache = REPO_ROOT / ".cache" / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))

    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


def _l2_normalize(vectors: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.asarray(vectors / np.maximum(norms, 1e-12), dtype=np.float32)


class SiglipEmbedder:
    """SigLIP(2) image/text embedder via transformers (lazy torch import)."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        configure_huggingface()
        import torch
        from transformers import AutoModel, AutoProcessor

        self.name = model_name
        self._torch = torch
        self._device = device
        self._model = AutoModel.from_pretrained(model_name).to(device).eval()
        self._processor = AutoProcessor.from_pretrained(model_name)
        self.dim = int(self._model.config.vision_config.hidden_size)

    def embed_images(self, images: Sequence[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        rgb = [np.ascontiguousarray(img[:, :, ::-1]) for img in images]  # cv2 BGR -> RGB
        inputs = self._processor(images=rgb, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        return _l2_normalize(feats.cpu().numpy())

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray[Any, Any]:
        # SigLIP's text tower expects fixed-length padding.
        inputs = self._processor(
            text=list(texts), padding="max_length", truncation=True, return_tensors="pt"
        ).to(self._device)
        with self._torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        return _l2_normalize(feats.cpu().numpy())
