"""Load the detection model the API serves.

Two sources, in priority order:

1. ``SERVING_WEIGHTS`` — an explicit local ``.pt`` path. Deterministic ("which model am
   I serving?" has exactly one answer) and the only option on machines without the
   synced ``mlruns/`` store.
2. The MLflow registry — resolve ``models:/<name>@<alias>`` and download its ``best.pt``.

The registry stores artifact locations as absolute ``file://`` URIs from the machine
that trained the model (the GPU server), so on any other machine the recorded path does
not exist. :func:`_localize_source_uri` remaps such URIs onto the local ``mlruns/``
directory before downloading.

The loaded model is cached as a module singleton (one load per process). The default
``sqlite:///mlruns/mlflow.db`` tracking URI is CWD-relative, so run ``serve`` from the
repo root — the same constraint training already has.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from nuscenes_data_engine.config import Settings, get_settings
from nuscenes_data_engine.training.runtime import REPO_ROOT, configure_ultralytics

logger = logging.getLogger("nuscenes_data_engine")

_cache: tuple[Any, str] | None = None


def load_production_model(settings: Settings | None = None) -> tuple[Any, str]:
    """Return ``(model, version)`` for the model to serve, loading it on first call.

    ``version`` is the registry version number, or ``"local:<filename>"`` when loaded
    from ``SERVING_WEIGHTS``.
    """
    global _cache
    if _cache is not None:
        return _cache

    settings = settings or get_settings()
    configure_ultralytics()  # before importing ultralytics: keep its writes in the repo
    from ultralytics import YOLO

    if settings.serving_weights:
        weights = Path(settings.serving_weights)
        if not weights.is_file():
            raise FileNotFoundError(f"SERVING_WEIGHTS points at {weights}, which does not exist.")
        _cache = (YOLO(str(weights)), f"local:{weights.name}")
        logger.info("Loaded serving model from local weights %s", weights)
        return _cache

    _cache = _load_from_registry(settings)
    return _cache


def reset_model_cache() -> None:
    """Drop the cached model so the next load re-reads settings (used by tests)."""
    global _cache
    _cache = None


def _load_from_registry(settings: Settings) -> tuple[Any, str]:
    """Resolve ``models:/<name>@<alias>``, download its weights, and load them."""
    import mlflow
    from ultralytics import YOLO

    alias_uri = f"models:/{settings.serving_model_name}@{settings.serving_model_alias}"
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias(
            settings.serving_model_name, settings.serving_model_alias
        )
        if not mv.source:
            raise FileNotFoundError(f"Model version {mv.version} has no source URI")
        source = _localize_source_uri(mv.source, REPO_ROOT / "mlruns")
        local_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=source))
        weights = next(local_dir.rglob("best.pt"), None) or next(local_dir.rglob("*.pt"), None)
        if weights is None:
            raise FileNotFoundError(f"No .pt file under downloaded artifacts at {local_dir}")
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {alias_uri} from the MLflow registry "
            f"({settings.mlflow_tracking_uri}): {exc}. "
            "Set SERVING_WEIGHTS to a local .pt checkpoint to serve without the registry."
        ) from exc
    logger.info("Loaded serving model %s v%s from %s", alias_uri, mv.version, weights)
    return YOLO(str(weights)), str(mv.version)


def _localize_source_uri(source: str, mlruns_dir: Path) -> str:
    """Remap a ``file://`` artifact URI recorded on another machine onto local mlruns.

    Registry entries store the training machine's absolute artifact path. If that path
    does not exist here but contains an ``mlruns/`` segment, rebase everything after it
    onto ``mlruns_dir``. Non-``file://`` URIs and locally-valid paths pass through.
    """
    if not source.startswith("file://"):
        return source
    path = Path(unquote(urlparse(source).path))
    if path.exists():
        return source
    try:
        tail = path.parts[path.parts.index("mlruns") + 1 :]
    except ValueError:
        return source
    return (mlruns_dir.joinpath(*tail)).as_uri()
