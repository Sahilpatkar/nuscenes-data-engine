"""Central runtime configuration.

Loaded from environment variables and an optional `.env` file (see `.env.example`).
Path-like settings are exposed as `pathlib.Path` for convenience.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide settings, sourced from the environment / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- nuScenes source data (read-only) ---
    nuscenes_dataroot: Path = Field(default=Path("/data/ggare/datasets/nuscenes"))
    nuscenes_version: str = Field(default="v1.0-trainval")

    # --- Pipeline output locations ---
    data_dir: Path = Field(default=Path("./data"))
    processed_dir: Path = Field(default=Path("./data/processed"))

    # --- MinIO / MLflow live on the LOCAL INFRA MACHINE, not this GPU server. ---
    # This server runs compute only (ingest/train/evaluate) and writes plain files;
    # these endpoints matter only when a run is pointed at the infra machine via .env.
    minio_endpoint: str = Field(default="http://localhost:9000")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin")
    minio_bucket: str = Field(default="nuscenes-data-engine")

    # MLflow defaults to a local file store so training needs no server here. The runs
    # dir (./mlruns) is synced to the infra machine, whose MLflow server owns the UI and
    # the model registry. Override with MLFLOW_TRACKING_URI=http://<infra-host>:5000.
    mlflow_tracking_uri: str = Field(default="file:./mlruns")


def get_settings() -> Settings:
    """Return a freshly-loaded :class:`Settings` instance."""
    return Settings()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the top of {path}, got {type(data).__name__}")
    return data
