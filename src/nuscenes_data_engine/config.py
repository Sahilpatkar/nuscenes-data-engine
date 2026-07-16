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

    # --- MinIO (local S3-compatible object store; DVC remote) ---
    minio_endpoint: str = Field(default="http://localhost:9000")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin")
    minio_bucket: str = Field(default="nuscenes-data-engine")

    # --- MLflow ---
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")


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
