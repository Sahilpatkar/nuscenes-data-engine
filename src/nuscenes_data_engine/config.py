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

    # MLflow defaults to a local SQLite backend so training needs no server here (MLflow
    # 3.x no longer accepts a plain file store, and SQLite also unlocks the model
    # registry). The mlruns/ dir (db + artifacts) is synced to the infra machine, whose
    # MLflow server owns the UI/registry. Override with MLFLOW_TRACKING_URI=http://<infra-host>:5000.
    mlflow_tracking_uri: str = Field(default="sqlite:///mlruns/mlflow.db")

    # --- Weights & Biases (cloud experiment tracking; complements local MLflow) ---
    # Set WANDB_API_KEY in .env (or run `uv run wandb login`). WANDB_MODE=offline logs
    # locally to ./wandb for later `wandb sync`; "disabled" turns W&B off entirely.
    wandb_api_key: str = Field(default="")
    wandb_entity: str = Field(default="")
    wandb_project: str = Field(default="nuscenes-data-engine")
    wandb_mode: str = Field(default="online")

    # --- Serving (Phase 4) ---
    # The API loads `models:/<serving_model_name>@<serving_model_alias>` from the MLflow
    # registry. SERVING_WEIGHTS (a local .pt path, kept as str: an empty env var must mean
    # "unset", and Path("") coerces to ".") bypasses the registry entirely — the dev/test
    # path on machines without mlruns/.
    serving_model_name: str = Field(default="nuscenes-yolo-detector")
    serving_model_alias: str = Field(default="production")
    serving_imgsz: int = Field(default=960)
    serving_conf: float = Field(default=0.25)
    serving_device: str = Field(default="cpu")
    serving_weights: str = Field(default="")


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
