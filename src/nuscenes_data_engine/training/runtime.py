"""Keep all Ultralytics writes inside the repo (workspace boundary is strict).

Ultralytics otherwise writes its settings to ``~/.config/Ultralytics`` and downloads
weights/datasets under ``$HOME`` — outside the writable root. Call
:func:`configure_ultralytics` *before* importing ``ultralytics`` to redirect its config
dir into the repo, then point run/weight outputs at repo paths.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/nuscenes_data_engine/training/runtime.py -> repo root is parents[3].
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "runs"
WEIGHTS_DIR = REPO_ROOT / "weights"


def configure_ultralytics(*, enable_wandb: bool = False) -> None:
    """Redirect Ultralytics writes into the repo and select its integrations.

    Call before importing ``ultralytics``. Sets the config dir (keeps ``settings.json`` in
    the repo), anchors weight downloads, disables telemetry sync, and turns off
    Ultralytics' built-in MLflow callback (we log MLflow explicitly). Its W&B callback is
    enabled only when ``enable_wandb`` is set.
    """
    cfg = REPO_ROOT / ".cache" / "ultralytics"
    cfg.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cfg)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WEIGHTS_DIR", str(WEIGHTS_DIR))

    # Use certifi's CA bundle so downloads (weights, W&B) work in SSL-strict envs
    # (the system CA store here rejects github's chain).
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

    from ultralytics import settings as ul_settings

    for key, value in {
        "sync": False,
        "mlflow": False,
        "wandb": enable_wandb,
        "tensorboard": False,
    }.items():
        if key in ul_settings:
            ul_settings.update({key: value})
