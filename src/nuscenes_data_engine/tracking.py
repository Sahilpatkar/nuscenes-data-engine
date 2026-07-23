"""Lightweight Weights & Biases run tracking for pipeline stages.

Training streams to W&B through the Ultralytics callback; every other stage (embed,
autolabel, evaluate, monitor) uses :func:`wandb_run` from here. The contract is
no-op-by-default-safe: if ``wandb`` isn't installed (it ships in the train extra), no
key is configured, or ``WANDB_MODE=disabled``, the context yields ``None`` and the
stage runs exactly as before.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from nuscenes_data_engine.config import get_settings

logger = logging.getLogger("nuscenes_data_engine")


def _wandb_available() -> Any | None:
    """Export the configured W&B env (as training does) and import wandb, or None."""
    settings = get_settings()
    if settings.wandb_mode == "disabled":
        return None
    if settings.wandb_api_key:
        os.environ.setdefault("WANDB_API_KEY", settings.wandb_api_key)
    os.environ.setdefault("WANDB_MODE", settings.wandb_mode)
    if settings.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", settings.wandb_entity)
    if settings.wandb_mode != "offline" and not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb
    except ImportError:
        return None
    return wandb


@contextmanager
def wandb_run(
    job_type: str,
    *,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    enabled: bool | None = None,
) -> Iterator[Any]:
    """Yield a W&B run for a pipeline stage, or ``None`` when tracking is off.

    ``enabled=None`` means auto: on whenever W&B is configured. Callers guard logging
    with ``if run is not None`` and never fail the stage on tracking problems.
    """
    if enabled is False:
        yield None
        return
    wandb = _wandb_available()
    if wandb is None:
        if enabled is True:
            logger.warning(
                "W&B requested but unavailable (install the train extra and set "
                "WANDB_API_KEY, or WANDB_MODE=offline)."
            )
        yield None
        return

    settings = get_settings()
    run = wandb.init(
        project=settings.wandb_project,
        entity=settings.wandb_entity or None,
        job_type=job_type,
        name=name,
        config=config or {},
    )
    try:
        yield run
    finally:
        run.finish()


def log_table(run: Any, name: str, frame: Any) -> None:
    """Log a pandas DataFrame as a W&B table (no-op when run is None)."""
    if run is None:
        return
    import wandb

    run.log({name: wandb.Table(dataframe=frame)})
