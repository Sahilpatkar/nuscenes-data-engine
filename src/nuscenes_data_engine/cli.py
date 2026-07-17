"""Typer CLI entry point.

One subcommand per pipeline stage. All stages are stubs at scaffold time and log a
"not implemented yet" message; each is filled in during its corresponding build phase.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.logging import RichHandler

from nuscenes_data_engine import __version__

app = typer.Typer(
    name="nuscenes-data-engine",
    help="MLOps pipeline for AV perception on the nuScenes dataset.",
    no_args_is_help=True,
)

logger = logging.getLogger("nuscenes_data_engine")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
    version: bool = typer.Option(  # consumed by the eager callback, not the body
        False,
        "--version",
        help="Show the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Top-level options shared by all subcommands."""
    _configure_logging(verbose)


def _todo(stage: str, phase: int) -> None:
    logger.info("[%s] not implemented yet — arrives in Phase %d.", stage, phase)


@app.command()
def ingest(
    config: Path = typer.Option(
        Path("configs/data.yaml"), "--config", "-c", help="Path to data.yaml."
    ),
    limit_scenes: int | None = typer.Option(
        None, "--limit-scenes", help="Only process the first N scenes (fast dev runs)."
    ),
) -> None:
    """Phase 1: parse nuScenes into Parquet metadata + 2D box projections."""
    from nuscenes_data_engine.ingestion.ingest import run_ingestion

    summary = run_ingestion(config, limit_scenes=limit_scenes)
    logger.info(
        "Done: %d images, %d annotations -> %s",
        summary["images"],
        summary["annotations"],
        summary["samples_parquet"].rsplit("/", 1)[0],
    )


@app.command()
def validate(
    processed_dir: Path = typer.Option(
        Path("data/processed"), "--processed-dir", help="Directory with the Parquet tables."
    ),
) -> None:
    """Phase 1: run Great Expectations suites over the processed dataset."""
    from nuscenes_data_engine.validation.expectations import validate_dataset

    passed = validate_dataset(processed_dir)
    raise typer.Exit(code=0 if passed else 1)


@app.command()
def prepare_dataset(
    config: Path = typer.Option(Path("configs/train.yaml"), "--config", "-c"),
    cameras: list[str] = typer.Option(None, "--camera", help="Restrict to camera(s); repeatable."),
    limit_scenes: int | None = typer.Option(None, "--limit-scenes", help="First N scenes only."),
) -> None:
    """Phase 2: build the YOLO dataset (image symlinks + labels + data.yaml)."""
    from nuscenes_data_engine.config import get_settings, load_yaml
    from nuscenes_data_engine.training.dataset import build_yolo_dataset

    settings = get_settings()
    cfg = load_yaml(config).get("data", {})
    processed_dir = Path(cfg.get("processed_dir", settings.processed_dir))
    yolo_dir = Path(cfg.get("yolo_dir", processed_dir / "yolo"))
    _, stats = build_yolo_dataset(
        processed_dir,
        Path(settings.nuscenes_dataroot),
        yolo_dir,
        cameras=cameras or None,
        limit_scenes=limit_scenes,
    )
    logger.info("Prepared: %s", stats)


@app.command()
def train(
    config: Path = typer.Option(Path("configs/train.yaml"), "--config", "-c"),
    cameras: list[str] = typer.Option(None, "--camera", help="Restrict to camera(s); repeatable."),
    limit_scenes: int | None = typer.Option(None, "--limit-scenes", help="First N scenes only."),
    epochs: int | None = typer.Option(None, "--epochs", help="Override configured epochs."),
    device: str | None = typer.Option(None, "--device", help="Ultralytics device, e.g. 0 or cpu."),
    wandb: bool | None = typer.Option(
        None, "--wandb/--no-wandb", help="Enable/disable Weights & Biases logging."
    ),
) -> None:
    """Phase 2: run the YOLO fine-tuning pipeline (prepare -> train -> log to MLflow/W&B)."""
    from nuscenes_data_engine.training.run import run_training

    summary = run_training(
        config,
        limit_scenes=limit_scenes,
        cameras=cameras or None,
        epochs=epochs,
        device=device,
        wandb_enabled=wandb,
    )
    logger.info("Run '%s' metrics: %s", summary["run_name"], summary["metrics"])
    if summary.get("wandb_url"):
        logger.info("W&B: %s", summary["wandb_url"])


@app.command()
def evaluate() -> None:
    """Phase 3: compute mAP and condition-sliced metrics."""
    _todo("evaluate", 3)


@app.command()
def serve() -> None:
    """Phase 4: launch the FastAPI serving app."""
    _todo("serve", 4)


@app.command()
def monitor() -> None:
    """Phase 5: generate Evidently drift reports."""
    _todo("monitor", 5)


if __name__ == "__main__":
    app()
