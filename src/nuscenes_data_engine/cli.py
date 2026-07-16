"""Typer CLI entry point.

One subcommand per pipeline stage. All stages are stubs at scaffold time and log a
"not implemented yet" message; each is filled in during its corresponding build phase.
"""

from __future__ import annotations

import logging

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
def ingest() -> None:
    """Phase 1: parse nuScenes into Parquet metadata + 2D box projections."""
    _todo("ingest", 1)


@app.command()
def validate() -> None:
    """Phase 1: run Great Expectations suites over the processed dataset."""
    _todo("validate", 1)


@app.command()
def train() -> None:
    """Phase 2: run the orchestrated YOLO fine-tuning pipeline."""
    _todo("train", 2)


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
