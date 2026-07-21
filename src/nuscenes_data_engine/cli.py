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
    rebuild: bool = typer.Option(False, "--rebuild", help="Force rebuild even if up to date."),
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
        force=rebuild,
    )
    logger.info("Prepared: %s", stats)


@app.command()
def train(
    config: Path = typer.Option(Path("configs/train.yaml"), "--config", "-c"),
    cameras: list[str] = typer.Option(None, "--camera", help="Restrict to camera(s); repeatable."),
    limit_scenes: int | None = typer.Option(None, "--limit-scenes", help="First N scenes only."),
    epochs: int | None = typer.Option(None, "--epochs", help="Override configured epochs."),
    batch: int | None = typer.Option(
        None, "--batch", help="Override batch size (total across GPUs)."
    ),
    model: str | None = typer.Option(None, "--model", help="Override weights, e.g. yolov8s.pt."),
    imgsz: int | None = typer.Option(None, "--imgsz", help="Override image size, e.g. 960."),
    device: str | None = typer.Option(None, "--device", help="Ultralytics device, e.g. 0 or 0,1."),
    wandb: bool | None = typer.Option(
        None, "--wandb/--no-wandb", help="Enable/disable Weights & Biases logging."
    ),
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Force YOLO-dataset rebuild even if up to date."
    ),
) -> None:
    """Phase 2: run the YOLO fine-tuning pipeline (prepare -> train -> log to MLflow/W&B)."""
    from nuscenes_data_engine.config import get_settings
    from nuscenes_data_engine.training.run import run_training

    summary = run_training(
        config,
        limit_scenes=limit_scenes,
        cameras=cameras or None,
        epochs=epochs,
        batch=batch,
        model=model,
        imgsz=imgsz,
        device=device,
        wandb_enabled=wandb,
        force_rebuild=rebuild,
    )
    logger.info("Run '%s' metrics: %s", summary["run_name"], summary["metrics"])
    if summary.get("wandb"):
        logger.info(
            "W&B: logged to project '%s' (see run URL above).", get_settings().wandb_project
        )


@app.command()
def evaluate(
    config: Path = typer.Option(Path("configs/eval.yaml"), "--config", "-c"),
    train_config: Path = typer.Option(Path("configs/train.yaml"), "--train-config"),
    weights: Path | None = typer.Option(None, "--weights", help="best.pt (default: latest run)."),
    device: str = typer.Option("0", "--device", help="Ultralytics device."),
    imgsz: int | None = typer.Option(
        None, "--imgsz", help="Eval image size (default: train.yaml)."
    ),
    register: bool = typer.Option(
        False, "--register", help="Register + promote (staging->production) in MLflow."
    ),
) -> None:
    """Phase 3: compute overall + condition-sliced mAP and (optionally) promote the model."""
    from nuscenes_data_engine.evaluation.evaluate import run_evaluation

    report = run_evaluation(
        config, train_config, weights=weights, device=device, imgsz=imgsz, register=register
    )
    o = report["overall"]
    logger.info(
        "== Overall == mAP50=%.3f mAP50-95=%.3f P=%.3f R=%.3f",
        o["mAP50"],
        o["mAP50-95"],
        o["precision"],
        o["recall"],
    )
    for name, m in report["slices"].items():
        logger.info(
            "== %-18s == mAP50=%.3f mAP50-95=%.3f (%d imgs)",
            name,
            m["mAP50"],
            m["mAP50-95"],
            m["n_images"],
        )
    logger.info("Promotion gate %s", "PASSED" if report["passed"] else "NOT MET")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev only)."),
) -> None:
    """Phase 4: launch the FastAPI serving app."""
    import uvicorn

    uvicorn.run("nuscenes_data_engine.serving.app:app", host=host, port=port, reload=reload)


monitor_app = typer.Typer(no_args_is_help=True, help="Phase 5: drift monitoring.")
app.add_typer(monitor_app, name="monitor")


@monitor_app.command("build-reference")
def monitor_build_reference(
    config: Path = typer.Option(Path("configs/monitoring.yaml"), "--config", "-c"),
    out: Path | None = typer.Option(None, "--out", help="Output parquet (default: config)."),
    condition: str = typer.Option("all", "--condition", help="all | day | night."),
    sample_images: int | None = typer.Option(None, "--sample-images", help="Override config."),
    no_images: bool = typer.Option(
        False, "--no-images", help="Metadata only (no dataset access); brightness = NaN."
    ),
) -> None:
    """Phase 5: build a drift-feature table from the processed dataset."""
    from nuscenes_data_engine.config import get_settings, load_yaml
    from nuscenes_data_engine.monitoring.features import build_reference

    settings = get_settings()
    cfg = load_yaml(config).get("reference", {})
    features = build_reference(
        Path(settings.processed_dir),
        None if no_images else Path(settings.nuscenes_dataroot),
        out or Path(cfg.get("out_path", "data/processed/monitoring_reference.parquet")),
        condition=condition,
        sample_images=sample_images or cfg.get("sample_images", 2000),
        seed=cfg.get("seed", 0),
    )
    logger.info(
        "%d rows (%s); brightness mean %.1f",
        len(features),
        condition,
        features["brightness"].mean(),
    )


@monitor_app.command("report")
def monitor_report(
    config: Path = typer.Option(Path("configs/monitoring.yaml"), "--config", "-c"),
    reference: Path | None = typer.Option(None, "--reference", help="Reference feature parquet."),
    current: Path | None = typer.Option(
        None, "--current", help="Feature parquet or serving JSONL (default: config serving_log)."
    ),
    simulate_night: bool = typer.Option(
        False, "--simulate-night", help="Use a night-slice of samples.parquet as current."
    ),
    out_dir: Path | None = typer.Option(None, "--out-dir", help="Report directory."),
) -> None:
    """Phase 5: generate an Evidently drift report (reference vs current)."""
    from nuscenes_data_engine.config import get_settings, load_yaml
    from nuscenes_data_engine.monitoring.drift import (
        build_drift_report,
        save_drift_report,
        summarize_drift,
    )
    from nuscenes_data_engine.monitoring.features import build_reference

    cfg = load_yaml(config)
    reference = reference or Path(cfg["reference"]["out_path"])
    out_dir = out_dir or Path(cfg["report"]["out_dir"])
    if simulate_night:
        settings = get_settings()
        current = out_dir / "night_current.parquet"
        build_reference(
            Path(settings.processed_dir), None, current, condition="night", sample_images=2000
        )
    current = current or Path(cfg["current"]["serving_log"])

    snapshot = build_drift_report(
        reference, current, drift_share=cfg.get("drift", {}).get("drift_share", 0.25)
    )
    summary = summarize_drift(snapshot)
    html_path, _ = save_drift_report(snapshot, out_dir)
    for column, verdict in summary["columns"].items():
        logger.info(
            "%-10s %s (score %.3g)",
            column,
            "DRIFT" if verdict.get("drift_detected") else "ok",
            verdict.get("score", float("nan")),
        )
    logger.info(
        "%s (%d/%d columns drifted) — report: %s",
        "DATASET DRIFT DETECTED" if summary["dataset_drift"] else "No dataset drift",
        summary["n_drifted"],
        len(summary["columns"]),
        html_path,
    )


if __name__ == "__main__":
    app()
