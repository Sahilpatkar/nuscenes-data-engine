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
    wandb: bool | None = typer.Option(None, "--wandb/--no-wandb", help="W&B run logging."),
) -> None:
    """Phase 3: compute overall + condition-sliced mAP and (optionally) promote the model."""
    from nuscenes_data_engine.evaluation.evaluate import run_evaluation
    from nuscenes_data_engine.tracking import wandb_run

    with wandb_run(
        "evaluate",
        config={"weights": str(weights) if weights else None, "imgsz": imgsz},
        enabled=wandb,
    ) as run:
        report = run_evaluation(
            config, train_config, weights=weights, device=device, imgsz=imgsz, register=register
        )
        if run is not None:
            run.log({f"overall_{k}": v for k, v in report["overall"].items()})
            for name, metrics in report["slices"].items():
                tag = name.split("/")[-1]
                run.log({f"{tag}_{k}": v for k, v in metrics.items()})
            run.summary["promotion_gate_passed"] = report["passed"]
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


@app.command()
def manifest(
    config: Path = typer.Option(Path("configs/engine.yaml"), "--config", "-c"),
    out: Path | None = typer.Option(None, "--out", help="Output parquet (default: config)."),
) -> None:
    """Phase 6: cross-check referenced sensor files against the filesystem."""
    from nuscenes_data_engine.config import get_settings, load_yaml
    from nuscenes_data_engine.validation.manifest import (
        build_manifest,
        camera_keyframes_complete,
        summarize_manifest,
    )

    settings = get_settings()
    cfg = load_yaml(config).get("manifest", {})
    out = out or Path(cfg.get("out_path", "data/processed/availability.parquet"))
    result = build_manifest(Path(settings.nuscenes_dataroot), settings.nuscenes_version, out)
    for row in summarize_manifest(result).to_dict("records"):
        logger.info(
            "%-18s keyframe=%-5s %8d referenced %8d present",
            row["channel"],
            row["is_key_frame"],
            row["n_referenced"],
            row["n_present"],
        )
    if not camera_keyframes_complete(result):
        logger.error("Camera keyframes (the working set) are incomplete!")
        raise typer.Exit(code=1)


@app.command()
def embed(
    config: Path = typer.Option(Path("configs/engine.yaml"), "--config", "-c"),
    limit_scenes: int | None = typer.Option(None, "--limit-scenes", help="First N scenes only."),
    rebuild: bool = typer.Option(False, "--rebuild", help="Drop and rebuild the vector store."),
    wandb: bool | None = typer.Option(None, "--wandb/--no-wandb", help="W&B run logging."),
) -> None:
    """Phase 6a: embed camera keyframes into the LanceDB frame store."""
    import time

    from nuscenes_data_engine.data_engine.embeddings import run_embedding
    from nuscenes_data_engine.tracking import wandb_run

    with wandb_run(
        "embed", config={"limit_scenes": limit_scenes, "rebuild": rebuild}, enabled=wandb
    ) as run:
        start = time.perf_counter()
        summary = run_embedding(config, limit_scenes=limit_scenes, rebuild=rebuild)
        if run is not None:
            duration = time.perf_counter() - start
            run.log(
                {
                    **{k: v for k, v in summary.items() if isinstance(v, int | float)},
                    "duration_s": duration,
                    "frames_per_s": summary["frames_added"] / duration if duration else 0,
                }
            )
            run.config.update({"model": summary["model"]})
    logger.info("Embed summary: %s", summary)


@app.command()
def search(
    query: str = typer.Argument("", help="Text query (omit when using --image/--similar)."),
    config: Path = typer.Option(Path("configs/engine.yaml"), "--config", "-c"),
    k: int = typer.Option(5, "-k", help="Number of results."),
    image: Path | None = typer.Option(None, "--image", help="Image file query."),
    similar: str | None = typer.Option(None, "--similar", help="sample_data_token query."),
) -> None:
    """Phase 6a: semantic frame search (text, image, or similar-to-frame)."""
    from nuscenes_data_engine.config import load_yaml
    from nuscenes_data_engine.data_engine.search import SearchEngine

    cfg = load_yaml(config)
    engine = SearchEngine(
        Path(cfg["lancedb"]["path"]),
        cfg["lancedb"]["table"],
        cfg["embedding"]["model_name"],
        device="cpu",
    )
    if similar is not None:
        results = engine.search_similar(similar, k)
    elif image is not None:
        results = engine.search_image(image.read_bytes(), k)
    elif query:
        results = engine.search_text(query, k)
    else:
        raise typer.BadParameter("Provide a text query, --image, or --similar.")
    for row in results:
        logger.info(
            "%.3f  %s  %-14s %-24s %s",
            row["score"],
            row["sample_data_token"],
            row["channel"],
            row["scene_name"],
            row["scene_description"][:60],
        )


@app.command()
def query(
    sql: str = typer.Argument(..., help="DuckDB SQL over samples/annotations/availability."),
    processed_dir: Path = typer.Option(Path("data/processed"), "--processed-dir"),
) -> None:
    """Phase 6: ad-hoc DuckDB analytics over the processed Parquet tables."""
    import duckdb

    con = duckdb.connect()
    for name in ("samples", "annotations", "availability"):
        path = processed_dir / f"{name}.parquet"
        if path.is_file():
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
    print(con.sql(sql))


autolabel_app = typer.Typer(no_args_is_help=True, help="Phase 6b: VLM auto-labeling.")
app.add_typer(autolabel_app, name="autolabel")


@autolabel_app.command("sample")
def autolabel_sample(
    config: Path = typer.Option(Path("configs/autolabel.yaml"), "--config", "-c"),
) -> None:
    """Draw the stratified labeling sample (run where data/processed lives)."""
    from nuscenes_data_engine.config import get_settings, load_yaml
    from nuscenes_data_engine.data_engine.autolabel.sampling import build_sample

    cfg = load_yaml(config)
    sample = build_sample(Path(get_settings().processed_dir), cfg.get("sample", {}))
    out = Path(cfg.get("state", {}).get("dir", "data/autolabel")) / "sample.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out, index=False)
    logger.info(
        "Sampled %d frames (%d in comparison subset) -> %s",
        len(sample),
        int(sample["in_opus_subset"].sum()),
        out,
    )


@autolabel_app.command("submit")
def autolabel_submit(
    config: Path = typer.Option(Path("configs/autolabel.yaml"), "--config", "-c"),
    yes: bool = typer.Option(False, "--yes", help="Confirm the estimated spend."),
    retry_missing: bool = typer.Option(
        False, "--retry-missing", help="Only frames without a terminal result."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Size and price only; no API calls."),
    provider: str | None = typer.Option(
        None, "--provider", help="anthropic (paid Batch API) | local (vLLM server, free)."
    ),
) -> None:
    """Phase 6b: run the labeling — Claude Batch API (paid, needs --yes) or local vLLM."""
    from nuscenes_data_engine.data_engine.autolabel.batch import run_submit
    from nuscenes_data_engine.tracking import wandb_run

    with wandb_run(
        "autolabel-submit",
        config={"retry_missing": retry_missing, "provider": provider},
        enabled=False if dry_run else None,
    ) as run:
        summary = run_submit(
            config, yes=yes, retry_missing=retry_missing, dry_run=dry_run, provider=provider
        )
        if run is not None:
            run.log(
                {"submitted": summary["submitted"], "estimated_cost_usd": summary["estimated_cost"]}
            )


@autolabel_app.command("status")
def autolabel_status(
    config: Path = typer.Option(Path("configs/autolabel.yaml"), "--config", "-c"),
    provider: str | None = typer.Option(None, "--provider", help="anthropic | local."),
) -> None:
    """Poll the processing status of submitted batches."""
    from nuscenes_data_engine.data_engine.autolabel.batch import run_status

    run_status(config, provider=provider)


@autolabel_app.command("collect")
def autolabel_collect(
    config: Path = typer.Option(Path("configs/autolabel.yaml"), "--config", "-c"),
    provider: str | None = typer.Option(None, "--provider", help="anthropic | local."),
) -> None:
    """Download ended batches and rebuild the validated labels table."""
    from nuscenes_data_engine.data_engine.autolabel.batch import run_collect
    from nuscenes_data_engine.tracking import wandb_run

    with wandb_run("autolabel-collect") as run:
        labels = run_collect(config, provider=provider)
        if run is not None and not labels.empty:
            counts = labels.groupby(["model", "parse_status"]).size()
            run.log({f"{model}/{status}": int(n) for (model, status), n in counts.items()})


@autolabel_app.command("eval")
def autolabel_eval(
    config: Path = typer.Option(Path("configs/autolabel.yaml"), "--config", "-c"),
) -> None:
    """Evaluate collected labels against nuScenes ground truth."""
    from nuscenes_data_engine.data_engine.autolabel.evaluate import run_eval
    from nuscenes_data_engine.tracking import wandb_run

    with wandb_run("autolabel-eval") as run:
        summary = run_eval(config)
        if run is not None:
            for model, metrics in summary.items():
                if isinstance(metrics, dict):
                    run.log(
                        {
                            f"{model}/{k}": v
                            for k, v in metrics.items()
                            if isinstance(v, int | float)
                        }
                    )
    logger.info("Eval summary: %s", summary)


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

    from nuscenes_data_engine.tracking import wandb_run

    with wandb_run("monitor-drift", config={"reference": str(reference)}) as run:
        if run is not None:
            metrics: dict[str, float] = {
                "dataset_drift": float(summary["dataset_drift"]),
                "n_drifted": float(summary["n_drifted"]),
                "share_drifted": float(summary["share_drifted"]),
            }
            for column, verdict in summary["columns"].items():
                metrics[f"{column}_drift"] = float(verdict.get("drift_detected", False))
                metrics[f"{column}_score"] = float(verdict.get("score", 0.0))
            run.log(metrics)
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
