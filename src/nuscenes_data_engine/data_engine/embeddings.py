"""Batch embedding job: samples.parquet -> SigLIP vectors + thumbnails in LanceDB.

Resumable by design: frames are written per scene, and scenes whose frames are all in
the store already are skipped on restart. Filters on the availability manifest when it
exists — never trust the metadata's file references.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from nuscenes_data_engine.config import get_settings, load_yaml
from nuscenes_data_engine.data_engine import store
from nuscenes_data_engine.data_engine.embedder import Embedder

logger = logging.getLogger("nuscenes_data_engine")

_METADATA_COLUMNS = [
    "sample_data_token",
    "sample_token",
    "scene_token",
    "scene_name",
    "scene_description",
    "channel",
    "filename",
    "timestamp",
    "location",
    "is_night",
    "is_rain",
    "n_boxes",
]


def _thumbnail(img: Any, max_px: int, quality: int) -> bytes:
    scale = max_px / max(img.shape[0], img.shape[1])
    if scale < 1.0:
        img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Failed to JPEG-encode thumbnail")
    return bytes(buf.tobytes())


def run_embedding(
    config_path: Path,
    *,
    limit_scenes: int | None = None,
    rebuild: bool = False,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Embed all (available) camera keyframes into the LanceDB frames table."""
    settings = get_settings()
    cfg = load_yaml(config_path)
    emb_cfg, db_cfg = cfg.get("embedding", {}), cfg.get("lancedb", {})
    db_path, table = Path(db_cfg.get("path", "data/lancedb")), db_cfg.get("table", "frames")
    batch_size = int(emb_cfg.get("batch_size", 64))
    max_px = int(emb_cfg.get("thumbnail_max_px", 256))
    quality = int(emb_cfg.get("thumbnail_jpeg_quality", 80))
    dataroot = Path(settings.nuscenes_dataroot)

    samples = pd.read_parquet(Path(settings.processed_dir) / "samples.parquet")
    manifest_path = Path(settings.processed_dir) / "availability.parquet"
    if manifest_path.is_file():
        manifest = pd.read_parquet(manifest_path, columns=["sample_data_token", "present"])
        before = len(samples)
        samples = samples.merge(manifest[manifest["present"]], on="sample_data_token")
        logger.info("Availability manifest: %d/%d referenced frames present", len(samples), before)
    else:
        logger.warning("No availability manifest at %s — trusting metadata paths", manifest_path)

    scene_order = samples.sort_values("scene_name")["scene_token"].unique()
    if limit_scenes is not None:
        scene_order = scene_order[:limit_scenes]
        samples = samples[samples["scene_token"].isin(scene_order)]

    if embedder is None:
        from nuscenes_data_engine.data_engine.embedder import SiglipEmbedder

        embedder = SiglipEmbedder(
            emb_cfg.get("model_name", "google/siglip2-base-patch16-256"),
            device=emb_cfg.get("device", "cpu"),
        )

    if rebuild:
        store.drop_frames_table(db_path, table)
    tbl = store.open_frames_table(db_path, table, embedder.dim, create=True)
    done = store.existing_tokens(tbl)

    n_scenes, n_skipped, n_frames, n_missing = 0, 0, 0, 0
    for scene_token in scene_order:
        scene = samples[samples["scene_token"] == scene_token]
        todo = scene[~scene["sample_data_token"].isin(done)]
        if todo.empty:
            n_skipped += 1
            continue
        rows: list[dict[str, Any]] = []
        for start in range(0, len(todo), batch_size):
            batch = todo.iloc[start : start + batch_size]
            images, kept = [], []
            for record in batch.to_dict("records"):
                img = cv2.imread(str(dataroot / record["filename"]))
                if img is None:
                    n_missing += 1
                    continue
                images.append(img)
                kept.append(record)
            if not images:
                continue
            vectors = embedder.embed_images(images)
            for vec, img, record in zip(vectors, images, kept, strict=True):
                row = {k: record[k] for k in _METADATA_COLUMNS}
                row["n_boxes"] = int(row["n_boxes"])
                row["vector"] = vec.tolist()
                row["thumbnail"] = _thumbnail(img, max_px, quality)
                rows.append(row)
        if rows:
            store.add_frames(tbl, rows)
            n_frames += len(rows)
        n_scenes += 1
        if n_scenes % 25 == 0:
            logger.info("Embedded %d scenes (%d frames) so far", n_scenes, n_frames)

    store.compact(tbl)
    total = len(store.existing_tokens(tbl))
    logger.info(
        "Embedding done: %d scenes processed, %d skipped (resume), %d new frames, "
        "%d unreadable images; store now holds %d frames",
        n_scenes,
        n_skipped,
        n_frames,
        n_missing,
        total,
    )
    return {
        "scenes_processed": n_scenes,
        "scenes_skipped": n_skipped,
        "frames_added": n_frames,
        "missing_images": n_missing,
        "total_frames": total,
        "model": embedder.name,
    }
