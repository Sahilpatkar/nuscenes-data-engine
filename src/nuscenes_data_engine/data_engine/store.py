"""LanceDB frame store: schema + open/add/search helpers.

No ML dependencies — importable wherever the engine extra is installed. Table handles
are typed ``Any`` at this boundary (lancedb's API is dynamically typed), the same
approach serving/model.py takes with YOLO.

Vectors are L2-normalized at write time and queried with the cosine metric; the
``thumbnail`` column carries a small JPEG so search results are renderable on machines
that don't have the dataset images.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa

logger = logging.getLogger("nuscenes_data_engine")


def frames_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("sample_data_token", pa.string()),
            pa.field("sample_token", pa.string()),
            pa.field("scene_token", pa.string()),
            pa.field("scene_name", pa.string()),
            pa.field("scene_description", pa.string()),
            pa.field("channel", pa.string()),
            pa.field("filename", pa.string()),
            pa.field("timestamp", pa.int64()),
            pa.field("location", pa.string()),
            pa.field("is_night", pa.bool_()),
            pa.field("is_rain", pa.bool_()),
            pa.field("n_boxes", pa.int32()),
            pa.field("thumbnail", pa.binary()),
        ]
    )


def open_frames_table(db_path: Path, table: str, dim: int, *, create: bool = False) -> Any:
    """Open (or with ``create=True``, create) the frames table."""
    if not create and not db_path.is_dir():
        # connect() would create the directory as a side effect — don't, when reading.
        raise FileNotFoundError(f"LanceDB store not found at {db_path}")
    db = lancedb.connect(str(db_path))
    if table in db.table_names():
        return db.open_table(table)
    if not create:
        raise FileNotFoundError(f"LanceDB table '{table}' not found under {db_path}")
    return db.create_table(table, schema=frames_schema(dim))


def drop_frames_table(db_path: Path, table: str) -> None:
    db = lancedb.connect(str(db_path))
    if table in db.table_names():
        db.drop_table(table)


def existing_tokens(tbl: Any) -> set[str]:
    """All sample_data_tokens already embedded (reads one column, not the vectors)."""
    n = tbl.count_rows()
    if n == 0:
        return set()
    frames = tbl.search().select(["sample_data_token"]).limit(n).to_pandas()
    return set(frames["sample_data_token"])


def add_frames(tbl: Any, rows: list[dict[str, Any]]) -> None:
    tbl.add(rows)


def vector_for(tbl: Any, token: str) -> np.ndarray[Any, Any] | None:
    """The stored vector for one frame, or None if the token is unknown."""
    frames = (
        tbl.search().where(f"sample_data_token = '{token}'").select(["vector"]).limit(1).to_pandas()
    )
    if frames.empty:
        return None
    return np.asarray(frames.iloc[0]["vector"], dtype=np.float32)


def search_frames(
    tbl: Any, vec: np.ndarray[Any, Any], k: int, *, exclude_token: str | None = None
) -> pd.DataFrame:
    """Cosine top-k over the store; returns rows with a ``score`` column (1 - distance)."""
    query = tbl.search(vec.tolist()).metric("cosine").limit(k + (1 if exclude_token else 0))
    if exclude_token is not None:
        query = query.where(f"sample_data_token != '{exclude_token}'")
    frames = query.to_pandas()
    frames["score"] = 1.0 - frames["_distance"]
    return frames.drop(columns=["_distance"]).head(k)


def compact(tbl: Any) -> None:
    """Merge small fragments after incremental writes (keeps rsync sane)."""
    try:
        tbl.optimize()
    except Exception:  # older lancedb spells it differently; compaction is best-effort
        logger.warning("LanceDB optimize() unavailable; skipping compaction", exc_info=True)
