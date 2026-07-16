"""Write flattened metadata records to columnar Parquet tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet(records: list[dict[str, Any]], path: Path) -> int:
    """Write a list of flat records to a Parquet file.

    Args:
        records: Denormalized rows (e.g. from :func:`.parse.flatten`).
        path: Destination ``.parquet`` path (parent dirs are created as needed).

    Returns:
        The number of rows written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)  # type: ignore[no-untyped-call]
    return int(table.num_rows)
