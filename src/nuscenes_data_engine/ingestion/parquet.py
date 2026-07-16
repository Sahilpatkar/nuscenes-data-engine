"""Write flattened metadata records to columnar Parquet tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_parquet(records: list[dict[str, Any]], path: Path) -> None:
    """Write a list of flat records to a Parquet file.

    Args:
        records: Denormalized rows (e.g. from :mod:`.parse`).
        path: Destination ``.parquet`` path (parent dirs created as needed).
    """
    # TODO(Phase 1): build a pyarrow.Table (or via pandas) and write with pyarrow.parquet.
    raise NotImplementedError
