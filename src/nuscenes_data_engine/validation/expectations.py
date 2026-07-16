"""Great Expectations suites for the processed nuScenes metadata.

Checks: schema conformance, null constraints, 2D boxes within image bounds,
valid category labels, and per-scene sample counts.
"""

from __future__ import annotations

from pathlib import Path


def build_annotation_suite() -> object:
    """Build the expectation suite for the flattened annotations table."""
    # TODO(Phase 1): define GE expectations (column types, non-null, box bounds,
    # category set membership).
    raise NotImplementedError


def validate_dataset(processed_dir: Path) -> bool:
    """Run all suites against the processed dataset; return True if all pass."""
    # TODO(Phase 1): run a GE checkpoint over the Parquet tables.
    raise NotImplementedError
