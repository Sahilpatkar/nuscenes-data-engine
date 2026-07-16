"""Data-drift monitoring with Evidently.

Tracks input-image statistics (brightness, resolution, detection-count distribution)
against the training reference and flags drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_drift_report(reference: Path, current: Path) -> Any:
    """Build an Evidently drift report comparing current inputs to the reference.

    Args:
        reference: Feature table computed from the training data.
        current: Feature table computed from recent serving inputs.

    Returns:
        An Evidently report object (renderable to HTML/JSON).
    """
    # TODO(Phase 5): assemble Evidently metrics and return the report.
    raise NotImplementedError
