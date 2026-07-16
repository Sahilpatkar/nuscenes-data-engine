"""Dagster job wiring the training stages into a reproducible pipeline.

Job graph: ``pull_data_version -> prepare_dataset -> train -> evaluate -> log_artifacts``.
"""

from __future__ import annotations

from typing import Any


def build_job() -> Any:
    """Construct and return the Dagster job for the training pipeline."""
    # TODO(Phase 2): define Dagster @op/@job with the five stages and return the job.
    raise NotImplementedError
