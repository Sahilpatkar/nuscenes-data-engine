"""MLflow Model Registry: register a model version and run staging -> production promotion.

Uses registry **aliases** (`staging`, `production`) — the current MLflow approach that
replaces the deprecated stage transitions. Every evaluated model is registered and
aliased `staging`; only models that clear the promotion gate also get `production`.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

logger = logging.getLogger("nuscenes_data_engine")


def register_and_promote(
    mlflow: Any,
    run_id: str,
    model_name: str,
    artifact_path: str,
    *,
    passed: bool,
) -> dict[str, Any]:
    """Register ``runs:/<run_id>/<artifact_path>`` and set registry aliases.

    Args:
        mlflow: The imported mlflow module (tracking URI already configured).
        run_id: MLflow run that logged the model weights.
        model_name: Registered-model name.
        artifact_path: Artifact sub-path holding the weights (e.g. ``"weights"``).
        passed: Whether the model cleared the promotion gate.

    Returns:
        ``{version, aliases}`` describing the new model version.
    """
    client = mlflow.MlflowClient()
    with contextlib.suppress(Exception):  # no-op if the registered model already exists
        client.create_registered_model(model_name)

    # MLflow 3.x's `register_model(runs:/...)` expects a logged-model flavor; we log the
    # weights as a plain artifact, so create the version directly from its artifact URI.
    source = f"{client.get_run(run_id).info.artifact_uri}/{artifact_path}"
    version = client.create_model_version(name=model_name, source=source, run_id=run_id).version

    aliases = ["staging"]
    client.set_registered_model_alias(model_name, "staging", version)
    if passed:
        client.set_registered_model_alias(model_name, "production", version)
        aliases.append("production")

    logger.info(
        "Registered %s v%s with aliases %s (%s)",
        model_name,
        version,
        aliases,
        "PROMOTED" if passed else "staging only — gate not met",
    )
    return {"version": version, "aliases": aliases}
