"""Smoke tests: the package imports and the CLI runs. Keeps the baseline green."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import nuscenes_data_engine
from nuscenes_data_engine.cli import app

runner = CliRunner()


def test_version_attribute() -> None:
    assert nuscenes_data_engine.__version__ == "0.1.0"


def test_cli_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # every pipeline stage should be listed
    for cmd in ("ingest", "validate", "train", "evaluate", "serve", "monitor"):
        assert cmd in result.output


def test_cli_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert nuscenes_data_engine.__version__ in result.output


def test_configs_present(configs_dir: Path) -> None:
    for name in ("data.yaml", "train.yaml", "eval.yaml"):
        assert (configs_dir / name).is_file()
