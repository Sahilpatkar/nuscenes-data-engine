"""pytest configuration and shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def configs_dir(repo_root: Path) -> Path:
    """Path to the ``configs/`` directory."""
    return repo_root / "configs"
