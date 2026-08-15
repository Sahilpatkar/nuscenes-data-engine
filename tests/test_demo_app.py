"""The demo app must stay deployable on Streamlit Cloud: no src/, no heavy deps."""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN = ("nuscenes_data_engine", "requests", "torch", "lancedb", "neo4j", "duckdb")
DEMO_DIR = Path(__file__).resolve().parents[1] / "app" / "demo"


def test_demo_app_never_imports_the_backend() -> None:
    # Static ast.Import/ImportFrom check only: dynamic imports (importlib, __import__)
    # slip past this. It's a contributor guardrail against the obvious mistake, not a
    # sandbox — don't rely on it to block a determined attempt to reach the backend.
    offenders = []
    for path in sorted(DEMO_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in FORBIDDEN:
                    offenders.append(f"{path.name}: {name}")
    assert not offenders, f"demo app imports forbidden modules: {offenders}"


def test_demo_requirements_stay_minimal() -> None:
    lines = [
        line.split("==")[0].split(">=")[0].strip()
        for line in (DEMO_DIR / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert set(lines) <= {"streamlit", "pandas", "pyarrow", "pillow"}
