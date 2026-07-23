"""Tests for the W&B tracking helper (stubbed wandb; no network, no wandb install)."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from nuscenes_data_engine.tracking import wandb_run


class _StubRun:
    def __init__(self) -> None:
        self.logged: list[dict[str, Any]] = []
        self.finished = False

    def log(self, payload: dict[str, Any]) -> None:
        self.logged.append(payload)

    def finish(self) -> None:
        self.finished = True


@pytest.fixture()
def stub_wandb(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("wandb")
    module.init_calls = []  # type: ignore[attr-defined]

    def init(**kwargs: Any) -> _StubRun:
        module.init_calls.append(kwargs)  # type: ignore[attr-defined]
        return _StubRun()

    module.init = init  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


class TestWandbRun:
    def test_disabled_explicitly_yields_none(self, stub_wandb: types.ModuleType) -> None:
        with wandb_run("embed", enabled=False) as run:
            assert run is None
        assert not stub_wandb.init_calls  # type: ignore[attr-defined]

    def test_noop_without_configuration(
        self, stub_wandb: types.ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in ("WANDB_API_KEY", "WANDB_MODE", "WANDB_ENTITY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("WANDB_MODE", "online")
        monkeypatch.setenv("WANDB_API_KEY", "")
        with wandb_run("embed") as run:
            assert run is None
        assert not stub_wandb.init_calls  # type: ignore[attr-defined]

    def test_runs_when_configured(
        self, stub_wandb: types.ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        monkeypatch.setenv("WANDB_MODE", "online")
        with wandb_run("embed", config={"limit_scenes": 5}) as run:
            assert run is not None
            run.log({"frames_added": 12})
        calls = stub_wandb.init_calls  # type: ignore[attr-defined]
        assert len(calls) == 1
        assert calls[0]["job_type"] == "embed"
        assert calls[0]["config"] == {"limit_scenes": 5}
        assert run.logged == [{"frames_added": 12}]
        assert run.finished  # finish() called even without errors

    def test_offline_mode_needs_no_key(
        self, stub_wandb: types.ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "")
        monkeypatch.setenv("WANDB_MODE", "offline")
        with wandb_run("monitor-drift") as run:
            assert run is not None

    def test_disabled_mode_never_imports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WANDB_MODE", "disabled")
        monkeypatch.setitem(sys.modules, "wandb", None)  # import would raise
        with wandb_run("embed") as run:
            assert run is None

    def test_finish_runs_on_exception(
        self, stub_wandb: types.ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        monkeypatch.setenv("WANDB_MODE", "online")
        captured: list[_StubRun] = []
        with pytest.raises(RuntimeError), wandb_run("embed") as run:
            assert run is not None
            captured.append(run)
            raise RuntimeError("stage failed")
        assert captured[0].finished
