# CAN-bus Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keyframe-aligned ego dynamics (speed/steering/brake/throttle/accel + `is_hard_braking`) from the nuScenes CAN-bus expansion, flowing Parquet → DuckDB → Neo4j `EgoPose` properties → chat, per the approved spec `docs/superpowers/specs/2026-08-05-canbus-ingestion-design.md`.

**Architecture:** Mirrors Phase B's `ingestion/geometry.py` exactly: pure alignment helpers (TDD, torch-free CI) + a `flatten_canbus` walking keyframes with the devkit's `NuScenesCanBus`, writing one ~34k-row `canbus.parquet`; then a DuckDB view, an idempotent Neo4j `SET` pass onto existing `EgoPose` nodes, and guard-prompt wiring. Ingestion runs on TRINITY (dataset + devkit); everything downstream runs on the infra Mac.

**Tech Stack:** Python 3.11, nuscenes-devkit (`NuScenesCanBus`), pandas/pyarrow, DuckDB, Neo4j 5 + Cypher, Typer CLI, pytest (torch-free), uv. Strict mypy + ruff; guards raise `ValueError`, never `assert`.

**Working branch:** `canbus-ingestion` (exists; spec committed as 917da34).

**Conventions:** Repo root `/Users/sahilpatkar/Curosr_repos/nuscenes-data-engine`. Tests: `uv run pytest tests/test_canbus.py -v` (new file) and the full `uv run pytest -q`. Lint/type: `uv run ruff check src tests && uv run mypy src`. Reference files to mirror: `src/nuscenes_data_engine/ingestion/geometry.py` (module layout), `tests/test_geometry.py` (test style), `src/nuscenes_data_engine/data_engine/graph/{model,builder,schema,guard}.py`, `src/nuscenes_data_engine/data_engine/chat/catalog.py`.

---

### Task 1: Pure alignment helpers

**Files:**
- Create: `src/nuscenes_data_engine/ingestion/canbus.py`
- Create: `tests/test_canbus.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_canbus.py`:

```python
"""Tests for CAN-bus ingestion: pure alignment helpers + fake-devkit flatten."""

from __future__ import annotations

import math
from typing import Any

import pytest

from nuscenes_data_engine.ingestion.canbus import (
    accel_window,
    nearest_message,
    scene_number,
)

# ---------------------------------------------------------------------------
# pure helpers — run in torch-free CI
# ---------------------------------------------------------------------------


def _msgs(*utimes: int) -> list[dict[str, Any]]:
    return [{"utime": u, "vehicle_speed": float(i)} for i, u in enumerate(utimes)]


def test_nearest_message_picks_closest_within_tolerance() -> None:
    messages = _msgs(1_000_000, 1_500_000, 2_100_000)
    msg = nearest_message(messages, t_us=1_600_000, tolerance_us=600_000)
    assert msg is not None and msg["utime"] == 1_500_000


def test_nearest_message_none_when_outside_tolerance() -> None:
    messages = _msgs(1_000_000)
    assert nearest_message(messages, t_us=2_000_000, tolerance_us=600_000) is None
    assert nearest_message([], t_us=0, tolerance_us=600_000) is None


def test_nearest_message_exact_tolerance_boundary_included() -> None:
    messages = _msgs(1_000_000)
    assert nearest_message(messages, t_us=1_600_000, tolerance_us=600_000) is not None


def test_accel_window_stats_and_nearest_speed() -> None:
    pose = [
        {"utime": 900_000, "accel": [-1.0, 0.0, 9.8], "vel": [10.0, 0.0, 0.0]},
        {"utime": 1_000_000, "accel": [-4.2, 0.1, 9.8], "vel": [8.0, 6.0, 0.0]},
        {"utime": 1_100_000, "accel": [0.5, 0.0, 9.8], "vel": [7.0, 0.0, 0.0]},
        {"utime": 9_000_000, "accel": [-9.0, 0.0, 9.8], "vel": [0.0, 0.0, 0.0]},  # outside
    ]
    accel_min, accel_max, speed = accel_window(pose, t_us=1_010_000, window_us=500_000)
    assert accel_min == pytest.approx(-4.2)
    assert accel_max == pytest.approx(0.5)
    assert speed == pytest.approx(10.0)  # nearest msg is utime=1_000_000 -> |[8,6,0]| = 10


def test_accel_window_empty_returns_nones() -> None:
    assert accel_window([], t_us=0, window_us=500_000) == (None, None, None)
    far = [{"utime": 9_000_000, "accel": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0]}]
    assert accel_window(far, t_us=0, window_us=500_000) == (None, None, None)


def test_scene_number() -> None:
    assert scene_number("scene-0161") == 161
    assert scene_number("scene-1094") == 1094
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_canbus.py -v`
Expected: FAIL at import — `ModuleNotFoundError`/`ImportError` for `nuscenes_data_engine.ingestion.canbus`.

- [ ] **Step 3: Implement the helpers**

Create `src/nuscenes_data_engine/ingestion/canbus.py`:

```python
"""CAN-bus ingestion: keyframe-aligned ego dynamics from the nuScenes can_bus expansion.

The pure helpers (nearest-message selection, windowed acceleration stats) carry the
alignment correctness and are unit-tested without the devkit. ``flatten_canbus`` walks
the loaded ``NuScenes`` handle plus a ``NuScenesCanBus`` handle to emit one record per
keyframe — JSON metadata only, so it runs where the dataset lives (TRINITY). Scenes in
the devkit's ``can_blacklist`` (or with missing CAN files) yield ``has_canbus=False``
rows with null signals; no keyframe is ever dropped.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger("nuscenes_data_engine")


def nearest_message(
    messages: list[dict[str, Any]], t_us: int, tolerance_us: int
) -> dict[str, Any] | None:
    """The message whose ``utime`` is closest to ``t_us``, or None if none within tolerance."""
    best: dict[str, Any] | None = None
    best_dt = tolerance_us + 1
    for msg in messages:
        dt = abs(int(msg["utime"]) - t_us)
        if dt < best_dt:
            best, best_dt = msg, dt
    return best


def accel_window(
    pose_msgs: list[dict[str, Any]], t_us: int, window_us: int
) -> tuple[float | None, float | None, float | None]:
    """(accel_long_min, accel_long_max, speed_mps at the nearest pose) over ``t±window``.

    Longitudinal acceleration is ``accel[0]`` (vehicle-frame x-forward); speed is the
    norm of the nearest in-window pose message's ``vel``. Empty window -> three Nones.
    """
    in_window = [m for m in pose_msgs if abs(int(m["utime"]) - t_us) <= window_us]
    if not in_window:
        return None, None, None
    longs = [float(m["accel"][0]) for m in in_window]
    nearest = min(in_window, key=lambda m: abs(int(m["utime"]) - t_us))
    speed = math.sqrt(sum(float(v) * float(v) for v in nearest["vel"]))
    return min(longs), max(longs), speed


def scene_number(scene_name: str) -> int:
    """``'scene-0161'`` -> ``161`` (the devkit blacklist holds scene numbers as ints)."""
    return int(scene_name.rsplit("-", 1)[1])
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_canbus.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuscenes_data_engine/ingestion/canbus.py tests/test_canbus.py
git commit -m "canbus: pure alignment helpers (nearest message, accel window)"
```

---

### Task 2: `flatten_canbus` against a fake devkit

**Files:**
- Modify: `src/nuscenes_data_engine/ingestion/canbus.py`
- Test: `tests/test_canbus.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_canbus.py`:

```python
# ---------------------------------------------------------------------------
# flatten_canbus — fake nusc + fake can handles (no devkit needed)
# ---------------------------------------------------------------------------


class _FakeNusc:
    """Two-keyframe scenes with the minimal fields flatten_canbus touches."""

    def __init__(self, scenes: list[dict[str, Any]]) -> None:
        self.scene = scenes
        self._samples: dict[str, dict[str, Any]] = {}
        self._logs = {"log-1": {"location": "singapore-onenorth"}}
        for scene in scenes:
            base = scene["name"]
            for i in range(2):
                token = f"{base}-kf{i}"
                self._samples[token] = {
                    "token": token,
                    "timestamp": 1_000_000 + i * 2_000_000,  # kf1 sits at t=3.0s
                    "next": f"{base}-kf{i + 1}" if i == 0 else "",
                }

    def get(self, table: str, token: str) -> dict[str, Any]:
        if table == "sample":
            return self._samples[token]
        if table == "log":
            return self._logs[token]
        raise KeyError(table)


def _scene(name: str) -> dict[str, Any]:
    return {
        "token": f"tok-{name}",
        "name": name,
        "log_token": "log-1",
        "description": "Night, heavy rain",
        "first_sample_token": f"{name}-kf0",
    }


class _FakeCan:
    can_blacklist = [161]

    def __init__(self, monitor: list[dict[str, Any]], pose: list[dict[str, Any]],
                 missing: set[str] = frozenset()) -> None:
        self._monitor, self._pose, self._missing = monitor, pose, missing

    def get_messages(self, scene_name: str, message: str) -> list[dict[str, Any]]:
        if scene_name in self._missing:
            raise Exception(f"no CAN data for {scene_name}")
        return self._monitor if message == "vehicle_monitor" else self._pose


_MONITOR = [{
    "utime": 1_050_000, "vehicle_speed": 14.7, "steering": 191.9, "steering_speed": 65.7,
    "brake": 20, "brake_switch": 1, "throttle": 55, "yaw_rate": 18.9,
    "left_signal": 0, "right_signal": 1,
}]
_POSE = [
    {"utime": 1_000_000, "accel": [-4.0, 0.0, 9.8], "vel": [3.0, 4.0, 0.0]},
    {"utime": 1_400_000, "accel": [1.0, 0.0, 9.8], "vel": [5.0, 0.0, 0.0]},
    {"utime": 2_900_000, "accel": [1.0, 0.0, 9.8], "vel": [6.0, 0.0, 0.0]},
]


def test_flatten_canbus_aligns_signals_and_flags() -> None:
    from nuscenes_data_engine.ingestion.canbus import flatten_canbus

    rows = flatten_canbus(
        _FakeNusc([_scene("scene-0001")]), _FakeCan(_MONITOR, _POSE),
        window_s=0.5, hard_braking_mps2=-3.0, monitor_tolerance_ms=600,
    )
    assert len(rows) == 2
    first, second = rows
    assert first["sample_token"] == "scene-0001-kf0" and first["has_canbus"] is True
    assert first["can_speed_kmh"] == pytest.approx(14.7)
    assert first["brake_pedal"] == 20 and first["right_signal"] == 1
    assert first["accel_long_min_mps2"] == pytest.approx(-4.0)
    assert first["accel_long_max_mps2"] == pytest.approx(1.0)
    assert first["is_hard_braking"] is True  # -4.0 <= -3.0
    assert first["can_vel_mps"] == pytest.approx(5.0)  # nearest pose |[3,4,0]| = 5
    assert first["is_night"] is True and first["is_rain"] is True
    assert first["scene_name"] == "scene-0001" and first["location"] == "singapore-onenorth"
    # Second keyframe (t=3.0s): the monitor msg is 1.95s stale (outside 600ms -> nulls)
    # but the utime=2_900_000 pose is in its ±0.5s window -> accel_min 1.0 -> no event.
    assert second["can_speed_kmh"] is None
    assert second["accel_long_min_mps2"] == pytest.approx(1.0)
    assert second["can_vel_mps"] == pytest.approx(6.0)
    assert second["is_hard_braking"] is False


def test_flatten_canbus_blacklisted_scene_yields_nulls() -> None:
    from nuscenes_data_engine.ingestion.canbus import flatten_canbus

    rows = flatten_canbus(_FakeNusc([_scene("scene-0161")]), _FakeCan(_MONITOR, _POSE))
    assert len(rows) == 2
    assert all(r["has_canbus"] is False for r in rows)
    assert all(r["can_speed_kmh"] is None for r in rows)
    assert all(r["is_hard_braking"] is None for r in rows)


def test_flatten_canbus_missing_files_treated_as_absent() -> None:
    from nuscenes_data_engine.ingestion.canbus import flatten_canbus

    rows = flatten_canbus(
        _FakeNusc([_scene("scene-0002")]),
        _FakeCan(_MONITOR, _POSE, missing={"scene-0002"}),
    )
    assert all(r["has_canbus"] is False and r["throttle"] is None for r in rows)


def test_flatten_canbus_monitor_outside_tolerance_is_null_but_pose_still_used() -> None:
    from nuscenes_data_engine.ingestion.canbus import flatten_canbus

    stale_monitor = [dict(_MONITOR[0], utime=99_000_000)]
    rows = flatten_canbus(_FakeNusc([_scene("scene-0003")]), _FakeCan(stale_monitor, _POSE))
    first = rows[0]
    assert first["has_canbus"] is True
    assert first["can_speed_kmh"] is None  # monitor too old
    assert first["accel_long_min_mps2"] == pytest.approx(-4.0)  # pose window still valid


def test_flatten_canbus_limit_scenes() -> None:
    from nuscenes_data_engine.ingestion.canbus import flatten_canbus

    rows = flatten_canbus(
        _FakeNusc([_scene("scene-0001"), _scene("scene-0004")]),
        _FakeCan(_MONITOR, _POSE),
        limit_scenes=1,
    )
    assert {r["scene_name"] for r in rows} == {"scene-0001"}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_canbus.py -k flatten -v`
Expected: FAIL — `ImportError: cannot import name 'flatten_canbus'`.

- [ ] **Step 3: Implement `flatten_canbus`**

Append to `src/nuscenes_data_engine/ingestion/canbus.py`:

```python
def flatten_canbus(
    nusc: Any,
    can: Any,
    *,
    window_s: float = 0.5,
    hard_braking_mps2: float = -3.0,
    monitor_tolerance_ms: int = 600,
    limit_scenes: int | None = None,
) -> list[dict[str, Any]]:
    """One record per keyframe: nearest vehicle_monitor signals + windowed pose accel.

    Blacklisted scenes (``can.can_blacklist`` holds scene numbers) and scenes whose CAN
    files are missing produce ``has_canbus=False`` rows with null signals.
    """
    from nuscenes_data_engine.ingestion.parse import _scene_conditions

    window_us = int(window_s * 1_000_000)
    tolerance_us = monitor_tolerance_ms * 1000
    blacklist = set(getattr(can, "can_blacklist", ()))
    rows: list[dict[str, Any]] = []
    scenes = nusc.scene[:limit_scenes] if limit_scenes else nusc.scene

    for si, scene in enumerate(scenes):
        log = nusc.get("log", scene["log_token"])
        is_night, is_rain = _scene_conditions(scene["description"])
        ctx = {
            "scene_token": scene["token"],
            "scene_name": scene["name"],
            "location": log["location"],
            "is_night": is_night,
            "is_rain": is_rain,
        }
        monitor: list[dict[str, Any]] = []
        pose: list[dict[str, Any]] = []
        has_canbus = scene_number(scene["name"]) not in blacklist
        if has_canbus:
            try:
                monitor = can.get_messages(scene["name"], "vehicle_monitor")
                pose = can.get_messages(scene["name"], "pose")
            except Exception:  # devkit raises plain Exception for absent scene files
                logger.warning("CAN data missing for %s; writing nulls", scene["name"])
                has_canbus = False

        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            t_us = int(sample["timestamp"])
            msg = nearest_message(monitor, t_us, tolerance_us) if has_canbus else None
            accel_min, accel_max, vel_mps = (
                accel_window(pose, t_us, window_us) if has_canbus else (None, None, None)
            )
            rows.append(
                {
                    "sample_token": sample_token,
                    "timestamp": t_us,
                    "has_canbus": has_canbus,
                    "can_speed_kmh": None if msg is None else float(msg["vehicle_speed"]),
                    "steering_deg": None if msg is None else float(msg["steering"]),
                    "steering_speed": None if msg is None else float(msg["steering_speed"]),
                    "brake_pedal": None if msg is None else int(msg["brake"]),
                    "brake_switch": None if msg is None else int(msg["brake_switch"]),
                    "throttle": None if msg is None else int(msg["throttle"]),
                    "yaw_rate": None if msg is None else float(msg["yaw_rate"]),
                    "left_signal": None if msg is None else int(msg["left_signal"]),
                    "right_signal": None if msg is None else int(msg["right_signal"]),
                    "accel_long_min_mps2": accel_min,
                    "accel_long_max_mps2": accel_max,
                    "can_vel_mps": vel_mps,
                    "is_hard_braking": None if accel_min is None else accel_min <= hard_braking_mps2,
                    **ctx,
                }
            )
            sample_token = sample["next"]
        if (si + 1) % 100 == 0 or si + 1 == len(scenes):
            logger.info("canbus scene %d/%d: %d keyframes so far", si + 1, len(scenes), len(rows))
    return rows
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_canbus.py -v`
Expected: 11 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuscenes_data_engine/ingestion/canbus.py tests/test_canbus.py
git commit -m "canbus: flatten_canbus keyframe walk (blacklist/missing tolerant)"
```

---

### Task 3: `run_canbus_ingestion` + speed cross-check + CLI + config

**Files:**
- Modify: `src/nuscenes_data_engine/ingestion/canbus.py`
- Modify: `src/nuscenes_data_engine/cli.py` (after the `ingest_geometry` command, ~line 104)
- Modify: `configs/data.yaml`
- Test: `tests/test_canbus.py`

- [ ] **Step 1: Write the failing test for the correlation helper**

Append to `tests/test_canbus.py`:

```python
def test_speed_correlation_matches_pandas() -> None:
    pd = pytest.importorskip("pandas")
    from nuscenes_data_engine.ingestion.canbus import speed_correlation

    canbus = pd.DataFrame(
        {"sample_token": ["a", "b", "c", "d"], "can_speed_kmh": [36.0, 18.0, 72.0, None]}
    )
    ego = pd.DataFrame(
        {"sample_token": ["a", "b", "c", "d"], "speed_mps": [10.0, 5.0, 20.0, 1.0]}
    )
    corr = speed_correlation(canbus, ego)
    assert corr == pytest.approx(1.0)  # kmh/3.6 exactly equals the GT speeds

    assert speed_correlation(canbus.head(1), ego.head(1)) is None  # <2 matched rows
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_canbus.py::test_speed_correlation_matches_pandas -v`
Expected: ImportError for `speed_correlation`.

- [ ] **Step 3: Implement `speed_correlation` and `run_canbus_ingestion`**

Append to `canbus.py`:

```python
def speed_correlation(canbus: Any, ego: Any) -> float | None:
    """Pearson corr of CAN speed (km/h -> m/s) vs GT-derived ego speed; None if <2 rows.

    A free wiring sanity-check: the two speeds come from independent sources
    (vehicle CAN vs consecutive GT poses), so high correlation validates alignment.
    """
    merged = canbus.merge(ego[["sample_token", "speed_mps"]], on="sample_token").dropna(
        subset=["can_speed_kmh", "speed_mps"]
    )
    if len(merged) < 2:
        return None
    return float(merged["can_speed_kmh"].div(3.6).corr(merged["speed_mps"]))


def run_canbus_ingestion(
    config_path: Path, *, limit_scenes: int | None = None
) -> dict[str, Any]:
    """Parse the CAN-bus expansion into the keyframe-aligned canbus Parquet table."""
    import pandas as pd

    from nuscenes_data_engine.config import get_settings, load_yaml
    from nuscenes_data_engine.ingestion.parquet import write_parquet
    from nuscenes_data_engine.ingestion.parse import load_nusc

    settings = get_settings()
    cfg = load_yaml(config_path)
    source = cfg.get("source", {})
    dataroot = Path(settings.nuscenes_dataroot or source.get("dataroot"))
    version = settings.nuscenes_version or source.get("version")
    can_cfg = cfg.get("canbus", {})
    out = cfg.get("output", {})
    processed_dir = Path(out.get("processed_dir", settings.processed_dir))
    canbus_path = processed_dir / out.get("parquet", {}).get("canbus", "canbus.parquet")

    from nuscenes.can_bus.can_bus_api import NuScenesCanBus

    nusc = load_nusc(dataroot, version)
    can = NuScenesCanBus(dataroot=str(dataroot))
    rows = flatten_canbus(
        nusc,
        can,
        window_s=float(can_cfg.get("window_s", 0.5)),
        hard_braking_mps2=float(can_cfg.get("hard_braking_mps2", -3.0)),
        monitor_tolerance_ms=int(can_cfg.get("monitor_tolerance_ms", 600)),
        limit_scenes=limit_scenes,
    )
    n_rows = write_parquet(rows, canbus_path)

    corr = None
    ego_path = processed_dir / "ego_pose.parquet"
    if ego_path.is_file():
        corr = speed_correlation(
            pd.DataFrame(rows), pd.read_parquet(ego_path, columns=["sample_token", "speed_mps"])
        )
    summary = {
        "version": version,
        "scenes_processed": limit_scenes if limit_scenes else len(nusc.scene),
        "canbus_rows": n_rows,
        "n_missing_canbus": sum(1 for r in rows if not r["has_canbus"]),
        "n_hard_braking": sum(1 for r in rows if r["is_hard_braking"]),
        "speed_correlation_vs_gt": corr,
        "canbus_parquet": str(canbus_path),
    }
    logger.info(
        "CAN ingestion wrote %d rows (%d without CAN, %d hard-braking; speed corr vs GT: %s)",
        n_rows, summary["n_missing_canbus"], summary["n_hard_braking"],
        "n/a" if corr is None else f"{corr:.3f}",
    )
    return summary
```

- [ ] **Step 4: Add the CLI command**

In `src/nuscenes_data_engine/cli.py`, directly after the `ingest_geometry` command:

```python
@app.command()
def ingest_canbus(
    config: Path = typer.Option(
        Path("configs/data.yaml"), "--config", "-c", help="Path to data.yaml."
    ),
    limit_scenes: int | None = typer.Option(
        None, "--limit-scenes", help="Only process the first N scenes (fast dev runs)."
    ),
) -> None:
    """CAN bus: keyframe-aligned ego dynamics (speed/steering/brake, is_hard_braking)."""
    from nuscenes_data_engine.ingestion.canbus import run_canbus_ingestion

    summary = run_canbus_ingestion(config, limit_scenes=limit_scenes)
    logger.info(
        "Done: %d canbus rows (%d hard-braking) -> %s",
        summary["canbus_rows"], summary["n_hard_braking"], summary["canbus_parquet"],
    )
```

- [ ] **Step 5: Add the config block**

In `configs/data.yaml`: under `output.parquet`, add `canbus: canbus.parquet` alongside the existing geometry entries; add a new top-level block (placed after the geometry-related config, matching file style):

```yaml
canbus:                      # CAN-bus keyframe alignment (docs/superpowers/specs/2026-08-05-canbus-ingestion-design.md)
  window_s: 0.5              # pose-accel window around each keyframe
  hard_braking_mps2: -3.0    # is_hard_braking threshold on accel_long_min
  monitor_tolerance_ms: 600  # max age of the nearest vehicle_monitor message
```

- [ ] **Step 6: Verify**

Run: `uv run pytest tests/test_canbus.py -v` (12 PASS) and `uv run nuscenes-data-engine ingest-canbus --help` (shows --limit-scenes; do NOT run without --help on this machine — no dataset here). `uv run python -c "from nuscenes_data_engine.config import load_yaml; from pathlib import Path; print(load_yaml(Path('configs/data.yaml'))['canbus'])"` prints the block.

- [ ] **Step 7: Commit**

```bash
git add src/nuscenes_data_engine/ingestion/canbus.py src/nuscenes_data_engine/cli.py configs/data.yaml tests/test_canbus.py
git commit -m "canbus: run_canbus_ingestion + ingest-canbus CLI + config (speed cross-check)"
```

---

### Task 4: Neo4j — `canbus_rows` projection, `SET` pass, index

**Files:**
- Modify: `src/nuscenes_data_engine/data_engine/graph/model.py` (after `ego_pose_rows`)
- Modify: `src/nuscenes_data_engine/data_engine/graph/builder.py` (new `_CANBUS` constant after `_EGO_POSE`; wiring inside `build_graph`'s geometry block)
- Modify: `src/nuscenes_data_engine/data_engine/graph/schema.py` (`_INDEXES`)
- Test: `tests/test_graph.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_graph.py` (next to the existing `ego_pose_rows` projection tests — read that section first and match its style; the tests below are complete as written):

```python
def test_canbus_rows_projection_and_null_handling() -> None:
    pd = pytest.importorskip("pandas")
    from nuscenes_data_engine.data_engine.graph.model import canbus_rows

    df = pd.DataFrame(
        [
            {
                "sample_token": "s1", "has_canbus": True, "can_speed_kmh": 14.7,
                "steering_deg": 191.9, "brake_pedal": 20.0, "throttle": 55.0,
                "yaw_rate": 18.9, "accel_long_min_mps2": -4.0,
                "accel_long_max_mps2": 0.5, "is_hard_braking": True,
            },
            {
                "sample_token": "s2", "has_canbus": False, "can_speed_kmh": None,
                "steering_deg": None, "brake_pedal": None, "throttle": None,
                "yaw_rate": None, "accel_long_min_mps2": None,
                "accel_long_max_mps2": None, "is_hard_braking": None,
            },
        ]
    )
    rows = canbus_rows(df)
    assert rows[0]["sample_token"] == "s1" and rows[0]["is_hard_braking"] is True
    assert rows[0]["accel_long_min_mps2"] == pytest.approx(-4.0)
    assert rows[0]["brake_pedal"] == pytest.approx(20.0)
    assert rows[1]["has_canbus"] is False
    assert rows[1]["can_speed_kmh"] is None and rows[1]["is_hard_braking"] is None


def test_schema_includes_canbus_index() -> None:
    from nuscenes_data_engine.data_engine.graph.schema import schema_statements

    stmts = "\n".join(schema_statements())
    assert "egopose_accel_long_min_mps2_idx" in stmts
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_graph.py -k "canbus or schema_includes" -v`
Expected: ImportError for `canbus_rows`; the schema test fails on the missing index.

- [ ] **Step 3: Implement**

`model.py` — add after `ego_pose_rows` (reuse the module's existing `_opt_float` helper; add `import pandas as pd` only if not already imported — it is):

```python
def canbus_rows(canbus: pd.DataFrame) -> list[dict[str, Any]]:
    """CAN-dynamics property payload per keyframe, SET onto the existing ``EgoPose``."""
    return [
        {
            "sample_token": str(r.sample_token),
            "has_canbus": bool(r.has_canbus),
            "can_speed_kmh": _opt_float(r.can_speed_kmh),
            "steering_deg": _opt_float(r.steering_deg),
            "brake_pedal": _opt_float(r.brake_pedal),
            "throttle": _opt_float(r.throttle),
            "yaw_rate": _opt_float(r.yaw_rate),
            "accel_long_min_mps2": _opt_float(r.accel_long_min_mps2),
            "accel_long_max_mps2": _opt_float(r.accel_long_max_mps2),
            "is_hard_braking": None if pd.isna(r.is_hard_braking) else bool(r.is_hard_braking),
        }
        for r in canbus.itertuples(index=False)
    ]
```

`builder.py` — add after the `_EGO_POSE` constant:

```python
_CANBUS = """
UNWIND $rows AS row
MATCH (:Sample {token: row.sample_token})-[:AT_POSE]->(e:EgoPose)
SET e.has_canbus = row.has_canbus,
    e.can_speed_kmh = row.can_speed_kmh, e.steering_deg = row.steering_deg,
    e.brake_pedal = row.brake_pedal, e.throttle = row.throttle,
    e.yaw_rate = row.yaw_rate,
    e.accel_long_min_mps2 = row.accel_long_min_mps2,
    e.accel_long_max_mps2 = row.accel_long_max_mps2,
    e.is_hard_braking = row.is_hard_braking
"""
```

and inside `build_graph`, at the END of the existing `if not skip_geometry and ego_path.is_file():` block (after the observations load), add:

```python
            canbus_path = processed / "canbus.parquet"
            if canbus_path.is_file():
                canbus = pd.read_parquet(canbus_path)
                if keep_scenes is not None:
                    canbus = canbus[canbus["scene_token"].isin(keep_scenes)]
                load("canbus", _CANBUS, model.canbus_rows(canbus))
```

(The pass is idempotent — plain `SET` — and requires the ego-pose pass to have run, which the enclosing block guarantees.)

`schema.py` — add to `_INDEXES`:

```python
    ("EgoPose", "accel_long_min_mps2"),
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_graph.py -q`
Expected: all pass (prior count + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/nuscenes_data_engine/data_engine/graph/model.py src/nuscenes_data_engine/data_engine/graph/builder.py src/nuscenes_data_engine/data_engine/graph/schema.py tests/test_graph.py
git commit -m "canbus: EgoPose CAN property pass + accel range index"
```

---

### Task 5: DuckDB view + chat guard wiring

**Files:**
- Modify: `src/nuscenes_data_engine/data_engine/chat/catalog.py` (`TABLES` + the schema-description dict)
- Modify: `src/nuscenes_data_engine/data_engine/graph/guard.py` (EgoPose property line + flagship example)
- Test: `tests/test_chat.py`, `tests/test_graph.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_chat.py`, find the existing tests that build a catalog from tmp parquets and that assert on the schema prompt (read them first; bind the import below to the module's actual prompt-builder function — it is the function that renders the table descriptions the `ego_pose`/`annotations_3d` entries live in). Add:

```python
def test_catalog_exposes_canbus_view(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    duckdb = pytest.importorskip("duckdb")
    del duckdb
    from nuscenes_data_engine.data_engine.chat.catalog import catalog_tables, open_catalog

    pd.DataFrame(
        {"sample_token": ["s1"], "has_canbus": [True], "can_speed_kmh": [14.7],
         "is_hard_braking": [True]}
    ).to_parquet(tmp_path / "canbus.parquet")
    con = open_catalog(tmp_path)
    assert "canbus" in catalog_tables(con)
    assert con.execute("SELECT count(*) FROM canbus WHERE is_hard_braking").fetchone()[0] == 1
```

and a schema-prompt assertion in the same style as the existing geometry-note test: with a `canbus.parquet` present the prompt must contain `"is_hard_braking"` and `"canbus"`; without it, it must not mention `canbus`.

In `tests/test_graph.py`, extend the existing guard-prompt content test (the one asserting the EgoPose line / distance example) with:

```python
    assert "is_hard_braking" in prompt
    assert "hard braking near pedestrians" in prompt.lower()
```

(bind `prompt` to however that existing test obtains the run_cypher schema prompt).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_chat.py -k canbus -v` and the extended guard test.
Expected: FAIL (view missing / prompt text missing).

- [ ] **Step 3: Implement**

`catalog.py`:
- `TABLES` becomes `("samples", "annotations", "availability", "ego_pose", "annotations_3d", "instances", "canbus")`.
- In the schema-description dict (where the `"ego_pose"` entry lives), add:

```python
        "canbus": (
            "canbus — one row per keyframe: CAN-bus ego dynamics (can_bus expansion):\n"
            "  sample_token (joins ego_pose/annotations_3d), has_canbus,\n"
            "  can_speed_kmh, steering_deg, steering_speed, brake_pedal (raw 0-126),\n"
            "  brake_switch, throttle, yaw_rate, left_signal, right_signal,\n"
            "  accel_long_min_mps2 / accel_long_max_mps2 (50 Hz window ±0.5 s),\n"
            "  can_vel_mps, is_hard_braking (accel_long_min <= -3.0 m/s²),\n"
            "  scene_name, location, is_night, is_rain\n"
        ),
```

- Follow the existing `has_geometry` conditional-note pattern: when `"canbus"` is in the available tables, append a note line `"Ego dynamics ARE available (canbus table): hard braking, steering, throttle."` to the prompt; when absent, no canbus text at all.

`guard.py`:
- Extend the `(:EgoPose {...})` property line to include `has_canbus, can_speed_kmh, steering_deg, brake_pedal, throttle, yaw_rate, accel_long_min_mps2, is_hard_braking`.
- After the existing distance example, add:

```python
        "Example (hard braking near pedestrians):\n"
        "  MATCH (e:EgoPose {is_hard_braking: true})<-[:AT_POSE]-(s:Sample)\n"
        "        -[:HAS_OBJECT]->(o:ObjectObservation)\n"
        "        -[:OF_CATEGORY]->(:Category {group: 'pedestrian'})\n"
        "  WHERE o.distance_to_ego_m < 10 RETURN count(DISTINCT s) AS n\n"
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_chat.py tests/test_graph.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/nuscenes_data_engine/data_engine/chat/catalog.py src/nuscenes_data_engine/data_engine/graph/guard.py tests/test_chat.py tests/test_graph.py
git commit -m "canbus: DuckDB view + chat/guard schema wiring (flagship braking example)"
```

---

### Task 6: CI-parity sweep

- [ ] **Step 1:** `uv run pytest -q` — all pass (prior count + ~14 new; report exact numbers). `uv run ruff check .` clean. Bare `uv run mypy` clean (63+1 files). Do NOT bulk-`ruff format` (CI does not gate it; pre-existing repo-wide drift).
- [ ] **Step 2:** Commit only if fixes were needed: `git commit -am "canbus: lint/type fixes"`.

---

### Task 7: TRINITY ingestion (operational; no commits)

The dataset, devkit, and `can_bus/` live on TRINITY. Use `scripts/gpu-run.sh` from this repo (it pushes the current branch — ensure all prior tasks are committed AND pushed: `git push -u origin canbus-ingestion` first).

- [ ] **Step 1: Smoke:** `scripts/gpu-run.sh ingest-canbus --limit-scenes 5` — expect a summary log with 5 scenes' keyframes, no exceptions. If the devkit's `NuScenesCanBus` import or blacklist behavior differs from the fake's assumptions, STOP and report — do not patch on TRINITY.
- [ ] **Step 2: Full run:** `scripts/gpu-run.sh ingest-canbus` (JSON-only; minutes). Record from the log: total rows (expect 34,149), `n_missing_canbus`, `n_hard_braking`, and `speed_correlation_vs_gt` (expect > 0.95 — if it is low or n/a, STOP and investigate alignment before proceeding).
- [ ] **Step 3: Sync down:** `rsync -a trinity-2-18:/home/mgaur/sahil/nuscenes_project/data/processed/canbus.parquet data/processed/`
- [ ] **Step 4: Local sanity:** row count 34,149; `has_canbus` false-count matches the log; a few `is_hard_braking` rows exist.

### Task 8: Graph pass + SQL/Cypher parity acceptance (operational; no commits)

- [ ] **Step 1:** `docker compose up -d neo4j` (the Phase B graph with EgoPose nodes must already be loaded — it is, from Phase B).
- [ ] **Step 2:** `uv run nuscenes-data-engine graph build --skip-knn` (or the narrower invocation the CLI offers — check `graph build --help`; the canbus pass runs inside the geometry block and is idempotent). Expect the summary line `canbus  34149` (rows written may be lower by the ~10 blacklisted scenes' keyframes if their Samples lack EgoPose — record the actual number).
- [ ] **Step 3: Flagship parity.** SQL:

```bash
uv run python - <<'EOF'
import duckdb
con = duckdb.connect()
n = con.execute("""
    SELECT count(DISTINCT c.sample_token)
    FROM read_parquet('data/processed/canbus.parquet') c
    JOIN read_parquet('data/processed/annotations_3d.parquet') a USING (sample_token)
    WHERE c.is_hard_braking AND a.category_group = 'pedestrian' AND a.distance_to_ego_m < 10
""").fetchone()[0]
print("SQL hard-braking-near-pedestrian keyframes:", n)
EOF
```

Cypher (same count expected, via cypher-shell):

```bash
docker compose exec -T neo4j cypher-shell -u neo4j -p neo4jpassword \
  "MATCH (e:EgoPose {is_hard_braking: true})<-[:AT_POSE]-(s:Sample)-[:HAS_OBJECT]->(o:ObjectObservation)-[:OF_CATEGORY]->(c:Category) WHERE c.group = 'pedestrian' AND o.distance_to_ego_m < 10 RETURN count(DISTINCT s)"
```

The two counts MUST match exactly. Record the number for the docs.

- [ ] **Step 4:** One chat smoke (optional but cheap): `uv run nuscenes-data-engine chat --once "how many keyframes have hard braking?"` and confirm the agent uses the canbus table.

### Task 9: Docs

**Files:** `docs/DATA.md`, `docs/GRAPH.md`, `docs/ANALYTICS.md`, `docs/DATASET_CHAT.md`, `docs/PROJECT.md`

- [ ] **Step 1:** `DATA.md` — add a `canbus.parquet` subsection to the Phase B tables: one row per keyframe (34,149), the column list with units (speed km/h, steering deg, accel m/s², pedal raw 0–126), the alignment rule (nearest monitor ≤ 600 ms; pose window ±0.5 s), the `is_hard_braking` definition, blacklist behavior, and the `canbus:` config block.
- [ ] **Step 2:** `GRAPH.md` — in the Phase B geo-spatial section: EgoPose's new CAN properties, the `accel_long_min_mps2` range index, and the flagship Cypher with the MEASURED count from Task 8 quoted (both SQL and Cypher, noting they match).
- [ ] **Step 3:** `ANALYTICS.md` — one worked DuckDB example: hard-braking keyframes by location/night with the real counts.
- [ ] **Step 4:** `DATASET_CHAT.md` — retire any "no dynamics" caveat; add the example question + the canbus table to the schema list.
- [ ] **Step 5:** `PROJECT.md` — §2 component map: mention canbus in the Phase-1/B ingestion line; §9: mark the CAN-bus roadmap item DONE with the headline numbers (n_hard_braking, speed correlation, flagship count).
- [ ] **Step 6:** Commit: `git add docs && git commit -m "canbus: docs (schema, graph properties, flagship query with measured counts)"`

---

### After the plan (not plan tasks)

1. Final whole-branch review, push, PR, CI green, merge (superpowers:finishing-a-development-branch).
2. `make sync-down` / rsync remains the path for other machines needing the parquet.
