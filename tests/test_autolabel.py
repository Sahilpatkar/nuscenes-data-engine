"""Tests for Phase 6b auto-labeling (fake transport + synthetic data; offline)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

pytest.importorskip("anthropic")  # engine extra

import cv2
import numpy as np
import pandas as pd
import yaml

from nuscenes_data_engine.data_engine.autolabel.batch import (
    build_request,
    chunk_requests,
    run_collect,
    run_submit,
)
from nuscenes_data_engine.data_engine.autolabel.evaluate import (
    eval_counts,
    eval_flags,
    gt_counts,
    model_agreement,
)
from nuscenes_data_engine.data_engine.autolabel.sampling import allocate, build_sample
from nuscenes_data_engine.data_engine.autolabel.schema import (
    COUNT_FIELDS,
    SceneLabel,
    structured_output_schema,
)

FORBIDDEN_SCHEMA_KEYS = {"minimum", "maximum", "title", "default", "minItems", "maxItems"}


def _valid_label(**overrides: Any) -> dict[str, Any]:
    label: dict[str, Any] = {
        "time_of_day": "day",
        "weather": "clear",
        "object_counts": {field: 0 for field in COUNT_FIELDS},
        "hazards": [],
        "notable_conditions": [],
        "label_confidence": "high",
    }
    label.update(overrides)
    return label


class TestSchema:
    def test_sanitized_schema_has_no_forbidden_keys(self) -> None:
        schema = structured_output_schema()

        def walk(node: Any) -> Iterator[dict[str, Any]]:
            if isinstance(node, dict):
                yield node
                for value in node.values():
                    yield from walk(value)
            elif isinstance(node, list):
                for item in node:
                    yield from walk(item)

        for node in walk(schema):
            assert not (set(node) & FORBIDDEN_SCHEMA_KEYS), node
            if node.get("type") == "object" and "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])

    def test_valid_label_round_trips(self) -> None:
        label = SceneLabel.model_validate_json(json.dumps(_valid_label()))
        assert label.time_of_day == "day"
        assert label.object_counts.cars == 0

    def test_negative_count_rejected(self) -> None:
        bad = _valid_label(object_counts={**{f: 0 for f in COUNT_FIELDS}, "cars": -1})
        with pytest.raises(ValueError):
            SceneLabel.model_validate_json(json.dumps(bad))


class TestAllocate:
    # Real strata counts from the trainval manifest (CAM_FRONT keyframes).
    REAL: ClassVar[dict[tuple[str, bool, bool], int]] = {
        ("bos", False, False): 12757,
        ("bos", False, True): 6028,
        ("hol", False, False): 770,
        ("hol", True, False): 2015,
        ("hol", True, True): 642,
        ("one", False, False): 7308,
        ("que", False, False): 3299,
        ("que", True, False): 1330,
    }

    def test_real_allocation(self) -> None:
        alloc = allocate(self.REAL, total=5000, floor=250)
        assert sum(alloc.values()) == 5000
        assert alloc[("hol", False, False)] == 250  # proportional share 113 -> floored
        assert alloc[("hol", True, True)] == 250
        assert alloc[("que", True, False)] == 250
        assert alloc[("bos", False, False)] == 1726  # matches the documented table

    def test_floor_exceeds_stratum_size(self) -> None:
        alloc = allocate({"a": 10, "b": 1000}, total=200, floor=50)
        assert alloc["a"] == 10  # whole stratum, smaller than the floor
        assert sum(alloc.values()) == 200

    def test_total_exceeds_population(self) -> None:
        counts = {"a": 5, "b": 7}
        assert allocate(counts, total=100, floor=3) == counts


def _write_dataset(root: Path, n_scenes: int = 4, frames_per_scene: int = 30) -> Path:
    """Synthetic samples/annotations parquet + tiny JPEGs across 2 strata."""
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    samples, annotations = [], []
    for s in range(n_scenes):
        night = s % 2 == 1
        for f in range(frames_per_scene):
            token = f"tok_s{s}f{f:02d}"
            filename = f"samples/CAM_FRONT/{token}.jpg"
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.full((90, 160, 3), 30 if night else 180, np.uint8))
            samples.append(
                {
                    "sample_data_token": token,
                    "channel": "CAM_FRONT",
                    "filename": filename,
                    "location": "loc",
                    "is_night": night,
                    "is_rain": False,
                    "scene_token": f"sc{s}",
                }
            )
            # GT: 2 cars in every frame, 1 pedestrian in night frames (visibility "1").
            annotations += [
                {
                    "sample_data_token": token,
                    "category_name": "vehicle.car",
                    "visibility_token": "4",
                },
                {
                    "sample_data_token": token,
                    "category_name": "vehicle.car",
                    "visibility_token": "3",
                },
            ]
            if night:
                annotations.append(
                    {
                        "sample_data_token": token,
                        "category_name": "human.pedestrian.adult",
                        "visibility_token": "1",
                    }
                )
    pd.DataFrame(samples).to_parquet(processed / "samples.parquet", index=False)
    pd.DataFrame(annotations).to_parquet(processed / "annotations.parquet", index=False)
    return processed


class TestSampling:
    def test_build_sample_stratified_and_deterministic(self, tmp_path: Path) -> None:
        processed = _write_dataset(tmp_path)
        cfg = {
            "size": 40,
            "opus_subset": 10,
            "seed": 6,
            "min_per_stratum": 10,
            "opus_min_per_stratum": 2,
            "channel": "CAM_FRONT",
        }
        first = build_sample(processed, cfg)
        second = build_sample(processed, cfg)
        assert len(first) == 40
        assert int(first["in_opus_subset"].sum()) == 10
        assert set(first[first["in_opus_subset"]]["sample_data_token"]) <= set(
            first["sample_data_token"]
        )
        pd.testing.assert_frame_equal(first, second)  # deterministic
        assert first.groupby("stratum").size().min() >= 10


class _FakeTransport:
    """Canned batch transport recording submissions."""

    def __init__(self, results_by_batch: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.submitted: list[list[dict[str, Any]]] = []
        self._results = results_by_batch or {}

    def submit(self, requests: list[dict[str, Any]]) -> str:
        self.submitted.append(requests)
        return f"batch_{len(self.submitted)}"

    def status(self, batch_id: str) -> tuple[str, dict[str, int]]:
        return "ended", {"succeeded": 1, "processing": 0, "errored": 0, "canceled": 0, "expired": 0}

    def results(self, batch_id: str) -> Iterator[dict[str, Any]]:
        yield from self._results.get(batch_id, [])


@pytest.fixture()
def autolabel_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    processed = _write_dataset(tmp_path)
    monkeypatch.setenv("NUSCENES_DATAROOT", str(tmp_path))
    monkeypatch.setenv("PROCESSED_DIR", str(processed))
    state_dir = tmp_path / "autolabel"
    config = tmp_path / "autolabel.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "sample": {
                    "size": 12,
                    "opus_subset": 4,
                    "seed": 6,
                    "min_per_stratum": 4,
                    "opus_min_per_stratum": 2,
                    "channel": "CAM_FRONT",
                },
                "models": {"primary": "claude-haiku-4-5", "comparison": "claude-opus-4-8"},
                "batch": {"chunk_size": 5, "max_batch_bytes": 10_000_000, "max_tokens": 800},
                "state": {"dir": str(state_dir)},
                "eval": {"visibility_min": "2", "out_dir": str(state_dir / "eval")},
            }
        )
    )
    from nuscenes_data_engine.config import get_settings
    from nuscenes_data_engine.data_engine.autolabel.sampling import build_sample as bs

    sample = bs(processed, yaml.safe_load(config.read_text())["sample"])
    state_dir.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(state_dir / "sample.parquet", index=False)
    assert get_settings().processed_dir == processed
    return config


class TestBatchPipeline:
    def test_build_request_shape(self, tmp_path: Path) -> None:
        processed = _write_dataset(tmp_path, n_scenes=1, frames_per_scene=1)
        row = pd.read_parquet(processed / "samples.parquet").iloc[0].to_dict()
        request = build_request(row, tmp_path, "claude-haiku-4-5", structured_output_schema(), 800)
        assert request["custom_id"] == row["sample_data_token"]
        params = request["params"]
        assert params["model"] == "claude-haiku-4-5"
        assert params["output_config"]["format"]["type"] == "json_schema"
        content = params["messages"][0]["content"]
        assert content[0]["type"] == "image"  # image before text
        assert content[0]["source"]["media_type"] == "image/jpeg"
        assert "\n" not in content[0]["source"]["data"]
        assert content[1]["type"] == "text"
        assert "temperature" not in params and "thinking" not in params

    def test_chunking_respects_count_and_bytes(self) -> None:
        requests = [{"custom_id": str(i), "params": {"x": "y" * 100}} for i in range(12)]
        by_count = chunk_requests(requests, chunk_size=5, max_bytes=10**9)
        assert [len(c) for c in by_count] == [5, 5, 2]
        by_bytes = chunk_requests(requests, chunk_size=100, max_bytes=300)
        assert all(len(c) <= 3 for c in by_bytes)
        assert sum(len(c) for c in by_bytes) == 12

    def test_submit_dry_run_calls_no_api(self, autolabel_env: Path) -> None:
        summary = run_submit(autolabel_env, dry_run=True)
        assert summary["dry_run"] is True
        assert summary["estimated_cost"] > 0

    def test_submit_requires_yes(self, autolabel_env: Path) -> None:
        with pytest.raises(SystemExit):
            run_submit(autolabel_env, yes=False, transport=_FakeTransport())

    def test_submit_chunks_and_persists_state(self, autolabel_env: Path) -> None:
        transport = _FakeTransport()
        summary = run_submit(autolabel_env, yes=True, transport=transport)
        assert summary["submitted"] == 12 + 4  # primary sample + comparison subset
        state = json.loads((autolabel_env.parent / "autolabel" / "batches.json").read_text())
        assert {batch["model"] for batch in state} == {"claude-haiku-4-5", "claude-opus-4-8"}
        assert all(batch["n_requests"] <= 5 for batch in state)

    def test_collect_parse_status_matrix_and_retry_missing(self, autolabel_env: Path) -> None:
        transport = _FakeTransport()
        run_submit(autolabel_env, yes=True, transport=transport)
        state_path = autolabel_env.parent / "autolabel" / "batches.json"
        state = json.loads(state_path.read_text())
        ok_text = json.dumps(_valid_label())
        results: dict[str, list[dict[str, Any]]] = {}
        outcomes = [
            ("succeeded", "end_turn", ok_text, None, "ok"),
            ("succeeded", "refusal", None, None, "refusal"),
            ("succeeded", "max_tokens", "{trunc", None, "truncated"),
            ("succeeded", "end_turn", "not json", None, "invalid_json"),
            ("succeeded", "end_turn", json.dumps({"nope": 1}), None, "schema_invalid"),
            ("errored", None, None, "invalid_request", "errored_bad_request"),
            ("errored", None, None, "api_error", "errored_server"),
            ("expired", None, None, None, "expired"),
        ]
        expected: dict[str, str] = {}
        tokens = iter(
            token
            for batch in state
            if batch["model"] == "claude-haiku-4-5"
            for token in batch["custom_ids"]
        )
        assignments = [(next(tokens), *outcome) for outcome in outcomes]
        for batch in state:
            batch_results = []
            for token, rtype, stop, text, err, exp in assignments:
                if token in batch["custom_ids"]:
                    batch_results.append(
                        {
                            "custom_id": token,
                            "result_type": rtype,
                            "stop_reason": stop,
                            "text": text,
                            "error_type": err,
                        }
                    )
                    expected[token] = exp
            results[batch["batch_id"]] = batch_results
        collecting = _FakeTransport(results)
        # Mark ended so collect downloads.
        from nuscenes_data_engine.data_engine.autolabel.batch import run_status

        run_status(autolabel_env, transport=collecting)
        labels = run_collect(autolabel_env, transport=collecting)
        by_token = labels[labels["model"] == "claude-haiku-4-5"].set_index("sample_data_token")
        for token, expected_status in expected.items():
            assert by_token.loc[token, "parse_status"] == expected_status, token

        # retry-missing: ok + bad_request are terminal; the rest (and never-returned) pend.
        retry = _FakeTransport()
        run_submit(autolabel_env, yes=True, retry_missing=True, transport=retry)
        resubmitted = {req["custom_id"] for chunk in retry.submitted for req in chunk}
        terminal = {
            t
            for t, s in expected.items()
            if s
            in {
                "ok",
                "refusal",
                "truncated",
                "invalid_json",
                "schema_invalid",
                "errored_bad_request",
            }
        }
        assert not (resubmitted & terminal)
        assert {t for t, s in expected.items() if s in {"errored_server", "expired"}} <= resubmitted

        # collect is idempotent
        labels_again = run_collect(autolabel_env, transport=collecting)
        assert len(labels_again) == len(labels)


class TestEval:
    def _labels(self, tokens: list[str], night_tokens: set[str]) -> pd.DataFrame:
        rows = []
        for token in tokens:
            label = _valid_label(
                time_of_day="night" if token in night_tokens else "day",
                object_counts={**{f: 0 for f in COUNT_FIELDS}, "cars": 2, "pedestrians": 0},
            )
            rows.append(
                {
                    "sample_data_token": token,
                    "model": "claude-haiku-4-5",
                    "parse_status": "ok",
                    "time_of_day": label["time_of_day"],
                    "weather": label["weather"],
                    "label_confidence": "high",
                    "hazards": "[]",
                    "notable_conditions": "[]",
                    **label["object_counts"],
                }
            )
        return pd.DataFrame(rows)

    def test_gt_counts_and_visibility_filter(self, tmp_path: Path) -> None:
        processed = _write_dataset(tmp_path, n_scenes=2, frames_per_scene=2)
        sample = pd.read_parquet(processed / "samples.parquet")
        tokens = list(sample["sample_data_token"])
        counts = gt_counts(processed / "annotations.parquet", tokens)
        assert (counts["cars"] == 2).all()
        night_tokens = set(sample[sample["is_night"]]["sample_data_token"])
        assert all(counts.loc[t, "pedestrians"] == (1 if t in night_tokens else 0) for t in tokens)
        # visibility filter drops the "1"-visibility pedestrians entirely
        filtered = gt_counts(processed / "annotations.parquet", tokens, visibility_min="2")
        assert (filtered["pedestrians"] == 0).all()
        assert (filtered["cars"] == 2).all()

    def test_flags_and_counts_metrics(self, tmp_path: Path) -> None:
        processed = _write_dataset(tmp_path, n_scenes=2, frames_per_scene=4)
        sample = pd.read_parquet(processed / "samples.parquet")
        sample["stratum"] = sample["is_night"].map({True: "night", False: "day"})
        tokens = list(sample["sample_data_token"])
        night_tokens = set(sample[sample["is_night"]]["sample_data_token"])
        # Predict night correctly for all but one night frame -> recall 3/4.
        wrong = sorted(night_tokens)[0]
        labels = self._labels(tokens, night_tokens - {wrong})
        flags = eval_flags(labels, sample)
        overall = flags[flags["scope"] == "overall"].iloc[0]
        assert overall["night_recall"] == pytest.approx(3 / 4)
        assert overall["night_precision"] == pytest.approx(1.0)

        counts = eval_counts(labels, gt_counts(processed / "annotations.parquet", tokens))
        cars = counts[counts["class"] == "cars"].iloc[0]
        assert cars["mae"] == 0.0 and cars["exact_rate"] == 1.0
        peds = counts[counts["class"] == "pedestrians"].iloc[0]
        assert peds["mae"] == pytest.approx(0.5)  # predicted 0, GT 1 on the 4 night frames
        assert peds["presence_recall"] == 0.0

    def test_model_agreement(self) -> None:
        tokens = ["t1", "t2", "t3", "t4"]
        a = self._labels(tokens, night_tokens={"t1", "t2"})
        b = self._labels(tokens, night_tokens={"t1"})
        agreement = model_agreement(a, b)
        assert agreement["n"] == 4
        assert agreement["time_of_day_agreement"] == pytest.approx(3 / 4)
        assert agreement["count_mae_between_models"] == 0.0
