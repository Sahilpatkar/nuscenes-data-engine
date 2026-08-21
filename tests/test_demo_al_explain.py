"""``demo/al_explain.py`` — the validated re-derivation of the `graph_rate_night` arm.

Four layers, each independently testable:

- **``route_failure_mass_per_frame``**: the same neighbour queries
  ``graph_mining.route_failure_mass`` makes, plus per-frame accumulators. Tested
  against the original as the ORACLE on identical inputs (a fake LanceDB table
  whose fluent ``search(...)`` chain returns canned neighbours), so the demo-side
  copy can never silently drift from the function the experiment actually ran.
- **``attribute_pick_pass``**: which of ``select_by_mass``'s two passes (plus the
  seeded night backfill) put each selected token in the set — re-derived with the
  real ``allocate_by_mass``, asserted to partition the selected list exactly.
- **``validate_reproduction``**: the gate. One swapped token or one off-by-one
  quota must flip it to False; ``mass`` is compared to 4 dp (the precision
  ``select_by_mass`` itself persists).
- **``build_explain_rows`` / ``run_al_explain`` / ``build._include_al_explain``**:
  the staged table's schema and ranks, the nothing-is-staged-on-mismatch rule, and
  the package's absent/partial/included input group.

Plus one ``@pytest.mark.integration`` test that re-derives the REAL arm against the
local Neo4j + LanceDB — self-skips (tests/test_demo_subgraphs.py's precedent) when
either is unavailable, so it is safe to leave enabled in the default test run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nuscenes_data_engine.active_learning import graph_mining
from nuscenes_data_engine.demo import al_explain

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _FakeQuery:
    """One fluent LanceDB query chain, for both callers of the fake table.

    ``mining.read_vectors`` builds ``search().where(...).select(...).limit(n)``;
    ``mining.mine_candidates`` builds ``search(vec).metric(...).where(...,
    prefilter=True).limit(k).select(...)`` — different orders, so every builder
    method returns ``self`` and ``to_pandas`` decides what to answer from whether a
    query vector was given.
    """

    def __init__(self, table: _FakeTable, vector: list[float] | None) -> None:
        self._table = table
        self._vector = vector
        self._where: str = ""
        self._limit: int | None = None

    def metric(self, _metric: str) -> _FakeQuery:
        return self

    def where(self, where: str, prefilter: bool = False) -> _FakeQuery:
        self._where = where
        return self

    def limit(self, k: int) -> _FakeQuery:
        self._limit = k
        return self

    def select(self, _columns: list[str]) -> _FakeQuery:
        return self

    def to_pandas(self) -> pd.DataFrame:
        if self._vector is None:  # read_vectors: "sample_data_token IN ('a', 'b')"
            tokens = re.findall(r"'([^']+)'", self._where)
            rows = [
                {"sample_data_token": token, "vector": self._table.vectors[token]}
                for token in tokens
                if token in self._table.vectors
            ]
            return pd.DataFrame(rows, columns=["sample_data_token", "vector"])
        # mine_candidates: one-hot query vectors identify which failure asked.
        self._table.where_clauses.append(self._where)
        index = int(np.argmax(np.asarray(self._vector)))
        neighbors = self._table.neighbors.get(index, [])
        if self._limit is not None:
            neighbors = neighbors[: self._limit]
        return pd.DataFrame(
            {
                "sample_data_token": neighbors,
                "scene_name": ["sP"] * len(neighbors),
                "is_night": [False] * len(neighbors),
                "is_rain": [False] * len(neighbors),
                "_distance": [0.1 * i for i in range(len(neighbors))],
            }
        )


class _FakeTable:
    def __init__(
        self, vectors: dict[str, list[float]], neighbors: dict[int, list[str]]
    ) -> None:
        self.vectors = vectors
        self.neighbors = neighbors
        self.where_clauses: list[str] = []

    def search(self, vector: list[float] | None = None) -> _FakeQuery:
        return _FakeQuery(self, vector)


def _failures_frame() -> pd.DataFrame:
    """Four failures; smoothed_rate scores order them f0 > f1 > f3 > f2."""
    return pd.DataFrame(
        {
            "sample_data_token": ["f0", "f1", "f2", "f3"],
            "scene_name": ["sV", "sV", "sV", "sV"],
            "is_night": [True, False, False, True],
            "n_gt": [3, 4, 10, 2],
            "n_matched": [0, 2, 9, 1],
            "n_fn": [3, 2, 1, 1],
            "n_low_conf": [0, 2, 0, 0],
            "failure_score": [3.0, 3.0, 1.0, 1.0],
        }
    )


# ---------------------------------------------------------------------------
# route_failure_mass_per_frame
# ---------------------------------------------------------------------------


def test_route_failure_mass_per_frame_matches_community_totals() -> None:
    failures = _failures_frame()
    # f3 deliberately has NO stored vector: the original skips it (contributing no
    # mass), and the per-frame copy must skip it identically.
    table = _FakeTable(
        vectors={"f0": [1.0, 0.0, 0.0], "f1": [0.0, 1.0, 0.0], "f2": [0.0, 0.0, 1.0]},
        neighbors={0: ["p1", "p2", "pX", "p3"], 1: ["p2", "p3"], 2: ["p1"]},
    )
    communities = {"p1": 10, "p2": 10, "p3": 20}  # pX is in no community
    pool_scenes = ["sP"]
    top_k, route_k = 3, 3

    oracle = graph_mining.route_failure_mass(
        table, failures, communities, pool_scenes, "CAM_FRONT", top_k, route_k
    )
    masses, per_frame = al_explain.route_failure_mass_per_frame(
        table, failures, communities, pool_scenes, "CAM_FRONT", top_k, route_k
    )

    assert set(masses) == set(oracle)
    for community, mass in oracle.items():
        assert abs(masses[community] - mass) < 1e-9

    score = (failures["n_fn"] + 0.5 * failures["n_low_conf"]) / (failures["n_gt"] + 0.5)
    s0, s1 = float(score.iloc[0]), float(score.iloc[1])
    # route_k=3 truncates f0's four canned neighbours to [p1, p2, pX].
    assert per_frame["p1"] == (1, pytest.approx(s0))
    assert per_frame["p2"] == (2, pytest.approx(s0 + s1))
    assert per_frame["p3"] == (1, pytest.approx(s1))
    assert "pX" not in per_frame  # outside every community -> no row at all
    assert "f2" not in per_frame  # below top_k, never routed

    # The per-frame rows sum, per community, to the same totals.
    totals: dict[int, float] = {}
    for token, (_n, mass) in per_frame.items():
        totals[communities[token]] = totals.get(communities[token], 0.0) + mass
    assert set(totals) == set(oracle)
    for community, mass in oracle.items():
        assert abs(totals[community] - mass) < 1e-9


# ---------------------------------------------------------------------------
# attribute_pick_pass
# ---------------------------------------------------------------------------

# Synthetic selection world, shared by the attribution / validation / rows tests.
# Community 1 holds two night members with different degrees so a quota of 2 can be
# checked against "top-degree night members first"; community 3 holds no night
# member at all, which forces the night pass short by one -> a seeded backfill draw
# from the disconnected night pool (z1/z2).
_COMMUNITIES = {
    "a1": 1, "a2": 1, "a3": 1,
    "b1": 2, "b2": 2,
    "c1": 3, "c2": 3, "c3": 3, "c4": 3,
}
_DEGREES = {
    "a1": 5.0, "a2": 9.0, "a3": 1.0,
    "b1": 4.0, "b2": 2.0,
    "c1": 8.0, "c2": 7.0, "c3": 6.0, "c4": 3.0,
}
_MASSES = {1: 10.0, 2: 5.0, 3: 1.0}
_NIGHT = {"a1", "a2", "b1", "z1", "z2"}
_NIGHT_FLOOR = 4
_N_MINE = 7


def _synthetic_selection() -> tuple[list[str], list[dict[str, Any]]]:
    return graph_mining.select_by_mass(
        _COMMUNITIES,
        _DEGREES,
        _MASSES,
        _N_MINE,
        night_tokens=_NIGHT,
        night_floor=_NIGHT_FLOOR,
        night_pool=sorted(_NIGHT),
        seed=64,
    )


def test_pick_pass_attribution_covers_selected_set_exactly() -> None:
    selected, _diagnostics = _synthetic_selection()
    assert len(selected) == _N_MINE

    passes = al_explain.attribute_pick_pass(
        _COMMUNITIES,
        _DEGREES,
        _MASSES,
        selected,
        night_tokens=_NIGHT,
        night_floor=_NIGHT_FLOOR,
        n_mine=_N_MINE,
    )

    assert set(passes) == set(selected)  # one label per selected token, no extras
    assert set(passes.values()) <= {"night", "main", "backfill"}
    night = {t for t, p in passes.items() if p == "night"}
    main = {t for t, p in passes.items() if p == "main"}
    backfill = {t for t, p in passes.items() if p == "backfill"}
    assert night | main | backfill == set(selected)  # a partition
    assert not (night & main) and not (night & backfill) and not (main & backfill)

    # Night pass: community 1's quota is 2, taken top-degree first (a2 deg 9 before
    # a1 deg 5); community 2's quota is 1 (b1); community 3 has no night member.
    assert night == {"a2", "a1", "b1"}
    # The floor of 4 could not be met from the communities -> exactly one seeded
    # backfill draw, from the disconnected night pool.
    assert len(backfill) == 1
    assert next(iter(backfill)) in {"z1", "z2"}
    assert main == {"a3", "b2", "c1"}


# ---------------------------------------------------------------------------
# validate_reproduction
# ---------------------------------------------------------------------------


def test_validation_rejects_one_token_difference_and_one_quota_difference() -> None:
    selected, diagnostics = _synthetic_selection()

    clean = al_explain.validate_reproduction(
        selected, diagnostics, persisted_tokens=list(selected),
        persisted_diagnostics=[dict(row) for row in diagnostics],
    )
    assert clean["selected_match"] is True
    assert clean["communities_match"] is True
    assert clean["n_selected"] == len(selected)
    assert clean["n_communities"] == len(diagnostics)
    assert clean["first_mismatch"] is None

    swapped = list(selected)
    swapped[-1] = "not-a-real-token"
    one_token = al_explain.validate_reproduction(
        selected, diagnostics, persisted_tokens=swapped,
        persisted_diagnostics=[dict(row) for row in diagnostics],
    )
    assert one_token["selected_match"] is False
    assert one_token["first_mismatch"] is not None
    assert "not-a-real-token" in one_token["first_mismatch"]

    off_by_one = [dict(row) for row in diagnostics]
    off_by_one[0]["quota"] = int(off_by_one[0]["quota"]) + 1
    one_quota = al_explain.validate_reproduction(
        selected, diagnostics, persisted_tokens=list(selected),
        persisted_diagnostics=off_by_one,
    )
    assert one_quota["selected_match"] is True
    assert one_quota["communities_match"] is False
    assert "quota" in one_quota["first_mismatch"]

    # mass is compared to 4 dp: below that is the same number, at that scale it isn't.
    below = [dict(row) for row in diagnostics]
    below[0]["mass"] = float(below[0]["mass"]) + 1e-6
    assert al_explain.validate_reproduction(
        selected, diagnostics, persisted_tokens=list(selected),
        persisted_diagnostics=below,
    )["communities_match"] is True
    at_scale = [dict(row) for row in diagnostics]
    at_scale[0]["mass"] = float(at_scale[0]["mass"]) + 0.001
    at_scale_result = al_explain.validate_reproduction(
        selected, diagnostics, persisted_tokens=list(selected),
        persisted_diagnostics=at_scale,
    )
    assert at_scale_result["communities_match"] is False
    assert "mass" in at_scale_result["first_mismatch"]


# ---------------------------------------------------------------------------
# build_explain_rows
# ---------------------------------------------------------------------------

_EXPLAIN_COLUMNS = [
    "sample_data_token", "arm", "is_night", "scene_name",
    "community", "community_size", "community_night_members", "community_mass",
    "community_mass_rank", "community_quota",
    "degree", "degree_rank_in_community",
    "pick_pass", "n_failures_routed", "mass_routed",
]


def test_explain_rows_schema_and_ranks() -> None:
    selected, diagnostics = _synthetic_selection()
    passes = al_explain.attribute_pick_pass(
        _COMMUNITIES, _DEGREES, _MASSES, selected,
        night_tokens=_NIGHT, night_floor=_NIGHT_FLOOR, n_mine=_N_MINE,
    )
    per_frame = {"a2": (3, 1.5), "c1": (1, 0.25)}
    meta = pd.DataFrame(
        {
            "sample_data_token": sorted(set(_COMMUNITIES) | {"z1", "z2"}),
            "scene_name": ["sP"] * (len(_COMMUNITIES) + 2),
            "is_night": [
                token in _NIGHT for token in sorted(set(_COMMUNITIES) | {"z1", "z2"})
            ],
        }
    ).set_index("sample_data_token")

    rows = al_explain.build_explain_rows(
        selected, _COMMUNITIES, _DEGREES, _MASSES, per_frame, passes, diagnostics, meta,
        arm="graph_rate_night",
    )

    assert list(rows.columns) == _EXPLAIN_COLUMNS
    assert len(rows) == len(selected)
    assert list(rows["sample_data_token"]) == sorted(selected)
    assert set(rows["arm"]) == {"graph_rate_night"}

    by_token = rows.set_index("sample_data_token")
    # community 1 is the heaviest (mass 10) -> rank 1; 2 -> 2; 3 -> 3.
    assert int(by_token.loc["a1", "community_mass_rank"]) == 1
    assert int(by_token.loc["b1", "community_mass_rank"]) == 2
    assert int(by_token.loc["c1", "community_mass_rank"]) == 3
    # degree rank 1 = highest degree in the community (a2 deg 9 over a1 deg 5).
    assert int(by_token.loc["a2", "degree_rank_in_community"]) == 1
    assert int(by_token.loc["a1", "degree_rank_in_community"]) == 2
    assert int(by_token.loc["a3", "degree_rank_in_community"]) == 3
    assert int(by_token.loc["c1", "degree_rank_in_community"]) == 1
    assert float(by_token.loc["a2", "degree"]) == 9.0
    assert int(by_token.loc["a1", "community_size"]) == 3
    assert int(by_token.loc["a1", "community_night_members"]) == 2
    assert float(by_token.loc["a1", "community_mass"]) == 10.0
    assert int(by_token.loc["a1", "community_quota"]) == 3  # 2 night + 1 main
    assert by_token.loc["a2", "pick_pass"] == "night"
    assert by_token.loc["c1", "pick_pass"] == "main"
    assert int(by_token.loc["a2", "n_failures_routed"]) == 3
    assert float(by_token.loc["a2", "mass_routed"]) == 1.5
    assert int(by_token.loc["a1", "n_failures_routed"]) == 0
    assert float(by_token.loc["a1", "mass_routed"]) == 0.0

    # The backfilled token belongs to no community -> the -1 sentinel.
    backfilled = [t for t, p in passes.items() if p == "backfill"]
    assert len(backfilled) == 1
    assert int(by_token.loc[backfilled[0], "community"]) == -1
    assert bool(by_token.loc[backfilled[0], "is_night"]) is True


# ---------------------------------------------------------------------------
# run_al_explain — nothing is staged unless the re-derivation reproduces the run
# ---------------------------------------------------------------------------


def _write_al_inputs(tmp_path: Path, *, persisted_tokens: list[str]) -> dict[str, Path]:
    """A miniature AL state dir + processed dir + config for run_al_explain."""
    processed = tmp_path / "processed"
    processed.mkdir()
    al_dir = tmp_path / "al"
    al_dir.mkdir()

    tokens = ["t0", "t1", "t2", "t3", "t4"]
    pd.DataFrame(
        {
            "sample_data_token": [*tokens, "b0"],
            "scene_name": ["sP"] * 5 + ["sB"],
            "channel": ["CAM_FRONT"] * 6,
            "is_night": [True, False, False, False, False, False],
            "is_rain": [False] * 6,
        }
    ).to_parquet(processed / "samples.parquet", index=False)
    pd.DataFrame(
        {
            "scene_name": ["sP", "sB"],
            "is_night": [True, False],
            "n_frames": [5, 1],
            "role": ["pool", "baseline"],
        }
    ).to_parquet(al_dir / "split.parquet", index=False)
    _failures_frame().to_parquet(al_dir / "failures.parquet", index=False)
    pd.DataFrame({"sample_data_token": persisted_tokens}).to_parquet(
        al_dir / "graph_rate_night.parquet", index=False
    )
    (al_dir / "communities_graph_rate_night.json").write_text(
        json.dumps(
            [
                {"community": 1, "size": 2, "mass": 2.0, "night_members": 1, "quota": 2},
                {"community": 2, "size": 3, "mass": 1.0, "night_members": 0, "quota": 1},
            ]
        )
    )

    config = tmp_path / "active_learning.yaml"
    config.write_text(
        "split: {channel: CAM_FRONT}\n"
        "mining: {n_mine: 3, seed: 64}\n"
        "sweep: {top_k_failures: 3}\n"
        "graph_mining:\n"
        "  route_k: 3\n"
        "  arms:\n"
        "    graph_rate_night: {weighting: rate_mass, quotas: {is_night: 1}}\n"
        "state: {dir: unused}\n"
        f"engine_config: {tmp_path / 'engine.yaml'}\n"
    )
    (tmp_path / "engine.yaml").write_text("lancedb: {path: unused, table: frames}\n")
    return {"processed": processed, "al_dir": al_dir, "config": config}


def _patch_al_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the Neo4j/LanceDB seams with deterministic synthetic answers."""
    communities = {"t0": 1, "t1": 1, "t2": 2, "t3": 2, "t4": 2}
    degrees = {"t0": 5.0, "t1": 4.0, "t2": 3.0, "t3": 2.0, "t4": 1.0}
    masses = {1: 2.0, 2: 1.0}
    per_frame = {"t0": (1, 2.0), "t2": (1, 1.0)}

    monkeypatch.setattr(al_explain.connection, "get_driver", lambda _settings: object())
    monkeypatch.setattr(al_explain.connection, "close", lambda _driver: None)
    monkeypatch.setattr(al_explain, "_gds_version", lambda _driver, _database: "2.13.11")
    monkeypatch.setattr(al_explain, "_open_frames_table", lambda _cfg: object())
    monkeypatch.setattr(
        al_explain, "_louvain_communities",
        lambda _driver, _tokens, _database, seed=64: (communities, degrees),
    )
    monkeypatch.setattr(
        al_explain, "route_failure_mass_per_frame",
        lambda *args, **kwargs: (masses, per_frame),
    )
    monkeypatch.setattr(al_explain, "route_failure_mass", lambda *args, **kwargs: masses)


def test_run_al_explain_raises_and_stages_nothing_on_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The re-derivation selects [t0, t1, t2]; the persisted arm claims t3 instead of t2.
    paths = _write_al_inputs(tmp_path, persisted_tokens=["t0", "t1", "t3"])
    _patch_al_seams(monkeypatch)
    out_dir = tmp_path / "staged"
    out_dir.mkdir()

    with pytest.raises(ValueError, match="reproduc"):
        al_explain.run_al_explain(
            al_config_path=paths["config"],
            processed_dir=paths["processed"],
            al_dir=paths["al_dir"],
            out_dir=out_dir,
        )
    assert list(out_dir.iterdir()) == []


def test_run_al_explain_stages_both_files_when_it_reproduces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_al_inputs(tmp_path, persisted_tokens=["t0", "t1", "t2"])
    _patch_al_seams(monkeypatch)
    out_dir = tmp_path / "staged"

    summary = al_explain.run_al_explain(
        al_config_path=paths["config"],
        processed_dir=paths["processed"],
        al_dir=paths["al_dir"],
        out_dir=out_dir,
    )

    assert summary["selected_match"] is True
    assert summary["communities_match"] is True
    validation = json.loads((out_dir / "al_explain_validation.json").read_text())
    assert validation["arm"] == "graph_rate_night"
    assert validation["n_selected"] == 3
    assert validation["n_communities"] == 2
    assert validation["gds_version"] == "2.13.11"
    assert validation["config"] == {
        "n_mine": 3, "route_k": 3, "top_k": 3, "night_floor": 1,
        "seed": 64, "channel": "CAM_FRONT",
    }
    rows = pd.read_parquet(out_dir / "al_selection_explain.parquet")
    assert list(rows.columns) == _EXPLAIN_COLUMNS
    assert list(rows["sample_data_token"]) == ["t0", "t1", "t2"]
    assert list(rows["pick_pass"]) == ["night", "main", "main"]


def test_run_al_explain_rejects_a_non_rate_mass_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_al_inputs(tmp_path, persisted_tokens=["t0", "t1", "t2"])
    _patch_al_seams(monkeypatch)
    with pytest.raises(ValueError, match="rate_mass"):
        al_explain.run_al_explain(
            al_config_path=paths["config"],
            processed_dir=paths["processed"],
            al_dir=paths["al_dir"],
            out_dir=tmp_path / "staged",
            arm="graph",
        )


# ---------------------------------------------------------------------------
# build._include_al_explain
# ---------------------------------------------------------------------------


def test_include_al_explain_absent_partial_included(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo import build

    staging = tmp_path / "staging"
    out_dir = tmp_path / "demo_data"
    out_dir.mkdir()
    config: dict[str, Any] = {
        "curation": {"staging_dir": str(staging)},
        "al": {"arm": "graph_rate_night"},
    }

    # absent: no staging dir at all
    assert build._include_al_explain(config, out_dir) == "absent"
    assert list(out_dir.iterdir()) == []

    # partial: the parquet without its validation JSON
    explain_dir = staging / "al_explain"
    explain_dir.mkdir(parents=True)
    pd.DataFrame({"sample_data_token": ["t0"]}).to_parquet(
        explain_dir / "al_selection_explain.parquet", index=False
    )
    with pytest.raises(ValueError, match=r"al_explain_validation\.json"):
        build._include_al_explain(config, out_dir)
    assert list(out_dir.iterdir()) == []

    # included: both files, matching arm, both match flags True
    (explain_dir / "al_explain_validation.json").write_text(
        json.dumps({
            "arm": "graph_rate_night", "selected_match": True, "communities_match": True,
            "n_selected": 1, "n_communities": 1,
        })
    )
    assert build._include_al_explain(config, out_dir) == "included"
    assert (out_dir / "al_selection_explain.parquet").is_file()
    assert (out_dir / "al_explain_validation.json").is_file()


def test_include_al_explain_rejects_stale_or_unvalidated_staging(tmp_path: Path) -> None:
    from nuscenes_data_engine.demo import build

    staging = tmp_path / "staging"
    out_dir = tmp_path / "demo_data"
    out_dir.mkdir()
    explain_dir = staging / "al_explain"
    explain_dir.mkdir(parents=True)
    pd.DataFrame({"sample_data_token": ["t0"]}).to_parquet(
        explain_dir / "al_selection_explain.parquet", index=False
    )
    config: dict[str, Any] = {
        "curation": {"staging_dir": str(staging)},
        "al": {"arm": "graph_rate_night"},
    }

    # a staging for a DIFFERENT arm must never ship as this package's arm
    (explain_dir / "al_explain_validation.json").write_text(
        json.dumps({"arm": "graph_rate", "selected_match": True, "communities_match": True})
    )
    with pytest.raises(ValueError, match="graph_rate"):
        build._include_al_explain(config, out_dir)

    # a staging that did NOT reproduce the run must never ship either
    (explain_dir / "al_explain_validation.json").write_text(
        json.dumps({
            "arm": "graph_rate_night", "selected_match": True, "communities_match": False,
        })
    )
    with pytest.raises(ValueError, match="reproduc"):
        build._include_al_explain(config, out_dir)
    assert list(out_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# live re-derivation of the real arm (self-skips when Neo4j/LanceDB are absent)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_live_reproduction_of_graph_rate_night(tmp_path: Path) -> None:
    pytest.importorskip("neo4j")
    pytest.importorskip("lancedb")
    from nuscenes_data_engine.config import get_settings
    from nuscenes_data_engine.data_engine.graph import connection

    al_dir = Path("data/active_learning")
    processed = Path("data/processed")
    al_config = Path("configs/active_learning.yaml")
    required = [
        al_dir / "graph_rate_night.parquet",
        al_dir / "communities_graph_rate_night.json",
        al_dir / "failures.parquet",
        al_dir / "split.parquet",
        processed / "samples.parquet",
        al_config,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing or not Path("data/lancedb").is_dir():
        pytest.skip(f"local AL artifacts not available: {missing or 'data/lancedb'}")

    settings = get_settings()
    try:
        driver = connection.get_driver(settings)
        connection.read_query(driver, "RETURN 1 AS ok", database=settings.neo4j_database)
        connection.close(driver)
    except Exception as exc:  # no compose Neo4j running -> skipped, not failed
        pytest.skip(f"Neo4j not reachable: {exc}")

    summary = al_explain.run_al_explain(
        al_config_path=al_config,
        processed_dir=processed,
        al_dir=al_dir,
        out_dir=tmp_path,
    )

    validation = json.loads((tmp_path / "al_explain_validation.json").read_text())
    assert validation["selected_match"] is True
    assert validation["communities_match"] is True
    assert validation["first_mismatch"] is None
    assert validation["n_selected"] == 1500
    assert validation["n_communities"] == 97
    assert summary["selected_match"] is True

    rows = pd.read_parquet(tmp_path / "al_selection_explain.parquet")
    assert len(rows) == 1500
    assert list(rows.columns) == _EXPLAIN_COLUMNS
    assert set(rows["arm"]) == {"graph_rate_night"}
    counts = rows["pick_pass"].value_counts().to_dict()
    # The persisted diagnostics carry no community == -1 row, so the run's night
    # pass never backfilled -- the re-derivation must not either.
    assert counts.get("backfill", 0) == 0
    assert counts["night"] == validation["config"]["night_floor"]
    assert int(rows["is_night"].sum()) >= validation["config"]["night_floor"]
    # Every routed pool frame's mass sums back to the community total (the whole
    # point of the per-frame accumulators). The 1500 SELECTED frames are a subset of
    # the routed pool, so the staged table's own sum is bounded by it, not equal.
    assert summary["mass_routed_total"] == pytest.approx(validation["mass_total"], abs=1e-3)
    assert 0.0 < float(rows["mass_routed"].sum()) <= validation["mass_total"] + 1e-6
