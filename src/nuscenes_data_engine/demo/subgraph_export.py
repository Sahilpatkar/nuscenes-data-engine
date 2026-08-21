"""``demo/subgraph_export.py`` — per-event explanatory subgraphs from the live graph.

Neo4j is an *operational* dependency, handled like ``demo semsearch``: the ``demo
subgraphs`` CLI command (``cli.py``) stages ``graph_subgraphs/<preset>.json`` files
under ``data/demo_curation/``, which ``demo build`` (Task 2) copies into the package
when present. See docs/superpowers/specs/2026-08-20-demo-phase6-design.md §1.

Three layers, each independently testable:

- **Count queries** (``count_cypher``): one Cypher COUNT per dynamics preset (the
  flagship, ``hard_braking_near_pedestrians``, is asserted textually equal to
  ``docs/GRAPH.md``'s verbatim fenced query — modulo whitespace/``//`` comments — by
  ``tests/test_demo_subgraphs.py``, so it stays a literal ``10``/``'pedestrian'``
  rather than a ``$near`` parameter: parameterizing it would make the text diverge
  from the doc). Model-result presets (``fn_pedestrians_night``, ``low_conf_
  braking``) have no graph-side count — their verdict lives only in the staged
  prediction/GT parquets, not the graph — so ``count_cypher`` returns ``None`` for
  them.
- **The subgraph query** (``event_subgraph_cypher``): ONE query, parameterized only
  by ``$sample_token``, that returns everything ``assemble_subgraph`` needs for any
  preset (the Sample, its Scene/Location, its EgoPose, every ObjectObservation with
  its Category, and its NEXT-prev/next Sample tokens) — one row per
  ObjectObservation (or a single all-null-object row when the sample has none), with
  the Scene/Sample/EgoPose/Location/prev/next columns repeated across rows. Running
  the SAME query for every preset (rather than a preset-specific WHERE) is what lets
  one function serve all six: which objects are "on the matched path" is a pure
  Python decision (``assemble_subgraph``), not a Cypher filter.
- **Assembly** (``assemble_subgraph``): pure — turns those rows into agraph-ready
  ``{nodes, edges, path}``, deduping nodes by id, capping non-matching
  ObjectObservations to the ``cap_other`` nearest so a crowded frame stays legible,
  and marking ``on_path`` for the backbone (Scene→Sample→EgoPose) and each matching
  ObjectObservation→Category chain.

``export_subgraphs`` ties it together per preset: run the count query (dynamics) or
skip it (model), compare against the caller-supplied SQL-side ``sql_counts``
(``events.preset_counts``), assert parity for the flagship (raises on mismatch) and
flag-but-don't-raise for the other three (a schema-coverage gap there is a finding to
report, not hide), then assemble + write one subgraph per tagged event.

``run_query`` is injected (production: a thin wrapper around
``graph/connection.py::read_query``) so this module never imports the ``neo4j``
driver itself — tests inject a fake returning canned records.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from nuscenes_data_engine.demo.exporters import write_json

logger = logging.getLogger("nuscenes_data_engine")

# A run_query callable: (cypher_text, **params) -> rows (plain dicts), the same
# shape graph/connection.py::read_query returns. Kept loose (not a Protocol) since
# the only real implementation is a one-line lambda around read_query (cli.py).
RunQuery = Callable[..., list[dict[str, Any]]]

_DYNAMICS_PRESETS = ("hard_braking_near_pedestrians", "night_pedestrians", "fast_cyclists", "rain_vru")
_MODEL_PRESETS = ("fn_pedestrians_night", "low_conf_braking")
_ALL_PRESETS = (*_DYNAMICS_PRESETS, *_MODEL_PRESETS)

_MODEL_PRESET_NOTE = "the model verdict is not in the graph"

# Verbatim (modulo whitespace/`//` comments — tests/test_demo_subgraphs.py asserts
# this textually) with docs/GRAPH.md's flagship fenced Cypher. Deliberately a
# LITERAL "10" / 'pedestrian', not a $near parameter: the doc's fenced query has no
# parameter, and matching it verbatim is the whole point of the assertion.
_FLAGSHIP_CYPHER = """
MATCH (e:EgoPose {is_hard_braking: true})<-[:AT_POSE]-(s:Sample)
      -[:HAS_OBJECT]->(o:ObjectObservation)-[:OF_CATEGORY]->(c:Category)
WHERE c.group = 'pedestrian' AND o.distance_to_ego_m < 10
RETURN count(DISTINCT s)
"""

# The other three dynamics presets built from the same predicates demo/events.py's
# build_events uses (near_dist_m / high_speed_mps), parameterized so the query TEXT
# doesn't change when the config's thresholds do.
_NIGHT_PEDESTRIANS_CYPHER = """
MATCH (sc:Scene {is_night: true})<-[:IN_SCENE]-(s:Sample)
      -[:HAS_OBJECT]->(o:ObjectObservation)-[:OF_CATEGORY]->(c:Category)
WHERE c.group = 'pedestrian' AND o.distance_to_ego_m < $near
RETURN count(DISTINCT s)
"""

_FAST_CYCLISTS_CYPHER = """
MATCH (e:EgoPose)<-[:AT_POSE]-(s:Sample)
      -[:HAS_OBJECT]->(o:ObjectObservation)-[:OF_CATEGORY]->(c:Category)
WHERE e.speed_mps >= $speed AND c.group = 'bicycle' AND o.distance_to_ego_m < $near
RETURN count(DISTINCT s)
"""

_RAIN_VRU_CYPHER = """
MATCH (sc:Scene {is_rain: true})<-[:IN_SCENE]-(s:Sample)
      -[:HAS_OBJECT]->(o:ObjectObservation)-[:OF_CATEGORY]->(c:Category)
WHERE c.group IN ['pedestrian', 'bicycle'] AND o.distance_to_ego_m < $near
RETURN count(DISTINCT s)
"""

_DYNAMICS_CYPHER: dict[str, str] = {
    "hard_braking_near_pedestrians": _FLAGSHIP_CYPHER,
    "night_pedestrians": _NIGHT_PEDESTRIANS_CYPHER,
    "fast_cyclists": _FAST_CYCLISTS_CYPHER,
    "rain_vru": _RAIN_VRU_CYPHER,
}

# Which thresholds each dynamics preset's query needs present (validated up front by
# count_cypher, even though the flagship's own text doesn't reference them — a
# missing near_dist_m/high_speed_mps in the caller's config is a configuration bug
# worth catching before a query round-trip, not after).
_REQUIRED_THRESHOLD_KEYS: dict[str, tuple[str, ...]] = {
    "hard_braking_near_pedestrians": (),
    "night_pedestrians": ("near_dist_m",),
    "fast_cyclists": ("near_dist_m", "high_speed_mps"),
    "rain_vru": ("near_dist_m",),
}


def count_cypher(preset: str, thresholds: Mapping[str, float]) -> str | None:
    """The dynamics count query for ``preset``, or ``None`` for a model preset.

    ``thresholds`` isn't interpolated into the query TEXT (the other three presets
    use ``$near``/``$speed`` parameters, run via ``_count_params`` at query time) —
    it's validated here so a caller misconfigured ``configs/demo.yaml``'s
    ``presets:`` section fails at text-generation time, not after a round-trip to
    the database.
    """
    if preset in _MODEL_PRESETS:
        return None
    if preset not in _DYNAMICS_CYPHER:
        raise ValueError(f"subgraph_export.count_cypher: unknown preset {preset!r}")
    missing = [key for key in _REQUIRED_THRESHOLD_KEYS[preset] if key not in thresholds]
    if missing:
        raise ValueError(
            f"subgraph_export.count_cypher: preset {preset!r} needs thresholds {missing}"
        )
    return _DYNAMICS_CYPHER[preset]


def _count_params(preset: str, thresholds: Mapping[str, float]) -> dict[str, float]:
    """The ``$near``/``$speed`` parameter values for ``preset``'s count query."""
    if preset in ("night_pedestrians", "rain_vru"):
        return {"near": float(thresholds["near_dist_m"])}
    if preset == "fast_cyclists":
        return {"near": float(thresholds["near_dist_m"]), "speed": float(thresholds["high_speed_mps"])}
    return {}


def event_subgraph_cypher() -> str:
    """One query, parameterized by ``$sample_token``, serving every preset.

    Returns one row per ObjectObservation on the sample (or one all-null-object row
    when it has none), with the Sample/Scene/Location/EgoPose/prev/next columns
    repeated across rows — ``assemble_subgraph`` dedupes those by id. Which objects
    end up ``on_path`` is a Python decision (the preset predicate), not a Cypher
    WHERE, so this same text works for every preset.
    """
    return """
MATCH (s:Sample {token: $sample_token})
OPTIONAL MATCH (s)-[:IN_SCENE]->(scene:Scene)-[:IN_LOCATION]->(loc:Location)
OPTIONAL MATCH (s)-[:AT_POSE]->(ego:EgoPose)
OPTIONAL MATCH (prev:Sample)-[:NEXT]->(s)
OPTIONAL MATCH (s)-[:NEXT]->(nxt:Sample)
OPTIONAL MATCH (s)-[:HAS_OBJECT]->(o:ObjectObservation)-[:OF_CATEGORY]->(c:Category)
RETURN s AS sample, scene AS scene, loc AS location, ego AS ego,
       prev.token AS prev_token, nxt.token AS next_token,
       o AS object, c AS category
"""


def _matches_preset(
    preset: str, category_group: str | None, distance_m: float | None, thresholds: Mapping[str, float]
) -> bool:
    """Whether one ObjectObservation is "on the matched path" for ``preset``.

    Mirrors demo/events.py::build_events' predicates (group is the graph's raw
    car/truck/bus/pedestrian/bicycle taxonomy, not events.py's collapsed vehicle/
    cyclist context groups). Model presets have no graph-derivable predicate — their
    verdict isn't in the graph — so every object is "non-matching" for them (still
    shown, just none marked on_path).
    """
    if category_group is None or distance_m is None:
        return False
    near = float(thresholds.get("near_dist_m", 10.0))
    if preset in ("hard_braking_near_pedestrians", "night_pedestrians"):
        return category_group == "pedestrian" and distance_m < near
    if preset == "fast_cyclists":
        return category_group == "bicycle" and distance_m < near
    if preset == "rain_vru":
        return category_group in ("pedestrian", "bicycle") and distance_m < near
    return False


def assemble_subgraph(
    records: Sequence[Mapping[str, Any]],
    *,
    preset: str,
    thresholds: Mapping[str, float],
    cap_other: int = 12,
) -> dict[str, Any]:
    """Pure: ``event_subgraph_cypher()`` rows -> agraph-ready ``{nodes, edges, path}``.

    Nodes: ``{id, label, group, on_path, meta}`` — ``label`` is the node's graph type
    (Scene/Sample/EgoPose/ObjectObservation/Category/Location), ``group`` is the
    finer coloring key (the taxonomy group for ObjectObservation/Category, the
    lowercased label otherwise), deduped by ``id``. Edges: ``{source, target,
    label}`` referencing only ids already added. ``path``: a list of ordered id
    chains — ``[scene, sample, egopose]`` (the backbone) plus one
    ``[sample, object, category]`` chain per matching (``on_path``) object,
    NEAREST FIRST -- ``path[1]`` is the nearest matching object, the same one the
    Scenario card's caption quotes.

    Only Scene/Sample/EgoPose and the matching objects (+ their Category) are
    ``on_path``; Location and the prev/next temporal-context Samples are always
    off-path context. BOTH object lists are sorted by ``distance_to_ego_m`` (ties
    broken by token, deterministic); the non-matching one is additionally capped to
    the ``cap_other`` nearest so a crowded frame stays legible; their Categories are
    included too, also off-path.
    """
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    edge_keys: set[tuple[str, str, str]] = set()

    def add_node(node_id: str, label: str, group: str, on_path: bool, meta: dict[str, Any]) -> None:
        existing = nodes.get(node_id)
        if existing is None:
            nodes[node_id] = {
                "id": node_id, "label": label, "group": group, "on_path": on_path, "meta": meta,
            }
        elif on_path and not existing["on_path"]:
            existing["on_path"] = True

    def add_edge(source: str, target: str, label: str) -> None:
        key = (source, target, label)
        if key not in edge_keys:
            edge_keys.add(key)
            edges.append({"source": source, "target": target, "label": label})

    if not records:
        return {"nodes": [], "edges": [], "path": []}

    anchor = records[0]
    sample = anchor.get("sample")
    if not sample or "token" not in sample:
        return {"nodes": [], "edges": [], "path": []}
    sample_id = f"sample:{sample['token']}"

    scene = anchor.get("scene")
    location = anchor.get("location")
    ego = anchor.get("ego")
    prev_token = anchor.get("prev_token")
    next_token = anchor.get("next_token")

    add_node(
        sample_id, "Sample", "sample", True,
        {
            "timestamp": sample.get("timestamp"),
            "scene": scene.get("name") if scene else None,
            "is_night": scene.get("is_night") if scene else None,
            "is_rain": scene.get("is_rain") if scene else None,
        },
    )

    scene_id: str | None = None
    if scene and "token" in scene:
        scene_id = f"scene:{scene['token']}"
        add_node(
            scene_id, "Scene", "scene", True,
            {
                "name": scene.get("name"),
                "location": scene.get("location"),
                "is_night": scene.get("is_night"),
                "is_rain": scene.get("is_rain"),
            },
        )
        add_edge(sample_id, scene_id, "IN_SCENE")

    if location and "name" in location:
        location_id = f"location:{location['name']}"
        add_node(location_id, "Location", "location", False, {"name": location.get("name")})
        if scene_id is not None:
            add_edge(scene_id, location_id, "IN_LOCATION")

    ego_id: str | None = None
    if ego:
        ego_key = ego.get("token") or sample["token"]
        ego_id = f"egopose:{ego_key}"
        add_node(
            ego_id, "EgoPose", "egopose", True,
            {
                "speed_mps": ego.get("speed_mps"),
                "accel_long_min_mps2": ego.get("accel_long_min_mps2"),
                "is_hard_braking": ego.get("is_hard_braking"),
                "can_speed_kmh": ego.get("can_speed_kmh"),
            },
        )
        add_edge(sample_id, ego_id, "AT_POSE")

    if prev_token:
        prev_id = f"sample:{prev_token}"
        add_node(prev_id, "Sample", "sample", False, {"sample_token": prev_token})
        add_edge(prev_id, sample_id, "NEXT")
    if next_token:
        next_id = f"sample:{next_token}"
        add_node(next_id, "Sample", "sample", False, {"sample_token": next_token})
        add_edge(sample_id, next_id, "NEXT")

    # Dedupe ObjectObservations by their own token across rows (one row per object
    # by construction, but be defensive) before splitting matching/non-matching.
    objects: dict[str, dict[str, Any]] = {}
    categories: dict[str, dict[str, Any] | None] = {}
    order: list[str] = []
    for row in records:
        obj = row.get("object")
        if not obj or "token" not in obj:
            continue
        token = obj["token"]
        if token not in objects:
            order.append(token)
        objects[token] = obj
        categories[token] = row.get("category")

    matching: list[str] = []
    non_matching: list[str] = []
    for token in order:
        obj = objects[token]
        cat = categories.get(token)
        group = cat.get("group") if cat else None
        if _matches_preset(preset, group, obj.get("distance_to_ego_m"), thresholds):
            matching.append(token)
        else:
            non_matching.append(token)

    def _distance_key(token: str) -> tuple[float, str]:
        dist = objects[token].get("distance_to_ego_m")
        return (float(dist) if dist is not None else float("inf"), token)

    # BOTH lists are distance-sorted, not just the capped one: the page reads
    # `path[1]` for its "why this event" narrative and each Scenario card quotes the
    # NEAREST matching object, so appending matching objects in raw Neo4j row order
    # made the two disagree on 31/120 real cards. Sorting here also makes the edge
    # order deterministic by construction rather than by driver row order.
    matching.sort(key=_distance_key)
    non_matching.sort(key=_distance_key)
    kept_non_matching = non_matching[:cap_other]

    path: list[list[str]] = []
    if scene_id is not None and ego_id is not None:
        path.append([scene_id, sample_id, ego_id])

    def add_object(token: str, on_path: bool) -> None:
        obj = objects[token]
        cat = categories.get(token)
        obj_id = f"object:{token}"
        group = (cat.get("group") if cat else None) or "unknown"
        add_node(
            obj_id, "ObjectObservation", group, on_path,
            {
                "category": obj.get("category"),
                "group": group,
                "distance_to_ego_m": obj.get("distance_to_ego_m"),
                "ego_rel_x": obj.get("ego_rel_x"),
                "ego_rel_y": obj.get("ego_rel_y"),
                "visibility": obj.get("visibility"),
            },
        )
        add_edge(sample_id, obj_id, "HAS_OBJECT")
        cat_id: str | None = None
        if cat and "name" in cat:
            cat_id = f"category:{cat['name']}"
            add_node(
                cat_id, "Category", cat.get("group") or "unknown", on_path,
                {"name": cat.get("name"), "group": cat.get("group")},
            )
            add_edge(obj_id, cat_id, "OF_CATEGORY")
        if on_path and cat_id is not None:
            path.append([sample_id, obj_id, cat_id])

    for token in matching:
        add_object(token, True)
    for token in kept_non_matching:
        add_object(token, False)

    return {
        "nodes": sorted(nodes.values(), key=lambda n: str(n["id"])),
        "edges": edges,
        "path": path,
    }


def _first_scalar(rows: list[dict[str, Any]]) -> int:
    if not rows:
        raise ValueError("subgraph_export: count query returned no rows")
    return int(next(iter(rows[0].values())))


def export_subgraphs(
    *,
    run_query: RunQuery,
    events: pd.DataFrame,
    thresholds: Mapping[str, float],
    sql_counts: Mapping[str, int],
    flagship_expected: int,
    out_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Per preset: count parity (dynamics) + a subgraph for every tagged event.

    Writes ``out_dir/<preset>.json`` = ``{preset, count_cypher, sql_count,
    cypher_count, parity, note?, events}`` (sorted keys, indent 2, via
    ``exporters.write_json`` — deterministic bytes for the same inputs) for all six
    presets, and returns a ``{preset: {sql_count, cypher_count, parity, n_events}}``
    summary for the CLI to log.

    The flagship's parity is ASSERTED — a mismatch raises ``ValueError`` naming both
    counts, since it's the plan's headline claim ("the SQL and Cypher counts agree").
    The other three dynamics presets are recorded with a ``parity`` flag and a
    ``logger.warning`` on mismatch, never a raise: a schema-coverage gap there is a
    finding to report honestly, not something to hide or block the whole export on.
    Model presets get ``cypher_count: None`` and a note — their verdict isn't in the
    graph — but still get a full per-event subgraph (the Sample/Scene/EgoPose/
    objects are all graph data; only the model's verdict is absent).
    """
    out_dir = Path(out_dir)
    subgraph_cypher = event_subgraph_cypher()
    summary: dict[str, dict[str, Any]] = {}

    for preset in _ALL_PRESETS:
        cypher_text = count_cypher(preset, thresholds)
        cypher_count: int | None = None
        parity: bool | None = None
        note: str | None = None
        sql_count = int(sql_counts[preset])

        if cypher_text is not None:
            rows = run_query(cypher_text, **_count_params(preset, thresholds))
            cypher_count = _first_scalar(rows)
            parity = cypher_count == sql_count
            if preset == "hard_braking_near_pedestrians":
                if sql_count != flagship_expected:
                    raise ValueError(
                        f"subgraph_export: flagship SQL count {sql_count} != expected "
                        f"{flagship_expected} — re-verify before comparing to the graph"
                    )
                if cypher_count != sql_count:
                    raise ValueError(
                        "subgraph_export: flagship parity mismatch — sql="
                        f"{sql_count} cypher={cypher_count}"
                    )
            elif not parity:
                logger.warning(
                    "demo subgraphs: preset %r parity mismatch — sql=%d cypher=%s",
                    preset, sql_count, cypher_count,
                )
        else:
            note = _MODEL_PRESET_NOTE

        tagged = events[events["preset_tags"].apply(lambda tags, p=preset: p in tags)]
        event_subgraphs: dict[str, Any] = {}
        for _, row in tagged.iterrows():
            sample_data_token = str(row["sample_data_token"])
            sample_token = str(row["sample_token"])
            records = run_query(subgraph_cypher, sample_token=sample_token)
            event_subgraphs[sample_data_token] = assemble_subgraph(
                records, preset=preset, thresholds=thresholds
            )

        payload: dict[str, Any] = {
            "preset": preset,
            "count_cypher": cypher_text,
            "sql_count": sql_count,
            "cypher_count": cypher_count,
            "parity": parity,
            "events": event_subgraphs,
        }
        if note is not None:
            payload["note"] = note

        write_json(out_dir / f"{preset}.json", payload)
        summary[preset] = {
            "sql_count": sql_count,
            "cypher_count": cypher_count,
            "parity": parity,
            "n_events": len(event_subgraphs),
        }

    return summary
