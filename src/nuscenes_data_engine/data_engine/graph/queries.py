"""A small library of canned read-only Cypher queries.

Powers ``graph query --canned <name>`` and ``graph stats``, seeds the docs, and gives the
chat agent worked examples. Every query here passes the read-only guard.
"""

from __future__ import annotations

# name -> (human description, cypher). All read-only.
CANNED: dict[str, tuple[str, str]] = {
    "stats": (
        "Node counts by label.",
        "MATCH (n) UNWIND labels(n) AS label "
        "RETURN label, count(*) AS n ORDER BY n DESC",
    ),
    "rel_stats": (
        "Relationship counts by type.",
        "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY n DESC",
    ),
    "top_co_occurrence": (
        "Category pairs that co-occur in the most frames.",
        "MATCH (a:Category)-[c:CO_OCCURS_WITH]->(b:Category) "
        "RETURN a.name AS a, b.name AS b, c.n_frames AS frames ORDER BY frames DESC LIMIT 15",
    ),
    "night_categories": (
        "Most common categories in night frames.",
        "MATCH (f:Frame)-[:IN_SCENE]->(s:Scene) WHERE s.is_night "
        "MATCH (f)-[c:CONTAINS]->(cat:Category) "
        "RETURN cat.name AS category, sum(c.count) AS boxes ORDER BY boxes DESC LIMIT 15",
    ),
    "top_hazards": (
        "Most frequent VLM-reported hazards.",
        "MATCH (:Frame)-[:HAS_HAZARD]->(h:Hazard) "
        "RETURN h.text AS hazard, count(*) AS frames ORDER BY frames DESC LIMIT 15",
    ),
    "location_conditions": (
        "Scene counts per location, split by night.",
        "MATCH (s:Scene)-[:IN_LOCATION]->(l:Location) "
        "RETURN l.name AS location, s.is_night AS night, count(*) AS scenes "
        "ORDER BY location, night",
    ),
    "similar_bicycles": (
        "Frames most similar to bicycle-containing frames (SigLIP kNN).",
        "MATCH (:Category {name: 'vehicle.bicycle'})<-[:CONTAINS]-(f:Frame)-[s:SIMILAR_TO]->(n:Frame) "
        "RETURN f.token AS frame, n.token AS neighbour, s.score AS score "
        "ORDER BY score DESC LIMIT 15",
    ),
}
