"""Phase 6e: a Neo4j knowledge/context graph over the processed nuScenes data.

Built entirely from data already on disk (``data/processed/*.parquet``,
``data/autolabel/labels.parquet``, ``data/lancedb``) — no nuScenes devkit re-ingestion.
The graph turns the already-denormalized token columns back into an explicit property
graph so the chat agent can answer multi-relationship / path / co-occurrence questions,
the dataset can be browsed visually in Neo4j Browser, and active learning can sample for
graph diversity.
"""
