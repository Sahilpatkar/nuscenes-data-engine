# Phase 6c — Chat with the Dataset

Ask the dataset questions in natural language. A tool-calling LLM agent answers by
writing and running **DuckDB SQL** over the processed Parquet tables and by
**semantic vector search** over the SigLIP/LanceDB frame store (Phase 6a) — then
returns the numbers *with example frames*. Every query the agent runs is logged.

```
you: Which location has the most night driving, and what share of its scenes are rainy?
      [run_sql] SELECT location, ... GROUP BY location   → 4 rows
agent: singapore-hollandvillage has the most night scenes (66) ...
```

## Architecture

```
Streamlit "💬 Ask the dataset" ── POST /chat ──> FastAPI (serving/app.py)
                                                  └─ chat.agent.answer()
                                                       ├─ tool: run_sql        → DuckDB views over data/processed/*.parquet (+ VLM labels)
                                                       ├─ tool: search_frames  → SearchEngine.search_text (SigLIP + LanceDB)
                                                       ├─ tool: show_frames    → thumbnails by sample_data_token
                                                       └─ ChatTransport ── local (Ollama/vLLM, OpenAI-compatible) │ anthropic (Claude API)
```

- `src/nuscenes_data_engine/data_engine/chat/catalog.py` — DuckDB catalog + guarded SQL.
- `src/nuscenes_data_engine/data_engine/chat/transports.py` — the provider seam.
- `src/nuscenes_data_engine/data_engine/chat/agent.py` — tool loop, frame collection, JSONL logging.
- Server stays stateless: the Streamlit tab (or any client) holds the conversation
  history and sends it with each request.

## Providers & deployment story

| | local (default) | anthropic |
|---|---|---|
| Backend | Any OpenAI-compatible server: **Ollama on this Mac**, vLLM on a GPU box | Claude API (`claude-opus-4-8`) |
| Cost | $0 | ~cents/question |
| Config | `CHAT_BASE_URL`, `CHAT_MODEL` | `ANTHROPIC_API_KEY`, `CHAT_ANTHROPIC_MODEL` |
| When | Laptop demo | Cloud deployment (no GPU/Ollama needed) |

The agent speaks the OpenAI wire shape internally; `AnthropicTransport` translates
to/from Claude tool-use blocks. **Deploying = flipping `CHAT_PROVIDER`** — no code
changes. In docker compose the api container reaches host Ollama via
`host.docker.internal:11434`.

## Setup (local, $0)

```bash
brew install ollama
brew services start ollama
ollama pull qwen2.5:14b        # ~9 GB; solid tool calling on an M-series Mac

# CLI
uv run nuscenes-data-engine chat "How many night scenes are there per location?"
uv run nuscenes-data-engine chat -i          # REPL
# Full stack (API + Streamlit tab)
docker compose up -d api streamlit           # → http://localhost:8501
```

## SQL safety

The model's SQL runs against an in-memory DuckDB connection with read-only views.
`run_sql` enforces, before execution:

1. exactly **one** statement, parsed type **SELECT** (rejects COPY/DDL/DML — DuckDB's
   COPY can write files);
2. a denylist for what a bare SELECT can still do in DuckDB: `PRAGMA` (parses as
   SELECT), settings/extension escapes (`SET`, `ATTACH`, `INSTALL`, `LOAD`, `getenv`),
   and arbitrary-file reads (`read_parquet('/any/path')`, `FROM 'x.parquet'`,
   path-like string literals);
3. a 50-row result cap (the model is told when results are truncated).

Tool errors are returned to the model as data, so it repairs its own SQL instead of
failing the question.

## Query log

Every interaction appends one JSON line to `data/chat/log.jsonl` (question, every
executed SQL/tool call with a result summary, the answer, model, latency) — the
spec's "log every agent query for inspection". The Streamlit tab also shows each
answer's steps in an expander.

## Tables the agent can query

`samples` (204,894 camera keyframes), `annotations` (~1.1M projected 2D boxes),
`availability` (file-integrity manifest), and `labels` (5,000 Qwen2.5-VL scene
labels from Phase 6b) — schemas in [ANALYTICS.md](ANALYTICS.md) and in the agent's
system prompt.

**Known limitation:** no ego-pose/CAN-bus data is ingested, so distance-to-ego
questions (the project plan's "pedestrians within 5 m" example) are out of scope
for the current schema; the agent is told to say so rather than guess.

## Example questions (live transcripts, qwen2.5:14b on an M4 Pro)

All from `data/chat/log.jsonl`; latencies include every tool round-trip.

**1. "How many night scenes are there per location?"** — 10 s, one query
(`SELECT location, count(DISTINCT scene_token) FROM samples WHERE is_night GROUP BY 1`):
*"singapore-hollandvillage: 66 scenes, singapore-queenstown: 33 scenes. Night
driving is only present in Singapore locations."* ✓ matches ANALYTICS.md.

**2. "Which object class is hardest to see at night? Compare average visibility
day vs night by category_group."** — 61 s. First attempt hit a Catalog Error
(invented a CTE name), read the error, fixed itself, then answered with the real
per-class day/night visibility averages — honestly concluding the differences are
marginal (~3.4–3.6 on the 1–4 scale for every class).

**3. "Using the labels table joined to samples, how often do the VLM labels agree
with ground truth that a frame is night?"** — 32 s: *"agreement rate ≈ 98.5%"* —
independently consistent with the Phase 6b evaluation (night F1 0.99, AUTOLABEL_EVAL.md).

**4. "Show me a few frames of construction zones with traffic cones at night."** —
45 s: `search_frames("construction zone traffic cones night")` + `show_frames` →
6 example thumbnails attached in the UI/CLI.

**5. "Are any referenced camera files missing from disk? (CAM channels only)"** —
9 s: `COUNT(*) FILTER (WHERE NOT present) ... WHERE channel LIKE 'CAM%' HAVING missing > 0`
→ 0 rows: *"no referenced CAM channel files are missing."*

**6. "Find frames that look foggy, misty or have heavy lens glare."** — 37 s:
vector search (no SQL column could answer this), 2 example frames attached.

**7. "What share of camera frames have 20+ annotated objects, and where?"** — 51 s:
self-repaired through two binder errors, produced a *related but subtly different*
statistic (the location distribution **of** crowded boxes, not the share of frames
that are crowded). Kept here deliberately — see limitations.

### Observed local-model limitations (and what the harness does about them)

- **Language drift**: qwen2.5:14b occasionally answered in Thai; pinning the reply
  language at the top of the system prompt largely fixed it.
- **Double-escaped SQL**: the model sometimes emits literal `\n` inside tool-call
  JSON; the SQL tool normalizes this (it once burned the whole tool budget on
  parse errors before the fix).
- **Hallucinated frame tokens**: `show_frames` sometimes receives invented tokens;
  `frames_by_tokens` silently drops unknown ones, so nothing wrong is displayed.
- **Subtle statistical misreads** (Q7): the numbers come from real SQL, but the
  query may answer a neighboring question. The steps expander shows the exact SQL
  precisely so this is checkable.
- Error self-repair works well: binder/catalog errors are fed back and usually
  fixed in one retry. For higher reliability, `CHAT_PROVIDER=anthropic` swaps in
  Claude with no other changes.
