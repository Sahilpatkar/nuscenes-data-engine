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

## Example questions

_Transcripts to be captured after the live run — placeholders below._

1. How many night scenes are there per location?
2. Which class is hardest to see at night — compare average visibility day vs night.
3. What share of frames have 20+ annotated objects, and where are they?
4. Do the VLM labels agree with ground truth about which frames are night?
5. Show me construction zones near traffic cones.
6. Find foggy- or glare-looking frames the tables can't identify.
7. Are any referenced camera files missing from disk?
