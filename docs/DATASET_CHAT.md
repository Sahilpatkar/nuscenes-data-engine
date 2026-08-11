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
`availability` (file-integrity manifest), `labels` (5,000 Qwen2.5-VL scene
labels from Phase 6b), `ego_pose`/`annotations_3d`/`instances` (Phase B 3D geometry),
and `canbus` (Phase B CAN-bus ego dynamics — speed, steering, braking, `is_hard_braking`)
— schemas in [ANALYTICS.md](ANALYTICS.md) and in the agent's system prompt.

**Multi-hop questions:** when the Phase 6e knowledge graph is reachable, the agent
gains a fourth tool, `run_cypher`, for relationship / path / co-occurrence /
similarity / temporal-next questions that SQL joins express awkwardly (see
[GRAPH.md](GRAPH.md)). It degrades to SQL + vector only when Neo4j is down.

**Geometry (Phase B):** ego-pose + 3D object geometry are now ingested, so
distance-to-ego questions *are* answerable — e.g. "pedestrians within 5 m of ego at
night" via `annotations_3d.distance_to_ego_m` in SQL or the `ObjectObservation` nodes
in the graph (see [GRAPH.md](GRAPH.md)).

**Ego dynamics (Phase B, CAN-bus):** speed/steering/braking are also ingested (the
`canbus` table), so dynamics questions are answerable too — e.g. "how many keyframes
have hard braking near pedestrians?" joins `canbus.is_hard_braking` to
`annotations_3d.distance_to_ego_m` in SQL, or the same query via the `EgoPose`/
`ObjectObservation` graph nodes (see the flagship query in
[GRAPH.md](GRAPH.md#geo-spatial-layer-phase-b) — 30 keyframes, identical in both).

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

- **Language drift, measured**: 8 of 20 answers in the correctness eval (40%) came
  back in a non-Latin script (mostly Thai/Chinese) — including simple questions
  like "how many night scenes in holland village." This is not the occasional
  glitch the live transcripts above might suggest: pinning the reply language at
  the top of the system prompt reduces it but is nowhere near sufficient; see the
  eval below. 3 of 20 answers also called no tool at all, answering from memory
  instead of querying — Claude never did either (20/20 English, 20/20 tool-using).
- **Loose terminology**: it sometimes labels image counts as "scenes" — the SQL in
  the steps expander is the ground truth for what was actually counted.
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
  Claude with no other changes — see the measured pass-rate gap below.

## Answer-correctness evaluation

Live transcripts above are illustrative; this is the measured version. A case set
scores each answer against a **reference SQL**, deterministically — there is no LLM
judge. Ground truth for each case is its `reference_sql`, executed through the SAME
guarded catalog (`chat/catalog.py`) the agent itself queries, so a case can never ask
for something the agent is structurally unable to answer. A `reference_sql` that
errors or returns the wrong shape fails the whole suite loudly at preflight
(`evaluate.validate_cases`) rather than silently scoring as an agent failure.

Five checks, each a pure function over the agent's own output (`chat/evaluate.py`):

- **`numeric`** — any number cited in the answer is within tolerance of the
  reference value.
- **`english`** — the answer's alphabetic characters are ≥90% Latin script; an
  explicit heuristic, not a language model, sized to catch wholesale script drift.
- **`tool_use`** — the agent called at least one tool rather than answering from
  memory.
- **`grounded`** — every number cited in the answer traces to the agent's own tool
  output, accepting values rounded to the precision the answer states and constants
  echoed back from the question. `grounded` re-executes the agent's own SQL from
  its logged steps, because `ChatResult.steps` stores tool-output *summary strings*,
  not the underlying rows — so this check also independently verifies the query
  really produces what the prose claims, not just that the prose looks plausible.
- **`frames`** — retrieval-only cases returned at least one example frame.

**The case set**: `configs/chat_eval.yaml`, 20 cases (17 numeric + 3 retrieval),
seeded from the real questions in `data/chat/log.jsonl`, covering `samples`,
`annotations`, `labels`, `canbus`, `ego_pose`, `annotations_3d`/`instances`, and
`availability`. Two references cross-check against numbers documented independently
elsewhere: "pedestrians within 5 m of ego at night" (183) and "hard braking with a
pedestrian within 10 m" (30) both reproduce docs/GRAPH.md's SQL/Cypher parity
results.

One infrastructure fix came out of building this harness: the SQL guard's
file-path denylist had a false positive on legitimate SQL that divides between two
string literals — exactly the shape of a filtered percentage
(`count(*) FILTER (WHERE a='x') / count(*) ... WHERE b='y'`) — because the old
single regex could pair the closing quote of one literal with the opening quote of
an unrelated later one. That was silently pushing the agent toward computing
percentages in prose instead of retrieving them directly; fixed by matching each
`'...'` literal on its own rather than scanning the whole statement (`chat/catalog.py`).

### Results (2026-08-11, 20 cases, identical suite both providers)

| check | local (`qwen2.5:14b`) | anthropic (`claude-opus-4-8`) |
|---|---|---|
| **overall passed** | **4/20 (20%)** | **11/20 (55%)** |
| numeric | 7/17 | 16/17 |
| english | 12/20 | 20/20 |
| tool_use | 17/20 | 20/20 |
| grounded | 18/20 | 11/20 |
| frames | 1/3 | 3/3 |
| median latency | 11.4s | 8.0s |

2 cases pass under both providers; 9 pass only under Claude; 2 pass only under
local. 7 cases fail under both: `avg_visibility_pedestrian_night`,
`canbus_hard_braking_count`, `foggy_misty_glare_frames`,
`hard_braking_near_pedestrian_10m`, `labels_night_agreement_pct`,
`max_instance_keyframes`, `missing_cam_files`.

**Finding 1 — language drift is far worse than the live transcripts suggest.** 8
of 20 local answers (40%) came back in a non-Latin script, including simple
questions like "how many night scenes in holland village." Claude: 20/20 English.
3 local answers also called no tool at all, answering from memory; Claude always
queried (20/20 `tool_use`).

**Finding 2 — the grounding check penalises the stronger model, and that is a
harness limitation, not a model defect.** Claude is far more accurate (`numeric`
16/17 vs 7/17) yet scores *worse* on `grounded` (11/20 vs 18/20). Inspecting its 9
grounding failures shows why — the cited numbers are legitimate but not literally
present in tool output:

- derived arithmetic: "**4,986** of the VLM-labeled frames parsed successfully
  (out of 5,000 total, so **14** failed to parse)" — 14 = 5000 − 4986, computed,
  never retrieved;
- computed percentages: "**Agreement rate: 99.66%** (4,969 of 4,986 usable
  frames)" — 99.66 derived from two retrieved counts;
- constants quoted from the schema prompt: "94 keyframes show hard braking (peak
  deceleration ≤ **−3.0** m/s² within the **±0.5** s window)" — those thresholds
  come from the documented schema, not from a query.

In each of those three examples `numeric` passed (the headline figure was right)
while `grounded` failed. So the composite "passed" column **understates the
stronger model**: a capable agent that shows its arithmetic and explains its
thresholds is penalised for doing so.

To be explicit about what this project did and did not do: **the grader was not
loosened after seeing these results.** Changing a measurement instrument to
flatter an outcome is exactly what this project avoids elsewhere (see the AL
random-control gate). This is recorded as a known limitation, with a principled
fix for a future round, pre-registered here before any re-run: admit numbers
derivable from observed values by simple arithmetic, and seed the observed set
with constants from the schema prompt, the same way question-echoed constants are
already admitted. `grounded` remains a real signal for the local model, though —
where it caught genuine invention, not just unretrieved-but-correct arithmetic.

### Running it

```bash
uv run nuscenes-data-engine chat-eval --provider local       # Ollama, $0
uv run nuscenes-data-engine chat-eval --provider anthropic   # needs ANTHROPIC_API_KEY
# artifacts: data/chat/eval/{results,report}_<provider>.{jsonl,md}
```
