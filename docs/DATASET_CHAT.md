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
  eval below. 3 of 20 answers also failed to emit a usable tool call — not recalled
  statistics; a Thai refusal that echoes the question back, an answer that prints
  the SQL in prose without calling it, and one instance of the double-escaped
  tool-call bug below — so no tool ran. Claude never failed to emit a usable call
  (20/20 English, 20/20 tool-using).
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
seeded from, and extended beyond, the logged questions in `data/chat/log.jsonl`
(6 of the 20 case questions match a logged question verbatim; the rest cover the
same tables and question shapes), covering `samples`, `annotations`, `labels`,
`canbus`, `ego_pose`, `annotations_3d`/`instances`, and `availability`. Two
references cross-check against numbers documented independently elsewhere:
"pedestrians within 5 m of ego at night" (183) and "hard braking with a pedestrian
within 10 m" (30) both reproduce docs/GRAPH.md's SQL/Cypher parity results.

Two infrastructure fixes came out of building this harness, both in the SQL guard.
First, the file-path denylist had a false positive on legitimate SQL that divides
between two string literals — exactly the shape of a filtered percentage
(`count(*) FILTER (WHERE a='x') / count(*) ... WHERE b='y'`) — because the old
single regex could pair the closing quote of one literal with the opening quote of
an unrelated later one. On this shape the guard would have rejected the natural
form of a filtered-percentage query, pushing the agent toward computing the
percentage in prose instead of retrieving it directly — though no logged run
actually hit the rejection (`data/chat/log.jsonl` has no `Disallowed token` error;
this is a mechanism the fix pre-empted, not a failure caught in the wild). Fixed by
matching each `'...'` literal on its own rather than scanning the whole statement.
Second, and unrelated — a pre-existing gap, not a regression introduced by the
first fix — the denylist only ever tokenized single-quoted `'...'` literals, but
DuckDB reads a file path out of a double-quoted identifier or a `$$...$$`-quoted
string just as readily (`SELECT * FROM "x.parquet"`, `SELECT * FROM $$x.parquet$$`
both worked before this fix); the guard now tokenizes and checks all three forms
(`chat/catalog.py`).

### Results (2026-08-11, 20 cases, identical suite both providers)

| check | local (`qwen2.5:14b`) | anthropic (`claude-opus-4-8`) |
|---|---|---|
| **overall passed** | **4/20 (20%)** | **11/20 (55%)** |
| numeric | 7/17 | 16/17 |
| english | 12/20 | 20/20 |
| tool_use | 17/20 | 20/20 |
| grounded | 18/20 | 11/20 |
| frames | 1/3 | 3/3 |
| median latency | 10.8s | 8.0s |

**Caveat on the `frames` row**: `search_frames` failed on *every* call in *both*
runs, always with the same error — `search failed: No module named 'torch'` (the
Mac this eval ran on has no torch install; see `SearchEngine._encoder`). So `frames`
does not measure semantic retrieval at all here — it measures recovery via SQL
fallback against `labels`/`annotations_3d` after the vector-search tool errored out.
Claude reached frames in all three retrieval cases via that SQL fallback:
immediately for `bike_bus_singapore_night_frames` (never called `search_frames` at
all), after one failed attempt for `construction_cones_night_frames`, and after
two — the second with a reworded query — for `foggy_misty_glare_frames`.
`foggy_misty_glare_frames` — one of the seven fail-both cases below — never found a
working path for local: one failed `search_frames` call, no SQL fallback attempted,
so `frames` failed too (it fails overall for Claude as well, on `grounded`, despite
successfully attaching frames via fallback). None of the measured numbers change
because of this — it is a property of the environment the eval ran in, not the
grading logic — but this run says nothing about semantic search quality
specifically, only about fallback robustness when it is unavailable.

2 cases pass under both providers; 9 pass only under Claude; 2 pass only under
local. 7 cases fail under both: `avg_visibility_pedestrian_night`,
`canbus_hard_braking_count`, `foggy_misty_glare_frames`,
`hard_braking_near_pedestrian_10m`, `labels_night_agreement_pct`,
`max_instance_keyframes`, `missing_cam_files`.

**Finding 1 — language drift is far worse than the live transcripts suggest.** 8
of 20 local answers (40%) came back in a non-Latin script, including simple
questions like "how many night scenes in holland village." Claude: 20/20 English.
3 local answers also failed to emit a usable tool call, so no query ran — none of
the three recalled or fabricated a statistic instead: one is a Thai refusal that
echoes the question back, one prints SQL in a fenced block without calling it, and
one is a malformed tool call (the double-escaped tool-argument bug documented
above). Claude always emitted a usable call (20/20 `tool_use`).

**Finding 2 — the grounding check penalises the stronger model, and that is a
harness limitation, not a model defect.** Claude is far more accurate (`numeric`
16/17 vs 7/17) yet scores *worse* on `grounded` (11/20 vs 18/20). Inspecting its 9
grounding failures shows why — the cited numbers are legitimate but not literally
present in tool output:

- derived arithmetic: "**4,986** of the VLM-labeled frames parsed successfully
  (out of 5,000 total, so 14 failed to parse)" — 14 = 5000 − 4986, computed, never
  retrieved;
- computed percentages: "**Agreement rate: 99.66%** (4,969 of 4,986 usable
  VLM-labeled frames)" — 99.66 derived from two retrieved counts;
- constants quoted from the schema prompt: "**94 keyframes** show hard braking
  (where `is_hard_braking` is true, meaning peak longitudinal deceleration ≤ −3.0
  m/s² within the ±0.5 s CAN-bus window)" — those thresholds come from the
  documented schema, not from a query.

In each of those three examples `numeric` passed (the headline figure was right)
while `grounded` failed. So the composite "passed" column **understates the
stronger model**: a capable agent that shows its arithmetic and explains its
thresholds is penalised for doing so. Claude's one genuine `numeric` miss,
`avg_visibility_pedestrian_night` (2.78 vs. the reference's 3.64), arguably is not
one either: Claude read `annotations_3d` (2.78 — confirmed by querying it
directly), the reference reads the 2D `annotations` table (3.64), and both are
real, correct averages of a real `visibility_token` column on different tables —
the case question has since been reworded to say which table is meant
(`configs/chat_eval.yaml`). So Claude's 16/17 `numeric` likely understates it too.

To be explicit about what this project did and did not do: **the grader was not
loosened after seeing these results.** Changing a measurement instrument to
flatter an outcome is exactly what this project avoids elsewhere (see the AL
random-control gate). This is recorded as a known limitation, with a principled
fix for a future round, pre-registered here before any re-run: admit numbers
derivable from observed values by simple arithmetic, and seed the observed set
with constants from the schema prompt, the same way question-echoed constants are
already admitted. Pre-registering a fix rather than applying it quietly means
checking whether it would actually work: it would flip 7 of the 9 Claude failures
above to grounded — not all 9. It would not fix `max_instance_keyframes`, whose
unsupported values ("~20 s", "2 Hz") are nuScenes domain facts from the model's
own memory, nowhere in the schema prompt or in any retrieved row. Nor does it
change the outcome for `avg_visibility_pedestrian_night` — its ungrounded
citation ("4", for "fully visible") is exactly this kind of schema constant, but
the case still fails `numeric` for the table-ambiguity reason above, so fixing its
grounding would not flip its overall result either way.

The honest reading of the 18/20-vs-11/20 `grounded` split runs the other way from
"it caught the local model's mistakes." `is_grounded` returns True vacuously when
an answer cites no numbers at all, and 8 of the local model's 20 answers do
exactly that — a free pass, not a demonstration of care. Across all 20 questions
the local model cites 27 numbers in total; Claude cites 99, with no number-free
answer among them — a model that commits to fewer specifics is structurally harder
to catch being wrong about one. Local's own two `grounded` failures are not
invention either: one cites "4" for "fully visible" — the same schema constant
used to excuse Claude above, just echoed in Chinese ("其中 4 表示完全可见"); the
other is a stray "1" pulled out of `THEN 1 END` inside a SQL query the model
printed in prose but never actually called (that answer made no statistical claim
tied to it at all). `grounded` is a real signal in general — a number nowhere in
the agent's own retrieved data or the question is unsupported, whichever model
states it — but nothing in this run shows it catching a local-model invention; on
this run it mostly rewards saying less.

### Running it

```bash
uv run nuscenes-data-engine chat-eval --provider local       # Ollama, $0
uv run nuscenes-data-engine chat-eval --provider anthropic   # needs ANTHROPIC_API_KEY
# artifacts: data/chat/eval/{results,report}_<provider>.{jsonl,md}
```
