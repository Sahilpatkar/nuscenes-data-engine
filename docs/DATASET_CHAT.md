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

`samples` (204,894 camera keyframes), `annotations` (~1.0M projected 2D boxes),
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
- **`grounded`** (**v2**, 2026-08-11) — every number cited in the answer, at the
  precision the answer states, must match one of: an **observed value** (any cell,
  or per-column sum, from re-executing the answer's own SQL — excluded when the
  result was truncated); **one arithmetic step** over two observed measurements
  (`a+b`, `a−b`, a share-percentage `a/b×100` only when the denominator is ≥10 and
  ≥ the numerator, or an m/s↔km/h `×÷3.6` conversion — row counts are citable but
  never feed a derivation, since a `LIMIT 8`'s 8 is a property of the query text,
  not the world); a **schema-prompt constant** (extracted live from the same schema
  text the agent's system prompt embeds); or a **question echo** / the
  **attached-frame count**. Wrong arithmetic still fails: the historical log's
  98.5%-agreement answer, whose own retrieved counts give 99.66%, still fails
  `grounded` under v2 — a required regression, not an aspiration. `grounded`
  re-executes the agent's own SQL from its logged steps, because `ChatResult.steps`
  stores tool-output *summary strings*, not the underlying rows — so this check
  also independently verifies the query really produces what the prose claims, not
  just that the prose looks plausible. Design and the full rule taxonomy:
  `docs/superpowers/specs/2026-08-11-grounding-v2-design.md`.
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

### Results (2026-08-11 sweep, v2 grader, 20 identical cases)

| check | qwen2.5:14b (replay) | qwen2.5:32b (live) | claude-opus-4-8 (replay) |
|---|---|---|---|
| **composite** | **4/20** | **12/20** | **17/20** |
| numeric | 7/17 | 13/17 | 16/17 |
| english | 12/20 | **20/20** | 20/20 |
| tool_use | 17/20 | **20/20** | 20/20 |
| grounded | 20/20 | 19/20 | 18/20 |
| frames | 1/3 | 0/3 | 3/3 |
| median latency | 10.8s | 23.6s | 8.0s |

**The `frames` row is not comparable across columns** — the live (`qwen2.5:32b`)
run had frame *attachment* disabled entirely, not just semantic search; see the
caveat below. The comparable subset, the 17 cases that do not depend on the
attachment tool: **`qwen2.5:14b` 3/17, `qwen2.5:32b` 12/17, `claude-opus-4-8`
14/17.**

`qwen2.5:14b` and `claude-opus-4-8` are the original 2026-08-11 runs **replayed**
through the current grader, $0 and LLM-free (`chat-eval --regrade`, §3 of the
design spec) — same stored answers as the original run, only the instrument
changed. `qwen2.5:32b` is a fresh **live** run under the same grader.

<details>
<summary><b>v1 grader (superseded 2026-08-11)</b> — the original 2-provider table, for reference</summary>

| check | local (`qwen2.5:14b`) | anthropic (`claude-opus-4-8`) |
|---|---|---|
| **overall passed** | **4/20 (20%)** | **11/20 (55%)** |
| numeric | 7/17 | 16/17 |
| english | 12/20 | 20/20 |
| tool_use | 17/20 | 20/20 |
| grounded | 18/20 | 11/20 |
| frames | 1/3 | 3/3 |
| median latency | 10.8s | 8.0s |

Only `grounded`/composite changed under v2 — `numeric`/`english`/`tool_use`/`frames`
are grader-invariant and identical above. Under v1, `grounded` ran the wrong way
(18/20 local vs 11/20 Claude): see "Prediction scorecard" below for why, and how v2
fixes it.

</details>

**Three-point instrument comparison** (same stored Claude answers, three grader
states — isolates grader effect from model effect): `grounded` **11/20** under the
grader active when the run happened → **12/20** replayed at v2 development's start
(the delta is one case, `avg_visibility_pedestrian_night`, flipped by an
already-landed fix widening the question-echo extractor to catch hyphenated
constants like "the 1-4 scale" — its `numeric` still fails, so composite is
unaffected) → **18/20** under the full v2 rule set. Composite: **11 → 11 → 17**.
Same answers throughout; only the grader changed.

**32b headline:** doubling the local model to `qwen2.5:32b` eliminates the two
dominant `qwen2.5:14b` failure modes — language drift 8/20 → 0/20, unusable tool
calls 3/20 → 0/20 — at $0, for 2.2× the median latency (10.8s → 23.6s). Its 8
residual failures break down as **4 numeric-accuracy misses**, **1 `grounded`
miss** (`labels_night_agreement_pct`), and **3 `frames` misses** that — per the
caveat below — were structurally unattainable in this run, not a model gap.
**Decision rule: the cheapest local model that clears the bar.** On the comparable
17-case subset above, 32b (12/17) still trails Claude (14/17) but by a much
narrower margin than the raw 12/20-vs-17/20 composite suggests, since the raw
composite also charges 32b for 3 `frames` cases it could not structurally pass.
Whether 32b "clears the bar" is a judgment call, but an honest one: it triples
14b's composite (4/20 → 12/20) and its residual failures are mostly measurement
artifacts (3 `frames`) plus a real but narrower numeric-accuracy gap — not
language or tooling breakdowns. **Recommended local model: `qwen2.5:32b`** — the
config default (`chat_model` in `config.py`) stays `qwen2.5:14b`; flipping it is a
deployment decision, not made here, since 32b needs ~20 GB RAM and runs at 2.2×
the latency. `qwen2.5:14b` remains the config default and is still the right
choice for latency-critical demos; `qwen2.5:32b` is the right choice whenever
reliability matters more than the extra ~13 seconds.

**Caveat on the `frames` row — the live column carries no model signal.** The
error text and mechanism differ *by run*, not just by provider, because a probe
added partway through this branch's history (`cli.py`, commit `e110706`, 12:32)
changed what "vector search is unavailable" *means* for `chat-eval`:

- **`qwen2.5:14b` (11:34) and `claude-opus-4-8` (11:56) — both before the probe
  existed.** `chat-eval` built a real `SearchEngine` object unconditionally
  (construction alone never touches the encoder). `search_frames` failed
  *per call* with `search failed: No module named 'torch'` (the Mac this eval ran
  on has no torch install), but `show_frames` — which only looks up already-known
  `sample_data_token`s and never touches the encoder — kept working. Both models'
  answers show this: `search_frames` errors, then a SQL fallback finds tokens,
  then `show_frames` succeeds ("6 frames attached"). Claude reached frames in all
  three retrieval cases this way: immediately for `bike_bus_singapore_night_frames`
  (never called `search_frames` at all), after one failed attempt for
  `construction_cones_night_frames`, and after two — the second with a reworded
  query — for `foggy_misty_glare_frames`.
- **`qwen2.5:32b` (14:49) — after the probe landed.** `chat-eval` now calls
  `search_text` once *before* the run starts; that call raises (still no torch),
  so the CLI sets `search_engine=None` for the *entire* run. With the engine
  `None`, `agent.py` fails **both** `search_frames` *and* `show_frames` for every
  call, always with `Vector search is not available (LanceDB store not found)` —
  frame attachment was disabled outright, independent of anything the model did.
  For 2 of its 3 frames cases (`construction_cones_night_frames`,
  `foggy_misty_glare_frames`) 32b did exactly what Claude did — SQL fallback to 6
  valid `sample_data_token`s, then `show_frames` — and the tool call still failed,
  because the tool itself was off. Only `bike_bus_singapore_night_frames` is a
  genuine model failure: 32b fabricated markdown image links to a literal
  `fakeurl` placeholder instead of reporting that attachment had failed.

So **the live column ran with the frame-attachment tool disabled entirely, so its
`frames` row carries no model signal** — an instrument asymmetry of exactly the
kind this branch exists to correct, caught in final review. The comparable subset
above (17/20 cases, `frames` excluded) is the fair local/cloud comparison; the
`frames` row itself says nothing about 32b's retrieval ability, only that the
probe's all-or-nothing fallback ran during its eval and not the other two. (Not
implemented here, flagged as a follow-up: the probe should disable only
`search_text`, leaving token-based frame attachment available regardless of torch,
matching what the pre-probe runs actually exercised.) None of the measured
numbers change because of this — it is a property of the environment the eval ran
in, not the grading logic — but this run says nothing about semantic search
quality specifically, only about fallback robustness, and for `qwen2.5:32b`, about
an instrument change mid-branch.

**Finding 1 — language drift is far worse than the live transcripts suggest, and it
is model-scale-bound, not a property of local serving.** 8 of 20 `qwen2.5:14b`
answers (40%) came back in a non-Latin script, including simple questions like "how
many night scenes in holland village." Claude: 20/20 English. 3 local answers also
failed to emit a usable tool call, so no query ran — none of the three recalled or
fabricated a statistic instead: one is a Thai refusal that echoes the question
back, one prints SQL in a fenced block without calling it, and one is a malformed
tool call (the double-escaped tool-argument bug documented above). Claude always
emitted a usable call (20/20 `tool_use`). **`qwen2.5:32b` eliminates both failure
modes**: drift 8/20 → 0/20, unusable tool calls 3/20 → 0/20 (see the "32b
headline" note above), for the same $0 and 2.2× the latency — the limitation was a capacity
ceiling of the 14b weights, not something local-first serving structurally can't
fix. Recommended local model: `qwen2.5:32b` (the config default, `chat_model` in
`config.py`, stays `qwen2.5:14b` — flipping it is a deployment decision, not made
here, given the ~20 GB RAM and 2.2× latency cost); `qwen2.5:14b` remains useful
only where latency matters more than reliability (a live demo where drift can be
caught and retried on the spot).

**Finding 2 (v1, resolved by v2) — the v1 grounding check penalised the stronger
model.** Claude was far more accurate (`numeric` 16/17 vs 7/17) yet scored *worse*
on `grounded` (11/20 vs 18/20 local) under v1, because its 9 grounding failures
were legitimate but not literally present in tool output — derived arithmetic
("4,986 of 5,000, so 14 failed"), computed percentages ("99.66% = 4,969/4,986"),
and schema-quoted constants ("≤ −3.0 m/s² within ±0.5 s"). To be explicit about
what this project did and did not do: **the grader was not loosened after seeing
these results.** A principled fix was pre-registered *before* any re-run — admit
numbers derivable from observed values by simple arithmetic, and seed the allowed
set with schema-prompt constants — and its predicted effect was checked before it
was measured (`docs/superpowers/specs/2026-08-11-grounding-v2-design.md`, §2). The
scorecard below is that pre-registration graded against what the replay actually
showed.

**Prediction scorecard** (spec §2 + its dated amendment; replay = same stored
answers, new grader):

- Claude `grounded` 11 → **18/20**: **HELD** exactly.
- Claude composite 11 → **17/20** (pre-registered as 17, not 18 — the ninth
  grounding fix, `avg_visibility_pedestrian_night`, still fails `numeric`): **HELD**
  exactly.
- Local (`qwen2.5:14b`) composite 4/20 → 4/20, "barely moves": **HELD** — its
  `grounded` goes 18 → **20/20**, both v1 failures artifacts of the same mechanism
  as predicted (a schema constant echoed in Chinese; a stray `1` scraped from
  printed-but-uncalled SQL).
- No case flips `grounded` True → False under v2 (v2 only widens the allowlist,
  never narrows it): **HELD**, both providers — verified case-by-case, not just by
  the aggregate counts.
- Record 5 of the historical log (the 98.5%-for-99.66% miscalculation) still
  fails `grounded`: **HELD**.
- **COMPOSITION MISS**: the predicted surviving-ungrounded pair was
  `{max_instance_keyframes, foggy_misty_glare_frames}`; the actual survivors are
  `{max_instance_keyframes, labels_parse_ok_count}`. One half of this was
  pre-registered in the committed spec amendment *before* the replay ran (the
  `foggy` flip, below); the other was caught in review before measurement but
  never committed as a dated artifact at the time — the plan line has since been
  corrected (`docs/superpowers/plans/2026-08-11-grounding-v2.md`).
  - `foggy_misty_glare_frames` flipped to grounded — its residual "~50+" figure
    matched the schema text "50 Hz" (from the `canbus` table), units-blind against
    a hallucinated frame count that has nothing to do with sample rate. This was
    itself pre-registered as a MISSED prediction in the spec's dated amendment,
    *before* the schema-constant rule landed, from a live read of the extracted
    constants — so it is a documented limitation of a units-blind allowlist, not
    an unexpected collision.
  - `labels_parse_ok_count` stayed ungrounded because its cited 14 needs
    `5000 − 4986`, and 5000 is only a schema constant here (the case's own SQL
    never queries a row total) — derivations deliberately draw only on *observed*
    values, never on the constants pool. The original plan's claim that both
    operands were observed was simply wrong; caught in review, before measurement,
    not after.
- **ATTRIBUTION**: the frame-count rule (`len(frames)`) changed **zero** verdicts
  on real records (checked across all v2 runs) — every case citing "N examples
  attached" was already admitted by a schema constant. The "6 examples" citations
  in particular are admitted by schema constant `6.0`, which comes from the string
  "(Phase 6b)" in the `labels` table's schema prose, not from the frame-count rule
  at all. Stated plainly so the frame-count rule isn't credited for something a
  different rule already did.

### Known limitations of grounding v2

- **Units-blind constants.** A schema constant is matched on value alone, not
  units or meaning — the `foggy_misty_glare_frames` flip above (a hallucinated
  "~50 frames" admitted by "50 Hz") is the concrete case. A number that happens to
  equal a documented constant is treated as grounded whether or not the model
  actually meant that constant.
- **One-step derivations only.** Sum, difference, one guarded percentage form, and
  the m/s↔km/h conversion — no chains (a derived value is never re-derived), no
  products, no other quotients. This is a deliberate bound, not an oversight: it
  keeps the allowlist small enough to reason about and to bound collision risk
  (below).
- **The schema prose *is* the allowlist.** `catalog.schema_prompt()`'s text is
  parsed live for constants — editing a table docstring (adding, rewording, or
  removing a number) silently changes what `grounded` admits. A pinned test
  (`test_production_schema_constants_are_pinned`) makes that a visible CI diff
  instead of a silent instrument change — e.g. "(Phase 6b)" contributes `6.0`,
  "50 Hz" contributes `50.0`.
- **Measured collision cost** — the fraction of arbitrary integers 1–200 a
  record's own allowed values would admit, so a wrong figure can get lucky,
  measured across all 68 stored records (the v1 local + Claude runs plus the
  28-entry historical log) two ways, since the basis matters:

  | basis | mean | on the 8 records with ≥6 measurements | worst case |
  |---|---|---|---|
  | derivation rules alone (observed + one-step derivations + echoes, no schema constants) | 1.9% | 9.1% | 24.0% |
  | **full production allowlist (+ schema constants) — what the shipped grader actually uses** | 5.3% | 11.5% | **24.5%** |

  (Figures are one-decimal because one stored record's SQL — an unordered
  aggregate over a join — re-executes non-deterministically, moving the corpus
  means by ~±0.02pp between probe runs. The instability never touches a cited
  number: replayed verdicts are byte-identical across runs; only this 1..200
  integer sweep is sensitive.)

  The shipped grader always includes schema constants, so the second row is the
  one that matters: **24.5% < the 30% CI-enforced ceiling**, worst case at
  `max_instance_keyframes`. Its 19 observed keyframe counts span 34..10614; only
  8 of them (34–41) are close together, and it's those 8 that manufacture almost
  all of the sum/difference collisions.

The 18/20-vs-20/20 `grounded` inversion (Claude vs `qwen2.5:14b`) that remains
under v2 is understood, not a grader bug: `is_grounded` returns True vacuously
when an answer cites no numbers at all, and 8 of the local model's 20 v1 answers
do exactly that — a free pass, not a demonstration of care. Across all 20
questions the local model cites 27 numbers in total; Claude cites 99, with no
number-free answer among them — a model that commits to fewer specifics is
structurally harder to catch being wrong about one. `grounded` is a real signal in
general — a number nowhere in the agent's own retrieved data, the question, or the
schema is unsupported, whichever model states it — but this comparison mostly
rewards Claude for stating more and local for stating less.

### Running it

```bash
uv run nuscenes-data-engine chat-eval --provider local       # Ollama, $0
uv run nuscenes-data-engine chat-eval --provider anthropic   # needs ANTHROPIC_API_KEY
# artifacts: data/chat/eval/{results,report}_<provider>.{jsonl,md}

# Re-grade a stored run's answers with the CURRENT grader — no LLM calls, no cost:
uv run nuscenes-data-engine chat-eval --regrade data/chat/eval/results_local.jsonl
# writes <stem>_v2.jsonl / <stem>_v2.md next to the input, never overwrites it

# Run against a specific model — output filenames gain a model slug, so sweep runs
# don't clobber each other or the default artifacts:
uv run nuscenes-data-engine chat-eval --provider local --model qwen2.5:32b
# -> results_local_qwen2.5-32b.jsonl, report_local_qwen2.5-32b.md
```

Regrade reports carry grader provenance (the schema-constant count and the tables
it was built from, appended after the case table) so a replay run in a checkout
with different/missing data is visibly a different instrument, not a silent
number change.
