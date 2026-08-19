# Demo Phase 4 — chat upgrades on the local stack (probe fix, streaming, charts)

**Status:** approved 2026-08-19 (Phase 4 of the approved demo plan; Phases 1-3 merged
as PRs #14-#16). The parallel-safe workstream: the LIVE local-stack chat gets its
three long-queued upgrades. The public demo is untouched — Phase 8's replay gallery
will *record* streamed answers with charts, which is why this lands before it.

## Why

The chat agent answers well (measured: Claude 17/20, qwen2.5:32b 12/20 under the
v2 grader) but the experience is a blocking 600-second POST with a spinner, tool
results reach the UI only as summary strings (no tables, no charts), and one known
seam bug remains: the eval CLI's search probe nulls the whole engine when torch is
missing, killing token-based frame attachment that needs no torch — the documented
cause of the non-comparable frames row in the eval.

## 1. Probe fix (first commit — small, independent)

- `cli.py`'s `chat-eval` search probe: when `SearchEngine(...)` construction
  succeeds but the `search_text` probe raises, KEEP the engine (semantic search
  fails per-call as model-visible errors; `frames_by_tokens` still works). Null it
  only when construction itself fails. Warning text distinguishes the two.
- `agent.py`: wrap the `frames_by_tokens` call in the same try/except pattern as
  `search_frames` (a broken store degrades to a model-visible error, not an
  exception through the loop).
- `docs/DATASET_CHAT.md`'s frames-row caveat: updated to say the seam is fixed
  going forward; the stored runs' non-comparability note stays (history is
  history).
- Tests: probe-failure path keeps the engine (monkeypatched SearchEngine whose
  search_text raises); frames_by_tokens exception becomes a tool-error dict.

## 2. Streaming

**Agent seam (return shape FROZEN — eval + all callers unchanged):**
`agent.answer()` gains optional keyword-only callbacks `on_token: Callable[[str],
None] | None` and `on_step: Callable[[dict], None] | None`. `on_token` receives
final-answer text deltas as the transport produces them; `on_step` fires after each
tool execution with the same dict appended to `result.steps`. No callbacks = today's
behavior, byte-identical. The existing chat eval suite rerunning green is the
regression proof.

**Transports:**
- `OpenAICompatTransport`: a `complete_stream(messages, tools, on_token)` method —
  `stream: true` payload, `httpx client.stream`, SSE `data:` parsing, accumulating
  `delta.content` (forwarded to on_token) and index-keyed `delta.tool_calls`
  argument fragments; returns the same assembled message dict `complete` returns.
  The non-stream `complete` stays untouched.
- `AnthropicTransport`: `complete_stream` via the SDK's `messages.stream` context
  manager forwarding `text_stream` deltas, then `get_final_message()` through the
  existing content-block conversion (thinking blocks filtered — only `text` blocks
  reach on_token, matching the non-stream path's block handling).
- The Protocol gains `complete_stream` with a default fallthrough: transports that
  do not implement it (test fakes) fall back to `complete` + one on_token call with
  the full text. `answer()` calls `complete_stream` when `on_token` is set, else
  `complete`. IMPORTANT: no one can know in advance which turn is final (the model
  decides by not calling a tool), so both transports stream every turn's text
  deltas. The UI treats a non-final turn's text as interim narration and resets
  its buffer when the next turn starts — the SSE protocol carries a `turn` event
  for exactly this. Simple, honest, no lookahead required.

**Serving:** new `POST /chat/stream` (sync generator → `StreamingResponse`,
`media_type="text/event-stream"`) emitting SSE events: `turn` (reset), `token`
(text delta), `step` (tool executed), `final` (the complete ChatResponse body
JSON — frames, steps, charts — so terminal rendering is identical to /chat).
Resource setup stays OUTSIDE the generator (HTTP errors must map to status codes).
`/chat` is frozen for the CLI and eval harness.

**Streamlit (`app/streamlit_app.py`):** `requests.post(stream=True)` on the new
endpoint; `st.write_stream` renders token deltas (buffer reset on `turn` events);
the `final` event's body feeds the existing `_render_chat_answer` and session-state
`{content, body}` shape, so history replay is unchanged. Falls back to `/chat` if
`/chat/stream` 404s (older API container).

## 3. Charts

- New always-offered tool `make_chart` in `TOOL_SPECS`: the model calls it with
  `{"kind": "bar"|"line", "title": str, "columns": [x, y1, ...], "rows": [[...]]}`
  after retrieving data via run_sql. Side-effect-mutating like `show_frames`:
  appends to a new `ChatResult.charts: list[dict]` and returns a confirmation dict.
  Guards raise model-visible errors (unknown kind; ragged rows; >2000 cells).
- `SYSTEM_PROMPT` gains one bullet: chart numeric comparisons/trends the user asks
  to see, using data you actually retrieved.
- `schemas.py`: additive `ChatChart` model + `charts: list[ChatChart] = []` on
  `ChatResponse`. `/chat` and `/chat/stream` both carry it.
- Streamlit renders charts between the answer and frames: `st.bar_chart`/
  `st.line_chart` over `pd.DataFrame(rows, columns=columns)` — first column is the
  index (x-axis). No new dependencies.
- The chat eval is NOT extended to grade charts (out of scope; the grader ignores
  unknown ChatResult fields by construction — verified by the suite rerun).

## 4. Testing

- Transports: SSE parsing unit-tested with canned byte streams (content deltas,
  fragmented tool-call arguments across chunks, [DONE]); Anthropic path with a
  stubbed SDK stream object. Fallback path: a fake with only `complete` still
  works with on_token set.
- Agent: callbacks fire in order (turn deltas then steps) with a scripted
  stream-capable fake; no-callback path byte-identical (existing tests unchanged).
- make_chart: guards (bad kind, ragged rows, oversize); charts accumulate; chart
  present in the logged JSONL record.
- Serving: /chat/stream integration via the FastAPI TestClient with a scripted
  transport — event sequence turn→token*→step*→final, final body complete.
- CI stays LLM-free/network-free; the full chat eval suite reruns green (the
  frozen-shape proof). Manual: one live streamed chat with a chart against Ollama,
  transcript pasted into the PR/report.

## 5. Out of scope

Public-demo pages (Phase 8 records replays later); saved-questions gallery (Phase
8); LLM judging of charts; `/chat` behavior changes; docker-compose changes (the
same containers serve the new endpoint).
