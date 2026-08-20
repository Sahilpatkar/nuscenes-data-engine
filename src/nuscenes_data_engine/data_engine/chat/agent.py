"""The dataset-chat agent: a tool loop over guarded SQL + vector search.

The model sees compact JSON tool results (thumbnails withheld); frames surfaced by
``search_frames``/``show_frames`` are collected onto the result so the UI can render
example images. Every interaction is appended to a JSONL log for inspection.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nuscenes_data_engine.data_engine.chat import catalog
from nuscenes_data_engine.data_engine.chat.transports import stream_with_fallback
from nuscenes_data_engine.data_engine.graph import guard as graph_guard

logger = logging.getLogger("nuscenes_data_engine")

MAX_TURNS = 8
MAX_FRAMES = 12

SYSTEM_PROMPT = """\
You are the data analyst for a nuScenes autonomous-driving dataset (Boston +
Singapore, multi-camera keyframes). Always respond in the same language as the
user's question — English unless they write otherwise. Answer questions about the
dataset using your tools; never invent numbers.

- Use run_sql for anything countable/aggregable. If a query errors, read the error
  and fix your SQL. Prefer one solid query over many small ones.
- Use search_frames for visual/semantic questions the tables cannot answer
  ("foggy-looking scenes", "construction zones").
- When a question has concrete example frames (interesting rows with a
  sample_data_token, or search hits), call show_frames with up to 6 tokens so the
  user sees them; mention in the answer that examples are attached.
- Use make_chart to visualize data you retrieved (call after run_sql; never with
  invented numbers); charts render automatically — never embed image markdown.
- Answer concisely with the actual numbers; note assumptions or data limitations.

{schema}
{graph_schema}"""

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Run one read-only DuckDB SELECT over the dataset tables "
            "and get rows back (capped at 50).",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "A single SELECT."}},
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_frames",
            "description": "Semantic vector search over camera frames (SigLIP "
            "embeddings). For visual concepts the SQL tables don't capture.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language description."},
                    "k": {"type": "integer", "description": "Results (default 6, max 12)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_frames",
            "description": "Attach specific frames (by sample_data_token) to the "
            "answer as example images for the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_data_tokens": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["sample_data_tokens"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_chart",
            "description": "Render a chart from data you retrieved — call after "
            "run_sql, never with invented numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["bar", "line"]},
                    "title": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "rows": {
                        "type": "array",
                        "items": {"type": "array", "items": {}},
                        "description": "Row-major data; each row has one value per column.",
                    },
                },
                "required": ["kind", "title", "columns", "rows"],
            },
        },
    },
]

# Guard: total chart cells (rows * columns) beyond this are rejected as model-visible
# errors rather than silently truncated — keeps chart payloads small and forces the
# model to aggregate instead of dumping raw rows.
MAX_CHART_CELLS = 2000

# Offered only when a graph driver is available (composed per call in ``answer``).
GRAPH_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_cypher",
        "description": "Run one read-only Cypher query over the knowledge graph. Use it "
        "for relationship / path / co-occurrence / similarity / temporal-next questions "
        "that SQL joins express awkwardly. Results are capped.",
        "parameters": {
            "type": "object",
            "properties": {
                "cypher": {"type": "string", "description": "A single read-only Cypher query."},
                "params": {"type": "object", "description": "Optional query parameters."},
            },
            "required": ["cypher"],
        },
    },
}


@dataclass
class ChatResult:
    """One answered question, with the agent's working shown."""

    answer: str
    model: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    frames: list[dict[str, Any]] = field(default_factory=list)
    charts: list[dict[str, Any]] = field(default_factory=list)


def _safe_callback(callback: Callable[..., None], *args: Any, label: str) -> None:
    """Invoke a caller-supplied callback without letting it abort the turn or skip the
    JSONL log — mirrors ``_log``'s own OSError containment: a side-channel failure
    (here, UI-side callback code we don't control) must never cost the answer or the
    every-interaction-logged promise in this module's docstring."""
    try:
        callback(*args)
    except Exception as exc:
        logger.warning("%s callback failed: %s", label, exc, exc_info=True)


def _frame_meta(frame: dict[str, Any]) -> dict[str, Any]:
    """The model-visible part of a frame row (no thumbnail bytes)."""
    return {
        key: frame[key]
        for key in ("sample_data_token", "scene_name", "scene_description", "channel",
                    "location", "is_night", "is_rain", "score")
        if key in frame
    }


def answer(
    question: str,
    *,
    transport: Any,
    con: Any,
    search_engine: Any | None,
    history: list[dict[str, Any]] | None = None,
    max_turns: int = MAX_TURNS,
    log_path: Path | None = None,
    graph_driver: Any | None = None,
    graph_database: str = "neo4j",
    on_turn: Callable[[], None] | None = None,
    on_token: Callable[[str], None] | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> ChatResult:
    """Run the tool loop for one question and return the answer + working."""
    started = time.time()
    graph_schema = (
        graph_guard.graph_schema_prompt() + "\n" if graph_driver is not None else ""
    )
    system = SYSTEM_PROMPT.format(
        schema=catalog.schema_prompt(catalog.catalog_tables(con)), graph_schema=graph_schema
    )
    tools = TOOL_SPECS + ([GRAPH_TOOL_SPEC] if graph_driver is not None else [])
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    messages += list(history or [])
    messages.append({"role": "user", "content": question})

    result = ChatResult(answer="", model=transport.model)
    seen_calls: set[str] = set()
    for _ in range(max_turns):
        if on_turn:
            _safe_callback(on_turn, label="on_turn")
        reply = (
            stream_with_fallback(transport, messages, tools, on_token=on_token)
            if on_token
            else transport.complete(messages, tools)
        )
        tool_calls = reply.get("tool_calls") or []
        messages.append(
            {"role": "assistant", "content": reply.get("content"), "tool_calls": tool_calls}
            if tool_calls
            else {"role": "assistant", "content": reply.get("content")}
        )
        if not tool_calls:
            result.answer = str(reply.get("content") or "")
            break
        for call in tool_calls:
            name = call.get("function", {}).get("name", "")
            try:
                args = json.loads(call.get("function", {}).get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                args, output = {}, {"error": f"Bad tool arguments: {exc}"}
            else:
                signature = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
                if signature in seen_calls:
                    # Break tool-call loops: a smaller model sometimes re-issues an
                    # identical call instead of writing its answer (seen with show_frames).
                    output = {
                        "note": "You already ran this exact call and the result is "
                        "unchanged. Write your final answer now from the results you have."
                    }
                else:
                    seen_calls.add(signature)
                    output = _run_tool(
                        name, args, con, search_engine, result,
                        graph_driver=graph_driver, graph_database=graph_database,
                    )
            step = {"tool": name, "input": args, "output": _summarize(output)}
            result.steps.append(step)
            if on_step:
                _safe_callback(on_step, step, label="on_step")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": json.dumps(output, default=str),
                }
            )
    else:
        result.answer = "I couldn't finish within the tool-call budget — try a narrower question."

    _log(log_path, question, result, time.time() - started)
    return result


def _run_tool(
    name: str,
    args: dict[str, Any],
    con: Any,
    search_engine: Any | None,
    result: ChatResult,
    *,
    graph_driver: Any | None = None,
    graph_database: str = "neo4j",
) -> dict[str, Any]:
    """Execute one tool call; frames get collected onto the result as a side effect."""
    if name == "run_sql":
        sql = str(args.get("sql", ""))
        if "\\n" in sql and "\n" not in sql:
            # Local models sometimes double-escape whitespace in tool-call JSON.
            sql = sql.replace("\\n", "\n").replace("\\t", "\t")
        return catalog.run_sql(con, sql)
    if name == "run_cypher":
        if graph_driver is None:
            return {"error": "Knowledge graph is not available."}
        params = args.get("params")
        return graph_guard.run_cypher(
            graph_driver,
            str(args.get("cypher", "")),
            params if isinstance(params, dict) else None,
            database=graph_database,
        )
    if name in ("search_frames", "show_frames"):
        if search_engine is None:
            return {"error": "Vector search is not available (LanceDB store not found)."}
        if name == "search_frames":
            k = min(int(args.get("k") or 6), MAX_FRAMES)
            try:
                frames = search_engine.search_text(str(args.get("query", "")), k)
            except Exception as exc:  # encoder/store failures -> model-visible error
                return {"error": f"search failed: {exc}"}
            _collect(result, frames)
            return {"results": [_frame_meta(frame) for frame in frames]}
        tokens = [str(token) for token in args.get("sample_data_tokens") or []][:MAX_FRAMES]
        try:
            frames = search_engine.frames_by_tokens(tokens)
        except Exception as exc:  # store failures -> model-visible error, not a crash
            return {"error": f"frame lookup failed: {exc}"}
        _collect(result, frames)
        return {"attached": [_frame_meta(frame) for frame in frames]}
    if name == "make_chart":
        return _make_chart(args, result)
    return {"error": f"Unknown tool: {name}"}


def _make_chart(args: dict[str, Any], result: ChatResult) -> dict[str, Any]:
    """Validate + collect one chart request; guards are model-visible so the model
    can correct its own call rather than the answer silently losing the chart — and so
    the Streamlit renderer (``pd.DataFrame(rows, columns=columns).set_index(columns[0])``)
    never IndexErrors, silently collapses duplicate columns, or chokes on a
    non-scalar x value or a non-numeric y value.

    Deliberately reads ``columns``/``rows`` without an ``or []`` fallback: coercing a
    falsy-but-present value (e.g. a stray ``0`` or ``null``) to ``[]`` would silently
    hide a malformed call instead of surfacing it through the type check below.
    """
    kind = str(args.get("kind", ""))
    if kind not in ("bar", "line"):
        return {"error": f"Unknown chart kind {kind!r}: expected 'bar' or 'line'."}
    title = str(args.get("title", ""))
    columns = args.get("columns")
    rows = args.get("rows")
    if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
        return {"error": "columns must be a list of strings."}
    if len(columns) < 2:
        return {
            "error": f"columns needs at least 2 entries (one x-axis, one y-series); "
            f"got {len(columns)}."
        }
    if len(set(columns)) != len(columns):
        return {
            "error": f"columns must be unique — duplicate names make "
            f"set_index ambiguous and can silently empty the chart: {columns!r}."
        }
    if not isinstance(rows, list):
        return {"error": "rows must be a list of rows."}
    if len(rows) < 1:
        return {"error": "rows must not be empty."}
    if any(not isinstance(row, list) or len(row) != len(columns) for row in rows):
        return {"error": f"Every row must have exactly {len(columns)} cells (one per column)."}
    for row in rows:
        x_value = row[0]
        if isinstance(x_value, bool) or not isinstance(x_value, str | int | float):
            return {
                "error": f"Chart x-values (first cell of each row) must be a string, "
                f"int, or float; got {x_value!r} in row {row!r}."
            }
        for value in row[1:]:
            if isinstance(value, bool) or not isinstance(value, int | float):
                return {
                    "error": f"Chart y-values must be numeric; got {value!r} in row {row!r}."
                }
    total_cells = len(rows) * len(columns)
    if total_cells > MAX_CHART_CELLS:
        return {
            "error": f"Chart has {total_cells} cells, over the {MAX_CHART_CELLS} limit — "
            "aggregate the data further before charting."
        }
    result.charts.append({"kind": kind, "title": title, "columns": columns, "rows": rows})
    return {
        "charted": True,
        "title": title,
        "note": "Displayed to the user automatically — do not embed an image or link "
        "in your answer.",
    }


def _collect(result: ChatResult, frames: list[dict[str, Any]]) -> None:
    seen = {frame["sample_data_token"] for frame in result.frames}
    for frame in frames:
        if frame["sample_data_token"] not in seen and len(result.frames) < MAX_FRAMES:
            result.frames.append(frame)
            seen.add(frame["sample_data_token"])


def _summarize(output: dict[str, Any]) -> str:
    """Compact, human-readable step summary for the UI/log (not the model)."""
    if "error" in output:
        return f"error: {output['error']}"
    if "note" in output:
        return "repeat (skipped)"
    if "rows" in output:
        return f"{output['row_count']} rows" + (" (truncated)" if output["truncated"] else "")
    if "results" in output:
        return f"{len(output['results'])} frames found"
    if "attached" in output:
        return f"{len(output['attached'])} frames attached"
    if "charted" in output:
        return f"charted: {output['title']}"
    return "ok"


def _log(log_path: Path | None, question: str, result: ChatResult, latency: float) -> None:
    """Append the interaction to the JSONL query log (spec: log every agent query)."""
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "question": question,
            "model": result.model,
            "steps": result.steps,
            "n_frames": len(result.frames),
            "n_charts": len(result.charts),
            "answer": result.answer,
            "latency_s": round(latency, 2),
        }
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:  # logging must never fail the answer
        logger.warning("Chat log write failed: %s", exc)
