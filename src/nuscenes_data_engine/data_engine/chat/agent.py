"""The dataset-chat agent: a tool loop over guarded SQL + vector search.

The model sees compact JSON tool results (thumbnails withheld); frames surfaced by
``search_frames``/``show_frames`` are collected onto the result so the UI can render
example images. Every interaction is appended to a JSONL log for inspection.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nuscenes_data_engine.data_engine.chat import catalog

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
- Answer concisely with the actual numbers; note assumptions or data limitations.

{schema}
"""

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
]


@dataclass
class ChatResult:
    """One answered question, with the agent's working shown."""

    answer: str
    model: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    frames: list[dict[str, Any]] = field(default_factory=list)


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
) -> ChatResult:
    """Run the tool loop for one question and return the answer + working."""
    started = time.time()
    system = SYSTEM_PROMPT.format(schema=catalog.schema_prompt(catalog.catalog_tables(con)))
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    messages += list(history or [])
    messages.append({"role": "user", "content": question})

    result = ChatResult(answer="", model=transport.model)
    for _ in range(max_turns):
        reply = transport.complete(messages, TOOL_SPECS)
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
                output = _run_tool(name, args, con, search_engine, result)
            result.steps.append({"tool": name, "input": args, "output": _summarize(output)})
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
) -> dict[str, Any]:
    """Execute one tool call; frames get collected onto the result as a side effect."""
    if name == "run_sql":
        sql = str(args.get("sql", ""))
        if "\\n" in sql and "\n" not in sql:
            # Local models sometimes double-escape whitespace in tool-call JSON.
            sql = sql.replace("\\n", "\n").replace("\\t", "\t")
        return catalog.run_sql(con, sql)
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
        frames = search_engine.frames_by_tokens(tokens)
        _collect(result, frames)
        return {"attached": [_frame_meta(frame) for frame in frames]}
    return {"error": f"Unknown tool: {name}"}


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
    if "rows" in output:
        return f"{output['row_count']} rows" + (" (truncated)" if output["truncated"] else "")
    if "results" in output:
        return f"{len(output['results'])} frames found"
    if "attached" in output:
        return f"{len(output['attached'])} frames attached"
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
            "answer": result.answer,
            "latency_s": round(latency, 2),
        }
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:  # logging must never fail the answer
        logger.warning("Chat log write failed: %s", exc)
