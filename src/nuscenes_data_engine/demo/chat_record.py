"""``demo/chat_record.py`` — record ONE chat session for the demo's replay page.

The public demo never calls an LLM (spec
docs/superpowers/specs/2026-08-21-demo-phase8-design.md §1): the "Ask the Dataset"
page replays a session recorded here, through the same agent the local stack runs
(``agent.answer`` with the same tools, the same catalog, the same LanceDB store and,
when reachable, the same Neo4j driver). Everything chat-related on disk is gitignored
(``/data/chat/``), so the replay needs its own export into the committed package —
this module writes it, ``demo build`` copies and validates it.

Two question sets, answered in one session (showcase first, so a ``--limit`` dry run
covers the questions the page leads with):

- **graded** — the eval harness's own cases (``configs/chat_eval.yaml``), scored with
  the harness's own ``grade_case``. The recorder never re-implements a check, and
  stores the question it ACTUALLY SENT (the current config text): the page must never
  show a question the model did not get.
- **showcase** — hand-written questions from ``configs/demo.yaml``
  (``chat_replay.showcase``), each carrying an ``expect`` that this module ASSERTS
  against the result. A showcase whose expectation is missed fails the whole run: a
  dud showcase is re-recorded, never shipped.

Failure policy mirrors ``evaluate.run_eval`` — a question that raises is recorded with
its error and the run continues, because a transient API failure must not throw away
the answers already paid for. The run fails loudly only on a missed showcase
expectation, or when no question was answered at all.

Frames are projected to ``FRAME_COLUMNS`` (thumbnail bytes dropped — ``demo build``
exports the package's own thumbs, keyed by token) and coerced to plain JSON types:
search rows arrive through pandas, so ``is_night``/``score`` are numpy scalars that
``json.dumps`` refuses — a crash there would land AFTER every answer had been paid for.

Pure functions with injected collaborators (transport, catalog connection, search
engine, graph driver): nothing here opens a network connection, so the tests drive the
whole recorder with the chat module's scripted fakes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from nuscenes_data_engine.data_engine.chat import agent
from nuscenes_data_engine.data_engine.chat.evaluate import EvalCase, grade_case, validate_cases
from nuscenes_data_engine.demo.exporters import write_json

logger = logging.getLogger("nuscenes_data_engine")

# What a showcase question may claim its answer will demonstrate. Each is asserted by
# ``check_expectation`` against the recorded result — never taken on trust.
EXPECTATIONS = ("chart", "cypher", "frames", "search", "none")

# The frame columns that reach the package: the search engine's row minus the
# thumbnail bytes (size) and minus filename/timestamp/scene_description (the page
# renders thumbs by token and captions from scene/location/night/rain).
FRAME_COLUMNS = (
    "sample_data_token",
    "scene_name",
    "location",
    "is_night",
    "is_rain",
    "channel",
    "score",
)

KINDS = ("eval", "showcase")

# Written per replay; ``demo build`` validates against exactly this shape.
RECORD_KEYS = (
    "id", "kind", "question", "answer", "model", "provider", "steps", "frames",
    "charts", "checks", "latency_s", "error",
)


@dataclass(frozen=True)
class ShowcaseQuestion:
    """One hand-written demo question plus the claim the recorder must verify."""

    id: str
    question: str
    expect: str


@dataclass(frozen=True)
class _Queued:
    """One question to send: its identity, and how the answer will be judged."""

    id: str
    kind: str
    question: str
    case: EvalCase | None = None
    expect: str | None = None


def load_showcase(config: dict[str, Any]) -> list[ShowcaseQuestion]:
    """Parse ``chat_replay.showcase`` out of the loaded ``configs/demo.yaml``.

    Takes the WHOLE demo config (what ``load_yaml(configs/demo.yaml)`` returns), the
    way the CLI has it. An absent section or an empty list is allowed — a graded-only
    recording is a legitimate (if dull) run; a duplicate id or an unknown ``expect``
    is not, and fails here rather than after the paid run has answered five questions.
    """
    raw = (config.get("chat_replay") or {}).get("showcase") or []
    questions: list[ShowcaseQuestion] = []
    seen: set[str] = set()
    for entry in raw:
        if "id" not in entry:
            raise ValueError(f"showcase entry missing required field 'id': {entry!r}")
        question_id = str(entry["id"])
        if question_id in seen:
            raise ValueError(f"duplicate showcase id {question_id!r}")
        seen.add(question_id)
        if "question" not in entry:
            raise ValueError(f"showcase {question_id!r} missing required field 'question'")
        expect = str(entry.get("expect", ""))
        if expect not in EXPECTATIONS:
            raise ValueError(
                f"showcase {question_id!r}: expect must be one of {list(EXPECTATIONS)}, "
                f"got {expect!r}"
            )
        questions.append(
            ShowcaseQuestion(id=question_id, question=str(entry["question"]), expect=expect)
        )
    return questions


def _plain(value: Any) -> Any:
    """A JSON-serialisable version of one cell.

    LanceDB rows reach us through ``DataFrame.to_dict("records")``, so booleans and
    numbers are numpy scalars (``np.bool_``, ``np.float32``) that ``json.dumps``
    rejects. ``.item()`` is the numpy-scalar → Python-scalar conversion and exists on
    nothing we otherwise carry; anything still exotic is stringified rather than
    exploding at write time, after the answers have been paid for.
    """
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


def project_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The retrieved frames as the package stores them: ``FRAME_COLUMNS``, JSON-safe."""
    return [
        {column: _plain(frame[column]) for column in FRAME_COLUMNS if column in frame}
        for frame in frames
    ]


def _tool_steps(result: agent.ChatResult, tool: str) -> list[dict[str, Any]]:
    return [step for step in result.steps if step.get("tool") == tool]


def check_expectation(result: agent.ChatResult, expect: str) -> str | None:
    """``None`` when the showcase question delivered what it promised, else the reason.

    The claim is checked against the RESULT, not against the answer text: a question
    that says "show the Cypher" must have actually run ``run_cypher``, and a question
    that promises a chart must have produced one (a ``make_chart`` call whose arguments
    the agent's own validator rejected leaves a step behind but no chart).
    """
    if expect not in EXPECTATIONS:
        raise ValueError(f"unknown expect {expect!r}: expected one of {list(EXPECTATIONS)}")
    if expect == "none":
        return None
    if expect == "chart":
        if not _tool_steps(result, "make_chart"):
            return "expected a chart: the agent never called make_chart"
        if not result.charts:
            return "expected a chart: make_chart ran but produced no chart"
        return None
    if expect == "cypher":
        if not _tool_steps(result, "run_cypher"):
            return "expected a graph query: the agent never called run_cypher"
        return None
    if expect == "frames":
        if not result.frames:
            return "expected example frames: the answer attached no frame"
        return None
    searches = _tool_steps(result, "search_frames")
    if not searches:
        return "expected a semantic search: the agent never called search_frames"
    if all(str(step.get("output", "")).startswith("error") for step in searches):
        outputs = "; ".join(str(step.get("output", "")) for step in searches)
        return f"expected a semantic search: every search_frames call failed ({outputs})"
    return None


def build_record(
    *,
    id: str,
    kind: str,
    question: str,
    result: agent.ChatResult | None,
    provider: str,
    checks: dict[str, Any] | None,
    latency_s: float,
    error: str | None = None,
) -> dict[str, Any]:
    """One replay as the package stores it (``RECORD_KEYS`` exactly).

    ``result is None`` is the failed-question shape: no answer and no model, the error
    text instead. Keys are always present so the page and ``demo build`` can read a
    replay without probing for optional fields.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown replay kind {kind!r}: expected one of {list(KINDS)}")
    return {
        "id": id,
        "kind": kind,
        "question": question,
        "answer": None if result is None else result.answer,
        "model": None if result is None else result.model,
        "provider": provider,
        "steps": [] if result is None else result.steps,
        "frames": [] if result is None else project_frames(result.frames),
        "charts": [] if result is None else result.charts,
        "checks": checks,
        "latency_s": round(latency_s, 2),
        "error": error,
    }


def record_session(
    *,
    con: Any,
    transport: Any,
    search_engine: Any | None,
    graph_driver: Any | None,
    graph_database: str,
    cases: list[EvalCase],
    showcase: list[ShowcaseQuestion],
    provider: str,
    max_turns: int,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Answer every question through the real agent; return ``(records, summary)``.

    Showcase questions go first (config order), then the graded cases (config order),
    which is also the order the page renders them in. ``limit`` caps the TOTAL number
    of questions — a ``--limit 2`` dry run therefore costs two answers and still
    exercises a showcase question.

    Raises on a harness bug BEFORE spending anything (``validate_cases``), on a missed
    showcase expectation (all misses in one message, so one re-record fixes them all),
    and when no question was answered at all.
    """
    validate_cases(con, cases)  # a harness bug must never score as an agent failure

    queue: list[_Queued] = [
        _Queued(id=item.id, kind="showcase", question=item.question, expect=item.expect)
        for item in showcase
    ]
    queue += [
        _Queued(id=case.id, kind="eval", question=case.question, case=case) for case in cases
    ]
    if limit is not None:
        queue = queue[:limit]
    if not queue:
        raise ValueError(
            "demo chat-record: nothing to record — configure chat_replay.showcase "
            "and/or point chat_replay.cases at an eval-case file"
        )

    records: list[dict[str, Any]] = []
    misses: list[tuple[str, str]] = []
    model: str | None = None
    for index, item in enumerate(queue, start=1):
        started = time.time()
        try:
            result = agent.answer(
                item.question,
                transport=transport,
                con=con,
                search_engine=search_engine,
                max_turns=max_turns,
                log_path=None,  # the demo never writes to the live chat log
                graph_driver=graph_driver,
                graph_database=graph_database,
            )
        except Exception as exc:  # one broken question must not throw away the rest
            # ERROR for a showcase: it is recorded and the run continues (the answers
            # already paid for are worth more than the failed one), but an errored
            # showcase renders as an error box on the page — the operator should
            # re-record rather than ship it.
            log = logger.error if item.kind == "showcase" else logger.warning
            log("[%d/%d] %s (%s) raised: %s", index, len(queue), item.id, item.kind, exc)
            records.append(
                build_record(
                    id=item.id, kind=item.kind, question=item.question, result=None,
                    provider=provider,
                    # An eval case that never got an answer failed it; a showcase is
                    # not graded, so it stays null and the page labels it as errored.
                    checks={"passed": False} if item.case is not None else None,
                    latency_s=time.time() - started, error=str(exc),
                )
            )
            continue

        checks = (
            None
            if item.case is None
            else grade_case(
                con, item.case, answer=result.answer, steps=result.steps,
                n_frames=len(result.frames),
            )
        )
        if item.expect is not None:
            reason = check_expectation(result, item.expect)
            if reason is not None:
                misses.append((item.id, reason))
        if model is None:
            model = result.model
        records.append(
            build_record(
                id=item.id, kind=item.kind, question=item.question, result=result,
                provider=provider, checks=checks, latency_s=time.time() - started,
            )
        )
        logger.info(
            "[%d/%d] %s (%s): %s", index, len(queue), item.id, item.kind,
            "pass" if checks is None else ("pass" if checks.get("passed") else "FAIL"),
        )

    if misses:
        detail = "\n".join(f"  {question_id}: {reason}" for question_id, reason in misses)
        raise ValueError(
            "demo chat-record: showcase expectation missed — re-word the question(s) "
            f"and re-record rather than shipping a dud showcase:\n{detail}"
        )
    if all(record["error"] is not None for record in records):
        raise ValueError(
            "demo chat-record: no question was answered — every question raised "
            f"(first error: {records[0]['error']})"
        )

    summary = {
        "model": model if model is not None else getattr(transport, "model", None),
        "provider": provider,
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "n_eval": sum(1 for record in records if record["kind"] == "eval"),
        "n_passed": sum(
            1
            for record in records
            if record["kind"] == "eval" and (record["checks"] or {}).get("passed") is True
        ),
        "n_showcase": sum(1 for record in records if record["kind"] == "showcase"),
        # "the frame store was open", which is what the agent needed to attach frames
        # at all. A working store with a broken text encoder still shows up honestly:
        # the failed search_frames step is recorded, and a `search` showcase fails the
        # run above rather than shipping.
        "search_available": search_engine is not None,
        "graph_available": graph_driver is not None,
        "max_turns": max_turns,
    }
    return records, summary


def _git_sha() -> str:
    """The package repo's HEAD sha, via ``demo/build.py``'s own resolver.

    Imported lazily, not at module scope: ``demo build`` (Task 2) reads the files this
    module stages, and a top-level import here would make that a cycle the moment
    ``build.py`` wants anything from this module.
    """
    from nuscenes_data_engine.demo.build import _git_sha as git_sha

    return git_sha()


def write_staging(
    records: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path
) -> None:
    """Stage ``chat_replays.json`` + ``chat_replay_summary.json`` for ``demo build``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "chat_replays.json", records)
    write_json(out_dir / "chat_replay_summary.json", summary)
