"""Ask the Dataset (recorded) — a recorded session with the dataset chat agent,
replayed out of the committed package (design: docs/superpowers/specs/2026-08-21-
demo-phase8-design.md §3).

The public app never calls an LLM. Every question, answer, SQL/Cypher query, chart
and retrieved frame on this page comes from one `demo chat-record` run against the
real agent (`chat_replays.json` + `chat_replay_summary.json`); only the section
prose is written here.

Three honesty rules this page is built around:

- Nothing is generated at view time. The word-by-word reveal of an answer is
  COSMETIC — the recording stores finished text, not a token stream — and it runs
  once per browser session; after that the same stored text renders statically.
- Raw result rows are not stored. A step shows the query the agent ran and the row
  count it got back, which is exactly what the live chat UI shows; the page never
  reconstructs a table the package does not carry.
- Graded questions are shown WITH their verdicts, failures included: the selectbox
  marks every case ✓ / ✗, a failed one names the check it failed and the reference
  value it was graded against, and a question the provider errored on shows the
  error instead of an empty answer.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from typing import Any

import pandas as pd
import streamlit as st
from filters import frame_caption, replay_tool_counts, step_detail, verdict_line
from render import bar_chart, metric_cards, recorded_banner

from data import (
    chat_replay_available,
    frame_image_path,
    load_chat_replay_summary,
    load_chat_replays,
)

# What this page says when the package carries no recording -- `demo chat-record` is
# a paid, one-off run against a live provider, so a fresh clone (and every package
# built before the group existed) simply has none, which is not an error.
_ABSENT_NOTE = "no recorded sessions in this package (demo chat-record)"

_ROWS_NOTE = (
    "Raw SQL result rows are not stored — each step shows the query and its row "
    "count, as the live UI does."
)

# The ids whose answer has already been revealed in THIS browser session: the
# reveal is a one-time cosmetic effect, and re-typing an answer on every rerun
# (every selectbox change) would be an animation, not a chat.
_SEEN_KEY = "chat_replay_seen"

# A whole answer reveals in about _REVEAL_TOTAL_S however long it is (per-word
# delay capped, so a two-word answer doesn't crawl): a fixed per-word delay makes a
# 200-word recorded answer take ten seconds before the page is readable.
_REVEAL_TOTAL_S = 1.2
_REVEAL_MAX_STEP_S = 0.03

# Each chunk is one word WITH its own trailing whitespace, so the streamed text is
# character-identical to the stored answer -- splitting on whitespace and re-joining
# with " " would flatten the markdown (lists, paragraphs) of a real recorded answer,
# and the static re-render afterwards would then look different from the reveal.
_WORD_CHUNK = re.compile(r"\S+\s*")

_GALLERY_COLUMNS = 4


def _word_chunks(answer: str) -> Iterator[str]:
    chunks = _WORD_CHUNK.findall(answer)
    delay = min(_REVEAL_MAX_STEP_S, _REVEAL_TOTAL_S / len(chunks)) if chunks else 0.0
    for chunk in chunks:
        time.sleep(delay)
        yield chunk


def _reveal_answer(replay: dict[str, Any]) -> None:
    """The stored answer, typed out on its first display in this session."""
    answer = str(replay.get("answer") or "")
    seen: set[str] = st.session_state.setdefault(_SEEN_KEY, set())
    replay_id = str(replay.get("id", ""))
    if replay_id in seen:
        st.markdown(answer)
        return
    seen.add(replay_id)
    st.write_stream(_word_chunks(answer))


def _render_charts(replay: dict[str, Any]) -> None:
    """The charts the agent drew with its own `make_chart` tool, as recorded."""
    for chart in replay.get("charts") or []:
        columns = [str(column) for column in chart.get("columns") or []]
        rows = chart.get("rows") or []
        if len(columns) < 2 or not rows:
            # The recorder stores what the model asked for; a one-column or empty
            # payload is a real (rare) model mistake, said plainly rather than
            # crashed on or silently dropped.
            st.caption("chart payload not renderable")
            continue
        st.altair_chart(
            bar_chart(
                pd.DataFrame(rows, columns=columns),
                x=columns[0],
                y=columns[1],
                title=str(chart.get("title") or ""),
                mark="line" if chart.get("kind") == "line" else "bar",
                label_angle=-30,
            ),
            width="stretch",
        )


def _render_frames(replay: dict[str, Any]) -> None:
    """The frames the agent attached to this answer (thumbs from the package)."""
    frames = replay.get("frames") or []
    if not frames:
        return
    per_row = min(len(frames), _GALLERY_COLUMNS)
    columns = st.columns(per_row)
    for position, frame in enumerate(frames):
        with columns[position % per_row]:
            image_path = frame_image_path(str(frame.get("sample_data_token", "")))
            if image_path is not None:
                st.image(str(image_path))
            st.caption(frame_caption(frame))


def _render_steps(replay: dict[str, Any]) -> None:
    """What the agent actually ran to get there, in order."""
    steps = replay.get("steps") or []
    if not steps:
        return
    with st.expander(f"Agent steps ({len(steps)})"):
        for step in steps:
            outcome, code, language = step_detail(step)
            st.markdown(outcome)
            if code:
                st.code(code, language=language)
        st.caption(_ROWS_NOTE)


def _render_replay(replay: dict[str, Any]) -> None:
    """One recorded question end to end — the same renderer for showcase and
    graded replays, so a graded case is never presented more thinly than a
    showcase one."""
    st.markdown(f"**{replay.get('question', '')}**")
    error = replay.get("error")
    if error:
        st.error(str(error))
    elif replay.get("answer"):
        _reveal_answer(replay)
    else:
        st.caption("this question was recorded without an answer")
    _render_charts(replay)
    _render_frames(replay)
    _render_steps(replay)


def _render_header(replays: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """(1) What this recording is, and what it exercised."""
    model = str(summary.get("model") or "unknown model")
    recorded_at = str(summary.get("recorded_at") or "")
    recorded_banner(
        f"Recorded answers from the dataset chat agent ({model}, recorded "
        f"{recorded_at[:10]}) — the live agent runs in the local stack"
    )

    # Counted from the replays on screen rather than read off the summary's
    # n_eval/n_passed: `demo build` already refuses a summary that disagrees with
    # its records, so the two always match -- and deriving them keeps the cards
    # true to exactly what this page renders.
    graded = [replay for replay in replays if replay.get("kind") == "eval"]
    n_passed = sum(
        1
        for replay in graded
        if isinstance(replay.get("checks"), dict) and replay["checks"].get("passed")
    )
    counts = replay_tool_counts(replays)
    # Two cards only: "Model" (e.g. "claude-opus-4-8") and "Tools exercised"
    # ("SQL 27 · Cypher 1 · …") both overflowed st.metric's large-font value box, so
    # those figures move to a plain caption line below instead (consolidated
    # review, real-browser finding).
    metric_cards([
        ("Questions recorded", str(len(replays))),
        ("Graded pass rate", f"{n_passed}/{len(graded)}"),
    ])
    st.caption(
        f"Model `{model}` · tools exercised: SQL {counts['run_sql']} · "
        f"Cypher {counts['run_cypher']} · semantic search {counts['search_frames']} · "
        f"frames shown {counts['show_frames']} · charts {counts['charts']}"
    )
    st.caption(
        "The pass rate covers the graded cases only — the showcase questions below "
        "are demonstrations and were never graded."
    )

    unavailable = [
        text
        for key, text in (
            ("search_available", "without the semantic search engine"),
            ("graph_available", "without the graph"),
        )
        if summary.get(key) is False
    ]
    if unavailable:
        st.caption(
            "the recording ran " + " and ".join(unavailable) + " — the agent was "
            "never offered that tool, so no step below uses it"
        )


def _render_showcase(replays: list[dict[str, Any]]) -> None:
    """(2) Every showcase replay in full, in the order the recorder wrote them."""
    showcase = [replay for replay in replays if replay.get("kind") == "showcase"]
    if not showcase:
        return
    st.subheader("Showcase questions")
    st.caption(
        "Hand-picked questions recorded to show what the agent does with the "
        "dataset: the answer it gave, the charts it drew, the frames it retrieved "
        "and every tool call it made."
    )
    for position, replay in enumerate(showcase):
        if position:
            st.divider()
        _render_replay(replay)
    # Closes the section itself (rather than render() drawing the rule): a
    # recording with no showcase replays would otherwise leave two rules stacked
    # on top of each other.
    st.divider()


def _render_graded(replays: list[dict[str, Any]]) -> None:
    """(3) The eval harness's own cases, with their verdicts — failures included."""
    st.subheader("Graded questions")
    graded = [replay for replay in replays if replay.get("kind") == "eval"]
    if not graded:
        st.info("this recording has no graded questions")
        return

    by_id = {str(replay.get("id", "")): replay for replay in graded}
    labels = {
        key: (
            f"{'✓' if isinstance(replay.get('checks'), dict) and replay['checks'].get('passed') else '✗'} "
            f"{replay.get('question', '')}"
        )
        for key, replay in by_id.items()
    }
    st.caption(
        "The chat-eval harness's own cases, graded by its own checks (the question "
        "shown is the one the model was asked). Failed cases are here too — pick "
        "any ✗ to see what it answered and which check it missed."
    )
    chosen = st.selectbox(
        "Graded question",
        options=list(by_id),
        format_func=lambda key: labels[str(key)],
        key="chat_replay_graded",
    )
    replay = by_id[str(chosen)]
    _render_replay(replay)
    st.caption(verdict_line(replay))


def render() -> None:
    st.title("Ask the Dataset (recorded)")

    if not chat_replay_available():
        st.info(_ABSENT_NOTE)
        return
    replays = load_chat_replays()
    if not replays:
        st.info(_ABSENT_NOTE)
        return

    _render_header(replays, load_chat_replay_summary())
    st.divider()
    _render_showcase(replays)
    _render_graded(replays)
