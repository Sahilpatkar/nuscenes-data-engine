# Chat-agent answer-correctness eval harness

**Status:** approved 2026-08-11. Turns the dataset-chat agent's quality from a
qualitative claim into a measured number, and quantifies the documented local-model
limitation by running the same suite under both providers.

## Why, and what already fails

`docs/DATASET_CHAT.md` says local reliability "has measured limits — language drift,
occasional misread stats". That is asserted, not measured. The logged JSONL
(`data/chat/log.jsonl`, 28 rows / 20 unique questions) already contains a concrete
instance: the first record answers *"How many night scenes are there per location?"*
**in Thai**. A harness that cannot catch that is not worth building.

The log has questions but **no reference answers**, so it seeds the eval set rather
than serving as one.

## 1. The eval set — `configs/chat_eval.yaml`

~20–25 cases, version-controlled and diffable:

```yaml
cases:
  - id: night_scenes_by_location
    question: How many night scenes are there per location?
    reference_sql: |
      SELECT count(DISTINCT scene_token) FROM samples
      WHERE is_night AND location = 'singapore-onenorth'
    tolerance: 0            # absolute; use tolerance_pct for relative
  - id: foggy_frames_examples
    question: Find frames that look foggy or have heavy lens glare, and show examples.
    expect_frames: true     # retrieval case: no single number to check
```

Fields: `id`, `question`, optional `reference_sql`, optional `tolerance` (absolute,
default 0) **or** `tolerance_pct` (relative — exactly one of the two, never both),
optional `expect_frames`. Seeded from logged questions that have a checkable numeric
answer, topped up for coverage across `samples`, `annotations`, `labels`, `canbus`,
`ego_pose`.

**Reference SQL runs through the same guarded catalog the agent uses**, so a case can
never ask for something the agent is structurally unable to answer. A case whose
reference SQL errors or returns no rows fails the *suite* loudly at load time — a
broken reference is a harness bug, not an agent failure.

## 2. Grading — pure functions in `chat/evaluate.py`

Scoring is separated from running so it is unit-testable without an LLM.

- **numeric** — extract numeric literals from the answer text (handling `1,234`, `12.5`,
  and `45%`); pass if any is within tolerance of the reference value. Cases without
  `reference_sql` skip this check rather than counting as a pass.
- **english** — script-ratio test (share of alphabetic characters in Latin script above
  a threshold). Catches the observed Thai drift with no language model. Explicitly a
  heuristic, and documented as one.
- **tool_use** — the agent called at least one tool, i.e. it queried rather than
  answered from memory.
- **grounded** — every number in the answer appears in some tool output for that case.
  This is the honest hallucination check: the agent may not cite figures it never
  retrieved. Numbers matching the reference but absent from tool output still fail —
  being right by luck is not being grounded.
- **frames** — `expect_frames` cases pass when the result carries at least one frame.

Each case yields per-check verdicts plus an overall pass (all applicable checks), so a
failure names the property that broke instead of a bare score.

## 3. Running — `chat eval`

`chat eval [--config configs/chat_eval.yaml] [--provider local|anthropic] [--limit N]`
reuses `answer()` **unchanged** — no edits to `agent.py`. One case at a time; a case
that raises is recorded as a failure with the exception text rather than aborting the
suite.

Artifacts under `data/chat/eval/`:
- `results.jsonl` — per case: id, question, answer, steps, per-check verdicts, expected
  vs extracted values, latency, model.
- `report.md` — overall pass rate, per-check pass rates, a per-case table, median
  latency, and the provider/model that produced it.

Re-running with a different `--provider` writes to a provider-suffixed filename so the
two runs can be compared rather than overwriting each other.

## 4. The deliverable

Running the identical suite under `qwen2.5:14b` (local, $0) and `claude-opus-4-8`
replaces the docs' qualitative claim with a table: pass rate per check, per provider.
Whatever it shows — including the local model doing better than expected — is the
result, reported as measured.

## 5. Testing

Graders are pure and unit-tested on fixtures: a Thai answer fails `english`; an answer
citing a number absent from tool output fails `grounded`; tolerance boundaries (exactly
at, just outside) for both absolute and relative; an answer containing no digits; a
case with no `reference_sql` skips `numeric` rather than passing it; percent and
thousands-separator extraction. The runner is integration-tested with a stub transport
(the existing chat tests' pattern), so **CI stays LLM-free and torch-free** — no live
model, no network. Config loading is tested for the "exactly one tolerance field" rule
and for the broken-reference-SQL failure.

## 6. Explicitly out of scope

- Streaming responses, chart generation, and the saved-questions gallery — separate
  features, each with its own design.
- No LLM-as-judge. It would add an unvalidated component whose own accuracy would then
  need measuring; every check here is deterministic and inspectable.
- No CI gate on agent quality (CI tests the grading logic only) and no changes to
  `agent.py`, the SQL guard, or the Cypher guard.
