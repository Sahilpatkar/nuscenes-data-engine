# Grounding check v2 + replay re-grade + local-model sweep

**Status:** approved 2026-08-11. Executes the pre-registered fix recorded in
`docs/DATASET_CHAT.md`'s eval section: the v1 grounding check penalises answers that
show their arithmetic, which biases the published comparison against the stronger
model. v2 admits legitimately derived numbers while still rejecting miscalculation
and invention — and the change is measured by replaying the existing runs, not by
re-running models first.

## Why (measured, not asserted)

The final branch review categorised all nine of Claude's `grounded` failures against
the live catalog: 0 of 9 are invented figures. They are derived arithmetic
("4,986 of 5,000, so 14 failed"), computed percentages ("99.66% = 4,969/4,986"),
unit conversions (5.1049 m/s × 3.6 = 18.4 km/h), a column sum (1,183,790 = the six
per-channel counts), schema-prompt constants ("≤ −3.0 m/s² within ±0.5 s"), a frame
count the checker cannot see, and — in two cases — things that *should* stay failed
(model-memory domain facts, a hedged "~50+" estimate). Meanwhile the local model's
two `grounded` failures are artifacts of the same mechanism (a schema constant, a
`1` scraped from printed SQL), not catches. The check as shipped mostly measures how
much arithmetic a model narrates, inverted against capability.

## 1. Grounding v2 — `chat/evaluate.py` only

`agent.py`, `catalog.py`, and the other four checks are untouched. A cited number is
grounded when it matches, at the answer's own stated precision (the existing
`_matches_at_cited_precision`), any member of the allowed set:

1. **Observed values** (unchanged): every numeric cell and row count from
   re-executing the answer's own `run_sql` steps through the guarded catalog — now
   **plus the per-column sums** of each numeric result column (one extra value per
   column, motivated by `missing_cam_files`' channel-sum total).
2. **Targeted one-step derivations** of two observed values `a, b`:
   `a + b`, `a − b`, `a / b × 100`; and the unary m/s↔km/h conversions `a × 3.6`,
   `a / 3.6`. Exactly these — no products, no quotients other than the percentage
   form, no derivation chains (derived values are not re-derived). Each rule exists
   because a real failure used it; the taxonomy is the allowlist.
3. **Schema-prompt constants**: numbers extracted at grading time from the same
   `catalog.schema_prompt(catalog.catalog_tables(con))` text the agent's system
   prompt embeds, using the permissive extractor already used for question echoes.
   Never a hardcoded list — the source of truth is what the model was shown.
4. **Question echoes** (existing, unchanged) and the **attached-frame count**
   (`len(frames)`, passed by `grade_case`), closing the "6 examples attached" gap.

**The property that keeps the check honest:** v2 admits *correct* arithmetic and
still rejects *incorrect* arithmetic. The historical log's record 5 answered 98.5%
where its own retrieved counts give 4,969 / 4,986 = 99.66% — a genuine miscalculation.
That answer must still fail `grounded` under v2, because a wrong result matches no
valid derivation of the observed values. This is a required regression test, not an
aspiration.

**Collision risk, stated up front:** admitting pairwise sums/differences means a
wrong figure can coincide with some `a ± b`. The bound is deliberate — one step,
four rules, no closure — and the acceptance tests include a collision guard (a wrong
count adjacent to the true one must not pass). If replay shows a collision in real
data, it is reported as a v2 limitation, not silently accepted.

## 2. Pre-registered expectations (written before any re-grade runs)

- Claude's `grounded` goes from 11/20 to **18/20**: all nine failures flip **except**
  `max_instance_keyframes` (cites "~20 s at 2 Hz" — nuScenes domain facts from model
  memory, absent from the schema prompt; correctly ungrounded) and
  `foggy_misty_glare_frames` (the "~50+" hedged estimate was never computed by any
  query; correctly ungrounded).
- Claude's composite rises to **17/20** on replay, not 18: of the nine failures,
  eight were grounded-only (two of those stay failed by design → six flip), and the
  ninth (`avg_visibility_pedestrian_night`) also failed `numeric` in the stored run —
  its grounding flips but the replayed composite stays failed. A *fresh* run against
  the now-disambiguated question could pass it; that would show up in §5's live 
  sweep, not the replay.
- The local model's composite barely moves: its failures are `english`/`numeric`/
  `tool_use`; its two v1 `grounded` failures were artifacts and are expected to flip.
- Record 5 of the historical log still fails `grounded` (the 98.5% miscalculation).

Whatever the replay actually shows is the reported result. If a prediction is wrong,
the discrepancy is documented — the instrument is not adjusted again to meet it.

**Amendment (2026-08-11, pre-registered before Task 3 landed):** the Tasks-1-2 review
projected — by extracting the real schema-prompt constants — that §1 rule 3 will admit
both residual numbers of `foggy_misty_glare_frames` (5000, and 50 via the schema text
"50 Hz", matched units-blind against a hallucinated frame count). Prediction 1 is
therefore expected to be MISSED for `foggy` once schema constants land: it will flip
to grounded as a *documented limitation of a units-blind allowlist*, not a collision.
`max_instance_keyframes` stands as the surviving negative control (its residual 20 is
not a schema constant). Recorded here before Task 6 measures it, per this spec's own
rule that the instrument is not adjusted to meet predictions — nor predictions quietly
rewritten after the data is in.

## 3. Offline replay re-grade — `chat-eval --regrade <results.jsonl>`

Re-grades a stored run's answers+steps with the **current** grader. No LLM calls, no
network: the stored records already carry `id`, `question`, `answer`, `steps`, and
`n_frames`; reference values recompute from `configs/chat_eval.yaml` by case id, and
grounding re-executes the stored SQL against the current catalog. Writes
`<stem>_v2.jsonl` and `<stem>_v2.md` next to the input, never overwriting it.

This is the instrument-isolation measurement: same answers, new grader — any delta is
attributable to the grader alone. It also makes every future grader change replayable
against history. A record whose id is missing from the current config is skipped with
a logged warning and counted in neither numerator nor denominator.

## 4. Model-suffixed artifacts

When `--model` is passed explicitly, output filenames gain a slug
(`results_local_qwen2.5-32b.jsonl`, `report_local_qwen2.5-32b.md`; slug = model name
with `:`/`/` replaced by `-`). Without `--model`, filenames are unchanged, so
existing docs and scripts keep working. This stops sweep runs clobbering each other
(the 14b artifacts were hand-backed-up once already; that stops being necessary).

## 5. The sweep (operational; no code beyond §4)

After v2 lands and CI is green:

1. Replay-regrade `results_local.jsonl` (14b) and `results_anthropic.jsonl` → the
   grader-isolated before/after table.
2. Run the live suite on `qwen2.5:32b` (`--provider local --model qwen2.5:32b`;
   pulled already, 48 GB RAM holds the ~20 GB model).
3. Update `docs/DATASET_CHAT.md`'s results table with a model column: 14b (v2),
   32b (v2), claude-opus-4-8 (v2 replay). Decision rule, stated in the docs: the
   cheapest local model that clears an acceptable pass rate; whether 32b clears it is
   the finding either way.

## Testing

- One unit test per derivation rule, each constructed from its motivating real
  failure: `14 = 5000 − 4986`; `99.66 ≈ 4969/4986 × 100` at cited precision;
  `18.4 ≈ 5.1049 × 3.6`; the per-channel column sum.
- The record-5 regression: correct inputs, wrong arithmetic → still fails.
- Collision guard: with observed `{40, 2}`, a cited `43` fails (and the admitted
  `42 = 40 + 2` case is documented in the test as the accepted trade-off).
- Schema-constant extraction: `−3.0` and `0.5` from the real `schema_prompt` text
  are admitted; a number absent from it is not.
- `--regrade` integration test on a fixture JSONL with a stub-shaped record: correct
  artifacts written, input untouched, unknown-id record skipped with a warning.
- CI stays LLM-free and torch-free; all graders remain pure functions.

## Explicitly out of scope

- Agent-side fixes (English retry loop, malformed-tool-call repair shim): the next
  cycle, and only if the 32b measurement still warrants them.
- No LLM-as-judge; no `agent.py` or `catalog.py` changes; no new eval cases and no
  tolerance changes (the instrument changes, the test set does not).
- No CI gate on agent quality.
