# Demo Phase 8 — recorded chat replay, deployment, README

**Status:** approved 2026-08-21 (Phase 8 of the approved 8-phase demo plan; Phases 1–7
merged as PRs #14–#20, package v0.6). Goal: replace the last stub ("Ask the Dataset")
with a recorded chat-replay page, make the demo deployable on Streamlit Community
Cloud with zero secrets, and give the README its demo section, screenshots and live
link — after which the public demo is complete.

## Why

The five story pages exist and answer all ten success criteria locally. What remains
is delivery: everything chat-related is gitignored (`/data/chat/`), so a replay page
needs its own export into the committed package; the only Claude run on disk
(`results_anthropic_v2.jsonl`, 17/20) stores answers and SQL steps but no charts and
no frame metadata, and none of its 24 retrieved frames has a thumbnail in the
package; there is no deployment scaffolding (no root `requirements.txt`, no
`.streamlit/`); and the README never mentions `app/demo/`. Decisions (user,
2026-08-21): record a fresh Claude session over the graded cases plus a showcase set
(≈ $4 at Opus list price, measured from the recorded run's token footprint), commit
one screenshot per page, and ship static replays — the public app never calls an LLM.

## 1. Recording — `demo chat-record`

New CLI command + `src/nuscenes_data_engine/demo/chat_record.py`. Builds the agent
the way the `chat` / `chat-eval` commands do: `make_transport(settings, provider=…)`,
`open_catalog(processed_dir)`, a `SearchEngine` probe (record `search_available`
honestly if the probe fails — torch is installed, so it is expected to pass), and a
Neo4j driver when reachable (so `run_cypher` is offered; `graph_available` recorded).
Calls `agent.answer(question, transport=…, con=…, search_engine=…,
graph_driver=…, graph_database=…, log_path=None)` per question; the demo never
writes to `data/chat/log.jsonl`.

Two question sets:

- **Graded** — the cases of `configs/chat_eval.yaml` via `evaluate.load_cases`,
  graded with the harness's own `grade_case(con, case, answer=, steps=, n_frames=)`
  → the same `checks` dict `run_eval` writes (`passed`, `english`, `tool_use`,
  `grounded`, `numeric`/`expected`, `frames` where applicable). The **stored**
  question is the one the model saw (one config question was reworded after the
  original run; the page must never show a question the model did not get).
- **Showcase** — `configs/demo.yaml` `chat_replay.showcase`: 4–5 hand-written
  questions, each with `id`, `question`, `expect ∈ {chart, cypher, frames, search,
  none}`. The recorder **asserts** the expectation against the result (a `make_chart`
  step with a non-empty chart, a `run_cypher` step, ≥ 1 frame, a `search_frames` step
  that did not error) and fails the run on a miss — a dud showcase is re-recorded,
  never shipped. Showcase replays are not graded (`checks: null`, labelled so).

Per replay: `{id, kind: "eval" | "showcase", question, answer, model, provider,
steps: [{tool, input, output}], frames: [{sample_data_token, scene_name, location,
is_night, is_rain, channel, score}] (thumbnail bytes dropped; projected from the
search engine's result columns), charts: [{kind, title, columns, rows}], checks,
latency_s, error}`. A question that raises is recorded with `error` and the run
continues (as `run_eval` does); the run fails loudly only on a showcase expectation
miss or when no replay succeeded. Summary `chat_replay_summary.json`: `{model,
provider, recorded_at, git_sha, n_eval, n_passed, n_showcase, search_available,
graph_available, max_turns}`. Staged under `data/demo_curation/chat_replays/`
(`chat_replays.json` + `chat_replay_summary.json`). Options: `--provider`
(default from settings; `anthropic` for the shipped run), `--limit N` for dry runs,
`--model`. Cost note lives in the CLI help and `docs/DEMO.md`.

Amendment (2026-08-21): `search_available` records that the frame store opened
(token-based `show_frames` works regardless of the encoder); a failed encoder
shows up as an errored `search_frames` step, and a `search` showcase then fails
the run.

## 2. Build integration

`build._include_chat_replay(config, out_dir) -> str` — absent / partial-`ValueError` /
included, mirroring `_include_semsearch`. Validates before copying: unique ids; every
replay has `question`, `answer` (or `error`), `steps`; eval replays carry `checks`;
every frame token is alphanumeric; chart rows are rectangular. Copies both files flat
into `demo_data/`, exports thumbnails for every retrieved frame through
`_export_thumbs_deduped(context="chat_replay")` (a store failure degrades to a
warning, as for the other galleries), hashes both staged files as inputs and outputs,
records `validation.chat_replay ∈ {"absent", "included"}`, `n_replays`,
`n_replay_passed`, `n_replay_frames`. `_PACKAGE_VERSION = "0.7"`. No new app
dependency; size impact ≈ 0.4 MB.

## 3. Page — `app/demo/views/chat_replay.py`, "Ask the Dataset (recorded)"

Registered in `main.py` with `url_path="chat_replay"`; `views/stubs.py` is deleted.
`data.py` gains `chat_replay_available()`, `load_chat_replays()`,
`load_chat_replay_summary()` (graceful absence). Sections:

1. `recorded_banner("Recorded answers from the dataset chat agent (<model>, recorded
   <date>) — the live agent runs in the local stack")` and cards: model, questions
   recorded, graded pass rate (`n_passed/n_eval`, e.g. 17/20 — derived), tools
   exercised (SQL / Cypher / semantic search / charts, counted from steps).
2. **Showcase** — every showcase replay rendered in full: the question, the answer
   (markdown; revealed with a cosmetic `st.write_stream` on first display per session,
   static afterwards — nothing token-level is recorded), charts through
   `render.bar_chart` (kind `line` supported by the helper), a retrieved-frame gallery
   (thumbs via `frame_image_path`, captions `scene · location · night/day[, rain]`),
   and an "Agent steps (n)" expander: `tool → outcome`, SQL and Cypher in code blocks,
   search queries and shown-frame token counts as captions.
3. **Graded questions** — a selectbox over the eval replays labelled with ✓ / ✗; the
   chosen one rendered by the same renderer, plus a verdict line: passed checks, the
   failing check by name, and for numeric cases "reference 66 · answered 66". Failed
   cases are shown, never hidden.

   Amendment (2026-08-21, review): the verdict line shows the reference value
   only — `✓ passed (checks) · reference 66` / `✗ failed: numeric · reference
   2.91`; no "answered N" is extracted from the prose, the full answer is
   rendered above it.
4. Honesty lines: raw SQL result rows are not stored (the step shows the SQL and
   "n rows", as the live UI does); the summary's `search_available` /
   `graph_available` flags are stated when false; a replay with `error` renders the
   error text. Absent group → "no recorded sessions in this package
   (`demo chat-record`)". A pre-0.7 package renders the same note.

Pure helpers in `filters.py`: `replay_tool_counts(replays) -> dict`,
`verdict_line(replay) -> str`, `frame_caption(frame) -> str`, `step_detail(step) ->
tuple[str, str | None, str | None]` (outcome line, code, language) — unit-tested.

## 4. Deployment scaffolding (the deploy click is the user's — GitHub login)

- Root `requirements.txt` containing `-r app/demo/requirements.txt` (Community Cloud
  installs from the repo root; pip resolves the relative include), with a comment that
  it exists only for Cloud and that `uv sync` is the developer path. **Superseded — see the
  amendment at the end of this section.**
- `.streamlit/config.toml`: `[server] headless = true`, `[browser]
  gatherUsageStats = false`; `.gitignore` += `.streamlit/secrets.toml`.
- Overview's "how to read this demo" footer gains the master plan's credibility
  statement: "Results shown here were generated by the full offline pipeline; the
  public application serves curated experiment outputs for reproducibility and
  demonstration."
- `docs/DEMO.md` **Deploy** section: Community Cloud settings (repo
  `Sahilpatkar/nuscenes-data-engine`, branch `main`, main file `app/demo/main.py`,
  Python 3.11 per `.python-version`), no secrets, footprint (≈70 MB clone, 25 MB
  package, frames ≤ 1 MB in memory), cold-start note, and how to redeploy after a
  package rebuild. The tests' bare-venv smoke stays the local proxy for Cloud.
- After the user deploys and sends the URL, the README link is filled in (one small
  follow-up commit); until then the README shows `live link: pending deploy`.

**Amendment (2026-08-21, deploy scaffolding):** no root `requirements.txt` — Community
Cloud reads dependency files from the entrypoint's directory first, so
`app/demo/requirements.txt` is the one it uses; the root `uv.lock` is never consulted
while that file exists. Python 3.11 is selected in the deploy form's advanced settings
(Cloud defaults to 3.12).

## 5. README, screenshots, success-criteria walk

- `README.md` gains `## Public demo` directly after the intro paragraph: the live
  link (placeholder until deployed), one paragraph on what the demo shows (the six
  pages, every number derived from committed artifacts, recorded chat), six
  screenshots (`docs/img/demo-overview.png`, `-failures`, `-scenarios`,
  `-active-learning`, `-weak-supervision`, `-chat-replay`; 1200 px wide, ≤ 300 KB
  each), and a pointer to `docs/DEMO.md`. The "Status" paragraph, "Repo layout"
  (`app/demo/`, `docs/DEMO.md`) and "Build roadmap" (a demo row) are updated.
- Screenshots are produced by a committed `scripts/demo_screenshots.py` (Playwright
  driving the installed Chrome against a local `streamlit run`; documented as a manual
  tool in `docs/DEMO.md`, not a project dependency). They contain nuScenes-derived
  imagery and fall under the existing attribution section.
- `docs/DEMO.md` gains the **success-criteria walk**: the ten questions of
  `docs/DEMO_PLAN.md` → the page and section that answers each, checked locally at
  the end of this phase and against the live URL after deployment. Phase table row 8
  → shipped; the "Phase 8" forward references in `docs/` and `app/demo` are removed.

## 6. Testing

- Recorder: record assembly and frame projection from a `ChatResult` with fake frames
  (thumbnail bytes dropped, columns exact), expectation assertions (each `expect`
  kind passes/fails as specified), error continuation, summary fields, grading
  pass-through using the chat module's scripted-transport fakes (no network), and the
  stored-question rule.
- Build: absent / partial / included, schema validation failures (duplicate id, bad
  token, ragged chart), thumb export dedupe, manifest keys, determinism with the group
  staged.
- Page: AppTests — present (cards derived from the fixture, showcase chart + frames
  + steps expander, graded selectbox with a failing case showing its failing check and
  reference), absent, pre-0.7 package; `stubs.py` gone; no "Phase 8" promise left
  (`test_..._no_longer_promises_phase_8`). Requirements allowlist unchanged; AST
  guard unchanged.
- Same batched gates as Phases 6–7: consolidated review after the code tasks, final
  whole-branch review, browser check of the new page and of every page for the
  screenshots.
- Operational: `demo chat-record` (paid; `--limit 2` dry run first), `demo build` →
  v0.7, screenshots, docs/README, PR. Deployment and the live-URL commit follow the
  merge.

## 7. Out of scope

Live chat on the public app; token-level streaming replay; local-model recordings;
changes to the agent, the eval harness, or `app/streamlit_app.py`; Terraform; the
`docs/DEMO_PLAN.md` wording (user's document).

Amendment (2026-08-21): one scoped agent change landed — `_summarize` ordered
`charted` before `note`, fixing chart steps that were summarised as "repeat
(skipped)" since Phase 4 (surfaced by the recording); the shipped recording was
re-run after the fix.
