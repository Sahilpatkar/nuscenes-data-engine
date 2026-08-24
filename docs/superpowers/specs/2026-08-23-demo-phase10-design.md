# Demo Phase 10 — storytelling rework of the guided tour

**Status:** approved 2026-08-23 (plan-mode session over the user's brief
`docs/nuscenes_demo_storytelling_improvement_plan.md`, 46 sections; user decisions:
scope = the brief's highest + medium priorities, tour headline = "From model failure
to better training data"). App-only: package v0.8 and every exporter unchanged.
Phases 9a (merged, PR #22) and 9b (branch `demo-phase-9b`, 310849d) precede this.

## Why

The demo is technically strong but the guided tour reads as *metrics → image →
graph → metrics*. The brief asks for *problem → intervention → impact*: lead each
step with a headline and takeaway, fold mechanism detail behind expanders, use
readable experiment names in the tour (deep pages stay technical, brief §32), show
the payoff with hero numbers (absolute + relative together), and replace the prose
overlay legend with a compact component. No new features (§41): copy, composition,
and hierarchy only.

## Honesty rules (unchanged from Phases 9a/9b, restated for this phase)

- Every number on screen is derived from the package tables; the +41.8 % relative
  night-pedestrian gain is computed by one helper (`gain_text`/`relative_gain`),
  never hard-coded, and the relative clause is omitted when the base is ≤ 0.
- Superlatives and headlines are computed: the "Best intervention…" statement names
  whichever arm actually has the max `delta_night`; the "Closed the loop:
  … measurable improvement" subheader requires the arm's `delta_night > 0`; the
  similarity-0 %-night explanation renders only when the `mined` row is present with
  `night_share < 0.005`.
- The 0.135 low-confidence hero disclaimer chain survives in substance (the fact
  line's phrasing may swap the raw id for its story label; "below the 0.40 hit
  floor; the matching rule still counts it as a hit" stays). The negative
  weak-supervision answer and the worst-night-arm superlative are untouched.
- Story labels never replace raw ids inside precise experimental claims: pinned
  sentences that carry backticked arm ids keep them; the tour shows the raw id in
  small text (chart caption, provenance) wherever a story label fronts it.
- Readable names are tour-scope only (§15 + §32): `filters.arm_story_label` maps
  the nine mining arms; deep pages keep `model_label` / raw ids / the full 13-arm
  chart / the per-arm table.

## 1. Shared vocabulary (`filters.py`, `render.py`)

- `_ARM_STORY_LABELS`: baseline → "Baseline (no mined data)", random → "Random
  sample — control", mined → "Similarity mining", rate → "Failure-rate mining",
  strat → "Stratified mining", rate_strat → "Rate + stratified mining", graph →
  "Graph mining", graph_rate → "Graph + rate scoring", graph_rate_night → "Graph +
  night targeting". `arm_story_label(model)` falls back to `model_label()` so weak
  checkpoints keep their technical labels.
- `_SCORE_BASED_ARMS = ("rate", "rate_strat", "strat")`;
  `tour_strategies(arms, *, baseline, arm) -> dict[str, str]` — ordered
  {arm id: story label}: baseline, `random`, `mined`, the best score-based arm
  (max `delta_night` over those present — computed), the tour arm; ids absent from
  the table are omitted; duplicates collapse. Consumed by the existing
  `strategy_coverage`.
- `relative_gain(before, after) -> float | None` (None when `before <= 0`);
  `gain_text(metric, before, after) -> str` — "{metric} {before:.4f} →
  {after:.4f} (+{Δ:.4f} absolute, {rel:+.1%} relative)", relative clause omitted
  when None. On v0.8: "Night pedestrian mAP50-95 0.0826 → 0.1171 (+0.0345
  absolute, +41.8% relative)".
- `upgrade_callout(fixed: pd.DataFrame) -> tuple[str, str, str] | None` —
  (category_group, before, after) from a `fixed_boxes` frame, first pedestrian row
  else first row, values verbatim from the frame's own claim strings; None on empty.
- `render.LEGEND_ITEMS` / `PSEUDO_LEGEND_ITEM` / `legend_text(*, pseudo=False)` /
  `legend(*, pseudo=False)`: a one-line badge-chip legend (green — ground truth ·
  orange dashed — GT box the model missed · white — true positive · yellow dotted —
  low-confidence claim (below the hit floor) · red — false positive; blue —
  pseudo-label box), defined beside the `STYLE_*` constants, with a drift-guard
  unit test tying entries to `_PRED_STYLES` + `STYLE_GT`/`STYLE_FN`/`STYLE_PSEUDO`.
  The badge colour is a hint; the words carry the on-image description.
  (amendment 2026-08-23, review: `PSEUDO_LEGEND_ITEM` and the `pseudo=` parameter
  are removed — no page ever called them, and a pseudo chip on Weak Supervision's
  GT + pseudo image would claim prediction colours that image never draws. The
  signatures are `legend_text()` / `legend()`; the drift guard keeps
  `STYLE_PSEUDO` outside the legend contract.)

## 2. The tour, step by step (`views/tour.py`; keys, step count and degradation branches unchanged)

Page: `st.title("From model failure to better training data")` + subtitle caption
"See how the data engine finds a perception weakness, mines targeted AV scenarios,
and measures whether retraining fixes it." Sidebar `st.Page(title="Guided tour")`
stays. Step captions keep the "Step N of 7 · {title}" shape with new titles.
Takeaways use `render.learned()`.

| # | New title | Changes |
|---|---|---|
| 0 | We found a blind spot | subheader "Aggregate accuracy was hiding a night-driving blind spot."; 3 metric cards + bold miss-count line verbatim; the 4-clause paragraph → one derived sentence; `learned("Aggregate metrics hide important failure slices.")` |
| 1 | What the failure looks like | headline "Here the baseline misses a pedestrian at night."; radio kept with `format_func=arm_story_label`; `render.legend()` under the image; "Held-out validation frame — never in any training set" caption (only when the row's `split == "val"`); bridge sentence "Finding one failure is easy — the hard part is finding the rest of the dataset where the same thing happens."; HERO_CAPTION, fact lines (low-conf clause rephrased to the story label, floor + counts-as-hit kept), miss-to-hit sentence kept |
| 2 | Where else does this happen? | headline "Is this one bad photo, or a recurring driving scenario?"; layout/chips/filmstrip/CAN curves/parity/train-pool captions untouched; one mechanism sentence "The system searches driving context — night, braking, pedestrians near the ego — not just similar-looking images."; `learned("Perception failures must be analyzed in driving context.")` |
| 3 | What data should we add? | headline "Thousands of candidate training frames — which are worth labelling?"; `reason_chips` row unchanged; §9 one-liner "failed val frame → similarity community → representative train-pool frames → selected for retraining"; the seven `selection_factors` lines + causal closing sentence move into `st.expander("How selection works")`; flagship/weak-rejected sentences, "reproduced" provenance, deep link stay outside; architecture strip as one grey caption "System path: nuScenes → validated Parquet → SQL / Neo4j / CAN → failure analysis → scenario search → active learning → YOLO retraining → evaluation" |
| 4 | We changed the training data | cards unchanged; bold comparison "Targeted mining: **31% night** · random control: **13% night**" (derived; similarity clause when present); the 13-arm chart → the ~5-bar `strategy_coverage(tour_strategies(...))` chart (`delta_night`, highlight the arm, story labels on the axis, raw ids in a caption); fairness statement "Same detector · same {n:,}-frame budget · same training configuration · scored on the same held-out split — Random sample is the control." (budget clause only when all charted non-baseline arms share `n_train_images`); computed winner statement; conditional §37 similarity-0 %-night caption; scene-diversity sentence |
| 5 | Did it fix the failure? | derived before/after callout (two metric cards from `upgrade_callout`); two-column baseline-vs-arm overlays with `arm_story_label` captions; the `tour_exemplar_model` radio retires (the AL page keeps its radio; `tour_open_exemplar` serves inspection); `render.legend()`; `gain_text` abs+rel sentence; night-mAP sentence, `_HERO_HONESTY_LINE`, held-out caption, provenance, deep link kept; the `fixed_boxes` dataframe in `st.expander("Technical details — per-box claims")`; empty-table branch byte-identical |
| 6 | Closed the loop | guarded subheader ("… measurable improvement" iff `delta_night > 0`, else "… measured result"); three hero metric cards (frames added / night share vs control / night-ped relative gain — short values, each omitted on NA); the four bordered answers stay (answer 3 uses `gain_text`; answer 2 gains the §37 clause when its condition holds; answer 4 untouched); closing thesis "The system demonstrated a repeatable way to turn model failures into data decisions." |

(amendment 2026-08-23, review: step 4's bold comparison names every arm by its
story label, so one screen has one name per arm — "Graph + night targeting: **31%
night** · Random sample — control: **13% night**"; step 5's two overlay captions
carry the raw arm id beside the label; step 6's night-pedestrian card states its
absolute gain alone, the arrow staying in answer 3's `gain_text` sentence.)

Go-deeper labels become story sentences keeping the "PageName — clause" shape (no
numerals in static labels), e.g. "Failure Explorer — see every night frame the
baseline missed", "Scenario Search — find every braking-near-pedestrian event",
"Active Learning — inspect the full mined set, all 13 arms".

## 3. Legend rollout

`render.legend()` replaces the prose `_OVERLAY_LEGEND` caption on tour step 5, the
Active Learning before/after panel, and Weak Supervision's held-out row; the three
`_OVERLAY_LEGEND` constants are deleted and the copy-contract assertions replaced by
a source check that no view defines its own prose legend. Weak Supervision's
pseudo-box prose legend stays (it defines what a pseudo box *is*). GT-only frames
(tour steps 2–3) get no legend — it would claim prediction colours that are not
drawn.

(amendment 2026-08-23, review: the legend also renders on the Failure Explorer detail and the Overview hero — every full prediction overlay in the app now carries it.)

## 4. Out of scope / already satisfied

Deep-page renames and id-hiding (§32 keeps them technical); a second stage-chain on
the result screen (the all-lit breadcrumb is the chain); provenance quieting (§35 —
already small grey captions); Overview copy (§3/§43 — reworked in 9a); a drawn
selection diagram (§11 — chips + the fold carry it); any new data surface (§41).

## 5. Tests and docs

Every changed literal updates its pinned AppTest in the same commit (the walk tests
are partitioned by step range; the blast-radius map lives in the plan). New unit
tests: `arm_story_label` fallback, `tour_strategies` order/dedup/absent-skip,
`relative_gain` zero-guard, `gain_text` clause omission, `upgrade_callout`
pedestrian-first/empty, the legend↔styles drift guard, the injected-`mined`-row
§37 branch. Docs: `docs/DEMO.md` guided-tour section (new title, step-title table,
retired key, legend, folds) and any success-walk row naming a step title. Batched
gates: one consolidated review after the code tasks, a fix round, one final
whole-branch review; a browser walk (legend badges, side-by-side fit, ~one-viewport
steps) and a re-shot `docs/img/demo-tour.png`.
