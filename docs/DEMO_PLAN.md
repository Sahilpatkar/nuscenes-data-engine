# nuScenes Perception Data Engine — Demo Plan

## Objective

Build a polished, public-facing Streamlit demo that makes the project understandable in a few minutes.

The demo should **not** try to expose every backend component directly. Instead, it should tell one clear story:

> A perception model is trained, its weaknesses are identified, the system finds the most useful failure cases using semantic, spatial, temporal, and ego-dynamics context, targeted retraining is performed, and the results are evaluated.

The demo should make the project feel like a small internal autonomous-driving perception data platform rather than a static dashboard.

---

# Core Demo Story

The recommended narrative is:

**Data → Detection → Failure Analysis → Scenario Search → Active Learning → Retraining → Weak Supervision → Improvement**

The user should be able to understand the project through a guided sequence rather than having to inspect raw infrastructure.

---

# Recommended Navigation

Keep the public demo compact.

Suggested top-level pages:

1. **Overview**
2. **Failure Explorer**
3. **Active Learning**
4. **Scenario Search**
5. **Weak Supervision**

Optional additions can live within these pages rather than creating more top-level tabs.

---

# 1. Overview Page

## Purpose

Immediately explain what the project is and why it matters.

Suggested project description:

> **nuScenes Perception Data Engine**  
> A system for training, evaluating, diagnosing, and improving autonomous-driving perception models using active learning, semantic search, knowledge graphs, CAN-bus context, and weak supervision.

## Headline Scale Metrics

Show a few high-value numbers prominently:

- **204,894 images**
- **~1.1M 2D boxes**
- **1.17M 3D object observations**
- **34,149 keyframe-aligned CAN-bus rows**

## Headline Experimental Results

Highlight the strongest findings:

- **Best targeted night gain:** `+0.0101 night mAP50-95`
- **CAN / ego-speed validation:** `r = 0.999`
- **Weak supervision retained:** `18% of the ground-truth gain`
- **Flagship Graph + CAN query:** `30 matching events in both SQL and Cypher`

## Visual

Show one representative nuScenes frame with YOLO detections.

The first page should communicate within seconds:

- this is an autonomous-driving perception project,
- it goes beyond simple model training,
- the system performs controlled experiments and context-aware data mining.

---

# 2. Failure Explorer

## Purpose

Make model weaknesses visually tangible.

## Filters

Allow the user to filter failure cases by:

- day / night
- rain / clear
- object class
- object size
- distance from ego vehicle
- false negative
- false positive
- low-confidence detection

## Frame Display

When a frame is selected, show:

- camera image,
- YOLO predictions,
- ground-truth boxes,
- confidence scores,
- object distance,
- environmental condition,
- relevant metadata.

## Ground Truth vs Prediction Toggle

Add a toggle:

- **Ground Truth**
- **Prediction**
- **Overlay**

Or show both side by side.

Use clear visual distinctions for:

- correct detections,
- false negatives,
- false positives,
- low-confidence detections.

This should let someone immediately see *why* a frame is considered difficult.

---

# 3. Synchronized Event Viewer

This should be one of the most visually engaging parts of the demo.

## Purpose

Show the complete driving context around one perception event.

For a selected event, display:

### Camera frame

The current keyframe with detections and GT.

### Ego dynamics

- speed
- longitudinal acceleration
- braking intensity
- timestamp

### Scene context

- nearby pedestrians
- nearby cars
- cyclists
- distance-to-ego
- lighting
- weather

### Model context

- prediction confidence
- GT category
- detected / missed
- baseline vs retrained model result

## Timeline Scrubber

Add a short temporal slider around the event.

Example:

```text
t-2    t-1    [t]    t+1    t+2
```

As the user scrubs the timeline, update:

- camera frame,
- speed,
- acceleration,
- object positions,
- detector output,
- graph context.

This introduces temporal understanding without requiring the public demo to run a full temporal detector.

---

# 4. Scenario Search — Graph + CAN

## Purpose

Demonstrate why the Neo4j graph and CAN-bus integration are useful.

The graph should **not** be presented as a database viewer.

Instead, use it as a visual explanation of:

> **Why did this query return this driving event?**

---

## Example Query

> **Hard braking near pedestrians**

The system interprets the query as something like:

```text
Object type        = pedestrian
distance_to_ego    < threshold
longitudinal_accel < hard-braking threshold
```

Then the graph traversal can be shown visually.

Example conceptual graph:

```text
                ┌──────────────┐
                │ Scene        │
                └──────┬───────┘
                       │ CONTAINS
                ┌──────▼───────┐
                │ Sample       │
                └───┬──────┬───┘
                    │      │
          HAS_POSE  │      │ HAS_DYNAMICS
                    │      │
             ┌──────▼─┐  ┌─▼────────────┐
             │EgoPose │  │ CANReading   │
             │x,y,yaw │  │ speed        │
             └────┬───┘  │ acceleration │
                  │      └──────────────┘
             NEAR │
                  │
            ┌─────▼──────────┐
            │ Pedestrian     │
            │ distance=...   │
            └────────────────┘
```

---

# Interactive Graph Visualization

## Query-Focused Subgraph

Do **not** render the entire graph.

The production graph contains more than a million nodes, so the demo should render only the subgraph relevant to the selected query or event.

## Desired Behavior

When the user runs a query:

1. show the relevant `Sample`,
2. highlight the `CANReading`,
3. highlight the braking condition,
4. traverse to `EgoPose`,
5. highlight nearby `ObjectObservation` nodes,
6. fade irrelevant nodes,
7. keep matching paths visible.

Example traversal:

```text
Query submitted

Scene
  ↓
Sample
  ↓
CANReading
  ↓
EgoPose
  ↓
ObjectObservation
```

The animation should be subtle and explanatory, not decorative.

## Node Interactions

Users should be able to click nodes to inspect metadata.

For example:

### Sample

- timestamp
- scene
- lighting
- weather

### CANReading

- speed
- acceleration
- braking state

### EgoPose

- location
- orientation

### ObjectObservation

- object class
- distance to ego
- 3D position
- GT information
- model prediction

## Implementation Options

Potential libraries:

- **Cytoscape.js** — preferred for polished interaction and path highlighting
- `streamlit-agraph`
- PyVis

Cytoscape.js is the strongest option if custom Streamlit components are acceptable.

---

# 5. Scenario Builder

In addition to natural-language-style preset queries, provide a structured builder.

Example:

```text
Lighting:       Night
Object:         Pedestrian
Distance:       < 10 m
Acceleration:   < -4 m/s²
Model result:   False negative
```

Each filter update should dynamically change:

- number of matching events,
- displayed graph,
- event list,
- selected representative frames.

This demonstrates that the graph/data model supports flexible driving-scenario analysis.

---

# Preset Scenario Queries

Provide clickable presets so users do not need to understand the schema.

Examples:

- **Hard braking near pedestrians**
- **Night scenes with nearby pedestrians**
- **Low-confidence detections during braking**
- **False-negative pedestrians at night**
- **Nearby cyclists during high-speed driving**
- **Rain scenes with missed vulnerable road users**

The demo should always have good example results ready.

---

# 6. Matching Event Results

After a scenario query, show a ranked list or gallery of matching events.

Example:

```text
Event 1
Braking: -5.2 m/s²
Pedestrian distance: 3.8 m
Speed: 12.4 m/s
Lighting: Night
YOLO confidence: 0.31
Result: False negative
```

The user should be able to click an event and open the synchronized event viewer.

For the flagship query, show:

> **30 matching events**

Also display that the same logic returned:

- `30` in DuckDB SQL
- `30` in Neo4j Cypher

This is useful as a visible consistency check.

---

# 7. Semantic Search Gallery

## Purpose

Show the SigLIP semantic-search layer visually.

Provide search examples such as:

> crowded nighttime intersection with pedestrians

or:

> rainy road with nearby vehicles

Return a gallery of visually similar nuScenes frames.

For each result, show:

- thumbnail,
- similarity score,
- scene metadata,
- condition,
- key object categories.

Clicking a search result should open:

- the full event/frame viewer,
- graph context,
- model predictions,
- GT,
- failure metadata.

This connects semantic search to the rest of the platform instead of making it feel like an isolated feature.

---

# 8. Active Learning Page

## Purpose

Explain how the system selects useful training data.

Show the controlled acquisition experiments across rounds.

Include the major arms, especially:

- absolute failure score
- smoothed rate
- stratified acquisition
- rate + stratification
- graph-aware variants
- `graph_rate_night`

## Experimental Result

Highlight:

> **`graph_rate_night` produced the project's best targeted night improvement: +0.0101 night mAP50-95.**

---

# Why Was This Frame Selected?

For selected active-learning frames, provide an explanation panel.

Example:

```text
Selected by: graph_rate_night

Contributing factors:
✓ Night condition
✓ High failure rate
✓ Low-confidence / missed objects
✓ Graph-community diversity
✓ Relevant spatial context
```

If the acquisition score is available, show the score decomposition.

This makes active learning understandable to non-specialists.

---

# Before / After Model Comparison

This should connect failure mining to model improvement.

For the same frame, show:

### Baseline Model

```text
Pedestrian confidence: 0.22
Prediction: missed
```

### Retrained Model

```text
Pedestrian confidence: 0.61
Prediction: detected
```

Where possible, allow a toggle:

- baseline
- active-learning retrained model

The visual story becomes:

```text
Failure
   ↓
Selected for acquisition
   ↓
Added to training data
   ↓
Retrained model
   ↓
Improved prediction
```

This is one of the strongest ways to communicate the project's feedback loop.

---

# 9. Experiment Story View

Instead of showing only experiment tables, provide a visual causal narrative.

Example:

```text
Problem
Night performance is weak

↓

Hypothesis
Target difficult night scenes

↓

Acquisition
graph_rate_night

↓

Training
Controlled retraining

↓

Evaluation
Same held-out validation set

↓

Result
+0.0101 night mAP50-95
```

Each major experiment can use this structure.

The goal is to explain *why the experiment was run*, not just its metrics.

---

# 10. Weak Supervision Page

## Purpose

Show the VLM auto-labeling experiment as an empirical study rather than merely an LLM feature.

## Headline Result

> **Weak supervision retained only 18% of the gain achieved with ground-truth labels.**

## Loss Decomposition

Show:

- approximately **50%** of lost gain from dropped frames,
- approximately **32%** from label quality.

## Crowded-Frame Bias

Highlight the verifier bias:

- rejected frames: **7.61 GT boxes/frame**
- accepted frames: **3.87 GT boxes/frame**

This demonstrates that the weak-label verifier systematically rejects crowded scenes.

## Visual Comparison

For representative examples, display:

- original frame,
- GT labels,
- VLM-generated labels,
- verifier decision,
- accepted / rejected state,
- downstream model result.

This makes the weak-supervision limitation easy to understand.

---

# 11. Graph + CAN Validation

Show the two key consistency checks.

## Speed Alignment

CAN speed was cross-checked against ego-motion-derived speed.

Result:

> **Pearson correlation: r = 0.999**

This supports correct CAN/sample timestamp alignment.

## SQL vs Cypher

For the flagship query:

> **Hard braking near pedestrians**

Both data systems returned:

> **30 matching events**

This shows consistency between the relational analytics layer and graph representation.

---

# 12. Demo Data Architecture

The public app should **not** depend on the full production-scale stack being online.

The heavy workloads should remain offline.

Suggested package:

```text
demo_data/
├── overview_metrics.json
├── active_learning_results.parquet
├── weak_supervision_results.parquet
├── scenario_events.parquet
├── graph_subgraphs/
│   ├── hard_braking_pedestrians.json
│   ├── night_pedestrians.json
│   └── low_conf_braking.json
├── semantic_search_results.parquet
├── model_comparison.parquet
└── sample_frames/
    ├── failure_001.jpg
    ├── braking_ped_001.jpg
    ├── night_001.jpg
    └── ...
```

The Streamlit app can load these artifacts directly.

This makes the demo:

- deterministic,
- fast,
- cheap,
- resilient,
- easy to deploy.

---

# Public Demo Deployment Strategy

Use **Streamlit Community Cloud** as the primary public demo host.

The public app should serve curated artifacts generated by the real pipeline.

Heavy components should remain outside the public hosting environment:

- YOLO training
- full nuScenes dataset
- full Neo4j graph
- SigLIP indexing jobs
- VLM labeling jobs
- large-scale evaluation runs

The demo can clearly state:

> Results shown here were generated by the full offline pipeline. The public application serves curated experiment outputs for reproducibility and demonstration.

---

# Recommended UX Flow

A strong recruiter/interviewer flow could be:

## Step 1 — Overview

See scale, project objective, and headline metrics.

## Step 2 — View a failure

Open a nighttime pedestrian false negative.

## Step 3 — Explore the event

Inspect GT, prediction, speed, acceleration, nearby objects, and temporal context.

## Step 4 — Ask a scenario query

Select:

> Hard braking near pedestrians

## Step 5 — Watch the graph traversal

See how `Sample`, `CANReading`, `EgoPose`, and `ObjectObservation` connect.

## Step 6 — Inspect a matching event

See the frame and associated driving context.

## Step 7 — See why it was selected

Open the active-learning explanation.

## Step 8 — Compare models

View baseline vs retrained predictions.

## Step 9 — Review the experiment result

See the night-performance improvement.

## Step 10 — Explore weak supervision

Understand why VLM-generated labels captured only part of the GT benefit.

---

# Visual / Interactive Priority List

Recommended implementation order:

1. **Interactive graph query visualization**
2. **Synchronized event viewer**
3. **GT vs prediction toggle**
4. **Baseline vs retrained comparison**
5. **Scenario builder**
6. **Semantic-search image gallery**
7. **Active-learning selection explanation**
8. **Experiment story view**
9. **Weak-supervision comparison UI**
10. **Timeline scrubbing around selected events**

---

# Design Principles

## 1. Do not expose infrastructure for its own sake

Avoid raw:

- Neo4j dashboards
- MLflow tables
- W&B dashboards
- large SQL tables
- full graph visualizations

Convert them into user-facing explanations.

## 2. Every visual should answer a question

Examples:

- **Why did the model fail?**
- **Why was this frame selected?**
- **Why did this scenario match?**
- **Did retraining improve the result?**
- **Why did weak supervision lose performance?**

## 3. Prefer representative curated events

The public demo does not need every nuScenes frame.

Use a high-quality curated subset that demonstrates:

- success cases,
- failure cases,
- night scenes,
- crowded frames,
- hard-braking events,
- graph-aware active-learning examples.

## 4. Preserve experiment credibility

Clearly distinguish:

- real experiment outputs,
- precomputed public demo results,
- interactive filters over curated data.

Do not present mock values as live pipeline execution.

## 5. Keep the demo understandable in 2–3 minutes

A recruiter should leave with this understanding:

> This is not just YOLO fine-tuning. The system identifies model weaknesses, mines contextually useful driving scenarios, runs controlled retraining experiments, evaluates whether interventions work, and analyzes the tradeoff between ground-truth and weak supervision.

---

# Proposed Architecture

```text
                         ┌──────────────────────────┐
                         │      Streamlit Demo      │
                         └────────────┬─────────────┘
                                      │
                ┌─────────────────────┼─────────────────────┐
                │                     │                     │
                ▼                     ▼                     ▼
        Failure Explorer       Scenario Search       Active Learning
                │                     │                     │
                ▼                     ▼                     ▼
        GT vs Prediction      Graph Visualization    Selection Reason
                │                     │                     │
                └──────────────┬──────┴──────────────┬─────┘
                               │                     │
                               ▼                     ▼
                    Synchronized Event Viewer   Model Comparison
                               │                     │
                               └──────────┬──────────┘
                                          ▼
                                 Experiment Results

                     Public demo reads curated artifacts
                                          │
                                          ▼
                                ┌────────────────────┐
                                │  Offline Pipeline  │
                                ├────────────────────┤
                                │ nuScenes ingestion │
                                │ YOLO training      │
                                │ MLflow / W&B       │
                                │ Neo4j graph        │
                                │ DuckDB             │
                                │ CAN ingestion      │
                                │ SigLIP search      │
                                │ VLM supervision    │
                                │ Active learning    │
                                └────────────────────┘
```

---

# Demo Success Criteria

The demo is successful if someone can answer these questions after using it:

1. What problem does the project solve?
2. Where does the baseline perception model fail?
3. How does the system find difficult data?
4. Why is the graph useful?
5. What does CAN-bus data add?
6. How does active learning choose frames?
7. Did targeted retraining improve performance?
8. How well did VLM-generated supervision work?
9. Why did weak supervision underperform GT?
10. How does the project form a closed model-improvement loop?

---

# Recommended Next Step

Before Terraform or additional large backend features, build the public demo around the existing real results.

Suggested immediate sequence:

```text
Merge pending branches
        ↓
Prepare curated demo artifacts
        ↓
Build Streamlit demo shell
        ↓
Failure Explorer
        ↓
Graph + CAN Scenario Search
        ↓
Interactive Graph
        ↓
Synchronized Event Viewer
        ↓
Active Learning / Model Comparison
        ↓
Weak Supervision
        ↓
Deploy publicly
        ↓
README screenshots + live-demo link
        ↓
Terraform / optional extensions
```

The demo should be treated as the **presentation layer for the engineering and experimental work already completed**, not as a replacement for the underlying pipeline.
