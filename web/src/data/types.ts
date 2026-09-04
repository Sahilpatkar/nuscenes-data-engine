/**
 * The story bundle's schema — the exact shape `web/build_data.py` writes into the
 * eight `web/src/data/*.json` files.
 *
 * Every field here is DERIVED at export time from `demo_data/` (v0.8) by the
 * exporter, and `tests/test_web_export.py` re-runs it and asserts byte-equality
 * with the committed JSON. Components import the JSON and type it with these
 * interfaces; they never restate a number, a metric string or a headline of their
 * own. Keep the field names in sync with the exporter's payload keys.
 *
 * A `| null` field is one the exporter deliberately declines to write when the
 * package cannot support the claim (the tour's own honest-absence rule): render
 * nothing rather than a placeholder.
 */

/** A pre-rendered story asset under `web/public/`. */
export interface ImageRef {
  src: string;
  width: number;
  height: number;
  alt: string;
}

/** A readable arm name with the raw id it stands for (the spec's honesty rule). */
export interface ArmLabel {
  id: string;
  label: string;
}

/** Where a number came from, in the app's own provenance wording. */
export interface Provenance {
  kind: "recomputed" | "recorded" | "reproduced";
  sentence: string;
  detail: string;
}

/** One stat tile: a pre-formatted value string plus an optional delta line. */
export interface Card {
  label: string;
  value: string;
  delta?: string | undefined;
}

/** One overlay-legend chip. The RGB is the exact colour burned into the JPEG. */
export interface LegendItem {
  wording: string;
  rgb: [number, number, number] | number[];
}

// --- meta.json ---------------------------------------------------------------

export interface Attribution {
  dataset: string;
  license: string;
  citation: string;
  url: string;
}

export interface DeepLink {
  url_path: string;
  url: string;
  label: string;
}

export interface PackageInfo {
  built_at: string;
  git_sha: string;
  version: string;
}

export interface Meta {
  attribution: Attribution;
  deep_links: DeepLink[];
  exporter: string;
  package: PackageInfo;
  streamlit_base: string;
}

// --- blindspot.json (story contract §1.1) ------------------------------------

export interface Blindspot {
  baseline: ArmLabel;
  cards: Card[];
  derived_sentence: string;
  headline: string;
  miss_sentence: string;
  /** Fixed: the plain-language problem, before any number. */
  problem_sentence: string;
  provenance: Provenance[];
  /** Fixed: what the system is for. */
  purpose_sentence: string;
  takeaway: string;
}

// --- hero_frame.json ---------------------------------------------------------

export interface PedestrianClaim extends ArmLabel {
  claim: string;
}

export interface PedestrianFact {
  subject: string;
  claims: PedestrianClaim[];
}

export interface HeroFrame {
  bridge_sentence: string;
  caption: string;
  facts: PedestrianFact[];
  headline: string;
  held_out_caption: string | null;
  images: { baseline: ImageRef; arm: ImageRef };
  legend: LegendItem[];
  models: { baseline: ArmLabel; arm: ArmLabel };
  provenance: Provenance[];
  rarity_sentence: string;
  token: string;
}

// --- scenario.json -----------------------------------------------------------

export interface ScenarioEvent {
  chips: string[];
  facts: string;
  scene_name: string;
  severity_caption: string;
  token: string;
}

export interface FilmstripStep {
  label: string;
  token: string;
  can_speed_kmh: number | null;
  accel_mps2: number | null;
  is_current: boolean;
  image: ImageRef;
}

export interface Filmstrip {
  accel_title: string;
  caption: string;
  /** False on a pre-v0.8 package: the speed is then the GT ego pose, not CAN. */
  speed_is_can: boolean;
  speed_title: string;
  steps: FilmstripStep[];
}

export interface Scenario {
  event: ScenarioEvent;
  filmstrip: Filmstrip;
  headline: string;
  image: ImageRef;
  mechanism_sentence: string;
  night_sentence: string;
  parity_caption: string;
  preset: string;
  provenance: Provenance[];
  takeaway: string;
}

// --- why_frame.json (story contract §1.2, the centrepiece) -------------------

export interface SelectionFactor {
  label: string;
  value: string;
  /** A check/cross only where one reads as a fact; null for a quota or a rank. */
  flag: boolean | null;
}

export interface WhyFrame {
  architecture_strip: string;
  chips: string[];
  factors: SelectionFactor[];
  flagship_rank_sentence: string | null;
  headline: string;
  image: ImageRef;
  /** Fixed: the differentiator from "model failed → retrain". */
  lede_sentence: string;
  mechanism_sentence: string;
  provenance: Provenance[];
  /** Fixed: the plain-language chain, this section's visual device. */
  selection_chain: string[];
  /** The implementation-oriented chain, folded underneath. */
  selection_path: string;
  token: string;
  train_pool_note: string | null;
  weak_rejected_sentence: string | null;
}

// --- intervention.json (story contract §1.3) ---------------------------------

export interface StrategyBar {
  arm: string;
  strategy: string;
  delta_night: number | null;
  night_share: number | null;
  n_scenes: number | null;
  n_train_images: number | null;
  highlight: boolean;
}

export interface StrategyChart {
  bars: StrategyBar[];
  title: string;
  y_title: string;
}

export interface Intervention {
  cards: Card[];
  chart: StrategyChart;
  charted_ids_caption: string;
  /** The fairness statement's clauses; the budget clause is derived. */
  fairness_parts: string[];
  fairness_sentence: string;
  headline: string;
  night_share_line: string | null;
  provenance: Provenance[];
  similarity_caption: string | null;
  spread_sentence: string | null;
  winner_sentence: string | null;
}

// --- verdict.json (story contract §1.4) --------------------------------------

export interface UpgradeCallout {
  category: string;
  before: string;
  after: string;
}

export interface FixedBox {
  annotation_token: string;
  category_group: string;
  distance_to_ego_m: number | null;
  baseline_claim: string;
  arm_claim: string;
}

export interface Verdict {
  /** Tier two's label. */
  aggregate_label: string;
  callout: UpgradeCallout | null;
  callout_labels: { before: string; after: string };
  /** Tier one's caveat: illustrative, not the metric. */
  example_caveat: string;
  example_label: string;
  fixed_boxes: FixedBox[];
  headline: string;
  held_out_caption: string | null;
  hero_honesty_line: string | null;
  images: { baseline: ImageRef; arm: ImageRef };
  legend: LegendItem[];
  models: { baseline: ArmLabel; arm: ArmLabel };
  night_map_sentence: string;
  night_ped_sentence: string | null;
  per_box_fold_label: string;
  provenance: Provenance[];
  token: string;
}

// --- closed_loop.json (story contract §1.5) ----------------------------------

export interface ClosedLoopAnswer {
  question: string;
  sentences: string[];
}

export interface ClosedLoop {
  answers: ClosedLoopAnswer[];
  cards: Card[];
  /** Fixed: the thesis the four answers are the evidence for. */
  closing_thesis: string;
  /** Computed: "measurable improvement" only where the night delta is a gain. */
  headline: string | null;
  provenance: Provenance[];
}
