import { CompareWipe } from "../components/CompareWipe";
import { FIGURE, Figure } from "../components/Figure";
import { StatTile } from "../components/StatTile";

import { OverlayLegend } from "../../components/OverlayLegend";
import type { Verdict as VerdictData } from "../../data/types";

/**
 * Step 6 — did the targeted retraining fix the KIND of failure step 1 opened on?
 *
 * Story contract §1.4 makes the evidence hierarchy the section's structure, so
 * the answer to "did you just pick a flattering image?" is the layout itself. In
 * this edition the two tiers are the report's own sub-heads — small-caps labels
 * over a hairline, the loudest structural thing in the section:
 *
 *   TIER ONE — "One example". Labelled as one example, with the caveat printed
 *   next to the label rather than under the picture: one hand-approved frame,
 *   illustrative, not the metric. Then the two callout tiles that actually
 *   changed hands, and the frame drawn twice under a wipe (Figure 6).
 *
 *   TIER TWO — "The aggregate result". The arm-level numbers on the held-out
 *   split, set at the page's stat weight in the serif, with the per-box claims
 *   folded away underneath as the technical detail they are.
 *
 * The one emphasis in tier one is the After tile; the one emphasis in tier two is
 * the relative gain inside the sentence. Everything else stays quiet so that the
 * two tier labels carry the argument.
 */
export function Verdict({ data }: { data: VerdictData }): JSX.Element {
  const { callout } = data;

  return (
    <>
      <div className="tier">
        <p className="tier-label">{data.example_label}</p>
        <p className="tier-caveat muted">{data.example_caveat}</p>
      </div>

      {callout ? (
        <div className="stat-row">
          {/* The claim strings are the table's own, named exactly as the tour
              names them: "<category>: <claim>" under the before/after label. */}
          <StatTile
            label={data.callout_labels.before}
            value={`${callout.category}: ${callout.before}`}
          />
          <StatTile
            label={data.callout_labels.after}
            value={`${callout.category}: ${callout.after}`}
            accent
          />
        </div>
      ) : null}

      <Figure
        number={FIGURE.compareWipe}
        wide
        caption={data.held_out_caption ?? data.images.arm.alt}
      >
        <CompareWipe
          before={data.images.baseline}
          after={data.images.arm}
          beforeLabel={data.models.baseline}
          afterLabel={data.models.arm}
        />

        <OverlayLegend items={data.legend} />
      </Figure>

      {data.hero_honesty_line ? (
        <p className="muted measure">{data.hero_honesty_line}</p>
      ) : null}

      <div className="tier">
        <p className="tier-label">{data.aggregate_label}</p>
      </div>

      {data.night_ped_sentence ? (
        <p className="aggregate-line">
          <RelativeGain sentence={data.night_ped_sentence} />
        </p>
      ) : null}

      <p className="key-line">{data.night_map_sentence}</p>

      {data.fixed_boxes.length > 0 ? (
        <details className="fold">
          <summary>{data.per_box_fold_label}</summary>
          <div className="fold-body">
            <div className="overflow-x">
              <table className="box-table mono tnum">
                <thead>
                  <tr>
                    <th scope="col">Category</th>
                    <th scope="col">Distance to ego (m)</th>
                    <th scope="col">
                      {data.models.baseline.label} <span className="muted">claim</span>
                    </th>
                    <th scope="col">
                      {data.models.arm.label} <span className="muted">claim</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {data.fixed_boxes.map((box) => (
                    <tr key={box.annotation_token}>
                      <td>{box.category_group}</td>
                      <td>
                        {box.distance_to_ego_m === null ? "not recorded" : box.distance_to_ego_m}
                      </td>
                      <td>{box.baseline_claim}</td>
                      <td>{box.arm_claim}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </details>
      ) : null}
    </>
  );
}

/**
 * The aggregate sentence, with its relative gain picked out in accent.
 *
 * The tour hands Streamlit `**bold**` markers and the exporter strips them, so
 * the report styles its own emphasis — but the SENTENCE is the bundle's,
 * character for character. The split is anchored on the word "relative" and the
 * clause boundary before it (a comma or an opening bracket), never on a figure;
 * the three pieces concatenate back to exactly what arrived, and a sentence this
 * rule does not fit is rendered whole rather than re-punctuated to fit it.
 */
function RelativeGain({ sentence }: { sentence: string }): JSX.Element {
  const word = "relative";
  const at = sentence.lastIndexOf(word);
  const boundary = Math.max(sentence.lastIndexOf(",", at), sentence.lastIndexOf("(", at));
  if (at < 0 || boundary < 0) return <>{sentence}</>;

  let start = boundary + 1;
  while (start < at && sentence.charAt(start) === " ") start += 1;
  const end = at + word.length;

  return (
    <>
      {sentence.slice(0, start)}
      <span className="accent">{sentence.slice(start, end)}</span>
      {sentence.slice(end)}
    </>
  );
}
