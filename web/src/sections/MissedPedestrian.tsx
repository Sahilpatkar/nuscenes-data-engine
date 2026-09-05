import { useId, useRef, useState, type KeyboardEvent } from "react";

import { OverlayLegend } from "../components/OverlayLegend";
import { Ticks } from "../components/Ticks";
import type { HeroFrame } from "../data/types";

/**
 * Step 2 — the failure as a picture: one held-out night frame, drawn twice.
 *
 * The toggle is the whole point of the step, so it is a real tab pair (two
 * `role="tab"` buttons, roving tabindex, arrow keys) over two tab panels — not a
 * library, and not a link that swaps a `src`. BOTH images stay mounted; the
 * inactive panel is `hidden`, which keeps its image fetched but out of the
 * accessibility tree, so switching is instant with no flash and no reflow.
 *
 * Order mirrors the live tour's own hero step: the pair, the legend, the two
 * captions, the per-model claims, the rarity sentence, and the bridge into the
 * mining half of the story.
 *
 * The arm is shown first. It is the state the story is heading towards ("the
 * night-targeted retrain recovers the pedestrian"), and it is the frame the
 * landing already showed — arriving on the baseline would silently change the
 * picture under the reader.
 */
const ORDER = ["baseline", "arm"] as const;
type ModelKey = (typeof ORDER)[number];

export function MissedPedestrian({ data }: { data: HeroFrame }): JSX.Element {
  const [current, setCurrent] = useState<ModelKey>("arm");
  const tablist = useRef<HTMLDivElement>(null);
  const id = useId();

  /** Move the selection AND the focus together — the tab pattern's own rule. */
  function select(index: number): void {
    const next = ORDER[((index % ORDER.length) + ORDER.length) % ORDER.length];
    if (next === undefined) return;
    setCurrent(next);
    tablist.current?.querySelectorAll("button")[ORDER.indexOf(next)]?.focus();
  }

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>): void {
    const index = ORDER.indexOf(current);
    if (event.key === "ArrowRight" || event.key === "ArrowDown") select(index + 1);
    else if (event.key === "ArrowLeft" || event.key === "ArrowUp") select(index - 1);
    else if (event.key === "Home") select(0);
    else if (event.key === "End") select(ORDER.length - 1);
    else return;
    event.preventDefault();
  }

  return (
    <>
      <figure className="overlay-figure">
        <div
          className="overlay-tabs"
          role="tablist"
          aria-label="Overlay model"
          ref={tablist}
          onKeyDown={onKeyDown}
        >
          {ORDER.map((key) => {
            const model = data.models[key];
            const selected = key === current;
            return (
              <button
                key={key}
                type="button"
                role="tab"
                id={`${id}-tab-${key}`}
                className={selected ? "overlay-tab is-selected" : "overlay-tab"}
                aria-selected={selected}
                aria-controls={`${id}-panel-${key}`}
                tabIndex={selected ? 0 : -1}
                onClick={() => setCurrent(key)}
              >
                <span className="overlay-tab-label">{model.label}</span>
                <span className="overlay-tab-id mono">{model.id}</span>
              </button>
            );
          })}
        </div>

        {ORDER.map((key) => {
          const image = data.images[key];
          const selected = key === current;
          return (
            <div
              key={key}
              role="tabpanel"
              id={`${id}-panel-${key}`}
              aria-labelledby={`${id}-tab-${key}`}
              className="overlay-panel"
              tabIndex={0}
              hidden={!selected}
            >
              <img
                src={image.src}
                width={image.width}
                height={image.height}
                alt={image.alt}
                decoding="async"
              />
            </div>
          );
        })}

        <OverlayLegend items={data.legend} />

        <figcaption className="overlay-caption">
          <span>{data.caption}</span>
          {data.held_out_caption ? (
            <span className="mono muted">{data.held_out_caption}</span>
          ) : null}
        </figcaption>
      </figure>

      <ul className="claims">
        {data.facts.map((fact) => (
          <li key={fact.subject}>
            <strong>{fact.subject}</strong>
            <span className="claims-detail">
              {" — "}
              {fact.claims.map((claim) => `${claim.label}: ${claim.claim}`).join(" · ")}
            </span>
          </li>
        ))}
      </ul>

      <p className="measure">
        <Ticks text={data.rarity_sentence} />
      </p>

      <p className="beat">{data.bridge_sentence}</p>
    </>
  );
}
