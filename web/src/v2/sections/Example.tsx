import { useId, useRef, useState, type KeyboardEvent } from "react";

import { assetUrl } from "../../assetUrl";
import { FactList } from "../components/FactList";
import { FIGURE, Figure } from "../components/Figure";

import { OverlayLegend } from "../../components/OverlayLegend";
import type { HeroFrame, Report } from "../../data/types";

/**
 * 02. Example failure: the one frame, with a baseline / targeted-retrain toggle
 * (the original edition's tab mechanics, unchanged: arrow keys, Home/End, focus
 * follows selection), then what each model did as labelled facts and the caveat
 * in small type. No narration of what the image already shows.
 */
const ORDER = ["baseline", "arm"] as const;
type ModelKey = (typeof ORDER)[number];

export function Example({ data, report }: { data: HeroFrame; report: Report }): JSX.Element {
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
      <Figure
        number={FIGURE.heroOverlay}
        wide
        className="overlay-figure"
        caption={report.example.caption}
      >
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
                src={assetUrl(image.src)}
                width={image.width}
                height={image.height}
                alt={image.alt}
                decoding="async"
              />
            </div>
          );
        })}

        <OverlayLegend items={data.legend} />
      </Figure>

      <FactList facts={report.example.facts} />
      <p className="muted measure small">{report.example.caveat}</p>
    </>
  );
}
