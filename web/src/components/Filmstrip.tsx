import { useRef, type KeyboardEvent } from "react";

import type { FilmstripStep } from "../data/types";

/**
 * The event's t−2 … t+2 keyframes — the Scenario page's filmstrip, and the left
 * half of this step's synced pair: the same `active` step the CAN curves mark.
 *
 * Every thumb is a real `<button>`: hover, tap and focus all set the active step,
 * and the arrow keys walk it (wrapping, with Home/End) so the scrub is reachable
 * without a pointer. There is no roving tabindex — with five buttons, tabbing
 * through them is cheaper for a keyboard reader than learning a widget.
 *
 * `is_current` is the event's OWN frame, marked permanently; `active` is what the
 * reader is pointing at. Two states, two treatments — the ring never moves.
 */
export interface FilmstripProps {
  steps: readonly FilmstripStep[];
  active: number;
  onActivate: (index: number) => void;
}

export function Filmstrip({ steps, active, onActivate }: FilmstripProps): JSX.Element {
  const strip = useRef<HTMLDivElement>(null);

  function move(index: number): void {
    const count = steps.length;
    const next = ((index % count) + count) % count;
    onActivate(next);
    strip.current?.querySelectorAll("button")[next]?.focus();
  }

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>): void {
    if (steps.length === 0) return;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") move(active + 1);
    else if (event.key === "ArrowLeft" || event.key === "ArrowUp") move(active - 1);
    else if (event.key === "Home") move(0);
    else if (event.key === "End") move(steps.length - 1);
    else return;
    event.preventDefault();
  }

  return (
    <div className="filmstrip-scroll overflow-x">
      <div
        className="filmstrip"
        role="group"
        aria-label="Event keyframes"
        ref={strip}
        onKeyDown={onKeyDown}
      >
        {steps.map((step, index) => {
          const classes = ["film-thumb"];
          if (step.is_current) classes.push("is-current");
          if (index === active) classes.push("is-active");
          return (
            <button
              key={step.token}
              type="button"
              className={classes.join(" ")}
              aria-pressed={index === active}
              onClick={() => onActivate(index)}
              onMouseEnter={() => onActivate(index)}
              onFocus={() => onActivate(index)}
            >
              <img
                src={step.image.src}
                width={step.image.width}
                height={step.image.height}
                alt={step.image.alt}
                decoding="async"
              />
              <span className="film-label mono">{step.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
