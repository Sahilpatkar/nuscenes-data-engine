import { useEffect, useState } from "react";

import { CONTENTS_ID, COVER_ID } from "./Cover";
import { reportIndex } from "./SectionFrame";
import { ThemeToggle } from "./ThemeToggle";
import { useActiveSection } from "../hooks/useActiveSection";

/**
 * The report's running head: a slim sticky bar carrying the wordmark, the
 * scroll-spy readout ("03 / 07 — Where else does this happen?"), the theme
 * toggle, and a one-pixel accent rule that fills as the document is read.
 *
 * This edition has no rail. A printed report puts its position in the running
 * head and its map on the contents page, so that is where those two jobs live:
 * the bar states where you ARE, the cover's contents block is where you GO. The
 * bar therefore carries no menu — only two links, both real anchors that work
 * before React hydrates and are reachable by keyboard:
 *
 *   · the wordmark returns to the cover;
 *   · the readout points at the section it names, or — before any section has
 *     entered the scroll-spy's band — at the contents block, which is the honest
 *     answer while you are still on the cover.
 *
 * The only copy here is the site's own chrome (its name, its edition, the word
 * "Contents"). The seven titles are the tour's, passed down from App.
 */

const WORDMARK = "nuScenes Data Engine";
const EDITION = "Report edition";
/** What the readout says while the reader is still above the first section. */
const CONTENTS_LABEL = "Contents";
/** The index slot before the first section — a held place, not a "00". */
const NO_INDEX = "—";

export interface HeaderStep {
  id: string;
  step: number;
  title: string;
}

/**
 * How far down the document we are, 0–1.
 *
 * Read from the scroll position rather than from the active section, so the rule
 * advances continuously (a section is not a step function). rAF-throttled: the
 * scroll listener only ever schedules one measurement per frame, and layout is
 * read inside that frame, never in the event.
 */
function useScrollProgress(): number {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    let frame = 0;

    const measure = (): void => {
      frame = 0;
      const scrollable = document.documentElement.scrollHeight - window.innerHeight;
      // A document shorter than the viewport has no progress to report.
      if (scrollable <= 0) {
        setProgress(0);
        return;
      }
      setProgress(Math.min(1, Math.max(0, window.scrollY / scrollable)));
    };

    const schedule = (): void => {
      if (frame === 0) frame = window.requestAnimationFrame(measure);
    };

    measure();
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      if (frame !== 0) window.cancelAnimationFrame(frame);
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
    };
  }, []);

  return progress;
}

export function Header({ steps }: { steps: readonly HeaderStep[] }): JSX.Element {
  const active = useActiveSection(steps.map((entry) => entry.id));
  const current = steps.find((entry) => entry.id === active) ?? null;
  const progress = useScrollProgress();

  return (
    <header className="report-header">
      <div className="page report-header-inner">
        <a className="wordmark" href={`#${COVER_ID}`}>
          {WORDMARK}
          <span className="wordmark-sep" aria-hidden="true">
            {" · "}
          </span>
          <span className="wordmark-edition">{EDITION}</span>
        </a>

        <div className="header-right">
          <a className="header-readout" href={`#${current ? current.id : CONTENTS_ID}`}>
            <span className="mono header-readout-index">
              {current ? reportIndex(current.step) : NO_INDEX} / {reportIndex(steps.length)}
            </span>
            <span className="header-readout-dash" aria-hidden="true">
              {" — "}
            </span>
            <span className="header-readout-title">
              {current ? current.title : CONTENTS_LABEL}
            </span>
          </a>
          <ThemeToggle />
        </div>
      </div>

      {/* Decoration: the same fact the readout states in words. */}
      <div className="header-progress" aria-hidden="true">
        <div className="header-progress-fill" style={{ width: `${progress * 100}%` }} />
      </div>
    </header>
  );
}
