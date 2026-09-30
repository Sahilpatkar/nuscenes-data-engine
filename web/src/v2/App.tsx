import type { ReactNode } from "react";

import { Colophon } from "./components/Colophon";
import { Cover, type CoverEntry } from "./components/Cover";
import { Header } from "./components/Header";
import { SectionFrame } from "./components/SectionFrame";
import { Baseline } from "./sections/Baseline";
import { Example } from "./sections/Example";
import { Intervention as InterventionSection } from "./sections/Intervention";
import { Mining } from "./sections/Mining";
import { Results } from "./sections/Results";
import { Selection } from "./sections/Selection";

import heroFrameJson from "../data/hero_frame.json";
import interventionJson from "../data/intervention.json";
import reportJson from "../data/report.json";
import scenarioJson from "../data/scenario.json";
import verdictJson from "../data/verdict.json";
import whyFrameJson from "../data/why_frame.json";

import type { HeroFrame, Intervention, Report, Scenario, Verdict, WhyFrame } from "../data/types";

/* The bundle. `report.json` is this edition's own copy (every sentence and label
   the report states, each figure derived at export); the other files supply the
   images, the filmstrip, the strategy chart and the selection facts it shares
   with the original edition. Each import is asserted into its interface once. */
const report = reportJson as Report;
const heroFrame = heroFrameJson as HeroFrame;
const scenario = scenarioJson as Scenario;
const whyFrame = whyFrameJson as WhyFrame;
const intervention = interventionJson as Intervention;
const verdict = verdictJson as Verdict;

interface ReportSection {
  /** The anchor: the contents target and the section's `id`. */
  id: string;
  step: number;
  /** What the section does, stated technically. */
  title: string;
  body: ReactNode;
}

/* The six sections of the case study, in reading order: what fails, what it
   looks like, how related scenarios are found, how training data is chosen,
   what changed, and whether it worked. The report's own titles (the guided tour
   keeps its conversational seven); pinned by tests/test_web_v2.py. */
const SECTIONS: readonly ReportSection[] = [
  {
    id: "baseline",
    step: 1,
    title: "Baseline failure analysis",
    body: <Baseline data={report} />,
  },
  {
    id: "example",
    step: 2,
    title: "Example failure",
    body: <Example data={heroFrame} report={report} />,
  },
  {
    id: "mining",
    step: 3,
    title: "Context-aware failure mining",
    body: <Mining data={scenario} report={report} />,
  },
  {
    id: "selection",
    step: 4,
    title: "Targeted training-data selection",
    body: <Selection data={whyFrame} report={report} />,
  },
  {
    id: "intervention",
    step: 5,
    title: "Training intervention",
    body: <InterventionSection report={report} />,
  },
  {
    id: "results",
    step: 6,
    title: "Results",
    body: <Results report={report} intervention={intervention} verdict={verdict} />,
  },
];

/** The contents page and the running head index the same six things. */
const CONTENTS: readonly CoverEntry[] = SECTIONS.map(({ id, step, title }) => ({
  id,
  step,
  title,
}));

export default function App(): JSX.Element {
  return (
    <>
      <Header steps={CONTENTS} />

      <main className="report">
        <Cover entries={CONTENTS} />

        {SECTIONS.map((section) => (
          <SectionFrame key={section.id} id={section.id} step={section.step} title={section.title}>
            {section.body}
          </SectionFrame>
        ))}
      </main>

      <Colophon />
    </>
  );
}
