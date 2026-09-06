import type { ReactNode } from "react";

import { Colophon } from "./components/Colophon";
import { Cover, type CoverEntry } from "./components/Cover";
import { Header } from "./components/Header";
import { SectionFrame } from "./components/SectionFrame";
/* The section components share their names with the bundle interfaces below
   (both are named for the step they belong to), so each is aliased once here. */
import { Blindspot as BlindspotSection } from "./sections/Blindspot";
import { ClosedLoop as ClosedLoopSection } from "./sections/ClosedLoop";
import { Intervention as InterventionSection } from "./sections/Intervention";
import { MissedPedestrian as MissedPedestrianSection } from "./sections/MissedPedestrian";
import { Scenario as ScenarioSection } from "./sections/Scenario";
import { Verdict as VerdictSection } from "./sections/Verdict";
import { WhyFrame as WhyFrameSection } from "./sections/WhyFrame";

import blindspotJson from "../data/blindspot.json";
import closedLoopJson from "../data/closed_loop.json";
import heroFrameJson from "../data/hero_frame.json";
import interventionJson from "../data/intervention.json";
import scenarioJson from "../data/scenario.json";
import verdictJson from "../data/verdict.json";
import whyFrameJson from "../data/why_frame.json";

import type {
  Blindspot,
  ClosedLoop,
  HeroFrame,
  Intervention,
  Provenance,
  Scenario,
  Verdict,
  WhyFrame,
} from "../data/types";

/* The bundle — the SAME eight files the root edition reads, so the two editions
   cannot disagree on a number. The JSON's inferred types are widened (a `kind`
   field is `string`, not the union), so each import is asserted into its exported
   interface once, here, and never re-typed downstream. `meta.json` is read by the
   cover and the colophon, which are the only places its facts appear. */
const blindspot = blindspotJson as Blindspot;
const heroFrame = heroFrameJson as HeroFrame;
const scenario = scenarioJson as Scenario;
const whyFrame = whyFrameJson as WhyFrame;
const intervention = interventionJson as Intervention;
const verdict = verdictJson as Verdict;
const closedLoop = closedLoopJson as ClosedLoop;

interface StorySection {
  /** The anchor — also the contents target and the section's `id`. */
  id: string;
  step: number;
  /** The step's fixed title, mirroring the live tour's own step titles. */
  title: string;
  /** The closed-improvement-loop stage this step lights. */
  stage: string;
  /** The deck's act, so both tellings are structured the same way. */
  act: string;
  /** Where this section's numbers came from — set as its footnotes. */
  provenance: Provenance[];
  /** The section's own content: its ledes, figures, charts and evidence. */
  body: ReactNode;
}

/* The seven steps, in story order. Titles and stages are the live tour's own
   (app/demo/views/tour.py `_STEPS`), typed out here a second time exactly as the
   root edition types them: three front-ends, one set of names, pinned by test.
   The ids are the root edition's too, so an anchor written for one edition means
   the same section in the other. */
const SECTIONS: readonly StorySection[] = [
  {
    id: "blindspot",
    step: 1,
    title: "We found a blind spot",
    stage: "Diagnose",
    act: "Problem",
    provenance: blindspot.provenance,
    body: <BlindspotSection data={blindspot} />,
  },
  {
    id: "hero-frame",
    step: 2,
    title: "What the failure looks like",
    stage: "Diagnose",
    act: "Problem",
    provenance: heroFrame.provenance,
    body: <MissedPedestrianSection data={heroFrame} />,
  },
  {
    id: "scenario",
    step: 3,
    title: "Where else does this happen?",
    stage: "Mine",
    act: "Approach",
    provenance: scenario.provenance,
    body: <ScenarioSection data={scenario} />,
  },
  {
    id: "why-frame",
    step: 4,
    title: "What data should we add?",
    stage: "Mine",
    act: "Approach",
    provenance: whyFrame.provenance,
    body: <WhyFrameSection data={whyFrame} />,
  },
  {
    id: "intervention",
    step: 5,
    title: "We changed the training data",
    stage: "Train",
    act: "Approach",
    provenance: intervention.provenance,
    body: <InterventionSection data={intervention} />,
  },
  {
    id: "verdict",
    step: 6,
    title: "Did it fix the failure?",
    stage: "Evaluate",
    act: "Results",
    provenance: verdict.provenance,
    body: <VerdictSection data={verdict} />,
  },
  {
    id: "closed-loop",
    step: 7,
    title: "Closed the loop",
    // The result step lights every stage: by here the story has walked the whole
    // loop rather than one stage of it.
    stage: "The whole loop",
    act: "Results",
    provenance: closedLoop.provenance,
    body: <ClosedLoopSection data={closedLoop} />,
  },
];

/** The contents page and the running head index the same seven things. */
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
          <SectionFrame
            key={section.id}
            id={section.id}
            step={section.step}
            title={section.title}
            stage={section.stage}
            act={section.act}
            provenance={section.provenance}
          >
            {section.body}
          </SectionFrame>
        ))}
      </main>

      <Colophon />
    </>
  );
}
