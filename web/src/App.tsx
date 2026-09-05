import type { ReactNode } from "react";

import { DeepLink } from "./components/DeepLink";
import { Footer } from "./components/Footer";
import { Landing } from "./components/Landing";
import { Section } from "./components/Section";
import { StepRail, type RailStep } from "./components/StepRail";
/* The section components share their names with the bundle interfaces below
   (both are named for the step they belong to), so each is aliased once here. */
import { Blindspot as BlindspotSection } from "./sections/Blindspot";
import { ClosedLoop as ClosedLoopSection } from "./sections/ClosedLoop";
import { Intervention as InterventionSection } from "./sections/Intervention";
import { MissedPedestrian as MissedPedestrianSection } from "./sections/MissedPedestrian";
import { Scenario as ScenarioSection } from "./sections/Scenario";
import { Verdict as VerdictSection } from "./sections/Verdict";
import { WhyFrame as WhyFrameSection } from "./sections/WhyFrame";

import blindspotJson from "./data/blindspot.json";
import closedLoopJson from "./data/closed_loop.json";
import heroFrameJson from "./data/hero_frame.json";
import interventionJson from "./data/intervention.json";
import metaJson from "./data/meta.json";
import scenarioJson from "./data/scenario.json";
import verdictJson from "./data/verdict.json";
import whyFrameJson from "./data/why_frame.json";

import type {
  Blindspot,
  ClosedLoop,
  DeepLink as DeepLinkRef,
  HeroFrame,
  Intervention,
  Meta,
  Provenance,
  Scenario,
  Verdict,
  WhyFrame,
} from "./data/types";

/* The bundle. Every headline, sentence and number the site shows is read from
   these eight files, which `web/build_data.py` derives from the committed demo
   package and a pytest re-derives byte-for-byte. The JSON's inferred types are
   widened (a `kind` field is `string`, not the union), so each import is asserted
   into its exported interface once, here, and never re-typed downstream. */
const meta = metaJson as Meta;
const blindspot = blindspotJson as Blindspot;
const heroFrame = heroFrameJson as HeroFrame;
const scenario = scenarioJson as Scenario;
const whyFrame = whyFrameJson as WhyFrame;
const intervention = interventionJson as Intervention;
const verdict = verdictJson as Verdict;
const closedLoop = closedLoopJson as ClosedLoop;

/** The exporter's deep links, by their pinned Streamlit `url_path`. */
function deepLinks(...paths: readonly string[]): DeepLinkRef[] {
  return paths
    .map((path) => meta.deep_links.find((link) => link.url_path === path))
    .filter((link): link is DeepLinkRef => link !== undefined);
}

interface StorySection {
  /** The anchor — also the rail target and the section's `id`. */
  id: string;
  step: number;
  /** The step's fixed title, mirroring the live tour's own step titles. */
  title: string;
  /** The closed-improvement-loop stage this step lights. */
  stage: string;
  /** The deck's act, so both tellings are structured the same way. */
  act: string;
  headline: string | null;
  provenance: Provenance[];
  links: DeepLinkRef[];
  /** The section's own internals. */
  body: ReactNode;
  /** The centrepiece alone gets a surface band (story contract §1.2). */
  tone?: "panel";
}

/* The seven steps, in story order. Titles, stages and the deep-link labels are
   the live tour's own (app/demo/views/tour.py `_STEPS` and meta.deep_links), so
   the two front-ends name the same seven things the same way. */
const SECTIONS: readonly StorySection[] = [
  {
    id: "blindspot",
    step: 1,
    title: "We found a blind spot",
    stage: "Diagnose",
    act: "Problem",
    headline: blindspot.headline,
    provenance: blindspot.provenance,
    links: deepLinks("failures"),
    body: <BlindspotSection data={blindspot} />,
  },
  {
    id: "hero-frame",
    step: 2,
    title: "What the failure looks like",
    stage: "Diagnose",
    act: "Problem",
    headline: heroFrame.headline,
    provenance: heroFrame.provenance,
    links: deepLinks("failures"),
    body: <MissedPedestrianSection data={heroFrame} />,
  },
  {
    id: "scenario",
    step: 3,
    title: "Where else does this happen?",
    stage: "Mine",
    act: "Approach",
    headline: scenario.headline,
    provenance: scenario.provenance,
    links: deepLinks("scenarios"),
    body: <ScenarioSection data={scenario} />,
  },
  {
    id: "why-frame",
    step: 4,
    title: "What data should we add?",
    stage: "Mine",
    act: "Approach",
    headline: whyFrame.headline,
    provenance: whyFrame.provenance,
    links: deepLinks("active_learning"),
    body: <WhyFrameSection data={whyFrame} />,
    tone: "panel",
  },
  {
    id: "intervention",
    step: 5,
    title: "We changed the training data",
    stage: "Train",
    act: "Approach",
    headline: intervention.headline,
    provenance: intervention.provenance,
    links: deepLinks("active_learning"),
    body: <InterventionSection data={intervention} />,
  },
  {
    id: "verdict",
    step: 6,
    title: "Did it fix the failure?",
    stage: "Evaluate",
    act: "Results",
    headline: verdict.headline,
    provenance: verdict.provenance,
    links: deepLinks("active_learning"),
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
    headline: closedLoop.headline,
    provenance: closedLoop.provenance,
    links: deepLinks(
      "tour",
      "failures",
      "scenarios",
      "active_learning",
      "weak_supervision",
      "chat_replay",
    ),
    body: <ClosedLoopSection data={closedLoop} />,
  },
];

const RAIL_STEPS: readonly RailStep[] = SECTIONS.map(({ id, step, title }) => ({
  id,
  step,
  title,
}));

export default function App(): JSX.Element {
  return (
    <>
      <Landing />
      <StepRail steps={RAIL_STEPS} />

      <main className="shell" id="story">
        {SECTIONS.map((section) => (
          <Section
            key={section.id}
            id={section.id}
            step={section.step}
            stage={section.stage}
            act={section.act}
            title={section.title}
            headline={section.headline}
            tone={section.tone}
            provenance={section.provenance}
          >
            {section.body}

            {/* The go-deeper links close every section — and on the last one they
                are the last thing on the page after the closing thesis. */}
            {section.links.length > 0 ? (
              <div className="deep-links">
                {section.links.map((link) => (
                  <DeepLink key={link.url} href={link.url}>
                    {link.label}
                  </DeepLink>
                ))}
              </div>
            ) : null}
          </Section>
        ))}
      </main>

      <Footer />
    </>
  );
}
