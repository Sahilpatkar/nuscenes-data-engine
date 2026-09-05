import { StatTile } from "../components/StatTile";
import { Ticks } from "../components/Ticks";
import type { ClosedLoop as ClosedLoopData } from "../data/types";

/**
 * Step 7 — the loop, closed (story contract §1.5).
 *
 * The headline is the exporter's guarded one ("… → measurable improvement" only
 * where the night delta really is a gain), rendered by the section frame; where
 * the guard did not hold the frame falls back to the step's own title and this
 * body claims nothing extra.
 *
 * Then the three numbers the whole story earned, the four questions it can now
 * answer — including the one about what failed, which is written as plainly as
 * the ones about what worked — and the thesis those four answers are the evidence
 * for. The thesis is the site's last line before the deep links: the sentence the
 * viewer should leave with, so it gets the page's largest pull treatment and
 * nothing follows it but the ways to go deeper.
 *
 * The accented tile is the last card: the exporter writes the relative-gain card
 * last (`_result_hero_cards`), and the relative gain is the number the story has
 * been chasing since step 1's third tile — the two accents are the same claim,
 * six steps apart.
 */
export function ClosedLoop({ data }: { data: ClosedLoopData }): JSX.Element {
  const lastCard = data.cards.length - 1;

  return (
    <>
      <div className="stat-row">
        {data.cards.map((card, index) => (
          <StatTile
            key={card.label}
            label={card.label}
            value={card.value}
            detail={card.delta}
            accent={index === lastCard}
          />
        ))}
      </div>

      <div className="answers">
        {data.answers.map((answer) => (
          <section className="answer" key={answer.question}>
            <h3 className="answer-question">{answer.question}</h3>
            {answer.sentences.map((sentence) => (
              <p className="answer-line" key={sentence}>
                <Ticks text={sentence} />
              </p>
            ))}
          </section>
        ))}
      </div>

      <p className="thesis">{data.closing_thesis}</p>
    </>
  );
}
