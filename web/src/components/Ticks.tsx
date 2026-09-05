/**
 * Bundle copy, rendered with its backticked spans as code.
 *
 * Several exported sentences carry raw arm ids in markdown backticks — the tour
 * writes them that way and Streamlit renders them as inline code, so the site
 * has to render the same emphasis rather than printing the backticks. Nothing is
 * reworded, reordered or dropped: the characters between the ticks are shown
 * verbatim, the ticks themselves become the `<code>` element they stand for.
 *
 * An unbalanced string (an odd number of backticks) is rendered exactly as it
 * arrived — a half-open span is a bug in the copy, not licence to guess where it
 * was meant to close.
 */
export function Ticks({ text }: { text: string }): JSX.Element {
  const parts = text.split("`");
  if (parts.length % 2 === 0) return <>{text}</>;

  return (
    <>
      {parts.map((part, index) =>
        index % 2 === 1 ? (
          <code className="tick" key={`${index}-${part}`}>
            {part}
          </code>
        ) : (
          <span key={`${index}-${part}`}>{part}</span>
        ),
      )}
    </>
  );
}
