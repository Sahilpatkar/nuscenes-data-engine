/**
 * The step's takeaway, in the live app's own "What we learned" frame
 * (`app/demo/render.py::learned` — a bordered callout, label then sentence).
 * The sentence is the bundle's `takeaway`; the label is the app's.
 */
export function Learned({ text }: { text: string }): JSX.Element {
  return (
    <p className="learned">
      <span className="learned-label">What we learned</span>
      <span className="learned-text">{text}</span>
    </p>
  );
}
