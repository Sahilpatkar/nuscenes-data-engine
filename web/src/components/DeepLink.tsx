import type { ReactNode } from "react";

/**
 * "Go deeper in the live app" — a story sentence that happens to be a link.
 *
 * The label is always the exporter's own deep-link label (`meta.deep_links`), so
 * the site never invents a description of a page it does not own. The arrow is
 * decoration; the sentence carries the meaning.
 */
export function DeepLink({ href, children }: { href: string; children: ReactNode }): JSX.Element {
  return (
    <a className="deep-link" href={href} target="_blank" rel="noopener noreferrer">
      <span>{children}</span>
      <span className="deep-link-arrow" aria-hidden="true">
        →
      </span>
    </a>
  );
}
