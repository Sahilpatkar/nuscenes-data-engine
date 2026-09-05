import { useEffect, useState } from "react";

import { applyTheme, storedTheme, watchSystemTheme, type Theme } from "../../theme";

/**
 * Two-state theme button, forked from v1's for the report header.
 *
 * State, storage and precedence are the SHARED ones (`src/theme.ts`, the `"theme"`
 * key, the same pre-paint script): a choice made in either edition holds in the
 * other. Two things are v2's own —
 *
 *   · the label voice (mono small-caps, "Paper" / "Night", set by base.css), and
 *   · which theme "no explicit choice" means. The shared `systemTheme()` encodes
 *     v1's rule — dark unless the OS explicitly asks for light. This edition is
 *     light-first and its tokens.css asks the complementary question, so the
 *     unstamped reading below matches the CSS by asking `prefers-color-scheme:
 *     dark`. The two agree in every browser that reports a preference; they only
 *     differ under the spec's "no-preference" reply, where v2 must read light
 *     because that is what the page is actually painting.
 *
 * The label names the theme you are in; the accessible name names what pressing
 * it does.
 */

function systemThemeV2(): Theme {
  if (typeof window.matchMedia !== "function") return "light";
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** What is on screen right now: the stamped choice, else the OS reading above. */
function effectiveThemeV2(): Theme {
  const stamped = document.documentElement.dataset["theme"];
  return stamped === "light" || stamped === "dark" ? stamped : systemThemeV2();
}

export function ThemeToggle(): JSX.Element {
  const [theme, setTheme] = useState<Theme>(() => effectiveThemeV2());

  useEffect(
    () =>
      watchSystemTheme(() => {
        if (storedTheme() === null) setTheme(systemThemeV2());
      }),
    [],
  );

  const other: Theme = theme === "dark" ? "light" : "dark";

  return (
    <button
      type="button"
      className="theme-toggle"
      aria-label={`Switch to ${other} theme`}
      onClick={() => {
        applyTheme(other);
        setTheme(other);
      }}
    >
      <span aria-hidden="true">{theme === "dark" ? "◑" : "◐"}</span>
      <span>{theme === "dark" ? "Night" : "Paper"}</span>
    </button>
  );
}
