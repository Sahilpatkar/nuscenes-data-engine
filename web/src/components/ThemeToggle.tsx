import { useEffect, useState } from "react";
import { cycleTheme, effectiveTheme, storedTheme, watchSystemTheme, type Theme } from "../theme";

/**
 * Two-state theme button. The label names the theme you are in; the accessible
 * name names what pressing it does. Until the viewer presses it, nothing is
 * stored and the page follows the OS — so the label follows the OS too.
 */
export function ThemeToggle(): JSX.Element {
  const [theme, setTheme] = useState<Theme>(() => effectiveTheme());

  useEffect(
    () =>
      watchSystemTheme((next) => {
        if (storedTheme() === null) setTheme(next);
      }),
    [],
  );

  const other: Theme = theme === "dark" ? "light" : "dark";

  return (
    <button
      type="button"
      className="theme-toggle"
      aria-label={`Switch to ${other} theme`}
      onClick={() => setTheme(cycleTheme())}
    >
      <span aria-hidden="true">{theme === "dark" ? "◒" : "◓"}</span>
      <span>{theme === "dark" ? "Dark" : "Light"}</span>
    </button>
  );
}
