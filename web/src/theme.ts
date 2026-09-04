/**
 * Theme state: dark (design-primary) or light (warm paper).
 *
 * Three sources, in precedence order — the same order tokens.css encodes:
 *   1. `data-theme` on <html>  — an explicit choice, stamped pre-paint by the
 *      inline script in index.html and by `applyTheme` here
 *   2. the OS `prefers-color-scheme`
 *   3. dark, the design default
 */

export type Theme = "light" | "dark";

/** Must match the key read by the pre-paint script in index.html. */
const STORAGE_KEY = "theme";

const isTheme = (value: unknown): value is Theme => value === "light" || value === "dark";

/** The stored choice, or null when the viewer has never chosen (= follow the OS). */
export function storedTheme(): Theme | null {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    return isTheme(value) ? value : null;
  } catch {
    // private mode / blocked site data: no stored choice, not an error
    return null;
  }
}

/**
 * What the OS asks for. Mirrors the CSS exactly: only an explicit
 * `prefers-color-scheme: light` means light — "no preference" stays dark.
 */
export function systemTheme(): Theme {
  if (typeof window.matchMedia !== "function") return "dark";
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

/** What is actually on screen right now. */
export function effectiveTheme(): Theme {
  const stamped = document.documentElement.dataset["theme"];
  return isTheme(stamped) ? stamped : systemTheme();
}

/** Stamp a theme and remember it. */
export function applyTheme(theme: Theme): void {
  document.documentElement.dataset["theme"] = theme;
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // storage unavailable — the stamp still holds for this page view
  }
}

/** Flip to the other theme and return it. */
export function cycleTheme(): Theme {
  const next: Theme = effectiveTheme() === "dark" ? "light" : "dark";
  applyTheme(next);
  return next;
}

/** Subscribe to OS changes (only meaningful while nothing is stored). */
export function watchSystemTheme(onChange: (theme: Theme) => void): () => void {
  if (typeof window.matchMedia !== "function") return () => {};
  const query = window.matchMedia("(prefers-color-scheme: light)");
  const handler = (event: MediaQueryListEvent): void => onChange(event.matches ? "light" : "dark");
  query.addEventListener("change", handler);
  return () => query.removeEventListener("change", handler);
}
