/**
 * Resolve a bundle asset path against the site's base.
 *
 * The exporter writes image srcs document-relative — `"story/hero-arm.jpg"` —
 * which only resolves from the document at the site root. The two editions are
 * two documents: the report edition at the root and the original scroll-through
 * edition one directory down (`…/nuscenes-data-engine/v1/`), where the same
 * string resolves to `…/v1/story/hero-arm.jpg` and 404s. Prefixing
 * `import.meta.env.BASE_URL` (Vite's build-time base: `/` in dev and preview,
 * `/nuscenes-data-engine/` on Pages — always with a trailing slash) points every
 * `<img>` in either edition at the one copy of the assets in `web/public/`. No
 * duplication, nothing to keep in sync when the bundle is re-exported, and
 * nothing to change if the editions ever swap places again.
 *
 * Every `src={…}` in both editions goes through here (a source-level test
 * asserts it).
 */
export const assetUrl = (src: string): string => import.meta.env.BASE_URL + src;
