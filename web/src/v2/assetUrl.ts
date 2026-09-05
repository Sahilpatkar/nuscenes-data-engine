/**
 * Resolve a bundle asset path against the site's base.
 *
 * The exporter writes image srcs document-relative — `"story/hero-arm.jpg"` — which
 * the root edition can use as-is because its document IS the base. The report
 * edition is served one directory down (`…/nuscenes-data-engine/v2/`), where the
 * same string resolves to `…/v2/story/hero-arm.jpg` and 404s. Prefixing
 * `import.meta.env.BASE_URL` (Vite's build-time base: `/` in dev and preview,
 * `/nuscenes-data-engine/` on Pages — always with a trailing slash) points every
 * v2 `<img>` back at the one copy of the assets in `web/public/`. No duplication,
 * and nothing to keep in sync when the bundle is re-exported.
 *
 * Every `src={…}` in v2 goes through here (a source-level test asserts it).
 */
export const assetUrl = (src: string): string => import.meta.env.BASE_URL + src;
