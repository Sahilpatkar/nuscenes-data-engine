import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages serves this project site under /nuscenes-data-engine/; the deploy
// workflow sets BASE_PATH. Local dev and previews keep the root base.
export default defineConfig({
  base: process.env.BASE_PATH ?? "/",
  plugins: [react()],
  build: {
    rollupOptions: {
      /* Two documents, one build (a Vite MPA): the report edition (src/v2/) at the
         site root and the original scroll-through edition (src/) at /v1/. Vite
         keeps each input's path relative to the project root, so `v1/index.html`
         emits `dist/v1/index.html`, served at `…/nuscenes-data-engine/v1/`. Shared
         chunks land in `dist/assets/` and are referenced base-prefixed and
         root-absolute, which is correct from both documents. The report edition's
         first address, /v2/, is a static redirect stub under `public/v2/` that
         Vite copies through untouched. package.json is `"type": "module"`, so
         entries resolve through `import.meta.url` — `__dirname` does not exist
         here. */
      input: {
        main: fileURLToPath(new URL("index.html", import.meta.url)),
        v1: fileURLToPath(new URL("v1/index.html", import.meta.url)),
      },
    },
  },
});
