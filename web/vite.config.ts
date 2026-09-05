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
      /* Two documents, one build (a Vite MPA): the v1 story site at the root and
         the report edition at /v2/. Vite keeps each input's path relative to the
         project root, so `v2/index.html` emits `dist/v2/index.html`, served at
         `…/nuscenes-data-engine/v2/`. Shared chunks land in `dist/assets/` and are
         referenced base-prefixed and root-absolute, which is correct from both
         documents. package.json is `"type": "module"`, so entries resolve through
         `import.meta.url` — `__dirname` does not exist here. */
      input: {
        main: fileURLToPath(new URL("index.html", import.meta.url)),
        v2: fileURLToPath(new URL("v2/index.html", import.meta.url)),
      },
    },
  },
});
