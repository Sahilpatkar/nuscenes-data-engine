import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages serves this project site under /nuscenes-data-engine/; the deploy
// workflow sets BASE_PATH. Local dev and previews keep the root base.
export default defineConfig({
  base: process.env.BASE_PATH ?? "/",
  plugins: [react()],
});
