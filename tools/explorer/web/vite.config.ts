import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Local-first: Monaco is bundled (see src/monaco.ts) — no CDN. The dev server
// proxies /api to the FastAPI backend on 8765. `base: ""` makes the production
// build work when served by FastAPI at any mount point.
export default defineConfig({
  base: "",
  plugins: [react()],
  server: {
    port: 5990,
    strictPort: true,
    proxy: {
      "/api": { target: "http://127.0.0.1:8765", changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 4000,
  },
});
