import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Relative base + hash routing => the built site works on ANY static host
// (GitHub Pages project pages, an internal server, or opened from a subfolder).
export default defineConfig({
  base: "./",
  plugins: [react()],
  // One fixed, quiet port for both `npm run dev` and `npm run preview`.
  // strictPort: fail loudly instead of silently hopping to another port.
  server: { port: 6023, strictPort: true },
  preview: { port: 6023, strictPort: true },
  build: {
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        // split heavy deps into their own chunks so no single bundle is huge (function form for rolldown)
        manualChunks(id: string) {
          if (id.includes("node_modules/katex")) return "katex";
          if (id.includes("node_modules/react") || id.includes("react-router")) return "react";
        },
      },
    },
  },
});
