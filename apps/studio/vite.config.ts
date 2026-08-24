import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The bundle ships inside the Python wheel, so it is written straight into the
// package and every asset is referenced relatively.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { outDir: "../../src/ntb/_static", emptyOutDir: true, target: "es2022" },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8756",
      "/ws": { target: "ws://127.0.0.1:8756", ws: true },
    },
  },
});
