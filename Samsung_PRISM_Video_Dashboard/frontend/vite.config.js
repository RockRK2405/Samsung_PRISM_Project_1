import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies /api and /outputs to the FastAPI backend on :8000,
// so the frontend can call same-origin paths in both dev and production.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/outputs": "http://127.0.0.1:8000",
    },
  },
});
