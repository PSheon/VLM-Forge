import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies the API to a running `forge serve`; `npm run build` writes web/dist,
// which `forge serve` mounts at /.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8796" } },
  build: { outDir: "dist", emptyOutDir: true },
});
