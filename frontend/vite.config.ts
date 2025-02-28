import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import dts from "vite-plugin-dts";
import { fileURLToPath } from "url";

// Get the directory name in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    dts({
      include: ["src/components/**/*", "src/config.ts", "src/assets/**/*"],
      outDir: "dist/types",
      tsconfigPath: "tsconfig.json",
    }),
  ],
  // Use different configurations based on command
  build: {
    // When building as a library (build:lib)
    ...(process.env.BUILD_LIB === "true"
      ? {
          lib: {
            entry: path.resolve(__dirname, "src/components/plugin/index.ts"),
            name: "AptosChatbotPlugin",
            fileName: (format) => `index.${format}.js`,
          },
          rollupOptions: {
            external: ["react", "react-dom", "react/jsx-runtime"],
            output: {
              globals: {
                react: "React",
                "react-dom": "ReactDOM",
                "react/jsx-runtime": "jsxRuntime",
              },
            },
          },
        }
      : // Default build for web application deployment
        {
          outDir: "dist",
          emptyOutDir: true,
          sourcemap: true,
        }),
  },
});
