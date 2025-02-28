// eslint.config.js
import js from "@eslint/js";
import * as tseslint from "typescript-eslint";
import eslintConfigPrettier from "eslint-config-prettier";
import reactPlugin from "eslint-plugin-react";
import reactHooksPlugin from "eslint-plugin-react-hooks";
import prettierPlugin from "eslint-plugin-prettier";
import globals from "globals";

export default [
  {
    // Base configuration for all files
    files: ["**/*.{js,jsx,ts,tsx}"],
    plugins: {
      react: reactPlugin,
      "react-hooks": reactHooksPlugin,
      prettier: prettierPlugin,
      "@typescript-eslint": tseslint.plugin,
    },
    languageOptions: {
      ecmaVersion: 2024,
      sourceType: "module",
      parserOptions: {
        jsx: true,
      },
      globals: {
        ...globals.browser,
        ...globals.es2021,
        React: "readonly",
        JSX: "readonly",
      },
    },
    settings: {
      react: {
        version: "detect",
      },
    },
    rules: {
      ...js.configs.recommended.rules,
      ...eslintConfigPrettier.rules,
      "prettier/prettier": "error",
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
      "react/jsx-uses-react": "error",
      "react/jsx-uses-vars": "error",
      "no-unused-vars": "off",
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_" },
      ],
    },
  },

  // TypeScript rules for source files
  {
    files: ["src/**/*.ts", "src/**/*.tsx"],
    languageOptions: {
      parser: tseslint.parser,
      parserOptions: {
        project: "./tsconfig.app.json",
        tsconfigRootDir: ".",
      },
    },
    rules: {
      ...tseslint.configs.recommended.rules,
      ...tseslint.configs.strict.rules,
      "@typescript-eslint/explicit-module-boundary-types": "off",
    },
  },

  // TypeScript rules for config files
  {
    files: ["*.config.ts", "vite.config.ts"],
    languageOptions: {
      parser: tseslint.parser,
      parserOptions: {
        project: "./tsconfig.node.json",
        tsconfigRootDir: ".",
      },
    },
    rules: {
      "@typescript-eslint/no-var-requires": "off",
    },
  },

  // JavaScript config files (no TypeScript checking)
  {
    files: [
      "*.config.js",
      "postcss.config.js",
      "tailwind.config.js",
      "eslint.config.js",
    ],
    rules: {
      "@typescript-eslint/no-var-requires": "off",
    },
  },
];
