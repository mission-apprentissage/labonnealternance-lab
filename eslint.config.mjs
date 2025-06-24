import path from "node:path";
import { fileURLToPath } from "node:url";

import dwordDesignImportAlias from "@dword-design/eslint-plugin-import-alias";
import { fixupPluginRules, fixupConfigRules, includeIgnoreFile } from "@eslint/compat";
import { FlatCompat } from "@eslint/eslintrc";
import js from "@eslint/js";
import typescriptEslint from "@typescript-eslint/eslint-plugin";
import tsParser from "@typescript-eslint/parser";
import _import from "eslint-plugin-import";
import simpleImportSort from "eslint-plugin-simple-import-sort";
import globals from "globals";
import pluginReact from "eslint-plugin-react";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const compat = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: js.configs.recommended,
  allConfig: js.configs.all,
});
const gitignorePath = path.resolve(__dirname, ".gitignore");

export default [
  // Load ignore patterns from .gitignore
  includeIgnoreFile(gitignorePath),
  {
    linterOptions: {
      reportUnusedDisableDirectives: "off",
    },
  },
  {
    ignores: [
      "**/release.config.js",
      "**/eslint.config.mjs",
      "**/dist",
      ".gitignore",
      "**/.next",
      "**/next.config.mjs",
      "**/.husky",
      "**/.yarn",
    ],
  },
  ...compat
    .extends("eslint:recommended", "plugin:react/recommended", "plugin:@typescript-eslint/recommended")
    .map((config) => ({
      ...config,
      files: ["**/*.js", "**/*.jsx", "**/*.ts", "**/*.tsx"],
      ...pluginReact.configs.flat.recommended,
      languageOptions: {
        ...pluginReact.configs.flat.recommended.languageOptions,
        globals: {
          ...globals.serviceworker,
        },
      },
    })),
  {
    plugins: {
      "@typescript-eslint": typescriptEslint,
      "simple-import-sort": simpleImportSort,
      import: fixupPluginRules(_import),
      "@dword-design/import-alias": dwordDesignImportAlias,
    },
    languageOptions: {
      globals: {
        ...globals.node,
      },

      parser: tsParser,
      ecmaVersion: "latest",
      sourceType: "module",

      parserOptions: {
        ecmaFeatures: {
          jsx: true,
        },

        project: ["tsconfig.json", "server/tsconfig.json", "shared/tsconfig.json", "ui/tsconfig.json"],
      },
    },
    settings: {
      "import/resolver": {
        typescript: {
          project: ["server/tsconfig.json", "shared/tsconfig.json", "ui/tsconfig.json"],
        },
      },
      react: {
        version: "detect",
      },
    },
    rules: {
      "simple-import-sort/imports": "error",
      "react/react-in-jsx-scope": "off",

      "import/no-cycle": [
        "error",
        {
          ignoreExternal: true,
        },
      ],

      "import/no-relative-packages": "error",
      "import/no-useless-path-segments": ["error"],
      "import/consistent-type-specifier-style": ["error", "prefer-top-level"],

      "import/no-extraneous-dependencies": [
        "error",
        {
          devDependencies: [
            "**/*.test.ts",
            "**/*.test.tsx",
            "**/tests/**/*.ts",
            "**/tests/*.ts",
            "**/fixtures/**/*.ts",
            "**/tsup.config.ts",
            "**/vitest.workspace.ts",
          ],
        },
      ],
      "@typescript-eslint/consistent-type-imports": [
        "error",
        {
          prefer: "type-imports",
        },
      ],
      "@typescript-eslint/ban-ts-comment": "off",
      "@typescript-eslint/no-import-type-side-effects": "error",
      "@typescript-eslint/promise-function-async": "error",
      "@typescript-eslint/switch-exhaustiveness-check": "error",
      "react/no-unescaped-entities": 0,
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          args: "all",
          argsIgnorePattern: "^_",
          caughtErrors: "all",
          caughtErrorsIgnorePattern: "^_",
          destructuredArrayIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          ignoreRestSiblings: true,
        },
      ],
    },
  },
];
