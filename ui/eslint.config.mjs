import { fixupConfigRules } from "@eslint/compat";
import path from "node:path";
import { fileURLToPath } from "node:url";
import js from "@eslint/js";
import { FlatCompat } from "@eslint/eslintrc";
import dwordDesignImportAlias from "@dword-design/eslint-plugin-import-alias";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const compat = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: js.configs.recommended,
  allConfig: js.configs.all,
});

// eslint-disable-next-line import/no-anonymous-default-export
export default [
  ...fixupConfigRules(
    compat.extends(
      "next",
      "next/core-web-vitals",
      "next/typescript",
      "plugin:import/recommended",
      "plugin:import/typescript",
      "plugin:react-hooks/recommended"
    )
  ),
  {
    settings: {
      next: {
        rootDir: "ui",
      },
    },
    plugins: {
      "@dword-design/import-alias": dwordDesignImportAlias,
    },
    rules: {
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
      "@dword-design/import-alias/prefer-alias": [
        "error",
        {
          alias: {
            "@": "./ui",
            shared: "./shared/src",
          },
        },
      ],
    },
  },
];
