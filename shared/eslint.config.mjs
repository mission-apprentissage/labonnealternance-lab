import { fixupConfigRules } from "@eslint/compat";
import zod from "eslint-plugin-zod";
import globals from "globals";
import path from "node:path";
import { fileURLToPath } from "node:url";
import js from "@eslint/js";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const compat = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: js.configs.recommended,
  allConfig: js.configs.all,
});

export default [
  ...fixupConfigRules(
    compat.extends("plugin:n/recommended-module", "plugin:import/recommended", "plugin:import/typescript")
  ),
  {
    plugins: {
      zod,
    },

    languageOptions: {
      globals: {
        ...globals.node,
      },
    },

    rules: {
      "zod/prefer-enum": "error",
      "n/no-missing-import": 0,
      "n/no-extraneous-import": ["error"],
    },
  },
];
