// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//
//  SPDX-FileCopyrightText: 2026 Yash Sharma <https://www.linkedin.com/in/yashsharmaa/>
//  SPDX-License-Identifier: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Flat ESLint config (ESLint 9) for the web app. Doc H "Static checks" pairs the
//  TypeScript `tsc --noEmit` typecheck with ESLint as the second gate. We keep
//  this lint *type-aware-light*: the recommended @typescript-eslint rule set runs
//  without the (slower) type-checked rules so it stays fast on contributor
//  machines and in CI, while the strict type contract is already enforced by
//  `tsc --noEmit` (tsconfig has strict + noUnusedLocals + noUnusedParameters).
//
//  Scope: only the TypeScript sources under src/ and the two test trees. Build
//  output, vendored deps, and config files are ignored. The Playwright e2e specs
//  live OUTSIDE this package (../../tests/frontend_e2e) and are linted by their
//  own tsconfig at run time; they are not part of this app's compile graph.

import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  // 0) global ignores — never lint generated / vendored trees.
  {
    ignores: ["dist/**", "node_modules/**", "public/**", "*.config.js", "*.config.ts"],
  },

  // 1) JS recommended baseline (applies to any plain .js that slips in).
  js.configs.recommended,

  // 2) TypeScript recommended (non-type-checked: fast, no full program build).
  ...tseslint.configs.recommended,

  // 3) project-specific rules for the app sources.
  {
    files: ["src/**/*.ts"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: {
        // Browser + Vite runtime globals the Decision Studio runs against.
        window: "readonly",
        document: "readonly",
        navigator: "readonly",
        HTMLElement: "readonly",
        HTMLButtonElement: "readonly",
        HTMLCanvasElement: "readonly",
        SVGElementTagNameMap: "readonly",
        EventSource: "readonly",
        AbortController: "readonly",
        PointerEvent: "readonly",
        RequestInit: "readonly",
        Response: "readonly",
        fetch: "readonly",
        setTimeout: "readonly",
        clearTimeout: "readonly",
        setInterval: "readonly",
        clearInterval: "readonly",
        requestAnimationFrame: "readonly",
        console: "readonly",
        URLSearchParams: "readonly",
      },
    },
    rules: {
      // The console is the demo's debug surface; allow warn/error, flag stray
      // `console.log` (the panels already mark intentional ones with an
      // eslint-disable-next-line no-console comment).
      "no-console": ["warn", { allow: ["warn", "error"] }],
      // Unused vars are also caught by tsc; here we let a leading underscore mean
      // "intentionally unused" (loop indices, retained-for-disposal handles).
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      // The semantic read-model crosses a few `unknown`→typed seams (the SSE
      // patch merge, the window.__studio debug handle). Those are deliberate and
      // localized, so warn rather than error to keep the gate actionable.
      "@typescript-eslint/no-explicit-any": "warn",
      // Empty catch blocks that intentionally swallow (proxy/offline fallback)
      // must carry a comment; an entirely empty block is an error.
      "no-empty": ["error", { allowEmptyCatch: false }],
      "eqeqeq": ["error", "smart"],
    },
  },

  // 4) the in-repo runnable test harnesses under src/__tests__ are self-contained
  //    assert()+main() scripts (no test framework in this lane); they print
  //    pass/fail to stdout by design, so console.log is allowed there. The
  //    intentional disable comments those files carry are then redundant but
  //    harmless — silence the "unused directive" report so the lint stays clean.
  {
    files: ["src/**/__tests__/**/*.ts"],
    linterOptions: { reportUnusedDisableDirectives: "off" },
    rules: {
      "no-console": "off",
    },
  },
);
