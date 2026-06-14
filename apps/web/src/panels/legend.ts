// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Persistent HTML overlay panels: the legend, the state-provenance header, and
//  the footer credit. These live in the dom_overlay layer, NOT the scene graph
//  (CSS2DRenderer only supports translation and breaks off-100% zoom — doc G).
//
//  Because LinkedIn videos autoplay MUTED, these three persistent elements turn
//  the scene into a self-captioning demo: a now/predicted/alternative legend, a
//  provenance header that always shows model + policy version, and a footer that
//  always credits Yash Sharma.

import { el, type PanelHandle } from "./dom";
import type { StoreState } from "../state/store";
import { getTheme, toggleTheme } from "../theme";

/** The persistent legend strip (doc G "Persistent legend strip"). */
export function createLegend(parent: HTMLElement): PanelHandle {
  const items: Array<[string, string]> = [
    ["Now (grounded)", "#58c4ff"],
    ["Predicted", "#c9a4ff"],
    ["Alternative action", "#ffcf6b"],
    ["Blocked (exact)", "#ff6b7a"],
    ["Low confidence", "#8a93a6"],
  ];
  const root = el("div", "panel legend");
  root.appendChild(el("div", "panel-title", "Legend"));
  const list = el("div", "legend-list");
  for (const [label, color] of items) {
    const row = el("div", "legend-row");
    const swatch = el("span", "legend-swatch");
    swatch.style.background = color;
    row.append(swatch, el("span", "legend-label", label));
    list.appendChild(row);
  }
  root.appendChild(list);
  parent.appendChild(root);
  return { root, update() {} };
}

/**
 * State-provenance header (doc G): ALWAYS shows state_hash, generated_at, model
 * version, policy version, freshness, and a backend-health dot. Keeps every
 * recommendation/replay auditable.
 */
export function createProvenanceHeader(parent: HTMLElement): PanelHandle {
  const root = el("div", "panel provenance");
  const left = el("div", "prov-left");
  const title = el("span", "prov-brand", "GPU-Center World Model");
  const sub = el("span", "prov-sub", "Decision Studio · shadow-mode (offline replay)");
  left.append(title, sub);

  const right = el("div", "prov-right");
  const health = el("span", "prov-health");
  const model = el("span", "prov-chip");
  const policy = el("span", "prov-chip");
  const hash = el("span", "prov-chip mono");
  const fresh = el("span", "prov-chip");
  // light/dark theme toggle — integrated into the header, professional light default.
  const themeBtn = el("button", "prov-toggle") as HTMLButtonElement;
  themeBtn.title = "Toggle light / dark theme";
  themeBtn.setAttribute("aria-label", "Toggle light or dark theme");
  const paintTheme = (): void => {
    themeBtn.textContent = getTheme() === "light" ? "☾" : "☀";
  };
  paintTheme();
  themeBtn.addEventListener("click", () => {
    toggleTheme();
    paintTheme();
  });
  right.append(health, model, policy, hash, fresh, themeBtn);

  root.append(left, right);
  parent.appendChild(root);

  return {
    root,
    update(s: StoreState) {
      const snap = s.snapshot;
      health.textContent = s.backendOnline ? "● live" : "● mock";
      health.style.color = s.backendOnline ? "#5fe3a1" : "#ffcf6b";
      health.title = s.health;
      model.textContent = `model ${snap?.model_version ?? "—"}`;
      policy.textContent = `policy ${snap?.policy_version ?? "—"}`;
      hash.textContent = snap?.state_hash ?? "—";
      const age = snap?.freshness.gpu_telemetry_age_s ?? 0;
      const sh = snap?.freshness.worst_source_health ?? "—";
      fresh.textContent = `tel ${Math.round(age)}s · ${sh}`;
      fresh.style.color = sh === "observed" ? "#9fb0c8" : "#ffcf6b";
    },
  };
}

/** Footer credit — always visible, carries the Yash Sharma attribution. */
export function createFooterCredit(parent: HTMLElement): PanelHandle {
  const root = el("div", "footer-credit");
  const link = el("a", "footer-link") as HTMLAnchorElement;
  link.href = "https://www.linkedin.com/in/yashsharmaa/";
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = "Built by Yash Sharma — linkedin.com/in/yashsharmaa";
  root.appendChild(link);
  parent.appendChild(root);
  return { root, update() {} };
}
