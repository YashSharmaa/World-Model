// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  The branch delta ribbon (doc G overlay) — the compact, side-by-side action
//  comparison that, with the legend and the recommendation summary, turns the
//  whole scene into a self-captioning demo. It shows the four headline channels
//  per pinned branch, NORMALIZED to the baseline (hold) action, so a muted
//  viewer can read "compare futures" at a glance.

import { el, fmtDelta, fmtSeconds, type PanelHandle } from "./dom";
import type { Store, StoreState } from "../state/store";
import type { CandidateResultDTO } from "../api/client";

const COLS: Array<{
  key: string;
  label: string;
  unit: string;
  dir: "lowerBetter" | "higherBetter";
  isSeconds?: boolean;
}> = [
  { key: "queue_wait_delta_s", label: "Queue wait", unit: "s", dir: "lowerBetter", isSeconds: true },
  { key: "admitted_gpu_hours_delta", label: "Admitted GPU-h", unit: "", dir: "higherBetter" },
  { key: "topology_fragmentation_delta", label: "Fragmentation", unit: "", dir: "lowerBetter" },
];

function branchLabel(c: CandidateResultDTO): string {
  if (c.action_id === "hold") return "Hold";
  if (c.action_id.startsWith("admit-a")) return "Admit A";
  if (c.action_id.startsWith("admit-b")) return "Admit B";
  return c.action_id;
}

export function createDeltaRibbon(parent: HTMLElement, _store: Store): PanelHandle {
  const root = el("div", "panel delta-ribbon hidden");
  parent.appendChild(root);

  return {
    root,
    update(s: StoreState) {
      const cmp = s.compare;
      const show =
        !!cmp &&
        (s.mode === "counterfactual-compare" || s.mode === "decision-studio") &&
        s.pinnedActions.length > 0;
      root.classList.toggle("hidden", !show);
      if (!show || !cmp) return;

      root.innerHTML = "";
      for (const actionId of s.pinnedActions) {
        const c = cmp.candidates.find((x) => x.action_id === actionId);
        if (!c) continue;
        const col = el("div", "delta-col");
        const title = el("div", "delta-col-title");
        const recommended = actionId === cmp.recommended_action_id;
        title.append(
          el("span", "", branchLabel(c)),
          el(
            "span",
            recommended ? "delta-good" : c.safe ? "" : "delta-bad",
            recommended ? "★ rec" : c.safe ? `#${c.rank ?? "—"}` : "🔒",
          ),
        );
        col.appendChild(title);

        for (const def of COLS) {
          const v = (c.delta_kpis as Record<string, number | undefined>)[def.key];
          const row = el("div", "delta-metric");
          if (v == null) {
            row.append(el("span", "", def.label), el("span", "", "—"));
          } else {
            const good = def.dir === "lowerBetter" ? v <= 0 : v >= 0;
            row.append(
              el("span", "", def.label),
              el(
                "span",
                good ? "delta-good" : "delta-bad",
                def.isSeconds ? fmtSeconds(v) : fmtDelta(v, def.unit),
              ),
            );
          }
          col.appendChild(row);
        }

        // violation probability (calibrated, conformal upper) as the 4th line.
        const haz = c.uncertainty.hazards.find((h) => h.channel === "headroom_violation")
          ?? c.uncertainty.hazards.find((h) => h.channel === "hotspot");
        const vp = haz?.conformal_upper ?? 0;
        const vrow = el("div", "delta-metric");
        vrow.append(
          el("span", "", "Violation p≤"),
          el("span", vp > 0.15 ? "delta-bad" : "delta-good", vp.toFixed(2)),
        );
        col.appendChild(vrow);

        root.appendChild(col);
      }
    },
  };
}
