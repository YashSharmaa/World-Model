// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  The why-not card (doc G overlay). It explains a REJECTED action in one
//  screenful and — crucially — keeps the two reason families separate:
//
//    exact blockers  : the kernel's hard infeasibility certificate (doc S, the
//                      minimal IIS-analogue group set). These are facts.
//    predicted risks : the learned model's calibrated downside. These are
//                      forecasts with conformal bounds.
//
//  Mixing the two would break the project's spine ("safety is a GATE, not a
//  weighted metric"), so they are visually distinct sections.

import { WorldModelClient, type WhyNotDTO } from "../api/client";
import { el, type PanelHandle } from "./dom";
import type { Store, StoreState } from "../state/store";

export function createWhyNotCard(
  parent: HTMLElement,
  store: Store,
  client: WorldModelClient,
): PanelHandle {
  const root = el("div", "panel why-not hidden");
  parent.appendChild(root);

  let lastActionId: string | null = null;
  let cache: WhyNotDTO | null = null;

  async function render(actionId: string, s: StoreState): Promise<void> {
    if (!s.snapshot) return;
    if (actionId !== lastActionId || !cache) {
      cache = await client.whyNot({
        state_hash: s.snapshot.state_hash,
        candidate_action: { action_id: actionId, type: "admit_and_place", params: {} },
      });
      lastActionId = actionId;
    }
    paint(cache);
  }

  function paint(w: WhyNotDTO): void {
    root.innerHTML = "";
    const close = el("button", "close-x", "×");
    close.addEventListener("click", () => store.clearWhyNot());
    root.appendChild(close);
    root.appendChild(el("div", "panel-title", `Why not — ${w.action_id}`));

    // exact blockers (hard facts)
    root.appendChild(el("div", "why-section-title", "Exact blockers (kernel gate)"));
    if (!w.exact_blockers.length) {
      root.appendChild(el("div", "blocker-detail", "None — action is feasible."));
    }
    for (const b of w.exact_blockers) {
      const node = el("div", "blocker");
      node.append(
        el("div", "blocker-group", b.group.replace(/_/g, " ")),
        el("div", "blocker-detail", b.detail),
      );
      if (Object.keys(b.deficit).length) {
        const def = Object.entries(b.deficit)
          .map(([k, v]) => `${k}: ${v}`)
          .join(", ");
        node.append(el("div", "wl-meta", `deficit → ${def}`));
      }
      root.appendChild(node);
    }

    // minimal certificate
    if (w.minimal_groups.length) {
      root.appendChild(el("div", "why-section-title", "Minimal certificate"));
      root.appendChild(
        el("div", "blocker-detail", w.minimal_groups.map((g) => g.replace(/_/g, " ")).join(" + ")),
      );
    }

    // predicted risks (learned forecasts, separate section)
    if (w.predicted_risks.length) {
      root.appendChild(el("div", "why-section-title", "Predicted risks (learned)"));
      for (const r of w.predicted_risks) root.appendChild(el("div", "blocker-detail", `• ${r}`));
    }

    // policy reasons + rejected domains
    if (w.policy_reasons.length) {
      root.appendChild(el("div", "why-section-title", "Policy"));
      for (const r of w.policy_reasons) root.appendChild(el("div", "blocker-detail", `• ${r}`));
    }
    if (w.rejected_domain_ids.length) {
      root.appendChild(
        el("div", "wl-meta", `Rejected domains: ${w.rejected_domain_ids.join(", ")}`),
      );
    }
  }

  return {
    root,
    update(s: StoreState) {
      if (!s.whyNotActionId) {
        root.classList.add("hidden");
        lastActionId = null;
        cache = null;
        return;
      }
      root.classList.remove("hidden");
      void render(s.whyNotActionId, s);
    },
  };
}
