// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  The recommendation panel + the pending workload queue strip + the beat
//  caption (doc G overlays). The recommendation panel translates the world
//  model output into ONE recommendation while keeping the safety GATE explicit:
//  exact blockers and learned risk are shown SEPARATELY, never blended into one
//  weighted score.

import { chip, el, fmtDelta, fmtSeconds, type PanelHandle } from "./dom";
import type { Store, StoreState } from "../state/store";
import type { CandidateResultDTO, WorkloadDTO } from "../api/client";

/** Human labels for the KPI delta channels shown in the recommendation panel. */
const KPI_LABELS: Record<string, [string, string, "lowerBetter" | "higherBetter"]> = {
  queue_wait_delta_s: ["Queue wait", "s", "lowerBetter"],
  admitted_gpu_hours_delta: ["Admitted GPU-h", "", "higherBetter"],
  topology_fragmentation_delta: ["Fragmentation", "", "lowerBetter"],
  rack_thermal_headroom_c: ["Thermal headroom", "°C", "higherBetter"],
  completion_time_delta_s: ["Completion", "s", "lowerBetter"],
  cross_rack_rate: ["Cross-rack", "", "lowerBetter"],
};

function actionLabel(c: CandidateResultDTO): string {
  if (c.action_id === "hold") return "Hold one window";
  if (c.action_id.startsWith("admit-a")) return "Admit → Rack Group A";
  if (c.action_id.startsWith("admit-b")) return "Admit → Rack Group B";
  return c.action_id;
}

/** Pending workload strip (left). Selecting a card focuses that decision target. */
export function createQueueStrip(parent: HTMLElement, store: Store): PanelHandle {
  const root = el("div", "panel queue-strip");
  root.appendChild(el("div", "panel-title", "Pending Workloads"));
  const list = el("div", "wl-list");
  root.appendChild(list);
  parent.appendChild(root);

  function card(wl: WorkloadDTO, focused: boolean): HTMLElement {
    const c = el("div", `wl-card ${focused ? "focused" : ""}`);
    const title = el("div", "wl-card-title");
    title.append(el("span", "", wl.workload_uid), el("span", "", `${wl.gpu_req} GPU`));
    const meta = el(
      "div",
      "wl-meta",
      `${wl.tenant_id} · ${wl.topology_policy} · ${wl.comm_class} · age ${fmtSeconds(wl.age_s)}`,
    );
    c.append(title, meta);
    const chips = el("div", "wl-chips");
    chips.append(
      chip("Compare actions", () => {
        store.setFocusedWorkload(wl.workload_uid);
        void store.runCompare();
      }),
    );
    c.appendChild(chips);
    c.addEventListener("click", () => store.setFocusedWorkload(wl.workload_uid));
    return c;
  }

  return {
    root,
    update(s: StoreState) {
      list.innerHTML = "";
      const pend = (s.snapshot?.entities.workloads ?? []).filter(
        (w) => w.status === "pending",
      );
      if (!pend.length) {
        list.appendChild(el("div", "wl-meta", "No pending workloads."));
        return;
      }
      for (const wl of pend) list.appendChild(card(wl, wl.workload_uid === s.focusedWorkload));
    },
  };
}

/** Recommendation panel (right): ranked safe actions, best summary, deltas. */
export function createRecommendation(parent: HTMLElement, store: Store): PanelHandle {
  const root = el("div", "panel recommendation");
  root.appendChild(el("div", "panel-title", "Recommendation"));
  const best = el("div", "rec-best");
  const rows = el("div", "rec-rows");
  root.append(best, rows);
  parent.appendChild(root);

  return {
    root,
    update(s: StoreState) {
      const cmp = s.compare;
      if (!cmp) {
        best.innerHTML = "";
        rows.innerHTML = "";
        best.appendChild(
          el("div", "wl-meta", "Select a workload and compare safe actions."),
        );
        root.classList.toggle("hidden", s.mode === "replay-calibration");
        return;
      }
      root.classList.remove("hidden");

      // Best summary — abstain is shown as its own state, not a hidden penalty.
      best.innerHTML = "";
      if (cmp.abstain) {
        const b = el("span", "badge abstain", "ABSTAIN");
        best.append(b, el("div", "wl-meta", cmp.abstain_reason ?? "low confidence"));
      } else {
        const rec = cmp.candidates.find((c) => c.action_id === cmp.recommended_action_id);
        best.append(
          el("div", "wl-card-title", rec ? `Recommended: ${actionLabel(rec)}` : "No safe action"),
          el(
            "div",
            "wl-meta",
            rec
              ? `horizon ${Math.round(cmp.horizon_s / 60)} min · baseline = hold`
              : "all candidates filtered",
          ),
        );
      }

      // Ranked rows: safe first (by rank), then blocked with why-not link.
      rows.innerHTML = "";
      const sorted = [...cmp.candidates].sort((a, b) => {
        if (a.safe !== b.safe) return a.safe ? -1 : 1;
        return (a.rank ?? 99) - (b.rank ?? 99);
      });
      for (const c of sorted) rows.appendChild(rowFor(c, cmp.recommended_action_id, store));
    },
  };
}

function rowFor(
  c: CandidateResultDTO,
  recommendedId: string | null,
  store: Store,
): HTMLElement {
  const recommended = c.action_id === recommendedId;
  const rowClass = !c.safe
    ? "blocked"
    : c.uncertainty.abstain
      ? "abstain"
      : recommended
        ? "recommended"
        : "";
  const row = el("div", `rec-row ${rowClass}`);

  const head = el("div", "rec-head");
  head.style.display = "flex";
  head.style.alignItems = "center";
  head.style.gap = "8px";
  head.style.width = "100%";
  head.append(el("span", "rec-rank", c.safe ? `#${c.rank ?? "—"}` : "🔒"));
  head.append(el("span", "", actionLabel(c)));
  const spacer = el("span", "");
  spacer.style.flex = "1";
  head.append(spacer);

  // Abstain is its OWN visual state (doc G): a safe-but-too-uncertain candidate
  // shows an explicit abstain chip and cannot be the recommendation, rather than
  // being silently down-weighted.
  const abstaining = c.safe && c.uncertainty.abstain;
  if (!c.safe) {
    head.append(el("span", "badge blocked", "BLOCKED"));
  } else if (abstaining) {
    head.append(el("span", "badge abstain", "ABSTAIN"));
  } else {
    head.append(el("span", "badge safe", "SAFE"));
  }

  const actions = el("div", "rec-actions");
  if (!c.safe) {
    actions.append(chip("Why not?", () => void store.showWhyNot(c.action_id), "danger"));
  } else {
    actions.append(chip("Pin", () => store.togglePinned(c.action_id)));
  }

  const wrap = el("div", "");
  wrap.style.width = "100%";
  wrap.append(head);

  // KPI deltas (median heads) — learned forecast, shown distinct from the gate.
  const kpis = el("div", "rec-kpis");
  for (const [ch, val] of Object.entries(c.delta_kpis)) {
    const meta = KPI_LABELS[ch];
    if (!meta || val == null) continue;
    const [label, unit, dir] = meta;
    const good = dir === "lowerBetter" ? val <= 0 : val >= 0;
    const cell = el("span", "delta-metric");
    cell.append(
      el("span", "", label),
      el(
        "span",
        good ? "delta-good" : "delta-bad",
        ch.endsWith("_s") ? fmtSeconds(val) : fmtDelta(val, unit),
      ),
    );
    kpis.appendChild(cell);
  }
  // Uncertainty / hazard summary — calibrated risk lives next to the deltas.
  const haz = c.uncertainty.hazards.find((h) => h.channel === "hotspot");
  if (haz) {
    const cell = el("span", "delta-metric");
    const risky = haz.conformal_upper > 0.15;
    cell.append(
      el("span", "", "Hotspot p≤"),
      el("span", risky ? "delta-bad" : "delta-good", haz.conformal_upper.toFixed(2)),
    );
    kpis.appendChild(cell);
  }
  wrap.append(kpis, actions);
  row.append(wrap);
  return row;
}

/** Self-captioning beat caption (doc G: works when the video plays muted). */
export function createBeatCaption(parent: HTMLElement): PanelHandle & {
  set(text: string | null): void;
} {
  const root = el("div", "panel beat-caption hidden");
  parent.appendChild(root);
  return {
    root,
    update() {},
    set(text: string | null) {
      if (!text) {
        root.classList.add("hidden");
        return;
      }
      root.textContent = text;
      root.classList.remove("hidden");
    },
  };
}
