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
import type {
  CandidateResultDTO,
  QuantileForecastDTO,
  QueueDTO,
  WorkloadDTO,
} from "../api/client";

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number>,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

/**
 * Render a p05-p95 confidence ribbon around the p50 median for ONE quantile
 * forecast (doc G animation_rules "Uncertainty display": confidence bands in
 * charts). The fan is drawn as a filled p05-p95 band, a lighter p10-p90 inner
 * band, and a solid p50 median line, normalised within the candidate's own
 * channel range so the relative spread is legible at a glance. Returns null when
 * the forecast lacks the quantile heads needed for a band.
 */
function bandChart(forecast: QuantileForecastDTO, label: string): HTMLElement | null {
  const q = forecast.quantiles;
  const p05 = q["0.05"];
  const p50 = q["0.5"];
  const p95 = q["0.95"];
  if (p05 == null || p50 == null || p95 == null) return null;
  const p10 = q["0.1"] ?? p05;
  const p90 = q["0.9"] ?? p95;

  const lo = Math.min(p05, p50, p95);
  const hi = Math.max(p05, p50, p95);
  const span = Math.max(1e-6, hi - lo);
  const W = 132;
  const H = 26;
  const padX = 2;
  const x = (v: number): number => padX + ((v - lo) / span) * (W - padX * 2);

  const wrap = el("div", "band-chart");
  const cap = el("div", "band-cap");
  cap.append(
    el("span", "", label),
    el("span", "band-spread", `±${fmtDelta((p95 - p05) / 2, "", 0).replace("+", "")}`),
  );
  wrap.appendChild(cap);

  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
  const midY = H / 2;
  // outer p05-p95 band
  svg.appendChild(
    svgEl("rect", {
      x: x(p05),
      y: midY - 8,
      width: Math.max(1, x(p95) - x(p05)),
      height: 16,
      rx: 3,
      fill: "var(--predicted)",
      "fill-opacity": "0.18",
    }),
  );
  // inner p10-p90 band
  svg.appendChild(
    svgEl("rect", {
      x: x(p10),
      y: midY - 6,
      width: Math.max(1, x(p90) - x(p10)),
      height: 12,
      rx: 2,
      fill: "var(--predicted)",
      "fill-opacity": "0.34",
    }),
  );
  // p50 median tick
  svg.appendChild(
    svgEl("line", {
      x1: x(p50),
      y1: midY - 9,
      x2: x(p50),
      y2: midY + 9,
      stroke: "var(--grounded)",
      "stroke-width": 2,
    }),
  );
  wrap.appendChild(svg);
  return wrap;
}

/** Short human channel label for the band-chart caption. */
const CHANNEL_LABELS: Record<string, string> = {
  queue_wait_delta_s: "Queue wait band",
  rack_thermal_headroom_c: "Thermal headroom band",
  admitted_gpu_hours_delta: "Admitted GPU-h band",
  topology_fragmentation_delta: "Fragmentation band",
  completion_time_delta_s: "Completion band",
  cross_rack_rate: "Cross-rack band",
};

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

  function card(wl: WorkloadDTO, focused: boolean, queue: QueueDTO | null): HTMLElement {
    const c = el("div", `wl-card ${focused ? "focused" : ""}`);
    const title = el("div", "wl-card-title");
    title.append(el("span", "", wl.workload_uid), el("span", "", `${wl.gpu_req} GPU`));
    // Resource ask + queue age + PRIORITY are first-class card fields (doc G
    // "Workload queue strip" contents: resource ask, queue age, priority).
    const meta = el(
      "div",
      "wl-meta",
      `${wl.tenant_id} · ${wl.topology_policy} · ${wl.comm_class} · age ${fmtSeconds(wl.age_s)}`,
    );
    const prio = el("div", "wl-meta");
    prio.append(
      el("span", "wl-prio", `priority ${wl.priority}`),
      // Queue context: where this job sits in its local queue and the live wait.
      el(
        "span",
        "",
        queue
          ? ` · queue ${wl.local_queue_id}: ${queue.pending_ct} pending (${queue.pending_gpu} GPU) · wait p95 ${fmtSeconds(queue.wait_time_p95_s)}`
          : ` · queue ${wl.local_queue_id}`,
      ),
    );
    c.append(title, meta, prio);

    // Per-candidate action chips (doc G queue strip "candidate action chips"):
    // the focused workload exposes its kernel-safe alphabet (hold / admit A /
    // admit B) right on the card so the decision target stays on the strip.
    const chips = el("div", "wl-chips");
    chips.append(
      chip("Compare actions", () => {
        store.setFocusedWorkload(wl.workload_uid);
        void store.runCompare();
      }),
    );
    if (focused) {
      const cmp = store.get().compare;
      if (cmp) {
        for (const cand of [...cmp.candidates].sort(
          (a, b) => (a.rank ?? 99) - (b.rank ?? 99),
        )) {
          const cls = !cand.safe ? "danger" : cand.action_id === cmp.recommended_action_id ? "active" : "";
          chips.append(
            chip(
              `${cand.safe ? "▶" : "🔒"} ${actionLabel(cand)}`,
              () => {
                if (cand.safe) store.playBranch(cand.action_id);
                else void store.showWhyNot(cand.action_id);
              },
              cls,
            ),
          );
        }
      }
    }
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
      const queues = s.snapshot?.entities.queues ?? [];
      for (const wl of pend) {
        const queue =
          queues.find((q) => q.name === wl.local_queue_id) ?? null;
        list.appendChild(card(wl, wl.workload_uid === s.focusedWorkload, queue));
      }
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
      // In replay-calibration the right rail belongs to the calibration panel;
      // the recommendation panel always yields the slot so they never overlap.
      if (s.mode === "replay-calibration") {
        root.classList.add("hidden");
        return;
      }
      const cmp = s.compare;
      if (!cmp) {
        best.innerHTML = "";
        rows.innerHTML = "";
        best.appendChild(
          el("div", "wl-meta", "Select a workload and compare safe actions."),
        );
        root.classList.remove("hidden");
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

  // Epistemic / OOD confidence badges (doc G: "confidence badges"). Learned
  // uncertainty is shown as its OWN signal, distinct from the exact-kernel gate.
  const ep = c.uncertainty.epistemic_score;
  const ood = c.uncertainty.ood_score;
  head.append(confBadge("Epi", ep, 0.3, 0.45));
  head.append(confBadge("OOD", ood, 0.2, 0.35));

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
  wrap.append(kpis);

  // Confidence band charts (p05-p95 ribbon around p50) — one per forecast channel
  // the model returned. These are the learned forecast spread, kept visually
  // separate from the exact-kernel verdict above (doc G uncertainty display).
  if (c.uncertainty.quantiles.length) {
    const bands = el("div", "rec-bands");
    for (const fc of c.uncertainty.quantiles) {
      const chart = bandChart(fc, CHANNEL_LABELS[fc.channel] ?? fc.channel);
      if (chart) bands.appendChild(chart);
    }
    if (bands.childElementCount) wrap.append(bands);
  }

  wrap.append(actions);
  row.append(wrap);
  return row;
}

/** A small coloured confidence badge: green (low), amber (mid), red (high). */
function confBadge(label: string, value: number, mid: number, hi: number): HTMLElement {
  const cls = value >= hi ? "blocked" : value >= mid ? "abstain" : "safe";
  return el("span", `badge conf ${cls}`, `${label} ${value.toFixed(2)}`);
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
