// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  The bottom timeline (doc G overlay). It makes temporal causality explicit:
//  one shared time cursor (0..horizon) drives ALL branches and the realized
//  overlay, with play/pause, event pins, and a predicted-vs-realized toggle.
//
//  Timeline authority (doc G): the cursor is the single source of scene time.
//  All scene motion is driven by it; the renderer interpolates between backend
//  rollout points but never free-runs its own simulation.

import { el, type PanelHandle } from "./dom";
import type { Store, StoreState } from "../state/store";
import type { ReplayDTO, RolloutPoint } from "../api/client";

/**
 * Earliest timestamp where the realized hotspot/occupancy/queue diverges from
 * the predicted branch beyond a small tolerance (doc G replay divergence). The
 * timeline marks this with a fork glyph. Returns null when traces never split.
 */
function divergenceTimestamp(replay: ReplayDTO): number | null {
  const predAt = (t: number): RolloutPoint | null => {
    let chosen: RolloutPoint | null = null;
    for (const p of replay.predicted.points) {
      if (p.t_s <= t) chosen = p;
      else break;
    }
    return chosen;
  };
  // Per-rack gap so a fixed busy rack elsewhere never masks the admit's divergence.
  const maxGap = (a: Record<string, number>, b: Record<string, number>): number => {
    let g = 0;
    for (const k of Object.keys(b)) g = Math.max(g, Math.abs((b[k] ?? 0) - (a[k] ?? 0)));
    return g;
  };
  for (const rp of replay.realized.points) {
    const pp = predAt(rp.t_s);
    if (!pp) continue;
    const dHot = maxGap(pp.rack_hotspot, rp.rack_hotspot);
    const dOcc = maxGap(pp.rack_occupancy, rp.rack_occupancy);
    const dQ = Math.abs(rp.queue_depth - pp.queue_depth);
    if (dHot > 0.08 || dOcc > 0.05 || dQ >= 1) {
      return rp.t_s;
    }
  }
  return null;
}

export interface TimelineHandle extends PanelHandle {
  /** which trace the replay overlay shows. */
  getTrace(): "predicted" | "realized" | "both";
}

export function createTimeline(parent: HTMLElement, store: Store): TimelineHandle {
  const root = el("div", "panel timeline");

  const playBtn = el("button", "tl-btn", "▶") as HTMLButtonElement;
  const resetBtn = el("button", "tl-btn", "⟲") as HTMLButtonElement;

  const track = el("div", "tl-track");
  const rail = el("div", "tl-rail");
  const fill = el("div", "tl-fill");
  const cursor = el("div", "tl-cursor");
  track.append(rail, fill, cursor);
  const pinsLayer = el("div", "");
  track.append(pinsLayer);

  const toggle = el("div", "tl-toggle");
  let trace: "predicted" | "realized" | "both" = "both";
  const tPred = el("button", "chip active", "Predicted") as HTMLButtonElement;
  const tReal = el("button", "chip", "Realized") as HTMLButtonElement;
  toggle.append(tPred, tReal);

  const timeLabel = el("div", "tl-time", "0:00 / 0:00");

  root.append(playBtn, resetBtn, track, toggle, timeLabel);
  parent.appendChild(root);

  // ── interactions ──────────────────────────────────────────────────────
  playBtn.addEventListener("click", () => store.setPlaying(!store.get().playing));
  resetBtn.addEventListener("click", () => {
    store.setCursor(0);
    store.setPlaying(false);
  });
  track.addEventListener("click", (ev) => {
    const rect = track.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
    store.setCursor(frac * store.get().horizonS);
    store.setPlaying(false);
  });
  function setTrace(t: "predicted" | "realized" | "both"): void {
    trace = t;
    tPred.classList.toggle("active", t !== "realized");
    tReal.classList.toggle("active", t !== "predicted");
  }
  tPred.addEventListener("click", () => setTrace(trace === "predicted" ? "both" : "predicted"));
  tReal.addEventListener("click", () => setTrace(trace === "realized" ? "both" : "realized"));

  function fmt(s: number): string {
    const m = Math.floor(s / 60);
    const r = Math.round(s % 60);
    return `${m}:${r.toString().padStart(2, "0")}`;
  }

  return {
    root,
    getTrace: () => trace,
    update(s: StoreState) {
      const horizon = Math.max(1, s.horizonS);
      const frac = s.cursorS / horizon;
      fill.style.width = `${frac * 100}%`;
      cursor.style.left = `${frac * 100}%`;
      playBtn.textContent = s.playing ? "⏸" : "▶";
      timeLabel.textContent = `${fmt(s.cursorS)} / ${fmt(horizon)}`;

      // Event pins from rollouts / decision windows.
      pinsLayer.innerHTML = "";
      const addPin = (tS: number, label: string, realized: boolean) => {
        const pin = el("div", `tl-pin ${realized ? "realized" : ""}`);
        pin.style.left = `${(tS / horizon) * 100}%`;
        pin.title = label;
        pin.addEventListener("click", (e) => {
          e.stopPropagation();
          store.setCursor(tS);
          store.setPlaying(false);
        });
        pinsLayer.appendChild(pin);
      };

      if (s.replay) {
        for (const w of s.replay.decision_windows) addPin(w.t_s, w.label, false);
        for (const p of s.replay.realized.points)
          for (const ev of p.events) addPin(ev.t_s, `realized: ${ev.detail}`, true);
        for (const p of s.replay.predicted.points)
          for (const ev of p.events) addPin(ev.t_s, `predicted: ${ev.detail}`, false);

        // Divergence split (doc G replay divergence): a forked marker where the
        // predicted path splits from the realized path, only meaningful when both
        // traces are shown. The marker grows once the cursor passes it.
        if (trace === "both") {
          const dt = divergenceTimestamp(s.replay);
          if (dt != null) {
            const passed = s.cursorS >= dt;
            const fork = el("div", `tl-split ${passed ? "active" : ""}`);
            fork.style.left = `${(dt / horizon) * 100}%`;
            fork.title = `Predicted vs realized split @ t+${Math.round(dt)}s`;
            fork.textContent = "⋔";
            fork.addEventListener("click", (e) => {
              e.stopPropagation();
              store.setCursor(dt);
              store.setPlaying(false);
            });
            pinsLayer.appendChild(fork);
          }
        }
      } else if (s.compare) {
        const rec = s.compare.candidates.find(
          (c) => c.action_id === s.compare?.recommended_action_id,
        );
        for (const p of rec?.rollout?.points ?? [])
          for (const ev of p.events) addPin(ev.t_s, ev.detail, false);
      }

      // The predicted/realized toggle is only meaningful in replay mode.
      toggle.style.display = s.mode === "replay-calibration" ? "flex" : "none";
    },
  };
}
