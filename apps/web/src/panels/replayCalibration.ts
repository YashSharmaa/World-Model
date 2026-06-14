// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Replay & Calibration panel (doc G core_views "Replay and Calibration" →
//  panel_mode: "Error badges for queue wait, hotspot, and headroom").
//
//  It computes the PREDICTED-MINUS-REALIZED error per channel at the shared time
//  cursor and renders one calibration badge per channel, plus a divergence
//  summary that names the timestamp where predicted and realized split and the
//  delta at that point (doc G animation_rules "Replay divergence": split the
//  predicted path from the realized path at the divergence timestamp and
//  annotate the delta). The grounded scene shows the simultaneous ghost +
//  solid traces; this panel is the numeric, muted-legible read of the gap.

import { el, fmtDelta, fmtSeconds, type PanelHandle } from "./dom";
import type { Store, StoreState } from "../state/store";
import type { ReplayDTO, RolloutPoint } from "../api/client";

/** A per-channel calibration error: predicted vs realized at the cursor. */
interface ChannelError {
  key: string;
  label: string;
  predicted: number;
  realized: number;
  error: number; // predicted - realized
  unit: string;
  isSeconds?: boolean;
  /** tolerance band; |error| within tol reads as "calibrated" (green). */
  tol: number;
}

/** Linear-floor sample of a rollout's points at time t (matches the renderer). */
function sampleAt(points: RolloutPoint[], t: number): RolloutPoint | null {
  if (!points.length) return null;
  let chosen = points[0];
  for (const p of points) {
    if (p.t_s <= t) chosen = p;
    else break;
  }
  return chosen;
}

/**
 * The rack where predicted and realized hotspot diverge the most at this point —
 * the rack the calibration read should focus on (a per-rack gap, not a
 * difference of global maxima, so a fixed busy rack elsewhere never masks the
 * divergence the admit caused).
 */
function divergenceRack(pred: RolloutPoint, real: RolloutPoint): string | null {
  let best: string | null = null;
  let bestGap = -1;
  for (const rackId of Object.keys(real.rack_hotspot)) {
    const gap = Math.abs(
      (real.rack_hotspot[rackId] ?? 0) - (pred.rack_hotspot[rackId] ?? 0),
    );
    if (gap > bestGap) {
      bestGap = gap;
      best = rackId;
    }
  }
  return best;
}

/**
 * Compute predicted-minus-realized errors for the three calibration channels the
 * doc names (queue wait, hotspot, headroom) at time `t`, evaluated on the rack
 * where the two traces diverge the most.
 */
function computeErrors(replay: ReplayDTO, t: number): ChannelError[] {
  const pred = sampleAt(replay.predicted.points, t);
  const real = sampleAt(replay.realized.points, t);
  if (!pred || !real) return [];

  // Queue wait proxy: queue_depth (pending count) — predicted vs realized.
  const predWait = pred.queue_depth;
  const realWait = real.queue_depth;

  const rack = divergenceRack(pred, real);
  const predHot = rack ? (pred.rack_hotspot[rack] ?? 0) : 0;
  const realHot = rack ? (real.rack_hotspot[rack] ?? 0) : 0;

  // Headroom proxy: 1 - rack occupancy (more occupied = less headroom).
  const predHead = 1 - (rack ? (pred.rack_occupancy[rack] ?? 0) : 0);
  const realHead = 1 - (rack ? (real.rack_occupancy[rack] ?? 0) : 0);

  return [
    {
      key: "queue_wait",
      label: "Queue wait",
      predicted: predWait,
      realized: realWait,
      error: predWait - realWait,
      unit: " jobs",
      tol: 0.5,
    },
    {
      key: "hotspot",
      label: "Hotspot",
      predicted: predHot,
      realized: realHot,
      error: predHot - realHot,
      unit: "",
      tol: 0.08,
    },
    {
      key: "headroom",
      label: "Headroom",
      predicted: predHead,
      realized: realHead,
      error: predHead - realHead,
      unit: "",
      tol: 0.08,
    },
  ];
}

/**
 * Find the earliest timestamp where predicted and realized diverge beyond a
 * small tolerance on ANY of the calibration channels, and the signed hotspot
 * delta there. Returns null when the traces never split.
 */
function findDivergence(
  replay: ReplayDTO,
): { t_s: number; hotspotDelta: number } | null {
  const realPts = replay.realized.points;
  for (const rp of realPts) {
    const pp = sampleAt(replay.predicted.points, rp.t_s);
    if (!pp) continue;
    const rack = divergenceRack(pp, rp);
    const dHot = rack ? (rp.rack_hotspot[rack] ?? 0) - (pp.rack_hotspot[rack] ?? 0) : 0;
    const dOcc = rack ? (rp.rack_occupancy[rack] ?? 0) - (pp.rack_occupancy[rack] ?? 0) : 0;
    const dQ = rp.queue_depth - pp.queue_depth;
    if (Math.abs(dHot) > 0.08 || Math.abs(dOcc) > 0.05 || Math.abs(dQ) >= 1) {
      // report predicted - realized on the divergence rack (badge convention).
      return {
        t_s: rp.t_s,
        hotspotDelta: rack ? (pp.rack_hotspot[rack] ?? 0) - (rp.rack_hotspot[rack] ?? 0) : 0,
      };
    }
  }
  return null;
}

export function createReplayCalibration(parent: HTMLElement, _store: Store): PanelHandle {
  const root = el("div", "panel replay-calibration hidden");
  root.appendChild(el("div", "panel-title", "Calibration · predicted − realized"));
  const badges = el("div", "calib-badges");
  const split = el("div", "calib-split");
  root.append(badges, split);
  parent.appendChild(root);

  return {
    root,
    update(s: StoreState) {
      const show = s.mode === "replay-calibration" && !!s.replay;
      root.classList.toggle("hidden", !show);
      if (!show || !s.replay) return;

      const errors = computeErrors(s.replay, s.cursorS);
      badges.innerHTML = "";
      for (const e of errors) {
        const calibrated = Math.abs(e.error) <= e.tol;
        const cell = el("div", "calib-badge");
        const top = el("div", "calib-badge-top");
        top.append(
          el("span", "", e.label),
          el(
            "span",
            calibrated ? "badge safe" : "badge blocked",
            calibrated ? "calibrated" : "miss",
          ),
        );
        const errTxt = e.isSeconds
          ? fmtSeconds(e.error)
          : fmtDelta(e.error, e.unit, e.unit ? 0 : 2);
        const detail = el(
          "div",
          "calib-badge-detail",
          `pred ${fmtNum(e.predicted, e)} · real ${fmtNum(e.realized, e)} · err ${errTxt}`,
        );
        cell.append(top, detail);
        badges.appendChild(cell);
      }

      // Divergence split annotation (doc G replay divergence): name the timestamp
      // where the predicted path splits from realized and the delta there.
      const div = findDivergence(s.replay);
      split.innerHTML = "";
      if (div) {
        const passed = s.cursorS >= div.t_s;
        split.append(
          el("span", "calib-split-dot", passed ? "◆" : "◇"),
          el(
            "span",
            "",
            `Paths split at t+${Math.round(div.t_s)}s — hotspot under-called by ${Math.abs(div.hotspotDelta).toFixed(2)} (predicted ${div.hotspotDelta >= 0 ? "above" : "below"} realized)`,
          ),
        );
      } else {
        split.append(el("span", "", "Predicted and realized stay aligned."));
      }
    },
  };
}

function fmtNum(v: number, e: ChannelError): string {
  if (e.isSeconds) return fmtSeconds(v);
  if (e.unit) return `${Math.round(v)}${e.unit}`;
  return v.toFixed(2);
}
