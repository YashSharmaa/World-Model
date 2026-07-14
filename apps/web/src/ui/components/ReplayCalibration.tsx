// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio (React overlay)
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//  Replay & Calibration panel: predicted-minus-realized error per channel at the
//  shared cursor, plus a divergence summary naming where the traces split.

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ReplayDTO, RolloutPoint } from "../../api/client";
import { fmtDelta } from "../format";
import { useStore } from "../studio-context";

interface ChannelError {
  key: string;
  label: string;
  predicted: number;
  realized: number;
  error: number;
  unit: string;
  tol: number;
}

function sampleAt(points: RolloutPoint[], t: number): RolloutPoint | null {
  if (!points.length) return null;
  let chosen = points[0];
  for (const p of points) {
    if (p.t_s <= t) chosen = p;
    else break;
  }
  return chosen;
}

function divergenceRack(pred: RolloutPoint, real: RolloutPoint): string | null {
  let best: string | null = null;
  let bestGap = -1;
  for (const rackId of Object.keys(real.rack_hotspot)) {
    const gap = Math.abs((real.rack_hotspot[rackId] ?? 0) - (pred.rack_hotspot[rackId] ?? 0));
    if (gap > bestGap) {
      bestGap = gap;
      best = rackId;
    }
  }
  return best;
}

function computeErrors(replay: ReplayDTO, t: number): ChannelError[] {
  const pred = sampleAt(replay.predicted.points, t);
  const real = sampleAt(replay.realized.points, t);
  if (!pred || !real) return [];
  const rack = divergenceRack(pred, real);
  const predHot = rack ? (pred.rack_hotspot[rack] ?? 0) : 0;
  const realHot = rack ? (real.rack_hotspot[rack] ?? 0) : 0;
  const predHead = 1 - (rack ? (pred.rack_occupancy[rack] ?? 0) : 0);
  const realHead = 1 - (rack ? (real.rack_occupancy[rack] ?? 0) : 0);
  return [
    { key: "queue_wait", label: "Queue wait", predicted: pred.queue_depth, realized: real.queue_depth, error: pred.queue_depth - real.queue_depth, unit: " jobs", tol: 0.5 },
    { key: "hotspot", label: "Hotspot", predicted: predHot, realized: realHot, error: predHot - realHot, unit: "", tol: 0.08 },
    { key: "headroom", label: "Headroom", predicted: predHead, realized: realHead, error: predHead - realHead, unit: "", tol: 0.08 },
  ];
}

function findDivergence(replay: ReplayDTO): { t_s: number; hotspotDelta: number } | null {
  for (const rp of replay.realized.points) {
    const pp = sampleAt(replay.predicted.points, rp.t_s);
    if (!pp) continue;
    const rack = divergenceRack(pp, rp);
    const dHot = rack ? (rp.rack_hotspot[rack] ?? 0) - (pp.rack_hotspot[rack] ?? 0) : 0;
    const dOcc = rack ? (rp.rack_occupancy[rack] ?? 0) - (pp.rack_occupancy[rack] ?? 0) : 0;
    const dQ = rp.queue_depth - pp.queue_depth;
    if (Math.abs(dHot) > 0.08 || Math.abs(dOcc) > 0.05 || Math.abs(dQ) >= 1) {
      return { t_s: rp.t_s, hotspotDelta: rack ? (pp.rack_hotspot[rack] ?? 0) - (rp.rack_hotspot[rack] ?? 0) : 0 };
    }
  }
  return null;
}

function fmtNum(v: number, e: ChannelError): string {
  if (e.unit) return `${Math.round(v)}${e.unit}`;
  return v.toFixed(2);
}

export function ReplayCalibration() {
  const s = useStore();
  if (s.mode !== "replay-calibration" || !s.replay) return null;
  const errors = computeErrors(s.replay, s.cursorS);
  const div = findDivergence(s.replay);

  return (
    <Card className="panel-surface pointer-events-auto absolute right-3 top-[84px] z-20 max-h-[64vh] w-80 overflow-auto border-0 p-0">
      <CardHeader className="pb-2">
        <CardTitle>Calibration · predicted − realized</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {errors.map((e) => {
          const calibrated = Math.abs(e.error) <= e.tol;
          return (
            <div key={e.key} className="rounded-md border border-border p-2.5">
              <div className="flex items-center justify-between font-semibold">
                <span>{e.label}</span>
                <Badge variant={calibrated ? "safe" : "blocked"}>{calibrated ? "calibrated" : "miss"}</Badge>
              </div>
              <div className="mt-1 font-mono text-[11px] text-muted-foreground">
                pred {fmtNum(e.predicted, e)} · real {fmtNum(e.realized, e)} · err{" "}
                {e.unit ? fmtDelta(e.error, e.unit, 0) : e.error.toFixed(2)}
              </div>
            </div>
          );
        })}
        <div className="mt-1 flex items-start gap-2 text-[11.5px]">
          {div ? (
            <>
              <span className="text-blocked">{s.cursorS >= div.t_s ? "◆" : "◇"}</span>
              <span>
                Paths split at t+{Math.round(div.t_s)}s — hotspot under-called by{" "}
                {Math.abs(div.hotspotDelta).toFixed(2)} (predicted {div.hotspotDelta >= 0 ? "above" : "below"} realized)
              </span>
            </>
          ) : (
            <span>Predicted and realized stay aligned.</span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default ReplayCalibration;
