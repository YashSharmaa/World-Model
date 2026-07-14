// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio (React overlay)
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//  The bottom timeline: ONE shared cursor (0..horizon) drives all branches and
//  the realized overlay, with play/pause, event pins, a divergence fork, and a
//  predicted-vs-realized toggle. Timeline authority (doc G): the cursor is the
//  single source of scene time; the renderer only interpolates between points.

import * as React from "react";
import { Pause, Play, RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";
import type { ReplayDTO, RolloutPoint } from "../../api/client";
import { type ReplayTrace, useStore, useStudio } from "../studio-context";
import { fmtClock } from "../format";

function predAt(replay: ReplayDTO, t: number): RolloutPoint | null {
  let chosen: RolloutPoint | null = null;
  for (const p of replay.predicted.points) {
    if (p.t_s <= t) chosen = p;
    else break;
  }
  return chosen;
}

function maxGap(a: Record<string, number>, b: Record<string, number>): number {
  let g = 0;
  for (const k of Object.keys(b)) g = Math.max(g, Math.abs((b[k] ?? 0) - (a[k] ?? 0)));
  return g;
}

function divergenceTimestamp(replay: ReplayDTO): number | null {
  for (const rp of replay.realized.points) {
    const pp = predAt(replay, rp.t_s);
    if (!pp) continue;
    if (
      maxGap(pp.rack_hotspot, rp.rack_hotspot) > 0.08 ||
      maxGap(pp.rack_occupancy, rp.rack_occupancy) > 0.05 ||
      Math.abs(rp.queue_depth - pp.queue_depth) >= 1
    ) {
      return rp.t_s;
    }
  }
  return null;
}

interface Pin {
  t: number;
  label: string;
  realized: boolean;
}

export function Timeline() {
  const s = useStore();
  const { store, setTrace } = useStudio();
  const [trace, setTraceState] = React.useState<ReplayTrace>("both");
  const horizon = Math.max(1, s.horizonS);

  const applyTrace = (t: ReplayTrace) => {
    setTraceState(t);
    setTrace(t);
  };

  const pins: Pin[] = [];
  if (s.replay) {
    for (const w of s.replay.decision_windows) pins.push({ t: w.t_s, label: w.label, realized: false });
    for (const p of s.replay.realized.points) for (const ev of p.events) pins.push({ t: ev.t_s, label: `realized: ${ev.detail}`, realized: true });
    for (const p of s.replay.predicted.points) for (const ev of p.events) pins.push({ t: ev.t_s, label: `predicted: ${ev.detail}`, realized: false });
  } else if (s.compare) {
    const rec = s.compare.candidates.find((c) => c.action_id === s.compare?.recommended_action_id);
    for (const p of rec?.rollout?.points ?? []) for (const ev of p.events) pins.push({ t: ev.t_s, label: ev.detail, realized: false });
  }
  const fork = s.replay && trace === "both" ? divergenceTimestamp(s.replay) : null;

  return (
    <Card className="panel-surface pointer-events-auto absolute inset-x-3 bottom-3 z-20 flex items-center gap-3 border-0 px-4 py-2">
      <Button size="icon" variant="secondary" className="size-8" onClick={() => store.setPlaying(!s.playing)}>
        {s.playing ? <Pause className="size-4" /> : <Play className="size-4" />}
      </Button>
      <Button
        size="icon"
        variant="secondary"
        className="size-8"
        onClick={() => {
          store.setCursor(0);
          store.setPlaying(false);
        }}
      >
        <RotateCcw className="size-4" />
      </Button>

      <div className="relative flex-1">
        <Slider
          min={0}
          max={horizon}
          step={1}
          value={[Math.min(s.cursorS, horizon)]}
          onValueChange={(v) => {
            store.setCursor(v[0] ?? 0);
            store.setPlaying(false);
          }}
        />
        {/* event pins + divergence fork (decorative, click to seek) */}
        <div className="pointer-events-none absolute inset-x-0 top-0 h-full">
          {pins.map((p, i) => (
            <button
              key={i}
              title={p.label}
              onClick={() => {
                store.setCursor(p.t);
                store.setPlaying(false);
              }}
              className={cn(
                "pointer-events-auto absolute top-1 size-2.5 -translate-x-1/2 cursor-pointer rounded-full border border-card",
                p.realized ? "bg-blocked" : "bg-predicted",
              )}
              style={{ left: `${(p.t / horizon) * 100}%` }}
            />
          ))}
          {fork != null && (
            <button
              title={`Predicted vs realized split @ t+${Math.round(fork)}s`}
              onClick={() => {
                store.setCursor(fork);
                store.setPlaying(false);
              }}
              className={cn(
                "pointer-events-auto absolute -top-1 -translate-x-1/2 cursor-pointer text-[15px] leading-none",
                s.cursorS >= fork ? "font-bold text-blocked" : "text-lowconf",
              )}
              style={{ left: `${(fork / horizon) * 100}%` }}
            >
              ⋔
            </button>
          )}
        </div>
      </div>

      {s.mode === "replay-calibration" && (
        <div className="flex gap-1.5">
          <Button
            size="xs"
            variant={trace !== "realized" ? "default" : "secondary"}
            onClick={() => applyTrace(trace === "predicted" ? "both" : "predicted")}
          >
            Predicted
          </Button>
          <Button
            size="xs"
            variant={trace !== "predicted" ? "default" : "secondary"}
            onClick={() => applyTrace(trace === "realized" ? "both" : "realized")}
          >
            Realized
          </Button>
        </div>
      )}

      <span className="min-w-[86px] text-right text-[11px] text-muted-foreground">
        {fmtClock(s.cursorS)} / {fmtClock(horizon)}
      </span>
    </Card>
  );
}

export default Timeline;
