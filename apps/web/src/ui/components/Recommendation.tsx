// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio (React overlay)
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//  Recommendation panel (right): translates the world-model output into ONE
//  recommendation while keeping the safety GATE explicit — exact blockers and
//  learned risk/uncertainty are shown SEPARATELY, never blended into one score.

import { Lock } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { CandidateResultDTO, QuantileForecastDTO } from "../../api/client";
import { fmtDelta, fmtSeconds } from "../format";
import { actionLabel } from "../labels";
import { useStore, useStudio } from "../studio-context";

const KPI_LABELS: Record<string, [string, string, "lowerBetter" | "higherBetter"]> = {
  queue_wait_delta_s: ["Queue wait", "s", "lowerBetter"],
  admitted_gpu_hours_delta: ["Admitted GPU-h", "", "higherBetter"],
  topology_fragmentation_delta: ["Fragmentation", "", "lowerBetter"],
  rack_thermal_headroom_c: ["Thermal headroom", "°C", "higherBetter"],
  completion_time_delta_s: ["Completion", "s", "lowerBetter"],
  cross_rack_rate: ["Cross-rack", "", "lowerBetter"],
};

const CHANNEL_LABELS: Record<string, string> = {
  queue_wait_delta_s: "Queue wait band",
  rack_thermal_headroom_c: "Thermal headroom band",
  admitted_gpu_hours_delta: "Admitted GPU-h band",
  topology_fragmentation_delta: "Fragmentation band",
  completion_time_delta_s: "Completion band",
  cross_rack_rate: "Cross-rack band",
};

export function Recommendation() {
  const s = useStore();
  if (s.mode === "replay-calibration") return null;
  const cmp = s.compare;

  return (
    <Card className="panel-surface pointer-events-auto absolute right-3 top-[84px] z-20 max-h-[64vh] w-80 overflow-auto border-0 p-0">
      <CardHeader className="pb-2">
        <CardTitle>Recommendation</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2.5">
        {!cmp ? (
          <p className="rounded-md border border-safe/40 bg-[var(--safe-bg)] p-2.5 text-[12px] text-muted-foreground">
            Select a workload and compare safe actions.
          </p>
        ) : (
          <>
            <BestSummary cmp={cmp} />
            {[...cmp.candidates]
              .sort((a, b) => (a.safe !== b.safe ? (a.safe ? -1 : 1) : (a.rank ?? 99) - (b.rank ?? 99)))
              .map((c) => (
                <Row key={c.action_id} c={c} recommendedId={cmp.recommended_action_id} />
              ))}
          </>
        )}
      </CardContent>
    </Card>
  );
}

function BestSummary({ cmp }: { cmp: NonNullable<ReturnType<typeof useStore>["compare"]> }) {
  if (cmp.abstain) {
    return (
      <div className="rounded-md border border-dashed border-lowconf p-2.5">
        <Badge variant="abstain">ABSTAIN</Badge>
        <p className="mt-1 text-[11px] text-muted-foreground">{cmp.abstain_reason ?? "low confidence"}</p>
      </div>
    );
  }
  const rec = cmp.candidates.find((c) => c.action_id === cmp.recommended_action_id);
  return (
    <div className="rounded-md border border-safe/50 bg-[var(--safe-bg)] p-2.5">
      <div className="font-semibold">{rec ? `Recommended: ${actionLabel(rec.action_id)}` : "No safe action"}</div>
      <div className="text-[11px] text-muted-foreground">
        {rec ? `horizon ${Math.round(cmp.horizon_s / 60)} min · baseline = hold` : "all candidates filtered"}
      </div>
    </div>
  );
}

function ConfBadge({ label, value, mid, hi }: { label: string; value: number; mid: number; hi: number }) {
  const variant = value >= hi ? "blocked" : value >= mid ? "abstain" : "safe";
  return (
    <Badge variant={variant} className="font-mono">
      {label} {value.toFixed(2)}
    </Badge>
  );
}

function Row({ c, recommendedId }: { c: CandidateResultDTO; recommendedId: string | null }) {
  const { store } = useStudio();
  const recommended = c.action_id === recommendedId;
  const abstaining = c.safe && c.uncertainty.abstain;
  const hotspot = c.uncertainty.hazards.find((h) => h.channel === "hotspot");

  return (
    <div
      className={cn(
        "rounded-md border border-border p-2.5",
        !c.safe && "border-blocked",
        recommended && "border-safe",
        abstaining && "border-dashed border-lowconf",
      )}
    >
      <div className="flex items-center gap-2">
        <span className="w-6 font-bold">{c.safe ? `#${c.rank ?? "—"}` : <Lock className="size-3.5 text-blocked" />}</span>
        <span className="flex-1">{actionLabel(c.action_id)}</span>
        {!c.safe ? (
          <Badge variant="blocked">BLOCKED</Badge>
        ) : abstaining ? (
          <Badge variant="abstain">ABSTAIN</Badge>
        ) : (
          <Badge variant="safe">SAFE</Badge>
        )}
        <ConfBadge label="Epi" value={c.uncertainty.epistemic_score} mid={0.3} hi={0.45} />
        <ConfBadge label="OOD" value={c.uncertainty.ood_score} mid={0.2} hi={0.35} />
      </div>

      {/* KPI deltas (learned forecast medians) */}
      <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
        {Object.entries(c.delta_kpis).map(([ch, val]) => {
          const meta = KPI_LABELS[ch];
          if (!meta || val == null) return null;
          const [label, unit, dir] = meta;
          const good = dir === "lowerBetter" ? val <= 0 : val >= 0;
          return (
            <div key={ch} className="flex justify-between">
              <span>{label}</span>
              <span className={good ? "text-safe" : "text-blocked"}>
                {ch.endsWith("_s") ? fmtSeconds(val) : fmtDelta(val, unit)}
              </span>
            </div>
          );
        })}
        {hotspot && (
          <div className="flex justify-between">
            <span>Hotspot p≤</span>
            <span className={hotspot.conformal_upper > 0.15 ? "text-blocked" : "text-safe"}>
              {hotspot.conformal_upper.toFixed(2)}
            </span>
          </div>
        )}
      </div>

      {/* p05–p95 confidence band charts */}
      {c.uncertainty.quantiles.length > 0 && (
        <div className="mt-2 flex flex-col gap-1.5">
          {c.uncertainty.quantiles.map((fc) => (
            <BandChart key={`${fc.channel}-${fc.horizon_s}`} forecast={fc} label={CHANNEL_LABELS[fc.channel] ?? fc.channel} />
          ))}
        </div>
      )}

      <div className="mt-2 flex gap-1.5">
        {c.safe ? (
          <Button size="xs" variant="secondary" onClick={() => store.togglePinned(c.action_id)}>
            Pin
          </Button>
        ) : (
          <Button size="xs" variant="secondary" className="text-blocked" onClick={() => void store.showWhyNot(c.action_id)}>
            Why not?
          </Button>
        )}
      </div>
    </div>
  );
}

function BandChart({ forecast, label }: { forecast: QuantileForecastDTO; label: string }) {
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
  const W = 200;
  const H = 22;
  const padX = 2;
  const x = (v: number) => padX + ((v - lo) / span) * (W - padX * 2);
  const midY = H / 2;

  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex justify-between text-[10px] text-muted-foreground">
        <span>{label}</span>
        <span className="font-mono text-predicted">±{Math.round((p95 - p05) / 2)}</span>
      </div>
      <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="block">
        <rect x={x(p05)} y={midY - 7} width={Math.max(1, x(p95) - x(p05))} height={14} rx={3} fill="var(--predicted)" fillOpacity={0.18} />
        <rect x={x(p10)} y={midY - 5} width={Math.max(1, x(p90) - x(p10))} height={10} rx={2} fill="var(--predicted)" fillOpacity={0.34} />
        <line x1={x(p50)} y1={midY - 8} x2={x(p50)} y2={midY + 8} stroke="var(--grounded)" strokeWidth={2} />
      </svg>
    </div>
  );
}

export default Recommendation;
