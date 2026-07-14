// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio (React overlay)
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//  The branch delta ribbon: a compact side-by-side action comparison normalised
//  to the hold baseline, so a muted viewer can read "compare futures" at a glance.

import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { fmtDelta, fmtSeconds } from "../format";
import { branchLabel } from "../labels";
import { useStore } from "../studio-context";

const COLS: Array<{ key: string; label: string; unit: string; dir: "lowerBetter" | "higherBetter"; seconds?: boolean }> = [
  { key: "queue_wait_delta_s", label: "Queue wait", unit: "s", dir: "lowerBetter", seconds: true },
  { key: "admitted_gpu_hours_delta", label: "Admitted GPU-h", unit: "", dir: "higherBetter" },
  { key: "topology_fragmentation_delta", label: "Fragmentation", unit: "", dir: "lowerBetter" },
];

export function DeltaRibbon() {
  const s = useStore();
  const cmp = s.compare;
  const show =
    !!cmp &&
    (s.mode === "counterfactual-compare" || s.mode === "decision-studio") &&
    s.pinnedActions.length > 0;
  if (!show || !cmp) return null;

  return (
    <Card className="panel-surface pointer-events-auto absolute bottom-[84px] left-1/2 z-20 flex -translate-x-1/2 gap-4 border-0 p-3">
      {s.pinnedActions.map((actionId) => {
        const c = cmp.candidates.find((x) => x.action_id === actionId);
        if (!c) return null;
        const recommended = actionId === cmp.recommended_action_id;
        const haz =
          c.uncertainty.hazards.find((h) => h.channel === "headroom_violation") ??
          c.uncertainty.hazards.find((h) => h.channel === "hotspot");
        const vp = haz?.conformal_upper ?? 0;
        const delta = c.delta_kpis as Record<string, number | undefined>;
        return (
          <div key={actionId} className="min-w-[120px]">
            <div className="mb-1.5 flex justify-between text-[11px] text-muted-foreground">
              <span>{branchLabel(c.action_id)}</span>
              <span className={recommended ? "text-safe" : c.safe ? "" : "text-blocked"}>
                {recommended ? "★ rec" : c.safe ? `#${c.rank ?? "—"}` : "🔒"}
              </span>
            </div>
            {COLS.map((def) => {
              const v = delta[def.key];
              const good = v == null ? false : def.dir === "lowerBetter" ? v <= 0 : v >= 0;
              return (
                <div key={def.key} className="flex justify-between py-0.5 text-[11.5px]">
                  <span>{def.label}</span>
                  <span className={cn(v != null && (good ? "text-safe" : "text-blocked"))}>
                    {v == null ? "—" : def.seconds ? fmtSeconds(v) : fmtDelta(v, def.unit)}
                  </span>
                </div>
              );
            })}
            <div className="flex justify-between py-0.5 text-[11.5px]">
              <span>Violation p≤</span>
              <span className={vp > 0.15 ? "text-blocked" : "text-safe"}>{vp.toFixed(2)}</span>
            </div>
          </div>
        );
      })}
    </Card>
  );
}

export default DeltaRibbon;
