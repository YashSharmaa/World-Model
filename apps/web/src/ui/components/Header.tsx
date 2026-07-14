// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio (React overlay)
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//  The top header bar: ONE aligned, vertically-centred band that contains the
//  brand, the centred mode tabs + live indicator, the provenance chips
//  (model / policy / state-hash / telemetry freshness), the Play-demo CTA, and
//  the light/dark theme toggle — so every control reads as part of the header.

import { Moon, Play, Sun } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { toggleTheme } from "../../theme";
import type { StudioMode } from "../../state/store";
import { useStore, useStudio, useThemeName } from "../studio-context";

const MODE_TABS: Array<{ id: StudioMode; label: string }> = [
  { id: "decision-studio", label: "Decision Studio" },
  { id: "counterfactual-compare", label: "Counterfactual Compare" },
  { id: "replay-calibration", label: "Replay & Calibration" },
  { id: "inspect-drawer", label: "Inspect" },
];

function ProvChip({ children, mono = false, title }: { children: React.ReactNode; mono?: boolean; title?: string }) {
  return (
    <span
      title={title}
      className={cn(
        "rounded-full border border-border bg-secondary px-2.5 py-1 text-[11px] text-muted-foreground",
        mono && "max-w-[150px] truncate font-mono",
      )}
    >
      {children}
    </span>
  );
}

export function Header() {
  const s = useStore();
  const { store, playDemo } = useStudio();
  const theme = useThemeName();
  const snap = s.snapshot;

  const onTab = (id: StudioMode) => {
    store.setMode(id);
    if (id === "replay-calibration") void store.loadReplay("golden-path-admit-a");
    else if (id === "counterfactual-compare" && !store.get().compare) void store.runCompare();
  };

  const age = Math.round(snap?.freshness.gpu_telemetry_age_s ?? 0);
  const srcHealth = snap?.freshness.worst_source_health ?? "—";

  return (
    <header className="panel-surface pointer-events-auto absolute inset-x-3 top-3 z-30 flex items-center gap-3 px-4 py-2">
      {/* brand */}
      <div className="flex min-w-0 shrink-0 flex-col leading-tight">
        <span className="text-sm font-semibold">GPU-Center World Model</span>
        <span className="text-[11px] text-muted-foreground">
          Decision Studio · shadow-mode (offline replay)
        </span>
      </div>

      {/* centred mode tabs + live indicator */}
      <div className="flex flex-1 items-center justify-center gap-3">
        <Tabs value={s.mode} onValueChange={(v) => onTab(v as StudioMode)}>
          <TabsList>
            {MODE_TABS.map((t) => (
              <TabsTrigger key={t.id} value={t.id}>
                {t.label}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
        <span
          title={s.health}
          className={cn(
            "flex items-center gap-1 text-[11px] font-semibold",
            s.backendOnline ? "text-safe" : "text-alt",
          )}
        >
          <span className="text-[8px]">●</span>
          {s.backendOnline ? "live" : "mock"}
        </span>
      </div>

      {/* provenance chips + actions */}
      <div className="flex shrink-0 items-center gap-2">
        <ProvChip>model {snap?.model_version ?? "—"}</ProvChip>
        <ProvChip>policy {snap?.policy_version ?? "—"}</ProvChip>
        <ProvChip mono title={snap?.state_hash ?? undefined}>
          {snap?.state_hash ?? "—"}
        </ProvChip>
        <ProvChip>
          tel {age}s · {srcHealth}
        </ProvChip>
        <Button size="sm" onClick={playDemo} className="gap-1.5">
          <Play className="size-3.5" /> Play demo
        </Button>
        <Button
          size="icon"
          variant="outline"
          className="size-8 rounded-full"
          title="Toggle light / dark theme"
          aria-label="Toggle light or dark theme"
          onClick={() => toggleTheme()}
        >
          {theme === "light" ? <Moon className="size-4" /> : <Sun className="size-4" />}
        </Button>
      </div>
    </header>
  );
}

export default Header;
