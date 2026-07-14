// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio (React overlay)
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//  Scene controls: horizon selector (from snapshot.horizon_options_s), replay
//  scenario picker (from GET /replays), and the read-only context-layer toggles
//  (topology / power / cooling) wired to the live scene graph via the bridge.

import * as React from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import type { ReplayIndexEntry } from "../../api/client";
import { type DomainLayer, useStore, useStudio } from "../studio-context";

const LAYERS: DomainLayer[] = ["topology", "power", "cooling"];
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

function Group({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-[10px] uppercase tracking-[0.06em] text-muted-foreground">{label}</span>
      <div className="flex flex-wrap gap-1.5">{children}</div>
    </div>
  );
}

export function Controls() {
  const s = useStore();
  const { store, api, setLayer, initialLayers } = useStudio();
  const [catalog, setCatalog] = React.useState<ReplayIndexEntry[]>([]);
  const [layers, setLayers] = React.useState<Record<DomainLayer, boolean>>(initialLayers);

  React.useEffect(() => {
    let alive = true;
    void api.listReplays().then((idx) => {
      if (alive) setCatalog(idx.scenarios);
    });
    return () => {
      alive = false;
    };
  }, [api]);

  const horizons = s.snapshot?.horizon_options_s ?? [];
  const activeReplay = s.replay?.scenario_id ?? null;

  return (
    <Card className="panel-surface pointer-events-auto w-full border-0">
      <CardContent className="flex flex-col gap-3 p-3.5">
        <Group label="Horizon">
          {horizons.map((h) => (
            <Button
              key={h}
              size="xs"
              variant={h === s.horizonS ? "default" : "secondary"}
              onClick={() => void store.setHorizonAndCompare(h)}
            >
              {Math.round(h / 60)}m
            </Button>
          ))}
        </Group>

        {catalog.length > 0 && (
          <Group label="Replay scenario">
            {catalog.map((sc) => (
              <Button
                key={sc.scenario_id}
                size="xs"
                variant={sc.scenario_id === activeReplay ? "default" : "secondary"}
                title={`${sc.description} · ${Math.round(sc.duration_s / 60)} min`}
                onClick={() => void store.loadReplay(sc.scenario_id)}
              >
                {sc.title}
              </Button>
            ))}
          </Group>
        )}

        <Group label="Context layers">
          {LAYERS.map((layer) => (
            <Button
              key={layer}
              size="xs"
              variant={layers[layer] ? "default" : "secondary"}
              onClick={() => {
                const next = !layers[layer];
                setLayers((prev) => ({ ...prev, [layer]: next }));
                setLayer(layer, next);
              }}
            >
              {cap(layer)}
            </Button>
          ))}
        </Group>
      </CardContent>
    </Card>
  );
}

export default Controls;
