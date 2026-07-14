// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio (React overlay)
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//  The inspect drawer: clicking a rack / node / GPU / workload / domain slides in
//  a detail panel on the right WITHOUT a route change — relations, assigned
//  workloads, recent events, and a short-horizon forecast for the selection.

import * as React from "react";
import { X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { RolloutDTO, RolloutEvent, StateSnapshotDTO } from "../../api/client";
import type { Selection, StoreState } from "../../state/store";
import { fmtSeconds } from "../format";
import { useStore, useStudio } from "../studio-context";

function KV({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-3 border-b border-dashed border-border py-1 text-[12px]">
      <b className="font-medium text-muted-foreground">{label}</b>
      <span className="text-right">{value}</span>
    </div>
  );
}

function Section({ children }: { children: React.ReactNode }) {
  return <div className="mt-3 mb-1 text-[11px] uppercase tracking-[0.06em] text-primary">{children}</div>;
}

function rackOf(snap: StateSnapshotDTO, sel: Selection): string | null {
  if (sel.kind === "rack") return sel.id;
  if (sel.kind === "domain") return sel.id.replace(/^dom-/, "");
  if (sel.kind === "node") return snap.entities.nodes.find((n) => n.node_uid === sel.id)?.rack_id ?? null;
  if (sel.kind === "gpu") {
    const g = snap.entities.gpus.find((x) => x.gpu_uuid === sel.id);
    return g ? (snap.entities.nodes.find((n) => n.node_uid === g.node_uid)?.rack_id ?? null) : null;
  }
  return null;
}

function pickRollout(s: StoreState): RolloutDTO | null {
  if (s.mode === "replay-calibration" && s.replay) return s.replay.realized;
  if (s.compare) {
    const recId = s.pinnedActions[0] ?? s.compare.recommended_action_id;
    return s.compare.candidates.find((c) => c.action_id === recId)?.rollout ?? null;
  }
  return null;
}

function detailRows(snap: StateSnapshotDTO, sel: Selection): Array<[string, string]> {
  if (sel.kind === "rack") {
    const r = snap.entities.racks.find((x) => x.rack_id === sel.id);
    if (!r) return [];
    return [
      ["Block", r.block_id],
      ["Power domain", r.power_domain_id],
      ["Cooling zone", r.cooling_zone_id],
      ["Designed GPU cap", String(r.designed_gpu_cap)],
      ["Rack power", `${r.rack_power_kw.toFixed(1)} kW`],
      ["Power headroom", `${r.rack_headroom_kw.toFixed(1)} kW`],
      ["Thermal margin", `${r.inferred_thermal_margin_c.toFixed(1)} °C`],
      ["Hotspot risk (learned)", (r.hotspot_score ?? 0).toFixed(2)],
    ];
  }
  if (sel.kind === "node") {
    const n = snap.entities.nodes.find((x) => x.node_uid === sel.id);
    if (!n) return [];
    return [
      ["Rack", n.rack_id],
      ["Switch", n.switch_id],
      ["Flavor", n.flavor_id],
      ["GPU cap", String(n.gpu_cap)],
      ["Running pods", String(n.running_pods)],
      ["Ready", String(n.ready)],
      ["Schedulable", String(n.schedulable)],
      ["Health", n.node_health],
    ];
  }
  if (sel.kind === "gpu") {
    const g = snap.entities.gpus.find((x) => x.gpu_uuid === sel.id);
    if (!g) return [];
    return [
      ["Node", g.node_uid],
      ["Allocated to", g.allocated_to ?? "free"],
      ["Util", `${Math.round(g.util * 100)}%`],
      ["Temp", `${g.temp_c.toFixed(1)} °C`],
      ["Power", `${Math.round(g.power_w)} W`],
      ["Xid", g.xid_code ? String(g.xid_code) : "none"],
      ["Health (filtered)", g.health_state],
    ];
  }
  if (sel.kind === "workload") {
    const w = snap.entities.workloads.find((x) => x.workload_uid === sel.id);
    if (!w) return [];
    return [
      ["Tenant", w.tenant_id],
      ["Local queue", w.local_queue_id],
      ["Cluster queue", w.cluster_queue_id],
      ["Priority", String(w.priority)],
      ["GPU ask", String(w.gpu_req)],
      ["Topology policy", w.topology_policy],
      ["Comm class", w.comm_class],
      ["Status", w.status],
      ["Age", fmtSeconds(w.age_s)],
      ["Est remaining", fmtSeconds(w.est_remaining_s)],
      ["Ready fraction", `${Math.round(w.ready_fraction * 100)}%`],
      ["Admitted domain", w.admitted_domain_id ?? "—"],
    ];
  }
  // domain
  const d = snap.entities.topology_domains.find((x) => x.domain_id === sel.id);
  const r = snap.entities.racks.find((x) => `dom-${x.rack_id}` === sel.id);
  const rows: Array<[string, string]> = [];
  if (d) {
    rows.push(
      ["Level", d.level],
      ["Parent", d.parent_domain_id ?? "—"],
      ["Free GPU", String(d.free_gpu)],
      ["Headroom score", d.headroom_score.toFixed(2)],
      ["Hotspot score", d.hotspot_score.toFixed(2)],
    );
  }
  if (r) rows.push(["Backing rack", r.rack_id]);
  return rows;
}

export function InspectDrawer() {
  const s = useStore();
  const { store } = useStudio();
  const sel = s.selection;
  const snap = s.snapshot;
  const open = !!sel && !!snap;

  let relationRows: Array<[string, string]> = [];
  let assigned: Array<[string, string]> = [];
  let events: Array<[string, string]> = [];
  let forecast: { title: string; rows: Array<[string, string]> } | null = null;

  if (open && sel && snap) {
    const id = sel.id;
    for (const rel of snap.relations) {
      for (const [src, dst] of rel.pairs) {
        if (src === id) relationRows.push([`${rel.type} →`, dst]);
        else if (dst === id) relationRows.push([`← ${rel.type}`, src]);
      }
    }
    relationRows = relationRows.slice(0, 8);

    const rackId = rackOf(snap, sel);
    if (rackId) {
      const domainId = `dom-${rackId}`;
      assigned = snap.entities.workloads
        .filter((w) => w.admitted_domain_id === domainId)
        .map((w) => [w.workload_uid, `${w.gpu_req} GPU · ${w.status} · ${fmtSeconds(w.est_remaining_s)} left`]);
    }

    const matches = (ev: RolloutEvent) => ev.entity_id === id || ev.entity_id === rackId;
    let evs: RolloutEvent[] = [];
    if (s.mode === "replay-calibration" && s.replay) {
      evs = [...s.replay.realized.points, ...s.replay.predicted.points].flatMap((p) => p.events);
    } else if (s.compare) {
      const recId = s.pinnedActions[0] ?? s.compare.recommended_action_id;
      const rec = s.compare.candidates.find((c) => c.action_id === recId);
      evs = (rec?.rollout?.points ?? []).flatMap((p) => p.events);
    }
    events = evs
      .filter(matches)
      .filter((ev) => ev.t_s <= s.cursorS + 1)
      .sort((a, b) => b.t_s - a.t_s)
      .slice(0, 5)
      .map((ev) => [`t+${Math.round(ev.t_s)}s · ${ev.kind}`, ev.detail]);

    const rollout = pickRollout(s);
    if (rackId && rollout && rollout.points.length) {
      const previewT = Math.min(rollout.points.at(-1)?.t_s ?? 0, s.cursorS + Math.min(300, s.horizonS));
      let point = rollout.points[0];
      for (const p of rollout.points) {
        if (p.t_s <= previewT) point = p;
        else break;
      }
      forecast = {
        title: `Forecast (+${Math.round(previewT - s.cursorS)}s)`,
        rows: [
          ["Predicted occupancy", `${Math.round((point.rack_occupancy[rackId] ?? 0) * 100)}%`],
          ["Predicted hotspot", (point.rack_hotspot[rackId] ?? 0).toFixed(2)],
          ["Queue depth", String(point.queue_depth)],
          ["Epistemic / OOD", `${point.uncertainty.epistemic.toFixed(2)} / ${point.uncertainty.ood.toFixed(2)}`],
        ],
      };
    }
  }

  return (
    <aside
      className={cn(
        "panel-surface pointer-events-auto absolute right-0 top-0 bottom-0 z-40 w-[340px] overflow-auto rounded-none border-y-0 border-r-0 px-4 pt-[76px] pb-4 transition-transform duration-200 ease-out",
        open ? "translate-x-0" : "translate-x-full",
      )}
    >
      {open && sel && snap && (
        <>
          <Button size="icon" variant="ghost" className="absolute right-3 top-[72px] size-6" onClick={() => store.setSelection(null)}>
            <X className="size-4" />
          </Button>
          <div className="text-[11px] font-semibold uppercase tracking-[0.06em] text-muted-foreground">
            {sel.kind} · {sel.id}
          </div>
          <div className="mt-2">
            {detailRows(snap, sel).map(([k, v]) => (
              <KV key={k} label={k} value={v} />
            ))}
          </div>
          {relationRows.length > 0 && (
            <>
              <Section>Relations</Section>
              {relationRows.map(([k, v], i) => (
                <KV key={`${k}-${i}`} label={k} value={v} />
              ))}
            </>
          )}
          {assigned.length > 0 && (
            <>
              <Section>Assigned workloads</Section>
              {assigned.map(([k, v]) => (
                <KV key={k} label={k} value={v} />
              ))}
            </>
          )}
          {events.length > 0 && (
            <>
              <Section>Recent events</Section>
              {events.map(([k, v], i) => (
                <KV key={`${k}-${i}`} label={k} value={v} />
              ))}
            </>
          )}
          {forecast && (
            <>
              <Section>{forecast.title}</Section>
              {forecast.rows.map(([k, v]) => (
                <KV key={k} label={k} value={v} />
              ))}
            </>
          )}
        </>
      )}
    </aside>
  );
}

export default InspectDrawer;
