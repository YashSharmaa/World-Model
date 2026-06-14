// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  The inspect drawer (doc G overlay + mode). Clicking a rack / node / GPU /
//  workload / domain slides in a detail panel on the right WITHOUT a route
//  change — the camera stays fixed and the selected entity is highlighted, so
//  the scene context is never lost (doc G "Inspect Drawer").

import { el, fmtSeconds, type PanelHandle } from "./dom";
import type { Store, StoreState, Selection } from "../state/store";
import type {
  GPUDTO,
  NodeDTO,
  RackDTO,
  RolloutEvent,
  StateSnapshotDTO,
  WorkloadDTO,
} from "../api/client";

export function createInspectDrawer(parent: HTMLElement, store: Store): PanelHandle {
  const root = el("div", "panel inspect-drawer");
  parent.appendChild(root);

  function kv(label: string, value: string): HTMLElement {
    const row = el("div", "kv");
    row.append(el("b", "", label), el("span", "", value));
    return row;
  }

  function sectionTitle(text: string): HTMLElement {
    return el("div", "drawer-section", text);
  }

  function render(s: StoreState): void {
    root.innerHTML = "";
    const sel = s.selection;
    const snap = s.snapshot;
    if (!sel || !snap) return;

    const close = el("button", "close-x", "×");
    close.addEventListener("click", () => store.setSelection(null));
    root.appendChild(close);
    root.appendChild(el("div", "panel-title", `${sel.kind} · ${sel.id}`));

    if (sel.kind === "rack") renderRack(snap.entities.racks.find((r) => r.rack_id === sel.id));
    else if (sel.kind === "node") renderNode(snap.entities.nodes.find((n) => n.node_uid === sel.id));
    else if (sel.kind === "gpu") renderGpu(snap.entities.gpus.find((g) => g.gpu_uuid === sel.id));
    else if (sel.kind === "workload")
      renderWorkload(snap.entities.workloads.find((w) => w.workload_uid === sel.id));
    else if (sel.kind === "domain")
      renderDomain(sel.id, snap.entities.racks.find((r) => `dom-${r.rack_id}` === sel.id));

    // Doc G inspect-drawer contents: relation list, recent events, assigned
    // workloads, short-horizon forecast — for racks/nodes/GPUs/workloads/domains.
    renderRelations(snap, sel);
    renderAssignedWorkloads(snap, sel);
    renderRecentEvents(s, sel);
    renderForecast(s, sel);
  }

  // ── shared sections (doc G inspect-drawer "contents") ─────────────────────

  /** Resolve the rack a selection lives in, used to scope relations/workloads. */
  function rackOf(snap: StateSnapshotDTO, sel: Selection): string | null {
    if (sel.kind === "rack") return sel.id;
    if (sel.kind === "domain") return sel.id.replace(/^dom-/, "");
    if (sel.kind === "node")
      return snap.entities.nodes.find((n) => n.node_uid === sel.id)?.rack_id ?? null;
    if (sel.kind === "gpu") {
      const g = snap.entities.gpus.find((x) => x.gpu_uuid === sel.id);
      return g
        ? (snap.entities.nodes.find((n) => n.node_uid === g.node_uid)?.rack_id ?? null)
        : null;
    }
    return null;
  }

  /** Relation list: every snapshot relation whose src/dst id touches the entity. */
  function renderRelations(snap: StateSnapshotDTO, sel: Selection): void {
    const id = sel.id;
    const rows: HTMLElement[] = [];
    for (const rel of snap.relations) {
      for (const [src, dst] of rel.pairs) {
        if (src === id) rows.push(kv(`${rel.type} →`, dst));
        else if (dst === id) rows.push(kv(`← ${rel.type}`, src));
      }
    }
    if (!rows.length) return;
    root.appendChild(sectionTitle("Relations"));
    // cap the list so a node's many CONNECTED_TO rows do not flood the drawer.
    for (const r of rows.slice(0, 8)) root.appendChild(r);
  }

  /** Assigned workloads: jobs admitted to / running on the selected rack/domain. */
  function renderAssignedWorkloads(snap: StateSnapshotDTO, sel: Selection): void {
    const rackId = rackOf(snap, sel);
    if (!rackId) return;
    const domainId = `dom-${rackId}`;
    const assigned = snap.entities.workloads.filter(
      (w) => w.admitted_domain_id === domainId,
    );
    if (!assigned.length) return;
    root.appendChild(sectionTitle("Assigned workloads"));
    for (const w of assigned)
      root.appendChild(
        kv(w.workload_uid, `${w.gpu_req} GPU · ${w.status} · ${fmtSeconds(w.est_remaining_s)} left`),
      );
  }

  /** Recent events: rollout/replay events at or before the cursor for this entity. */
  function renderRecentEvents(s: StoreState, sel: Selection): void {
    const snap = s.snapshot;
    if (!snap) return;
    const rackId = rackOf(snap, sel);
    const matches = (ev: RolloutEvent): boolean =>
      ev.entity_id === sel.id ||
      ev.entity_id === rackId ||
      (sel.kind === "workload" && ev.entity_id === sel.id);

    // Pull events from whichever rollout currently drives the scene.
    let events: RolloutEvent[] = [];
    if (s.mode === "replay-calibration" && s.replay) {
      events = [...s.replay.realized.points, ...s.replay.predicted.points]
        .flatMap((p) => p.events);
    } else if (s.compare) {
      const recId = s.pinnedActions[0] ?? s.compare.recommended_action_id;
      const rec = s.compare.candidates.find((c) => c.action_id === recId);
      events = (rec?.rollout?.points ?? []).flatMap((p) => p.events);
    }
    const recent = events
      .filter(matches)
      .filter((ev) => ev.t_s <= s.cursorS + 1)
      .sort((a, b) => b.t_s - a.t_s)
      .slice(0, 5);
    if (!recent.length) return;
    root.appendChild(sectionTitle("Recent events"));
    for (const ev of recent)
      root.appendChild(kv(`t+${Math.round(ev.t_s)}s · ${ev.kind}`, ev.detail));
  }

  /** Short-horizon forecast preview: the sampled rollout point for this entity. */
  function renderForecast(s: StoreState, sel: Selection): void {
    const snap = s.snapshot;
    if (!snap) return;
    const rackId = rackOf(snap, sel);
    if (!rackId) return;

    // Prefer the live driving rollout; sample a short-horizon point ahead.
    let rollout = null as ReturnType<typeof pickRollout> | null;
    rollout = pickRollout(s);
    if (!rollout) return;
    const previewT = Math.min(
      rollout.points.at(-1)?.t_s ?? 0,
      s.cursorS + Math.min(300, s.horizonS), // ~5 min lookahead
    );
    let point = rollout.points[0];
    for (const p of rollout.points) {
      if (p.t_s <= previewT) point = p;
      else break;
    }
    if (!point) return;
    root.appendChild(sectionTitle(`Forecast (+${Math.round(previewT - s.cursorS)}s)`));
    root.append(
      kv("Predicted occupancy", `${Math.round((point.rack_occupancy[rackId] ?? 0) * 100)}%`),
      kv("Predicted hotspot", (point.rack_hotspot[rackId] ?? 0).toFixed(2)),
      kv("Queue depth", String(point.queue_depth)),
      kv("Epistemic / OOD", `${point.uncertainty.epistemic.toFixed(2)} / ${point.uncertainty.ood.toFixed(2)}`),
    );
  }

  /** The rollout currently driving the scene (replay realized, or fronted branch). */
  function pickRollout(s: StoreState) {
    if (s.mode === "replay-calibration" && s.replay) return s.replay.realized;
    if (s.compare) {
      const recId = s.pinnedActions[0] ?? s.compare.recommended_action_id;
      return s.compare.candidates.find((c) => c.action_id === recId)?.rollout ?? null;
    }
    return null;
  }

  function renderRack(r?: RackDTO): void {
    if (!r) return;
    root.append(
      kv("Block", r.block_id),
      kv("Power domain", r.power_domain_id),
      kv("Cooling zone", r.cooling_zone_id),
      kv("Designed GPU cap", String(r.designed_gpu_cap)),
      kv("Rack power", `${r.rack_power_kw.toFixed(1)} kW`),
      kv("Power headroom", `${r.rack_headroom_kw.toFixed(1)} kW`),
      kv("Thermal margin", `${r.inferred_thermal_margin_c.toFixed(1)} °C`),
      kv("Hotspot risk (learned)", (r.hotspot_score ?? 0).toFixed(2)),
    );
  }

  function renderNode(n?: NodeDTO): void {
    if (!n) return;
    root.append(
      kv("Rack", n.rack_id),
      kv("Switch", n.switch_id),
      kv("Flavor", n.flavor_id),
      kv("GPU cap", String(n.gpu_cap)),
      kv("Running pods", String(n.running_pods)),
      kv("Ready", String(n.ready)),
      kv("Schedulable", String(n.schedulable)),
      kv("Health", n.node_health),
    );
  }

  function renderGpu(g?: GPUDTO): void {
    if (!g) return;
    root.append(
      kv("Node", g.node_uid),
      kv("Allocated to", g.allocated_to ?? "free"),
      kv("Util", `${Math.round(g.util * 100)}%`),
      kv("Temp", `${g.temp_c.toFixed(1)} °C`),
      kv("Power", `${Math.round(g.power_w)} W`),
      kv("Xid", g.xid_code ? String(g.xid_code) : "none"),
      kv("Health (filtered)", g.health_state),
    );
  }

  function renderWorkload(w?: WorkloadDTO): void {
    if (!w) return;
    root.append(
      kv("Tenant", w.tenant_id),
      kv("Local queue", w.local_queue_id),
      kv("Cluster queue", w.cluster_queue_id),
      kv("Priority", String(w.priority)),
      kv("GPU ask", String(w.gpu_req)),
      kv("Topology policy", w.topology_policy),
      kv("Comm class", w.comm_class),
      kv("Status", w.status),
      kv("Age", fmtSeconds(w.age_s)),
      kv("Est remaining", fmtSeconds(w.est_remaining_s)),
      kv("Ready fraction", `${Math.round(w.ready_fraction * 100)}%`),
      kv("Admitted domain", w.admitted_domain_id ?? "—"),
    );
  }

  function renderDomain(domainId: string, r?: RackDTO): void {
    const d = store.get().snapshot?.entities.topology_domains.find((x) => x.domain_id === domainId);
    if (d) {
      root.append(
        kv("Level", d.level),
        kv("Parent", d.parent_domain_id ?? "—"),
        kv("Free GPU", String(d.free_gpu)),
        kv("Headroom score", d.headroom_score.toFixed(2)),
        kv("Hotspot score", d.hotspot_score.toFixed(2)),
      );
    }
    if (r) root.append(kv("Backing rack", r.rack_id));
  }

  return {
    root,
    update(s: StoreState) {
      const open = !!s.selection;
      root.classList.toggle("open", open);
      if (open) render(s);
    },
  };
}
