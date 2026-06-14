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
import type { Store, StoreState } from "../state/store";
import type { GPUDTO, NodeDTO, RackDTO, WorkloadDTO } from "../api/client";

export function createInspectDrawer(parent: HTMLElement, store: Store): PanelHandle {
  const root = el("div", "panel inspect-drawer");
  parent.appendChild(root);

  function kv(label: string, value: string): HTMLElement {
    const row = el("div", "kv");
    row.append(el("b", "", label), el("span", "", value));
    return row;
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
