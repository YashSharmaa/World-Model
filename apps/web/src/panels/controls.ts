// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Scene controls strip (doc G state contract + GET /replays + domain layers):
//
//    - Horizon selector : chips built from `snapshot.horizon_options_s`. Picking
//      one calls store.setHorizonAndCompare(), re-running the comparison so the
//      rollouts/deltas/uncertainty refresh for the chosen 5-60 min window.
//    - Replay picker    : chips built from the live GET /replays catalog so the
//      SECOND recorded scenario is actually loadable (not just the auto-loaded
//      first one). Picking one calls store.loadReplay().
//    - Layer toggles    : topology / power / cooling read-only context layers,
//      each wired to setDomainLayerVisibility via the supplied callback.
//
//  These are operator affordances, not part of the muted self-captioning grammar,
//  so they sit in their own compact strip and never fight the storyboard panels.

import { el, type PanelHandle } from "./dom";
import type { Store, StoreState } from "../state/store";
import type { WorldModelClient, ReplayIndexEntry } from "../api/client";

export type DomainLayer = "topology" | "power" | "cooling";

export interface ControlsHandle extends PanelHandle {
  /** current visibility of each read-only context layer. */
  getLayerVisibility(): Record<DomainLayer, boolean>;
}

export function createControls(
  parent: HTMLElement,
  store: Store,
  client: WorldModelClient,
  onToggleLayer: (layer: DomainLayer, visible: boolean) => void,
): ControlsHandle {
  const root = el("div", "panel controls");

  // ── horizon selector ──────────────────────────────────────────────────
  const horizonGroup = el("div", "control-group");
  horizonGroup.appendChild(el("div", "control-label", "Horizon"));
  const horizonChips = el("div", "control-chips");
  horizonGroup.appendChild(horizonChips);

  // ── replay scenario picker ────────────────────────────────────────────
  const replayGroup = el("div", "control-group");
  replayGroup.appendChild(el("div", "control-label", "Replay scenario"));
  const replayChips = el("div", "control-chips");
  replayGroup.appendChild(replayChips);

  // ── layer toggles ─────────────────────────────────────────────────────
  const layerGroup = el("div", "control-group");
  layerGroup.appendChild(el("div", "control-label", "Context layers"));
  const layerChips = el("div", "control-chips");
  const layers: DomainLayer[] = ["topology", "power", "cooling"];
  const layerVisible: Record<DomainLayer, boolean> = {
    topology: true,
    power: true,
    cooling: true,
  };
  const layerButtons = new Map<DomainLayer, HTMLButtonElement>();
  for (const layer of layers) {
    const b = el("button", "chip active", capitalize(layer)) as HTMLButtonElement;
    b.addEventListener("click", () => {
      layerVisible[layer] = !layerVisible[layer];
      b.classList.toggle("active", layerVisible[layer]);
      onToggleLayer(layer, layerVisible[layer]);
    });
    layerButtons.set(layer, b);
    layerChips.appendChild(b);
  }
  layerGroup.appendChild(layerChips);

  root.append(horizonGroup, replayGroup, layerGroup);
  parent.appendChild(root);

  // Replay catalog is loaded ONCE from GET /replays so both recorded scenarios
  // are selectable; it is independent of the live snapshot stream.
  let replayCatalog: ReplayIndexEntry[] = [];
  void client.listReplays().then((idx) => {
    replayCatalog = idx.scenarios;
    renderReplayChips();
  });

  function renderReplayChips(): void {
    replayChips.innerHTML = "";
    const activeId = store.get().replay?.scenario_id ?? null;
    for (const sc of replayCatalog) {
      const b = el(
        "button",
        `chip ${sc.scenario_id === activeId ? "active" : ""}`,
        sc.title,
      ) as HTMLButtonElement;
      b.title = `${sc.description} · ${Math.round(sc.duration_s / 60)} min`;
      b.addEventListener("click", () => void store.loadReplay(sc.scenario_id));
      replayChips.appendChild(b);
    }
  }

  let lastHorizonOpts = "";
  let lastHorizon = -1;
  let lastReplayId: string | null = null;

  return {
    root,
    getLayerVisibility: () => ({ ...layerVisible }),
    update(s: StoreState) {
      // Horizon chips track the snapshot's declared options + the active value.
      const opts = s.snapshot?.horizon_options_s ?? [];
      const optsKey = opts.join(",");
      if (optsKey !== lastHorizonOpts || s.horizonS !== lastHorizon) {
        lastHorizonOpts = optsKey;
        lastHorizon = s.horizonS;
        horizonChips.innerHTML = "";
        for (const h of opts) {
          const b = el(
            "button",
            `chip ${h === s.horizonS ? "active" : ""}`,
            `${Math.round(h / 60)}m`,
          ) as HTMLButtonElement;
          b.addEventListener("click", () => void store.setHorizonAndCompare(h));
          horizonChips.appendChild(b);
        }
      }
      // Re-mark the active replay chip when the loaded scenario changes.
      const replayId = s.replay?.scenario_id ?? null;
      if (replayId !== lastReplayId) {
        lastReplayId = replayId;
        renderReplayChips();
      }
    },
  };
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
