// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Tiny DOM helpers + the panel stylesheet (injected once). The heavy UI lives
//  in regular HTML overlay panels rather than the scene graph, per doc G.

import type { StoreState } from "../state/store";

/** Every panel exposes a root element and a store-driven update(). */
export interface PanelHandle {
  root: HTMLElement;
  update(s: StoreState): void;
}

/** Create an element with a class and optional text. */
export function el(tag: string, className = "", text = ""): HTMLElement {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

/** Create a clickable chip button. */
export function chip(label: string, onClick: () => void, className = ""): HTMLButtonElement {
  const b = document.createElement("button");
  b.className = `chip ${className}`.trim();
  b.textContent = label;
  b.addEventListener("click", onClick);
  return b;
}

/** Format seconds as a compact mm:ss / +Ns label. */
export function fmtSeconds(s: number): string {
  const sign = s < 0 ? "-" : "+";
  const a = Math.abs(Math.round(s));
  if (a < 60) return `${sign}${a}s`;
  const m = Math.floor(a / 60);
  const r = a % 60;
  return `${sign}${m}m${r ? `${r}s` : ""}`;
}

/** Format a delta with a leading sign and unit. */
export function fmtDelta(v: number, unit = "", digits = 1): string {
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(digits)}${unit}`;
}

/** Inject the overlay stylesheet exactly once. */
export function injectStyles(): void {
  if (document.getElementById("studio-styles")) return;
  const style = document.createElement("style");
  style.id = "studio-styles";
  style.textContent = CSS;
  document.head.appendChild(style);
}

const CSS = `
.panel {
  position: absolute;
  background: var(--panel);
  border: 1px solid var(--stroke);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  backdrop-filter: blur(8px);
  color: var(--text);
  font-size: 12.5px;
  line-height: 1.45;
  padding: 12px 14px;
}
.panel-title {
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 8px;
}
.mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; }

/* legend (bottom-left) */
.legend { left: 18px; bottom: 92px; width: 196px; }
.legend-list { display: grid; gap: 6px; }
.legend-row { display: flex; align-items: center; gap: 8px; }
.legend-swatch { width: 14px; height: 14px; border-radius: 4px; flex: none; }
.legend-label { color: var(--text); }

/* provenance header (top, full width) */
.provenance {
  top: 14px; left: 18px; right: 18px;
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 16px;
}
.prov-left { display: flex; flex-direction: column; }
.prov-brand { font-weight: 600; font-size: 14px; }
.prov-sub { color: var(--muted); font-size: 11px; }
.prov-right { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.prov-health { font-weight: 600; }
.prov-chip {
  background: var(--chip-bg);
  border: 1px solid var(--stroke);
  border-radius: 999px;
  padding: 3px 9px;
  color: var(--muted);
  font-size: 11px;
}
/* light/dark theme toggle — lives INSIDE the header (prov-right) */
.prov-toggle {
  width: 26px; height: 26px; border-radius: 999px; flex: none;
  display: grid; place-items: center; cursor: pointer; line-height: 1;
  background: var(--chip-bg); border: 1px solid var(--stroke);
  color: var(--text); font-size: 13px;
}
.prov-toggle:hover { border-color: var(--accent); }

/* mode tabs — centered nav INTEGRATED into the header band (not a stuck pill).
   Transparent so it reads as part of the top bar; overrides the .panel look. */
.mode-tabs {
  top: 17px; left: 50%; transform: translateX(-50%);
  display: flex; gap: 4px; padding: 0;
  background: transparent; border: none; box-shadow: none; backdrop-filter: none;
}
.mode-tab {
  background: transparent; color: var(--muted);
  border: 1px solid transparent; border-radius: 8px;
  padding: 6px 12px; font-size: 12px; cursor: pointer;
}
.mode-tab.active { background: var(--accent-soft); color: var(--text); border-color: var(--stroke); }
.mode-tab:hover { color: var(--text); }

/* workload queue strip (left) */
.queue-strip { left: 18px; top: 116px; width: 248px; max-height: 46vh; overflow:auto; }
.wl-card {
  border: 1px solid var(--stroke); border-radius: 8px;
  padding: 9px 10px; margin-bottom: 8px; cursor: pointer;
  background: var(--surface-2);
}
.wl-card.focused { border-color: var(--grounded); box-shadow: inset 0 0 0 1px var(--grounded); }
.wl-card-title { display:flex; justify-content: space-between; font-weight:600; }
.wl-meta { color: var(--muted); font-size: 11px; margin-top: 2px; }
.wl-chips { display:flex; gap:6px; margin-top:8px; flex-wrap:wrap; }

/* recommendation panel (right) */
.recommendation { right: 18px; top: 116px; width: 312px; max-height: 60vh; overflow:auto; }
.rec-best {
  border: 1px solid var(--safe); border-radius: 8px;
  padding: 9px 10px; margin-bottom: 10px;
  background: var(--safe-bg);
}
.rec-row {
  display:flex; align-items:center; justify-content: space-between;
  border: 1px solid var(--stroke); border-radius: 8px;
  padding: 8px 10px; margin-bottom: 7px;
}
.rec-row.blocked { border-color: var(--blocked); }
.rec-row.recommended { border-color: var(--safe); }
.rec-row.abstain { border-color: var(--lowconf); border-style: dashed; }
.rec-rank { font-weight:700; width: 22px; }
.rec-actions { display:flex; gap:6px; }
.rec-kpis { display:grid; grid-template-columns: 1fr 1fr; gap:4px 10px; margin-top:6px; font-size:11px; color:var(--muted); }
.badge { border-radius: 999px; padding: 2px 7px; font-size: 10px; }
.badge.safe { background: var(--safe-bg); color: var(--safe); }
.badge.blocked { background: var(--blocked-bg); color: var(--blocked); }
.badge.abstain { background: var(--chip-bg); color: var(--lowconf); }

/* why-not card (centered modal-ish) */
.why-not {
  right: 18px; top: 116px; width: 340px;
  border-color: var(--blocked);
}
.why-not .panel-title { color: var(--blocked); }
.blocker { border-left: 3px solid var(--blocked); padding: 4px 0 4px 8px; margin: 6px 0; }
.blocker-detail { font-size: 11px; color: var(--text); }
.blocker-group { font-size: 10px; text-transform: uppercase; letter-spacing: .05em; color: var(--blocked); }
.why-section-title { font-size: 11px; color: var(--muted); margin-top: 10px; text-transform: uppercase; letter-spacing:.05em; }
.close-x { float:right; cursor:pointer; color: var(--muted); border:none; background:none; font-size:16px; }

/* delta ribbon (bottom-center, above timeline) */
.delta-ribbon {
  left: 50%; transform: translateX(-50%); bottom: 92px;
  display:flex; gap: 10px; padding: 10px 14px; align-items: stretch;
}
.delta-col { min-width: 120px; }
.delta-col-title { font-size:11px; color: var(--muted); margin-bottom: 6px; display:flex; justify-content:space-between; }
.delta-metric { display:flex; justify-content: space-between; font-size: 11.5px; padding: 2px 0; }
.delta-good { color: var(--safe); }
.delta-bad { color: var(--blocked); }

/* timeline (bottom, full width) */
.timeline {
  left: 18px; right: 18px; bottom: 36px;
  display:flex; align-items:center; gap: 12px; padding: 8px 14px;
}
.tl-btn { background: var(--chip-bg); border:1px solid var(--stroke); color: var(--text); border-radius:6px; padding:5px 10px; cursor:pointer; }
.tl-track { position: relative; flex:1; height: 26px; }
.tl-rail { position:absolute; top:12px; left:0; right:0; height:3px; background: var(--rail); border-radius:2px; }
.tl-fill { position:absolute; top:12px; left:0; height:3px; background: var(--grounded); border-radius:2px; }
.tl-cursor { position:absolute; top:6px; width:3px; height:14px; background: var(--alt); transform: translateX(-1px); }
.tl-pin { position:absolute; top:4px; width:10px; height:10px; border-radius:50%; background: var(--predicted); transform: translate(-5px,0); cursor:pointer; border:1px solid var(--panel-solid); }
.tl-pin.realized { background: var(--blocked); }
.tl-time { color: var(--muted); font-size: 11px; min-width: 86px; text-align:right; }
.tl-toggle { display:flex; gap:6px; }

/* inspect drawer (right slide-over) */
.inspect-drawer {
  right: -360px; top: 0; bottom: 0; width: 340px;
  border-radius: 0; border-top: none; border-bottom: none; border-right:none;
  transition: right 220ms ease; padding-top: 70px; overflow:auto;
}
.inspect-drawer.open { right: 0; }
.kv { display:flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dashed var(--stroke); font-size: 12px; }
.kv b { color: var(--muted); font-weight: 500; }

/* footer credit (bottom-right) */
.footer-credit { position:absolute; right: 18px; bottom: 10px; font-size: 11px; }
.footer-link { color: var(--muted); }
.footer-link:hover { color: var(--accent); }

/* chips */
.chip {
  background: var(--chip-bg);
  border: 1px solid var(--stroke);
  border-radius: 999px;
  color: var(--text);
  font-size: 11px; padding: 4px 10px; cursor: pointer;
}
.chip:hover { border-color: var(--accent); }
.chip.active { background: var(--accent-soft); border-color: var(--accent); }
.chip.danger { color: var(--blocked); border-color: rgba(255,107,122,0.4); }

/* play-demo control — runs the 8-beat golden-path storyboard on demand.
   Sits top-center under the mode tabs so it never collides with the side panels. */
.play-demo {
  position: absolute;
  top: 58px; left: 50%; transform: translateX(-50%);
  z-index: 5;
  background: var(--accent-soft); border-color: var(--accent);
  font-weight: 600; padding: 5px 14px;
}
.play-demo:hover { border-color: var(--accent); filter: brightness(1.08); }

/* caption (self-captioning beat text, bottom-center top of timeline area) */
.beat-caption {
  left: 50%; transform: translateX(-50%); top: 108px;
  background: var(--panel-strong); border:1px solid var(--stroke);
  box-shadow: var(--shadow); backdrop-filter: blur(8px);
  border-radius: 8px; padding: 6px 14px; font-size: 13px; font-weight: 600;
  color: var(--text); pointer-events: none; max-width: 60vw; text-align:center;
}
.hidden { display: none !important; }
`;
