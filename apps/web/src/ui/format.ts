// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio (React overlay)
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//  Compact value formatters shared by the overlay panels (ported from the
//  original vanilla DOM helpers so labels read identically).

/** Format seconds as a compact +Ns / +Nm[Ms] label with a leading sign. */
export function fmtSeconds(s: number): string {
  const sign = s < 0 ? "-" : "+";
  const a = Math.abs(Math.round(s));
  if (a < 60) return `${sign}${a}s`;
  const m = Math.floor(a / 60);
  const r = a % 60;
  return `${sign}${m}m${r ? `${r}s` : ""}`;
}

/** Format a numeric delta with a leading sign + unit. */
export function fmtDelta(v: number, unit = "", digits = 1): string {
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(digits)}${unit}`;
}

/** Format seconds as a m:ss clock (timeline). */
export function fmtClock(s: number): string {
  const m = Math.floor(s / 60);
  const r = Math.round(s % 60);
  return `${m}:${r.toString().padStart(2, "0")}`;
}
