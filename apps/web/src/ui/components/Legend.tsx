// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio (React overlay)
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//  The persistent legend (doc G): a self-captioning key so the muted scene reads
//  on its own. Colours match the scene palette + the Tailwind semantic tokens.

import { Card, CardContent } from "@/components/ui/card";

const ITEMS: Array<[string, string]> = [
  ["Now (grounded)", "var(--grounded)"],
  ["Predicted", "var(--predicted)"],
  ["Alternative action", "var(--alt)"],
  ["Blocked (exact)", "var(--blocked)"],
  ["Low confidence", "var(--lowconf)"],
];

export function Legend() {
  return (
    <Card className="panel-surface pointer-events-auto w-full border-0">
      <CardContent className="flex flex-col gap-1.5 p-3.5">
        {ITEMS.map(([label, color]) => (
          <div key={label} className="flex items-center gap-2 text-[12px]">
            <span className="size-3.5 rounded-[4px]" style={{ background: color }} />
            <span>{label}</span>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export default Legend;
