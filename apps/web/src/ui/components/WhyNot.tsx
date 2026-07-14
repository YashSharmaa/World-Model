// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio (React overlay)
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//  The why-not card: explains a REJECTED action in one screenful, keeping the two
//  reason families separate — exact blockers (kernel facts) vs predicted risks
//  (calibrated learned forecasts). "Safety is a GATE, not a weighted metric."

import * as React from "react";
import { X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { WhyNotDTO } from "../../api/client";
import { useStore, useStudio } from "../studio-context";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <>
      <div className="mt-2.5 text-[11px] uppercase tracking-[0.05em] text-muted-foreground">{title}</div>
      {children}
    </>
  );
}

export function WhyNot() {
  const s = useStore();
  const { store, api } = useStudio();
  const actionId = s.whyNotActionId;
  const [data, setData] = React.useState<WhyNotDTO | null>(null);

  React.useEffect(() => {
    if (!actionId || !s.snapshot) {
      setData(null);
      return;
    }
    let alive = true;
    const candidate =
      store.candidateAction(actionId) ?? {
        action_id: actionId,
        type: "admit_and_place" as const,
        params: { workload_uid: s.focusedWorkload ?? undefined },
      };
    void api.whyNot({ state_hash: s.snapshot.state_hash, candidate_action: candidate }).then((w) => {
      if (alive) setData(w);
    });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [actionId, s.snapshot?.state_hash]);

  if (!actionId || !data) return null;

  return (
    <Card className="panel-surface pointer-events-auto absolute right-3 top-[84px] z-40 max-h-[64vh] w-[22rem] overflow-auto border border-blocked p-0">
      <CardHeader className="flex-row items-center justify-between pb-2">
        <CardTitle className="text-blocked">Why not — {data.action_id}</CardTitle>
        <Button size="icon" variant="ghost" className="size-6" onClick={() => store.clearWhyNot()}>
          <X className="size-4" />
        </Button>
      </CardHeader>
      <CardContent className="text-[12px]">
        <Section title="Exact blockers (kernel gate)">
          {data.exact_blockers.length === 0 && <div className="text-muted-foreground">None — action is feasible.</div>}
          {data.exact_blockers.map((b, i) => (
            <div key={i} className="my-1.5 border-l-[3px] border-blocked pl-2">
              <div className="text-[10px] uppercase tracking-[0.05em] text-blocked">{b.group.replace(/_/g, " ")}</div>
              <div>{b.detail}</div>
              {Object.keys(b.deficit).length > 0 && (
                <div className="text-[11px] text-muted-foreground">
                  deficit → {Object.entries(b.deficit).map(([k, v]) => `${k}: ${v}`).join(", ")}
                </div>
              )}
            </div>
          ))}
        </Section>

        {data.minimal_groups.length > 0 && (
          <Section title="Minimal certificate">
            <div>{data.minimal_groups.map((g) => g.replace(/_/g, " ")).join(" + ")}</div>
          </Section>
        )}

        {data.predicted_risks.length > 0 && (
          <Section title="Predicted risks (learned)">
            {data.predicted_risks.map((r, i) => (
              <div key={i}>• {r}</div>
            ))}
          </Section>
        )}

        {data.policy_reasons.length > 0 && (
          <Section title="Policy">
            {data.policy_reasons.map((r, i) => (
              <div key={i}>• {r}</div>
            ))}
          </Section>
        )}

        {data.rejected_domain_ids.length > 0 && (
          <div className="mt-2 text-[11px] text-muted-foreground">
            Rejected domains: {data.rejected_domain_ids.join(", ")}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default WhyNot;
