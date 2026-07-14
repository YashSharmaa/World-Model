// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio (React overlay)
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//  Pending workload strip (left). Selecting a card focuses that decision target;
//  the focused card exposes its kernel-safe action alphabet inline.

import { Lock, Play } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { QueueDTO, WorkloadDTO } from "../../api/client";
import { fmtSeconds } from "../format";
import { actionLabel } from "../labels";
import { useStore, useStudio } from "../studio-context";

export function PendingWorkloads() {
  const s = useStore();
  const pending = (s.snapshot?.entities.workloads ?? []).filter((w) => w.status === "pending");
  const queues = s.snapshot?.entities.queues ?? [];

  return (
    <Card className="panel-surface pointer-events-auto absolute left-3 top-[84px] z-20 max-h-[46vh] w-72 overflow-auto border-0 p-0">
      <CardHeader className="pb-2">
        <CardTitle>Pending Workloads</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {pending.length === 0 ? (
          <p className="text-[11px] text-muted-foreground">No pending workloads.</p>
        ) : (
          pending.map((wl) => (
            <WorkloadCard
              key={wl.workload_uid}
              wl={wl}
              focused={wl.workload_uid === s.focusedWorkload}
              queue={queues.find((q) => q.name === wl.local_queue_id) ?? null}
            />
          ))
        )}
      </CardContent>
    </Card>
  );
}

function WorkloadCard({
  wl,
  focused,
  queue,
}: {
  wl: WorkloadDTO;
  focused: boolean;
  queue: QueueDTO | null;
}) {
  const s = useStore();
  const { store } = useStudio();
  const cmp = focused ? s.compare : null;

  return (
    <div
      onClick={() => store.setFocusedWorkload(wl.workload_uid)}
      className={cn(
        "cursor-pointer rounded-md border border-border bg-card/60 p-2.5 transition-colors",
        focused && "border-grounded ring-1 ring-grounded",
      )}
    >
      <div className="flex justify-between font-semibold">
        <span>{wl.workload_uid}</span>
        <span>{wl.gpu_req} GPU</span>
      </div>
      <div className="mt-0.5 text-[11px] text-muted-foreground">
        {wl.tenant_id} · {wl.topology_policy} · {wl.comm_class} · age {fmtSeconds(wl.age_s)}
      </div>
      <div className="text-[11px] text-muted-foreground">
        <span className="font-semibold text-foreground">priority {wl.priority}</span>
        {queue
          ? ` · queue ${wl.local_queue_id}: ${queue.pending_ct} pending (${queue.pending_gpu} GPU) · wait p95 ${fmtSeconds(queue.wait_time_p95_s)}`
          : ` · queue ${wl.local_queue_id}`}
      </div>

      <div className="mt-2 flex flex-wrap gap-1.5" onClick={(e) => e.stopPropagation()}>
        <Button
          size="xs"
          variant="secondary"
          onClick={() => {
            store.setFocusedWorkload(wl.workload_uid);
            void store.runCompare();
          }}
        >
          Compare actions
        </Button>
        {focused &&
          cmp &&
          [...cmp.candidates]
            .sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99))
            .map((cand) => (
              <Button
                key={cand.action_id}
                size="xs"
                variant={cand.action_id === cmp.recommended_action_id ? "default" : "secondary"}
                className={cn(!cand.safe && "text-blocked")}
                onClick={() =>
                  cand.safe
                    ? store.playBranch(cand.action_id)
                    : void store.showWhyNot(cand.action_id)
                }
              >
                {cand.safe ? <Play className="size-3" /> : <Lock className="size-3" />}
                {actionLabel(cand.action_id)}
              </Button>
            ))}
      </div>
    </div>
  );
}

export default PendingWorkloads;
