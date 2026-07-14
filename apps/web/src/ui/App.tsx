// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio (React overlay)
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//  The 2D overlay root. A full-screen, pointer-events:none layer over the
//  three.js canvas; each panel re-enables pointer events so clicks in the gaps
//  fall through to the scene (orbit / raycast). Layout is region-based:
//  header (top), workloads (left), recommendation/why-not/calibration (right),
//  controls+legend (bottom-left, stacked so they never overlap), delta ribbon +
//  timeline (bottom), inspect drawer (right slide-over).

import { Controls } from "./components/Controls";
import { DeltaRibbon } from "./components/DeltaRibbon";
import { Header } from "./components/Header";
import { InspectDrawer } from "./components/InspectDrawer";
import { Legend } from "./components/Legend";
import { PendingWorkloads } from "./components/PendingWorkloads";
import { Recommendation } from "./components/Recommendation";
import { ReplayCalibration } from "./components/ReplayCalibration";
import { Timeline } from "./components/Timeline";
import { WhyNot } from "./components/WhyNot";
import { useBeat } from "./studio-context";

function BeatCaption() {
  const text = useBeat();
  if (!text) return null;
  return (
    <div className="panel-surface pointer-events-none absolute left-1/2 top-[92px] z-20 max-w-[60vw] -translate-x-1/2 rounded-lg px-3.5 py-1.5 text-center text-[13px] font-semibold">
      {text}
    </div>
  );
}

function Footer() {
  return (
    <a
      className="pointer-events-auto absolute bottom-1.5 right-4 z-20 text-[11px] text-muted-foreground hover:text-primary"
      href="https://www.linkedin.com/in/yashsharmaa/"
      target="_blank"
      rel="noopener noreferrer"
    >
      Built by Yash Sharma — linkedin.com/in/yashsharmaa
    </a>
  );
}

export function App() {
  return (
    <div className="pointer-events-none fixed inset-0 z-10 text-foreground">
      <Header />
      <PendingWorkloads />
      <Recommendation />
      <WhyNot />
      <ReplayCalibration />
      <InspectDrawer />

      {/* bottom-left: controls over legend, stacked in one column (no overlap) */}
      <div className="pointer-events-none absolute bottom-[84px] left-3 z-20 flex w-64 flex-col gap-2">
        <Controls />
        <Legend />
      </div>

      <DeltaRibbon />
      <Timeline />
      <BeatCaption />
      <Footer />
    </div>
  );
}

export default App;
